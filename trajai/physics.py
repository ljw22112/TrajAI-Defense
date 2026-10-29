"""Flight dynamics for TrajAI-Defense.

3-DOF point-mass model in a local flat-earth ENU frame (x: downrange, y: crossrange, z: altitude).
All functions are vectorised over a leading batch dimension so that thousands of trajectories
can be integrated at once with NumPy.

Models
------
* gravity        g(h) = g0 * (Re / (Re + h))^2
* density        rho(h) = rho0 * exp(-h / H)                    (exponential fit)
* speed of sound a(h) = sqrt(gamma * R * T(h)),  T(h) from U.S. Standard Atmosphere 1976 (0-51 km)
* drag           C_D(M) / C_D,ref : subsonic plateau, transonic rise, supersonic decay
* ballistic      dv/dt = -rho |v| v / (2 beta) * C_D(M)/C_D,ref - g e_z,   beta = m / (C_D,ref A)
* powered        m dv/dt = T v_hat - 1/2 rho C_D A |v| v - m g e_z,  dm/dt = -T / (g0 Isp)
"""
from __future__ import annotations

import numpy as np

G0 = 9.80665          # m/s^2
RE = 6_371_000.0      # m
RHO0 = 1.225          # kg/m^3
HS = 8_500.0          # m, density scale height
GAMMA_AIR = 1.4
R_AIR = 287.053       # J/(kg K)


# ----------------------------------------------------------------------------- environment
def gravity(h):
    r = RE / (RE + np.asarray(h, dtype=float))
    return G0 * r * r


def density(h):
    return RHO0 * np.exp(-np.maximum(h, 0.0) / HS)


def temperature(h):
    """USSA-1976 temperature profile up to 51 km, isothermal above."""
    h = np.maximum(np.asarray(h, dtype=float), 0.0)
    return np.select(
        [h < 11_000, h < 20_000, h < 32_000, h < 47_000],
        [288.15 - 0.0065 * h, 216.65 + 0.0 * h, 216.65 + 0.001 * (h - 20_000), 228.65 + 0.0028 * (h - 32_000)],
        default=270.65,
    )


def speed_of_sound(h):
    return np.sqrt(GAMMA_AIR * R_AIR * temperature(h))


def cd_ratio(mach):
    """C_D(M) / C_D,ref. Continuous, piecewise: 0.75 (M<0.8) -> 1.35 (M=1.2) -> 0.8 (M>>1)."""
    m = np.asarray(mach, dtype=float)
    return np.where(
        m < 0.8, 0.75,
        np.where(m < 1.2, 0.75 + (m - 0.8) / 0.4 * 0.6, 0.8 + 0.55 * np.exp(-(m - 1.2) / 1.2)),
    )


# ----------------------------------------------------------------------------- dynamics
def ballistic_accel(pos, vel, beta):
    """Acceleration of an unpowered body. pos, vel: (..., 3); beta: (...,) [kg/m^2]."""
    h = pos[..., 2]
    v = np.linalg.norm(vel, axis=-1)
    safe_v = np.maximum(v, 1e-6)
    k = density(h) * v * cd_ratio(safe_v / speed_of_sound(h)) / (2.0 * beta)
    acc = -k[..., None] * vel
    acc[..., 2] -= gravity(h)
    return acc


def _deriv(state, beta):
    out = np.empty_like(state)
    out[..., :3] = state[..., 3:6]
    out[..., 3:6] = ballistic_accel(state[..., :3], state[..., 3:6], beta)
    return out


def rk4_step(state, beta, dt):
    """One RK4 step of the ballistic model. state: (..., 6) = [r, v]."""
    k1 = _deriv(state, beta)
    k2 = _deriv(state + 0.5 * dt * k1, beta)
    k3 = _deriv(state + 0.5 * dt * k2, beta)
    k4 = _deriv(state + dt * k3, beta)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def propagate_to_ground(state0, beta, dt=0.25, t0=0.0, t_max=1200.0, record_every=None):
    """Integrate a batch of ballistic states until each reaches z = 0.

    Returns dict with impact time 't', impact position 'r' (N, 3) and, if record_every is given,
    a list of recorded (t, positions(N,3), alive mask) snapshots.
    """
    s = np.array(state0, dtype=float, copy=True)
    beta = np.broadcast_to(np.asarray(beta, dtype=float), s.shape[:-1]).copy()
    n = s.shape[0]
    alive = s[:, 2] > 0
    t_imp = np.full(n, np.nan)
    r_imp = np.full((n, 3), np.nan)
    t = t0
    snaps = []
    step = 0
    while alive.any() and t < t_max:
        idx = np.nonzero(alive)[0]
        prev = s[idx]
        new = rk4_step(prev, beta[idx], dt)
        hit = new[:, 2] <= 0
        if hit.any():
            f = prev[hit, 2] / (prev[hit, 2] - new[hit, 2])
            hi = idx[hit]
            t_imp[hi] = t + f * dt
            r_imp[hi] = prev[hit, :3] + f[:, None] * (new[hit, :3] - prev[hit, :3])
            r_imp[hi, 2] = 0.0
            alive[hi] = False
        s[idx] = new
        t += dt
        step += 1
        if record_every and step % record_every == 0:
            snaps.append((t, s[:, :3].copy(), alive.copy()))
    # anything still alive at t_max: report last state
    left = np.isnan(t_imp)
    t_imp[left] = t
    r_imp[left] = s[left, :3]
    return {"t": t_imp, "r": r_imp, "snaps": snaps}


def simulate_ballistic(state0, beta, dt=0.1, out_dt=0.5, t_max=1200.0):
    """Integrate a batch of ballistic trajectories and sample them every out_dt seconds.

    Returns times (K,), positions (N, K, 3) with NaN after impact, impact time (N,), impact point (N, 3).
    """
    s = np.array(state0, dtype=float, copy=True)
    n = s.shape[0]
    beta = np.broadcast_to(np.asarray(beta, dtype=float), (n,)).copy()
    sub = int(round(out_dt / dt))
    alive = np.ones(n, bool)
    t_imp = np.full(n, np.nan)
    r_imp = np.full((n, 3), np.nan)
    pos = [s[:, :3].copy()]
    vel = [s[:, 3:].copy()]
    t = 0.0
    while alive.any() and t < t_max:
        for _ in range(sub):
            idx = np.nonzero(alive)[0]
            if idx.size == 0:
                break
            prev = s[idx]
            new = rk4_step(prev, beta[idx], dt)
            hit = new[:, 2] <= 0
            if hit.any():
                f = prev[hit, 2] / (prev[hit, 2] - new[hit, 2])
                hi = idx[hit]
                t_imp[hi] = t + f * dt
                r_imp[hi] = prev[hit, :3] + f[:, None] * (new[hit, :3] - prev[hit, :3])
                r_imp[hi, 2] = 0.0
                alive[hi] = False
            s[idx] = new
            t += dt
        p = s[:, :3].copy()
        v = s[:, 3:].copy()
        p[~alive] = np.nan
        v[~alive] = np.nan
        pos.append(p)
        vel.append(v)
    times = np.arange(len(pos)) * out_dt
    return times, np.stack(pos, 1), np.stack(vel, 1), t_imp, r_imp


def simulate_powered(v0, angle_deg, mass0, cd, area, burn_time, thrust, isp=220.0, dt=0.01, t_max=900.0):
    """Single powered + ballistic trajectory (boost phase included). Returns array [t, x, z, vx, vz, m].

    Replaces the explicit-Euler legacy generator with RK4 and a mass-flow law dm/dt = -T/(g0 Isp).
    """
    a = np.radians(angle_deg)
    y = np.array([0.0, 0.0, v0 * np.cos(a), v0 * np.sin(a), mass0])

    def f(t, y):
        x, z, vx, vz, m = y
        v = max(np.hypot(vx, vz), 1e-6)
        T = thrust if t <= burn_time else 0.0
        drag = 0.5 * density(z) * cd * float(cd_ratio(v / speed_of_sound(z))) / 0.3 * area * v * v
        ax = (T - drag) * vx / v / m
        az = (T - drag) * vz / v / m - float(gravity(z))
        mdot = -T / (G0 * isp)
        return np.array([vx, vz, ax, az, mdot])

    out = [[0.0, *y]]
    t = 0.0
    while t < t_max:
        k1 = f(t, y); k2 = f(t + dt / 2, y + dt / 2 * k1); k3 = f(t + dt / 2, y + dt / 2 * k2); k4 = f(t + dt, y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        if y[1] < 0:
            break
        out.append([t, *y])
    return np.array(out)
