"""Reduced-order 6-DOF truth: point-mass translation plus the angle-of-attack motion of a spinning re-entry body.

The 3-DOF model (trajai.physics) assumes the body flies nose-first (zero angle of attack). A real body separates with a
tip-off angle and cones about its velocity vector. Here the complex angle of attack  xi = alpha_y + i alpha_z  in the
plane transverse to the air-relative velocity follows the standard slowly-varying solution of the pitch/yaw equations:
  |xi| = alpha0 * min(1, sqrt(omega_c / omega)) * exp(-D),   dD/dt = zeta * omega,   d(phase)/dt = p + omega
where omega^2 = K_q * q (static stability scales with dynamic pressure q): the amplitude stays at alpha0 in vacuum,
shrinks as q grows (adiabatic invariant, |xi| ~ omega^-1/2) and is damped aerodynamically; the plane of the angle of
attack rotates with the roll rate p plus the pitch frequency. Aerodynamic effects of xi:
  drag   x (1 + k_D |xi|^2)          k_D = C_N_alpha / C_D0  (induced drag)
  lift   (rho V^2 C_D/(2 beta)) * k_D * |xi|  along the direction of xi in the transverse plane
A small body-fixed trim angle alpha_t (mass or shape asymmetry, ablation) adds  alpha_t * exp(i * roll angle): it rolls
with the body at rate p, so its lift averages out when the body spins fast and pushes the trajectory sideways when the
roll is slow. With alpha0 = alpha_t = 0 this reduces exactly to the 3-DOF ballistic model.
Parameters (per trajectory): alpha0 ~ U(2, 8) deg, alpha_t ~ U(0.2, 1.0) deg, p ~ U(0.1, 1.5) rev/s, pitch frequency
at q = 20 kPa ~ U(1.5, 4) Hz, zeta ~ U(0.03, 0.10), k_D = 6.
"""
from __future__ import annotations

import numpy as np

from . import physics

Q_REF = 20_000.0
OMEGA_C = 2 * np.pi * 0.2      # below this the restoring moment is negligible (vacuum coning)
K_D = 6.0


def sample(n, rng, alpha0_deg=(2.0, 8.0), trim_deg=(0.2, 1.0)):
    return dict(alpha0=np.radians(rng.uniform(*alpha0_deg, n)), trim=np.radians(rng.uniform(*trim_deg, n)),
                p=2 * np.pi * rng.uniform(0.1, 1.5, n), roll0=rng.uniform(0, 2 * np.pi, n),
                kq=(2 * np.pi * rng.uniform(1.5, 4.0, n)) ** 2 / Q_REF, zeta=rng.uniform(0.03, 0.10, n),
                phase0=rng.uniform(0, 2 * np.pi, n))


def _accel(s, beta, prm):
    """s: (N, 9) = [r, v, D, coning phase, roll angle]. Returns ds/dt."""
    pos, vel, D, ph, roll = s[:, :3], s[:, 3:6], s[:, 6], s[:, 7], s[:, 8]
    h = pos[:, 2]
    rho, a = physics.density(h), physics.speed_of_sound(h)
    V = np.maximum(np.linalg.norm(vel, axis=1), 1e-6)
    q = 0.5 * rho * V ** 2
    omega = np.sqrt(prm["kq"] * q)
    amp_c = prm["alpha0"] * np.minimum(1.0, np.sqrt(OMEGA_C / np.maximum(omega, 1e-9))) * np.exp(-D)
    xr = amp_c * np.cos(ph) + prm["trim"] * np.cos(roll)                 # complex angle of attack (coning + trim)
    xi = amp_c * np.sin(ph) + prm["trim"] * np.sin(roll)
    amp = np.hypot(xr, xi)
    k = rho * V * physics.cd_ratio(V / a) / (2.0 * beta)                   # |a_drag| / V for zero angle of attack
    vhat = vel / V[:, None]
    up = np.zeros_like(vel)
    up[:, 2] = 1.0
    e1 = up - np.sum(up * vhat, 1, keepdims=True) * vhat
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-6)
    e2 = np.cross(vhat, e1)
    acc = -(k * (1.0 + K_D * amp ** 2))[:, None] * vel
    lift = (k * V * K_D)[:, None] * (xr[:, None] * e1 + xi[:, None] * e2)
    acc = acc + lift
    acc[:, 2] -= physics.gravity(h)
    out = np.empty_like(s)
    out[:, :3] = vel
    out[:, 3:6] = acc
    out[:, 6] = prm["zeta"] * omega
    out[:, 7] = prm["p"] + omega
    out[:, 8] = prm["p"]
    return out


def simulate(state0, beta, prm, dt=0.02, out_dt=0.1, t_max=1200.0):
    """RK4 until every trajectory reaches z = 0. Returns times (K,), positions (N, K, 3) held after impact,
    impact time T (N,) and impact point (N, 3)."""
    n = state0.shape[0]
    s = np.zeros((n, 9))
    s[:, :6] = state0
    s[:, 7] = prm["phase0"]
    s[:, 8] = prm["roll0"]
    beta = np.asarray(beta, float)
    every = int(round(out_dt / dt))
    rec, times = [s[:, :3].copy()], [0.0]
    alive = np.ones(n, bool)
    T = np.full(n, np.nan)
    r_imp = np.full((n, 3), np.nan)
    t, step = 0.0, 0
    while alive.any() and t < t_max:
        idx = np.nonzero(alive)[0]
        p = {k: v[idx] for k, v in prm.items()}
        x, b = s[idx], beta[idx]
        k1 = _accel(x, b, p)
        k2 = _accel(x + 0.5 * dt * k1, b, p)
        k3 = _accel(x + 0.5 * dt * k2, b, p)
        k4 = _accel(x + dt * k3, b, p)
        new = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        hit = new[:, 2] <= 0
        if hit.any():
            f = x[hit, 2] / (x[hit, 2] - new[hit, 2])
            hi = idx[hit]
            T[hi] = t + f * dt
            r_imp[hi] = x[hit, :3] + f[:, None] * (new[hit, :3] - x[hit, :3])
            r_imp[hi, 2] = 0.0
            alive[hi] = False
            new[hit, :3] = r_imp[hi]
        s[idx] = new
        t += dt
        step += 1
        if step % every == 0:
            rec.append(s[:, :3].copy())
            times.append(t)
    return np.array(times), np.stack(rec, 1), T, r_imp
