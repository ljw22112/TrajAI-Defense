"""Batched EKF / UKF baselines with a 7-state model x = [r, v, ln beta] and polar radar measurements.

Every operation is vectorised over the batch of tracks, so a whole test split is filtered at once.
After the cut-off time the posterior N(x, P) is sampled and propagated to the ground (Monte Carlo)
to obtain the impact distribution, CEP50 and time of impact.
"""
from __future__ import annotations

import numpy as np

from . import physics, radar

NX = 7
LB0 = np.log(2450.0)       # prior mean of ln beta (geometric centre of the 600-10,000 kg/m^2 training range)
SLB0 = 1.0                 # prior std of ln beta
Q_ACC = 0.5                # white-noise acceleration PSD [m^2/s^3]
Q_LB = 2e-4                # ln beta random-walk PSD [1/s]
LB_MIN, LB_MAX = np.log(100.0), np.log(1e5)


def phi(x, dt, sub=2, atm=None):
    s = x[:, :6].copy()
    beta = np.exp(np.clip(x[:, 6], LB_MIN, LB_MAX))
    h = dt / sub
    for _ in range(sub):
        s = physics.rk4_step(s, beta, h, atm)
    return np.concatenate([s, x[:, 6:7]], 1)


def numerical_F(x, dt, fx, atm=None):
    n = x.shape[0]
    eps = np.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 1e-4])
    xp = np.repeat(x[None], NX, 0)                    # (7, N, 7)
    for j in range(NX):
        xp[j, :, j] += eps[j]
    fp = phi(xp.reshape(-1, NX), dt, atm=None if atm is None else atm.__class__(atm.bank, np.tile(atm.idx, NX))).reshape(NX, n, NX)
    F = (fp - fx[None]) / eps[:, None, None]          # (7=j, N, 7=i)
    return np.transpose(F, (1, 2, 0))                 # (N, i, j)


def Qd(dt):
    Q = np.zeros((NX, NX))
    a, b, c = Q_ACC * dt ** 3 / 3, Q_ACC * dt ** 2 / 2, Q_ACC * dt
    for k in range(3):
        Q[k, k], Q[k, k + 3], Q[k + 3, k], Q[k + 3, k + 3] = a, b, b, c
    Q[6, 6] = Q_LB * dt
    return Q


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _init(meas_c, dt, sigma_r, sigma_ang):
    c0, c1 = meas_c[:, 0], meas_c[:, 1]
    n = c0.shape[0]
    x = np.concatenate([c1, (c1 - c0) / dt, np.full((n, 1), LB0)], 1)
    rng = np.linalg.norm(c1 - radar.RADAR_SITE, axis=1)
    pv = sigma_r ** 2 + (rng * sigma_ang) ** 2
    P = np.zeros((n, NX, NX))
    for k in range(3):
        P[:, k, k] = pv
        P[:, k + 3, k + 3] = 2 * pv / dt ** 2 + 400.0
    P[:, 6, 6] = SLB0 ** 2
    return x, P


def _R(n, sigma_r, sigma_ang):
    """Per-track measurement covariance (N, 3, 3); sigma_* may be scalars or (N,) arrays."""
    sr = np.broadcast_to(np.asarray(sigma_r, float), (n,))
    sa = np.broadcast_to(np.asarray(sigma_ang, float), (n,))
    R = np.zeros((n, 3, 3))
    R[:, 0, 0], R[:, 1, 1], R[:, 2, 2] = sr ** 2, sa ** 2, sa ** 2
    return R


def ekf(meas_pol, meas_c, n_meas, dt, sigma_r, sigma_ang, atm=None):
    """Run the EKF on a batch. Returns x (N,7), P (N,7,7), mean NIS (N,), ln-beta history (N, K, 2).
    sigma_r / sigma_ang may be scalars or per-track arrays (N,)."""
    x, P = _init(meas_c, dt, sigma_r, sigma_ang)
    n = x.shape[0]
    Rall = _R(n, sigma_r, sigma_ang)
    Q = Qd(dt)
    nis_sum = np.zeros(n)
    nis_cnt = np.zeros(n)
    hist = np.full((n, meas_c.shape[1], 2), np.nan)
    I = np.eye(NX)
    for k in range(2, int(n_meas.max())):
        act = k < n_meas
        if not act.any():
            break
        xa, Pa, R = x[act], P[act], Rall[act]
        aa = None if atm is None else atm[act]
        xp = phi(xa, dt, atm=aa)
        F = numerical_F(xa, dt, xp, aa)
        Pp = F @ Pa @ np.transpose(F, (0, 2, 1)) + Q
        J = radar.polar_jacobian(xp[:, :3])
        H = np.concatenate([J, np.zeros((xa.shape[0], 3, 4))], 2)
        nu = meas_pol[act, k] - radar.to_polar(xp[:, :3])
        nu[:, 1:] = _wrap(nu[:, 1:])
        S = H @ Pp @ np.transpose(H, (0, 2, 1)) + R
        Si = np.linalg.inv(S)
        K = Pp @ np.transpose(H, (0, 2, 1)) @ Si
        xu = xp + np.einsum("nij,nj->ni", K, nu)
        xu[:, 6] = np.clip(xu[:, 6], LB_MIN, LB_MAX)
        IKH = I - K @ H
        Pu = IKH @ Pp @ np.transpose(IKH, (0, 2, 1)) + K @ R @ np.transpose(K, (0, 2, 1))
        Pu = 0.5 * (Pu + np.transpose(Pu, (0, 2, 1)))
        e = np.einsum("ni,nij,nj->n", nu, Si, nu)
        idx = np.nonzero(act)[0]
        late = k >= 6
        nis_sum[idx] += e * late
        nis_cnt[idx] += late
        x[act], P[act] = xu, Pu
        hist[idx, k, 0] = xu[:, 6]
        hist[idx, k, 1] = np.sqrt(Pu[:, 6, 6])
    return x, P, nis_sum / np.maximum(nis_cnt, 1), hist


def ukf(meas_pol, meas_c, n_meas, dt, sigma_r, sigma_ang, alpha=1e-1, beta_ut=2.0, kappa=0.0, atm=None):
    """Unscented Kalman filter (additive noise, scaled sigma points)."""
    x, P = _init(meas_c, dt, sigma_r, sigma_ang)
    n = x.shape[0]
    L = NX
    lam = alpha ** 2 * (L + kappa) - L
    wm = np.full(2 * L + 1, 1 / (2 * (L + lam)))
    wc = wm.copy()
    wm[0] = lam / (L + lam)
    wc[0] = wm[0] + (1 - alpha ** 2 + beta_ut)
    Rall = _R(n, sigma_r, sigma_ang)
    Q = Qd(dt)
    nis_sum = np.zeros(n)
    nis_cnt = np.zeros(n)
    for k in range(2, int(n_meas.max())):
        act = k < n_meas
        if not act.any():
            break
        xa, Pa, R = x[act], P[act], Rall[act]
        m = xa.shape[0]
        Ls = np.linalg.cholesky((L + lam) * Pa + 1e-9 * np.eye(L))
        sig = np.concatenate([xa[:, None], xa[:, None] + np.transpose(Ls, (0, 2, 1)), xa[:, None] - np.transpose(Ls, (0, 2, 1))], 1)
        sp = phi(sig.reshape(-1, L), dt, atm=None if atm is None else atm[act].repeat(2 * L + 1)).reshape(m, 2 * L + 1, L)
        xp = np.einsum("s,nsi->ni", wm, sp)
        dX = sp - xp[:, None]
        Pp = np.einsum("s,nsi,nsj->nij", wc, dX, dX) + Q
        zs = radar.to_polar(sp[..., :3].reshape(-1, 3)).reshape(m, 2 * L + 1, 3)
        # angle-safe mean of measurement sigma points
        zref = zs[:, 0:1]
        dz = zs - zref
        dz[..., 1:] = _wrap(dz[..., 1:])
        zp = zref[:, 0] + np.einsum("s,nsi->ni", wm, dz)
        dZ = zs - zp[:, None]
        dZ[..., 1:] = _wrap(dZ[..., 1:])
        S = np.einsum("s,nsi,nsj->nij", wc, dZ, dZ) + R
        C = np.einsum("s,nsi,nsj->nij", wc, dX, dZ)
        Si = np.linalg.inv(S)
        K = C @ Si
        nu = meas_pol[act, k] - zp
        nu[:, 1:] = _wrap(nu[:, 1:])
        xu = xp + np.einsum("nij,nj->ni", K, nu)
        xu[:, 6] = np.clip(xu[:, 6], LB_MIN, LB_MAX)
        Pu = Pp - K @ S @ np.transpose(K, (0, 2, 1))
        Pu = 0.5 * (Pu + np.transpose(Pu, (0, 2, 1))) + 1e-9 * np.eye(L)
        e = np.einsum("ni,nij,nj->n", nu, Si, nu)
        idx = np.nonzero(act)[0]
        late = k >= 6
        nis_sum[idx] += e * late
        nis_cnt[idx] += late
        x[act], P[act] = xu, Pu
    return x, P, nis_sum / np.maximum(nis_cnt, 1), None


def monte_carlo_impact(x, P, t_c, n_particles=100, seed=0, dt=0.25, atm=None):
    """Sample N(x, P) and propagate each particle to the ground. Returns per-track impact statistics."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    Lc = np.linalg.cholesky(P + 1e-9 * np.eye(NX))
    e = rng.standard_normal((n, n_particles, NX))
    smp = x[:, None] + np.einsum("nij,npj->npi", Lc, e)
    smp[..., 2] = np.maximum(smp[..., 2], 1.0)
    beta = np.exp(np.clip(smp[..., 6], LB_MIN, LB_MAX))
    res = physics.propagate_to_ground(smp[..., :6].reshape(-1, 6), beta.reshape(-1), dt=dt, atm=None if atm is None else atm.repeat(n_particles))
    ti = res["t"].reshape(n, n_particles) + t_c[:, None]
    ri = res["r"].reshape(n, n_particles, 3)
    mpi = ri[..., :2].mean(1)
    rad = np.linalg.norm(ri[..., :2] - mpi[:, None], axis=-1)
    return dict(mpi=mpi, cep=np.median(rad, 1), tot=ti.mean(1), tot_std=ti.std(1), samples=ri[..., :2])


def nominal_on_grid(x, t_c, T_true, n_tau=129, steps_per_cell=8, atm=None):
    """Propagate the mean state over the true remaining time and sample at tau grid; holds impact point after landing."""
    s = x[:, :6].copy()
    beta = np.exp(np.clip(x[:, 6], LB_MIN, LB_MAX))
    h = (T_true - t_c) / ((n_tau - 1) * steps_per_cell)
    out = np.zeros((x.shape[0], n_tau, 3))
    out[:, 0] = s[:, :3]
    landed = np.zeros(x.shape[0], bool)
    hold = np.zeros((x.shape[0], 3))
    hh = h[:, None]
    for c in range(n_tau - 1):
        for _ in range(steps_per_cell):
            prev = s
            k1 = physics._deriv(s, beta, atm); k2 = physics._deriv(s + .5 * hh * k1, beta, atm)
            k3 = physics._deriv(s + .5 * hh * k2, beta, atm); k4 = physics._deriv(s + hh * k3, beta, atm)
            s = s + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            newly = (~landed) & (s[:, 2] <= 0)
            if newly.any():
                f = prev[newly, 2] / (prev[newly, 2] - s[newly, 2])
                hold[newly] = prev[newly, :3] + f[:, None] * (s[newly, :3] - prev[newly, :3])
                landed |= newly
        out[:, c + 1] = np.where(landed[:, None], hold, s[:, :3])
    return out
