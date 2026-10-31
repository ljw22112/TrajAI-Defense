"""Synthetic dataset generation (Monte Carlo over burnout states and ballistic coefficient).

Each sample is the post-burnout (ballistic) flight of one body. The sensor tracks it from t = 0 until
the cut-off time t_c = f * T_impact, where f is the observed fraction of flight. The network sees the
noisy track; the targets are the remaining trajectory, impact point, time of impact and beta.

Stored per sample (npz):
  meas      (N, K_MAX, 3)   converted (ENU) noisy measurements, zero padded
  meas_pol  (N, K_MAX, 3)   raw polar measurements [R, az, el]
  n_meas    (N,)            number of valid measurements
  t_c       (N,)            cut-off time [s]
  T         (N,)            time of impact [s]
  future    (N, N_TAU, 3)   true positions on tau grid, t = t_c + tau (T - t_c), tau in [0, 1]
  state_c   (N, 6)          true state at t_c
  impact    (N, 3)          true impact point
  beta      (N,)            ballistic coefficient [kg/m^2]
  params    (N, 5)          [h0, V0, gamma0, psi0, beta]
"""
from __future__ import annotations

import numpy as np

from . import physics, radar

DT_MEAS = 0.5
N_TAU = 129
K_MAX = 200

ID_RANGES = dict(h0=(8e3, 25e3), V0=(900.0, 1900.0), gamma=(30.0, 60.0), psi=(-10.0, 10.0), beta=(600.0, 10_000.0))
OOD_RANGES = dict(h0=(8e3, 25e3), V0=(1900.0, 2200.0), gamma=(25.0, 65.0), psi=(-10.0, 10.0), beta=(10_000.0, 16_000.0))


def sample_params(n, rng, ranges):
    u = lambda key: rng.uniform(*ranges[key], n)
    lb = rng.uniform(np.log(ranges["beta"][0]), np.log(ranges["beta"][1]), n)
    return np.stack([u("h0"), u("V0"), u("gamma"), u("psi"), np.exp(lb)], 1)


def initial_state(params):
    h0, V0, g, psi, _ = params.T
    g, psi = np.radians(g), np.radians(psi)
    vh = V0 * np.cos(g)
    return np.stack([0 * h0, 0 * h0, h0, vh * np.cos(psi), vh * np.sin(psi), V0 * np.sin(g)], 1)


def generate(n, seed, ranges=ID_RANGES, frac=(0.08, 0.20), noise=radar.NOISE_NOMINAL, chunk=4000, bank=None, mismatch=None):
    """bank: optional weather ProfileBank; each trajectory then flies through a randomly chosen profile
    (stored as 'wx_idx'). Without a bank the standard model (exponential density, no wind) is the truth."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ["meas", "meas_pol", "n_meas", "t_c", "T", "future", "state_c", "impact", "beta", "params", "sigma", "wx_idx"]}
    done = 0
    while done < n:
        m = min(chunk, n - done)
        p = sample_params(m, rng, ranges)
        s0 = initial_state(p)
        wx = rng.integers(0, len(bank), m) if bank is not None else np.full(m, -1)
        atm = bank[wx] if bank is not None else None
        if mismatch is not None:                      # truth-only model error, see trajai.mismatch
            from . import mismatch as mm
            atm = mm.sample(m, rng, **mismatch)
        times, pos, vel, T, r_imp = physics.simulate_ballistic(s0, p[:, 4], dt=0.1, out_dt=DT_MEAS, atm=atm)
        f = rng.uniform(*frac, m) if isinstance(frac, tuple) else np.full(m, frac)
        k_c = np.minimum(np.floor(f * T / DT_MEAS).astype(int), K_MAX - 1)
        k_c = np.maximum(k_c, 12)
        t_c = k_c * DT_MEAS
        meas = np.zeros((m, K_MAX, 3), np.float32)
        meas_pol = np.zeros((m, K_MAX, 3))
        future = np.zeros((m, N_TAU, 3), np.float32)
        state_c = np.zeros((m, 6))
        tau = np.linspace(0, 1, N_TAU)
        if noise == "mixed":   # augmentation: per-track noise level, log-uniform
            sr = np.exp(rng.uniform(np.log(10.0), np.log(60.0), m))
            sa = np.exp(rng.uniform(np.log(0.3e-3), np.log(1.8e-3), m))
        else:
            sr, sa = np.full(m, noise["sigma_r"]), np.full(m, noise["sigma_ang"])
        for i in range(m):
            k = k_c[i] + 1
            zp, zc = radar.measure(pos[i, :k], rng, sigma_r=sr[i], sigma_ang=sa[i])
            meas[i, :k] = zc
            meas_pol[i, :k] = zp
            state_c[i] = np.r_[pos[i, k_c[i]], vel[i, k_c[i]]]
            # true future on the tau grid: integrate from t_c (exact state) with fine steps
        # future positions: re-integrate from state_c and sample at tau grid (vectorised, piecewise linear on 0.1 s)
        t_rem = T - t_c
        future[:] = _future_on_grid(state_c, p[:, 4], t_rem, tau, atm)
        for key, val in zip(out.keys(), [meas, meas_pol.astype(np.float32), k_c + 1, t_c, T, future, state_c, r_imp, p[:, 4], p, np.stack([sr, sa], 1), wx]):
            out[key].append(val)
        done += m
    return {k: np.concatenate(v) for k, v in out.items()}


def _future_on_grid(state_c, beta, t_rem, tau, atm=None, dt=0.05):
    """Positions at t_c + tau * t_rem for each sample, by RK4 integration with per-sample step dt_i = t_rem/(steps)."""
    n = state_c.shape[0]
    steps_per_cell = 8
    n_cells = len(tau) - 1
    h = t_rem / (n_cells * steps_per_cell)          # per-sample step size
    s = state_c.copy()
    res = np.zeros((n, len(tau), 3))
    res[:, 0] = s[:, :3]
    for c in range(n_cells):
        for _ in range(steps_per_cell):
            s = _rk4_var(s, beta, h, atm)
        res[:, c + 1] = s[:, :3]
    res[:, -1, 2] = np.clip(res[:, -1, 2], -5.0, 5.0)   # lands at z ~ 0 (step-size dependent residue)
    return res


def _rk4_var(s, beta, h, atm=None):
    hh = h[:, None]
    k1 = physics._deriv(s, beta, atm)
    k2 = physics._deriv(s + 0.5 * hh * k1, beta, atm)
    k3 = physics._deriv(s + 0.5 * hh * k2, beta, atm)
    k4 = physics._deriv(s + hh * k3, beta, atm)
    return s + hh / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def make_all(out_dir, n_train=20_000, n_val=2_000, seed=2026, only=None):
    import os
    os.makedirs(out_dir, exist_ok=True)
    sets = {
        "train": dict(n=n_train, seed=seed, ranges=ID_RANGES, frac=(0.08, 0.20)),
        "val": dict(n=n_val, seed=seed + 1, ranges=ID_RANGES, frac=(0.08, 0.20)),
        "train_aug": dict(n=n_train, seed=seed + 10, ranges=ID_RANGES, frac=(0.08, 0.20), noise="mixed"),
        "val_aug": dict(n=n_val, seed=seed + 11, ranges=ID_RANGES, frac=(0.08, 0.20), noise="mixed"),
        "test_id_f10": dict(n=1000, seed=seed + 2, ranges=ID_RANGES, frac=0.10),
        "test_id_f15": dict(n=1000, seed=seed + 3, ranges=ID_RANGES, frac=0.15),
        "test_id_f05": dict(n=1000, seed=seed + 6, ranges=ID_RANGES, frac=0.05),
        "test_id_noise": dict(n=1000, seed=seed + 4, ranges=ID_RANGES, frac=0.12, noise=radar.NOISE_HIGH),
        "test_ood": dict(n=1000, seed=seed + 5, ranges=OOD_RANGES, frac=0.12),
    }
    for name, kw in sets.items():
        if only and name not in only:
            continue
        d = generate(**kw)
        np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **d)
        print(f"{name:14s} n={len(d['T']):6d}  T=[{d['T'].min():.0f},{d['T'].max():.0f}] s  "
              f"range=[{d['impact'][:,0].min()/1e3:.0f},{d['impact'][:,0].max()/1e3:.0f}] km  "
              f"n_meas=[{d['n_meas'].min()},{d['n_meas'].max()}]")
