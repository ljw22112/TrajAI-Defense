"""Feature extraction shared by training and inference (NumPy only, no torch dependency)."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from . import physics

L_POS = 1e5          # position scale [m]
T_SCALE = 100.0      # time scale [s]
V_SCALE = 1000.0     # velocity scale [m/s]
LB_MU, LB_SD = 7.8, 0.8
N_FEAT = 12
N_GLOBAL = 10


def track_features(meas, n_meas, dt=0.5, window=9):
    """meas (N, K, 3) ENU, n_meas (N,) -> feats (N, K, 12), mask (N, K) True = padding, origin (N, 3)."""
    n, K, _ = meas.shape
    feats = np.zeros((n, K, N_FEAT), np.float32)
    mask = np.ones((n, K), bool)
    origin = meas[:, 0].astype(float)
    for i in range(n):
        k = int(n_meas[i])
        m = meas[i, :k].astype(float)
        w = min(window, k - (1 - k % 2))
        v = savgol_filter(m, w, 2, deriv=1, delta=dt, axis=0)
        a = savgol_filter(m, w, 2, deriv=2, delta=dt, axis=0)
        z = m[:, 2]
        mach = np.linalg.norm(v, axis=1) / physics.speed_of_sound(z)
        t = np.arange(k) * dt
        rel = (m - origin[i]) / L_POS
        feats[i, :k] = np.column_stack([t / T_SCALE, rel, z / L_POS, v / V_SCALE, a / physics.G0, mach / 5.0])
        mask[i, :k] = False
    return feats, mask, origin


def global_features(x_ekf, P_ekf, origin):
    """EKF posterior at t_c -> 10 global features (hybrid model input)."""
    rel = (x_ekf[:, :3] - origin) / L_POS
    v = x_ekf[:, 3:6] / V_SCALE
    lb = (x_ekf[:, 6:7] - LB_MU) / LB_SD
    sp = np.log(np.sqrt(np.trace(P_ekf[:, :3, :3], axis1=1, axis2=2)) + 1.0)[:, None] / 5
    sv = np.log(np.sqrt(np.trace(P_ekf[:, 3:6, 3:6], axis1=1, axis2=2)) + 1e-3)[:, None] / 5
    sl = np.sqrt(P_ekf[:, 6, 6])[:, None]
    return np.concatenate([rel, v, lb, sp, sv, sl], 1).astype(np.float32)


def targets(d, origin):
    fut = ((d["future"] - origin[:, None]) / L_POS).astype(np.float32)
    dT = (d["T"] - d["t_c"]).astype(np.float32)
    lb = ((np.log(d["beta"]) - LB_MU) / LB_SD).astype(np.float32)
    return fut, dT, lb


def track_features_rts(meas_pol, meas, n_meas, sigma_r, sigma_ang, dt=0.5, chunk=500):
    """Same 12-feature layout as track_features, but position/velocity/acceleration come from an EKF-RTS smoother
    instead of Savitzky-Golay differentiation (velocity error 7.2 -> 1.2 m/s on simulated tracks, §9.6).
    Acceleration is the central difference of the smoothed velocity; the two initialisation steps copy step 2."""
    from . import filters
    n, K, _ = meas.shape
    feats = np.zeros((n, K, N_FEAT), np.float32)
    mask = np.ones((n, K), bool)
    origin = meas[:, 0].astype(float)
    sr = np.broadcast_to(np.asarray(sigma_r, float), (n,))
    sa = np.broadcast_to(np.asarray(sigma_ang, float), (n,))
    for c0 in range(0, n, chunk):
        sl = slice(c0, min(c0 + chunk, n))
        xs, _ = filters.rts_smooth(meas_pol[sl].astype(float), meas[sl].astype(float), n_meas[sl], dt, sr[sl], sa[sl])
        for j, i in enumerate(range(sl.start, sl.stop)):
            k = int(n_meas[i])
            x = xs[j, :k].copy()
            x[:2] = x[2]                                   # steps 0-1 are the two-point initialisation
            p, v = x[:, :3], x[:, 3:6]
            a = np.gradient(v, dt, axis=0) if k > 2 else np.zeros_like(v)
            z = p[:, 2]
            mach = np.linalg.norm(v, axis=1) / physics.speed_of_sound(np.maximum(z, 0.0))
            t = np.arange(k) * dt
            rel = (p - origin[i]) / L_POS
            feats[i, :k] = np.column_stack([t / T_SCALE, rel, z / L_POS, v / V_SCALE, a / physics.G0, mach / 5.0])
            mask[i, :k] = False
    return feats, mask
