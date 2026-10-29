"""Threat assessment (REQ-F-04): impact probability against protected assets, threat level 1-5,
and engagement altitude windows on the predicted descent.

The asset list is notional (local ENU coordinates, km) and exists only to exercise the logic.
"""
from __future__ import annotations

import numpy as np

ASSETS = [
    dict(name="ASSET-A", x=60.0, y=2.0, radius=3.0),
    dict(name="ASSET-B", x=120.0, y=-6.0, radius=4.0),
    dict(name="ASSET-C", x=185.0, y=10.0, radius=3.0),
    dict(name="ASSET-D", x=240.0, y=-12.0, radius=5.0),
    dict(name="ASSET-E", x=310.0, y=4.0, radius=4.0),
]

LEVELS = [(0.5, 5), (0.2, 4), (0.05, 3), (0.01, 2)]          # P(hit) threshold -> level, else 1

BANDS = {"upper": (15_000.0, 40_000.0), "lower": (3_000.0, 15_000.0)}   # notional interceptor altitude bands [m]


def impact_probabilities(mpi_m, sigma_m, assets=ASSETS, n=20_000, seed=0):
    """mpi_m (2,), sigma_m (2,) in metres -> {asset: P(impact inside radius)} using a diagonal Gaussian."""
    rng = np.random.default_rng(seed)
    pts = mpi_m + rng.standard_normal((n, 2)) * sigma_m
    out = {}
    for a in assets:
        c = np.array([a["x"], a["y"]]) * 1000.0
        out[a["name"]] = float((np.linalg.norm(pts - c, axis=1) <= a["radius"] * 1000.0).mean())
    return out


def threat_level(probs):
    p = max(probs.values()) if probs else 0.0
    for thr, lvl in LEVELS:
        if p >= thr:
            return lvl
    return 1


def engagement_windows(times, alt, bands=BANDS):
    """Time intervals on the descending branch during which altitude lies inside each band.

    Returns {band: dict(t_in, t_out, t_pip, h_pip)} with the predicted intercept point (PIP) at mid-window.
    """
    times = np.asarray(times); alt = np.asarray(alt)
    apo = int(np.argmax(alt))
    t_d, h_d = times[apo:], alt[apo:]
    res = {}
    for name, (lo, hi) in bands.items():
        inside = (h_d >= lo) & (h_d <= hi)
        if not inside.any():
            res[name] = None
            continue
        idx = np.nonzero(inside)[0]
        t_in, t_out = float(t_d[idx[0]]), float(t_d[idx[-1]])
        t_pip = 0.5 * (t_in + t_out)
        res[name] = dict(t_in=t_in, t_out=t_out, t_pip=t_pip, h_pip=float(np.interp(t_pip, t_d, h_d[::1])))
    return res


def assess(mpi_m, sigma_m, times, alt):
    probs = impact_probabilities(np.asarray(mpi_m, float), np.asarray(sigma_m, float))
    return dict(probabilities=probs, level=threat_level(probs), windows=engagement_windows(times, alt))
