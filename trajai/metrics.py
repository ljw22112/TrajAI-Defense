"""Evaluation metrics (definitions match Table 10 of the technical report)."""
from __future__ import annotations

import numpy as np


def ade(pred_grid, true_grid):
    """Average displacement error over the prediction window [m]; (N, K, 3) each -> (N,)."""
    return np.linalg.norm(pred_grid - true_grid, axis=-1).mean(1)


def ip_miss(pred_ip, true_ip):
    """Horizontal impact-point miss distance [m]."""
    return np.linalg.norm(pred_ip[:, :2] - true_ip[:, :2], axis=-1)


def tot_error(pred_t, true_t):
    return np.abs(pred_t - true_t)


def cep_coverage(mpi, cep, true_ip):
    """Fraction of true impacts inside the predicted CEP50 circle. A calibrated estimator gives 0.5."""
    return (np.linalg.norm(true_ip[:, :2] - mpi, axis=-1) <= cep).astype(float)


def gaussian_cep(sx, sy, n=4000, seed=0):
    """CEP50 of a zero-mean 2D Gaussian with independent axes, by Monte Carlo (vectorised)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, 2))
    r = np.sqrt((e[None, :, 0] * sx[:, None]) ** 2 + (e[None, :, 1] * sy[:, None]) ** 2)
    return np.median(r, 1)


def summarize(d):
    """Median and mean of each metric array in dict d."""
    out = {}
    for k, v in d.items():
        v = np.asarray(v, float)
        out[k] = dict(median=float(np.nanmedian(v)), mean=float(np.nanmean(v)))
    return out
