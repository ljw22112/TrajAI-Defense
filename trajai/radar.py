"""Radar measurement model: range / azimuth / elevation with Gaussian noise, plus conversion to ENU."""
from __future__ import annotations

import numpy as np

RADAR_SITE = np.array([-30_000.0, 20_000.0, 0.0])   # notional sensor location in the local ENU frame [m]

NOISE_NOMINAL = dict(sigma_r=15.0, sigma_ang=0.4e-3)   # m, rad
NOISE_HIGH = dict(sigma_r=50.0, sigma_ang=1.5e-3)


def to_polar(pos, site=RADAR_SITE):
    d = pos - site
    rng = np.linalg.norm(d, axis=-1)
    az = np.arctan2(d[..., 1], d[..., 0])
    el = np.arctan2(d[..., 2], np.hypot(d[..., 0], d[..., 1]))
    return np.stack([rng, az, el], -1)


def to_cartesian(polar, site=RADAR_SITE):
    rng, az, el = polar[..., 0], polar[..., 1], polar[..., 2]
    ce = np.cos(el)
    return np.stack([rng * ce * np.cos(az), rng * ce * np.sin(az), rng * np.sin(el)], -1) + site


def measure(pos, rng: np.random.Generator, sigma_r, sigma_ang, site=RADAR_SITE):
    """Noisy polar measurements of positions (..., 3). Returns (polar, cartesian)."""
    z = to_polar(pos, site)
    noise = rng.standard_normal(z.shape) * np.array([sigma_r, sigma_ang, sigma_ang])
    zp = z + noise
    return zp, to_cartesian(zp, site)


def polar_jacobian(pos, site=RADAR_SITE):
    """d[R, az, el]/d[x, y, z] for positions (N, 3) -> (N, 3, 3)."""
    d = pos - site
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    rh2 = x * x + y * y
    rh = np.sqrt(rh2)
    r2 = rh2 + z * z
    r = np.sqrt(r2)
    J = np.zeros((pos.shape[0], 3, 3))
    J[:, 0] = np.stack([x / r, y / r, z / r], -1)
    J[:, 1] = np.stack([-y / rh2, x / rh2, 0 * x], -1)
    J[:, 2] = np.stack([-x * z / (r2 * rh), -y * z / (r2 * rh), rh / r2], -1)
    return J
