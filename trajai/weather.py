"""Date-specific atmosphere from the Open-Meteo API (pressure levels 1000-10 hPa, about 0-31 km).

fetch_profiles(start, end)  -> list of raw profiles (one per selected UTC hour), cached under data/weather/
build_grid(profiles)        -> ProfileBank: temperature, density, speed of sound and wind on a 0-80 km grid
ProfileBank[idx]            -> GridAtmosphere for a batch (per-row profile index; memory independent of batch size)

Above the top pressure level the temperature blends into USSA-1976 over 10 km and the wind decays linearly
to zero by 45 km (no data there). Density comes from the measured pressure levels (log-linear in height)
and from hydrostatic integration above them.
"""
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np

from . import physics

LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]   # hPa
API = "https://historical-forecast-api.open-meteo.com/v1/forecast"
SITE = dict(latitude=36.5, longitude=127.5)          # weather sampling point (inland test range, notional)
DH = 250.0
H_GRID = np.arange(0.0, 80_000.0 + DH, DH)
FEAT_ALT = np.array([0, 2, 5, 9, 12, 16, 20, 25, 30]) * 1000.0        # altitudes summarised as model features [m]
N_WX = 3 * len(FEAT_ALT)


def _url(start, end, lat, lon):
    v = ",".join(f"{q}_{l}hPa" for l in LEVELS for q in ("temperature", "wind_speed", "wind_direction", "geopotential_height"))
    return f"{API}?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}&hourly={v}&wind_speed_unit=ms&timezone=GMT"


def _download(start, end, lat, lon, cache_dir):
    path = os.path.join(cache_dir, f"{start}_{end}_{lat}_{lon}.json")
    if not os.path.exists(path):
        import time
        last = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(_url(start, end, lat, lon), timeout=180) as r:
                    raw = json.load(r)
                break
            except Exception as e:      # rate limit (429) or transient network error: back off and retry
                last = e
                time.sleep(70 if "429" in str(e) else 10)
        else:
            raise last
        time.sleep(3)
        json.dump(raw, open(path, "w"))
    return json.load(open(path))["hourly"]


def fetch_profiles(start, end, hours=(0, 6, 12, 18), cache_dir="data/weather", lat=SITE["latitude"], lon=SITE["longitude"]):
    """Download (or load cached) hourly pressure-level data in monthly chunks and keep the given UTC hours."""
    import datetime as dt
    os.makedirs(cache_dir, exist_ok=True)
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    h = {}
    cur = d0
    while cur <= d1:
        nxt = min(d1, (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1))
        part = _download(cur.isoformat(), nxt.isoformat(), lat, lon, cache_dir)
        for k, v in part.items():
            h.setdefault(k, []).extend(v)
        cur = nxt + dt.timedelta(days=1)
    out = []
    for i, t in enumerate(h["time"]):
        if int(t[11:13]) not in hours:
            continue
        try:
            z = np.array([h[f"geopotential_height_{l}hPa"][i] for l in LEVELS], float)
            T = np.array([h[f"temperature_{l}hPa"][i] for l in LEVELS], float) + 273.15
            s = np.array([h[f"wind_speed_{l}hPa"][i] for l in LEVELS], float)
            d = np.radians(np.array([h[f"wind_direction_{l}hPa"][i] for l in LEVELS], float))
        except (TypeError, KeyError):
            continue
        if np.any(np.isnan(z)) or np.any(np.isnan(T)) or np.any(np.isnan(s)):
            continue
        # meteorological direction = where the wind comes FROM; ENU frame with x = east, y = north
        out.append(dict(time=t, z=z, p=np.array(LEVELS, float) * 100.0, T=T, u=-s * np.sin(d), v=-s * np.cos(d)))
    return out


class ProfileBank:
    """K atmospheric profiles on a common altitude grid."""

    def __init__(self, T, rho, sos, u, v, times):
        self.T, self.rho, self.sos, self.u, self.v = T, rho, sos, u, v
        self.times = list(times)

    def __len__(self):
        return self.T.shape[0]

    def __getitem__(self, idx):
        return GridAtmosphere(self, np.asarray(idx))

    def features(self, idx):
        """(N, 27) summary used as model input: u/50, v/50, (T - T_ussa)/20 at FEAT_ALT."""
        k = (FEAT_ALT / DH).astype(int)
        idx = np.asarray(idx)
        t_std = physics.temperature(FEAT_ALT)
        return np.concatenate([self.u[idx][:, k] / 50.0, self.v[idx][:, k] / 50.0, (self.T[idx][:, k] - t_std) / 20.0], 1).astype(np.float32)

    def mean(self):
        """Climatological mean profile as a one-profile bank."""
        return ProfileBank(*(a.mean(0, keepdims=True) for a in (self.T, self.rho, self.sos, self.u, self.v)), ["climatology"])

    def perturbed(self, rng, sigma_wind=4.0, sigma_T=1.5, corr_km=4.0):
        """Analysis-error realisation: smooth random perturbations of wind and temperature (per profile)."""
        K, G = self.T.shape
        lag = np.arange(G) * DH / 1000.0
        C = np.exp(-0.5 * (lag[:, None] - lag[None]) ** 2 / corr_km ** 2) + 1e-6 * np.eye(G)
        L = np.linalg.cholesky(C)
        def smooth(sig):
            return (rng.standard_normal((K, G)) @ L.T) * sig
        T = self.T + smooth(sigma_T)
        u = self.u + smooth(sigma_wind)
        v = self.v + smooth(sigma_wind)
        rho = self.rho * (self.T / T)            # same pressure, perturbed temperature
        sos = np.sqrt(physics.GAMMA_AIR * physics.R_AIR * T)
        return ProfileBank(T, rho, sos, u, v, self.times)


def build_grid(profiles):
    K, G = len(profiles), len(H_GRID)
    T = np.zeros((K, G)); rho = np.zeros((K, G)); u = np.zeros((K, G)); v = np.zeros((K, G))
    t_std = physics.temperature(H_GRID)
    for k, pr in enumerate(profiles):
        z, p, Tm, um, vm = pr["z"], pr["p"], pr["T"], pr["u"], pr["v"]
        top = z[-1]
        Ti = np.interp(H_GRID, z, Tm)
        off = Tm[-1] - physics.temperature(top)
        w = np.clip((H_GRID - top) / 10_000.0, 0, 1)
        above = H_GRID > top
        Ti[above] = (t_std + off * (1 - w))[above]
        lnp = np.interp(H_GRID, z, np.log(p))
        # below the lowest level: hydrostatic extrapolation down to 0 m
        # above the top level: hydrostatic integration with the blended temperature
        pr_grid = np.exp(lnp)
        i_top = int(np.searchsorted(H_GRID, top))
        for i in range(i_top, G):
            hm = 0.5 * (H_GRID[i] + H_GRID[i - 1])
            pr_grid[i] = pr_grid[i - 1] * np.exp(-physics.gravity(hm) * DH / (physics.R_AIR * 0.5 * (Ti[i] + Ti[i - 1])))
        below = H_GRID < z[0]
        pr_grid[below] = p[0] * np.exp(physics.G0 * (z[0] - H_GRID[below]) / (physics.R_AIR * Tm[0]))
        T[k] = Ti
        rho[k] = pr_grid / (physics.R_AIR * Ti)
        wd = np.clip(1 - (H_GRID - top) / (45_000.0 - top), 0, 1)
        u[k] = np.interp(H_GRID, z, um) * np.where(above, wd, 1.0)
        v[k] = np.interp(H_GRID, z, vm) * np.where(above, wd, 1.0)
    sos = np.sqrt(physics.GAMMA_AIR * physics.R_AIR * T)
    return ProfileBank(T, rho, sos, u, v, [p["time"] for p in profiles])


def ussa_bank():
    """USSA-1976 standard atmosphere (hydrostatic, no wind) as a one-profile bank."""
    T = physics.temperature(H_GRID)
    p = np.zeros_like(H_GRID); p[0] = 101_325.0
    for i in range(1, len(H_GRID)):
        hm = 0.5 * (H_GRID[i] + H_GRID[i - 1])
        p[i] = p[i - 1] * np.exp(-physics.gravity(hm) * DH / (physics.R_AIR * 0.5 * (T[i] + T[i - 1])))
    rho = p / (physics.R_AIR * T)
    z = np.zeros((1, len(H_GRID)))
    return ProfileBank(T[None], rho[None], np.sqrt(physics.GAMMA_AIR * physics.R_AIR * T)[None], z, z.copy(), ["USSA-1976"])


def load_banks(seed=7):
    """train (previous year), test (target dates), and the perturbed 'truth' versions of both."""
    tr = build_grid(fetch_profiles("2025-10-01", "2026-09-30"))
    te = build_grid(fetch_profiles("2026-10-01", "2026-10-28"))
    rng = np.random.default_rng(seed)
    return dict(train=tr, test=te, train_truth=tr.perturbed(rng), test_truth=te.perturbed(rng))


class GridAtmosphere:
    """Batch view: row n uses profile bank[idx[n]]. Linear interpolation on the 250 m grid."""

    def __init__(self, bank, idx):
        self.bank, self.idx = bank, idx

    def __getitem__(self, sel):
        return GridAtmosphere(self.bank, self.idx[sel])

    def repeat(self, n):
        return GridAtmosphere(self.bank, np.repeat(self.idx, n))

    def _lookup(self, arr, h):
        x = np.clip(h, 0.0, H_GRID[-1] - 1e-6) / DH
        i = x.astype(int)
        f = x - i
        idx = np.broadcast_to(self.idx, h.shape) if self.idx.ndim else self.idx
        return arr[idx, i] * (1 - f) + arr[idx, i + 1] * f

    def rho(self, h):
        return self._lookup(self.bank.rho, h)

    def sos(self, h):
        return self._lookup(self.bank.sos, h)

    def wind(self, h):
        return np.stack([self._lookup(self.bank.u, h), self._lookup(self.bank.v, h), np.zeros_like(h)], -1)
