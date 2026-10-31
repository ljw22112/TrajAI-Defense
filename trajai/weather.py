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


# ---------------------------------------------------------------- user-defined atmosphere
CUSTOM_DEFAULT = dict(
    t_sfc_c=15.0,          # surface air temperature [degC]
    p_sfc_hpa=1013.25,     # surface pressure [hPa]
    rh_pct=50.0,           # surface relative humidity [%]
    lapse_k_per_km=6.5,    # tropospheric lapse rate [K/km]
    tropopause_km=11.0,    # tropopause height [km]
    wind_sfc_ms=5.0,       # surface wind speed [m/s]
    wind_sfc_dir=270.0,    # direction the surface wind blows FROM [deg, met. convention]
    jet_ms=40.0,           # jet-stream peak speed [m/s]
    jet_km=11.0,           # jet-stream core height [km]
    jet_width_km=3.0,      # 1-sigma half-thickness of the jet [km]
    jet_dir=270.0,         # direction the jet blows FROM [deg]
    condition="clear",     # clear | cloudy | fog | rain | heavy_rain | snow
)

# Precipitation and fog barely change the drag on a ballistic body (hydrometeor loading is < 0.1 % of air density),
# but they degrade a radar track: attenuation and clutter lower SNR, which raises range/angle noise. These factors
# are a coarse X-band-like assumption used to scale the sensor noise, not a radar propagation model.
CONDITION_NOISE = {"clear": 1.0, "cloudy": 1.0, "fog": 1.05, "snow": 1.4, "rain": 1.3, "heavy_rain": 2.0}


def saturation_vapour_pressure(t_c):
    """Magnus formula over water [Pa]."""
    return 610.94 * np.exp(17.625 * t_c / (t_c + 243.04))


def custom_levels(params: dict | None = None, z_top=32_000.0, dz=500.0) -> dict:
    """User-specified atmosphere as a level profile (same keys as a provider profile, plus Tv and q).

    Temperature: surface value, linear lapse to the tropopause, isothermal to 20 km, then +1 K/km (USSA-1976 shape).
    Humidity: surface specific humidity from RH (Magnus), decaying with a 2 km scale height; it enters through the
    virtual temperature Tv = T (1 + 0.608 q), which sets density (p / R Tv) and is integrated hydrostatically.
    Wind: surface wind blended linearly into a Gaussian jet core; direction turns linearly between the two.
    """
    p = {**CUSTOM_DEFAULT, **(params or {})}
    z = np.arange(0.0, z_top + dz, dz)
    zt, zs = p["tropopause_km"] * 1000.0, 20_000.0
    T0 = p["t_sfc_c"] + 273.15
    Tt = T0 - p["lapse_k_per_km"] * zt / 1000.0
    T = np.where(z <= zt, T0 - p["lapse_k_per_km"] * z / 1000.0, np.where(z <= zs, Tt, Tt + (z - zs) / 1000.0))
    e_s = saturation_vapour_pressure(p["t_sfc_c"])
    p0 = p["p_sfc_hpa"] * 100.0
    q0 = 0.622 * (p["rh_pct"] / 100.0) * e_s / (p0 - 0.378 * (p["rh_pct"] / 100.0) * e_s)
    q = q0 * np.exp(-z / 2000.0)
    Tv = T * (1.0 + 0.608 * q)
    pres = np.empty_like(z); pres[0] = p0
    for i in range(1, len(z)):
        hm = 0.5 * (z[i] + z[i - 1])
        pres[i] = pres[i - 1] * np.exp(-physics.gravity(hm) * dz / (physics.R_AIR * 0.5 * (Tv[i] + Tv[i - 1])))
    zj, wj = p["jet_km"] * 1000.0, max(p["jet_width_km"], 0.5) * 1000.0
    jet = p["jet_ms"] * np.exp(-0.5 * ((z - zj) / wj) ** 2)
    blend = np.clip(z / max(zj, 1.0), 0, 1)
    speed = np.maximum(p["wind_sfc_ms"] * (1 - blend), 0) + jet
    # direction: interpolate on the circle from the surface to the jet core, constant above
    d0, d1 = np.radians(p["wind_sfc_dir"]), np.radians(p["jet_dir"])
    dd = (d1 - d0 + np.pi) % (2 * np.pi) - np.pi
    dirn = d0 + dd * blend
    u, v = -speed * np.sin(dirn), -speed * np.cos(dirn)
    return dict(z=z.tolist(), p=pres.tolist(), T=T.tolist(), Tv=Tv.tolist(), q=q.tolist(), u=u.tolist(), v=v.tolist(),
                custom={k: p[k] for k in CUSTOM_DEFAULT})
