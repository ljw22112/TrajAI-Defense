"""Weather profiles: Open-Meteo pressure levels, cached in the weather_profiles table."""
from __future__ import annotations

import time
from datetime import date, datetime, timezone

import httpx
import numpy as np
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from trajai import weather as wx

from ..config import get_settings
from ..models import WeatherProfile


def _parse(hourly: dict, hour: int) -> dict:
    z, T, u, v = [], [], [], []
    for lvl in wx.LEVELS:
        try:
            zz = hourly[f"geopotential_height_{lvl}hPa"][hour]
            tt = hourly[f"temperature_{lvl}hPa"][hour]
            ss = hourly[f"wind_speed_{lvl}hPa"][hour]
            dd = np.radians(hourly[f"wind_direction_{lvl}hPa"][hour])
        except (KeyError, IndexError, TypeError):
            raise HTTPException(502, f"weather provider returned no data for level {lvl} hPa")
        if None in (zz, tt, ss):
            raise HTTPException(502, f"weather provider has a gap at {lvl} hPa for this time")
        z.append(zz); T.append(tt + 273.15); u.append(-ss * np.sin(dd)); v.append(-ss * np.cos(dd))
    return dict(p=[l * 100.0 for l in wx.LEVELS], z=z, T=T, u=[float(a) for a in u], v=[float(a) for a in v])


GRID_DEG = 0.25        # Open-Meteo pressure-level resolution; snapping keeps the cache effective


def snap(lat: float | None, lon: float | None) -> tuple[float, float]:
    """Requested location -> provider grid point (defaults to the configured engagement area)."""
    s = get_settings()
    lat = s.weather_lat if lat is None else lat
    lon = s.weather_lon if lon is None else lon
    return round(round(lat / GRID_DEG) * GRID_DEG, 2), round(round(lon / GRID_DEG) * GRID_DEG, 2)


_day_cache: dict[tuple, dict] = {}       # one provider call returns all 24 hours of a day; reuse it within the process


def fetch_levels(day: date, hour: int, lat: float, lon: float) -> dict:
    key = (day.isoformat(), lat, lon)
    if key not in _day_cache:
        _day_cache[key] = _fetch_day(day, lat, lon)
        if len(_day_cache) > 64:
            _day_cache.pop(next(iter(_day_cache)))
    return _parse(_day_cache[key], hour)


def _from_archive(day: date, lat: float, lon: float) -> dict | None:
    """Monthly provider downloads kept by scripts/weather_experiment.py (data/weather/*.json): no network, no rate limit."""
    import json
    from pathlib import Path
    d = Path(get_settings().weather_archive_dir)
    if not d.is_dir():
        return None
    for f in d.glob(f"*_{lat}_{lon}.json"):
        a, b = f.name.split("_")[:2]
        if a <= day.isoformat() <= b:
            hourly = json.loads(f.read_text())["hourly"]
            i0 = hourly["time"].index(f"{day.isoformat()}T00:00")
            return {k: v[i0: i0 + 24] for k, v in hourly.items()}
    return None


def _fetch_day(day: date, lat: float, lon: float) -> dict:
    arch = _from_archive(day, lat, lon)
    if arch is not None:
        return arch
    s = get_settings()
    if s.weather_offline:
        raise HTTPException(503, "weather provider disabled (offline mode)")
    url = wx._url(day.isoformat(), day.isoformat(), lat, lon)
    r, last = None, None
    for attempt in range(3):                         # transient network errors: retry with backoff
        try:
            r = httpx.get(url, timeout=httpx.Timeout(30.0, connect=15.0))
            break
        except httpx.TransportError as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    if r is None:
        raise HTTPException(502, f"weather provider unreachable: {last.__class__.__name__}")
    if r.status_code == 429:
        raise HTTPException(503, "weather provider rate limit reached, retry in a minute")
    if r.status_code != 200:
        raise HTTPException(502, f"weather provider error {r.status_code}")
    return r.json()["hourly"]


FETCHER = {"fn": fetch_levels}          # swappable (tests use a synthetic profile)


def get_or_fetch(db: Session, day: date, hour: int, lat: float | None = None, lon: float | None = None,
                 fetcher=None) -> WeatherProfile:
    lat, lon = snap(lat, lon)
    vt = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
    row = db.scalar(select(WeatherProfile).where(WeatherProfile.valid_time == vt, WeatherProfile.lat == lat,
                                                 WeatherProfile.lon == lon))
    if row:
        return row
    lv = (fetcher or FETCHER["fn"])(day, hour, lat, lon)
    speed = np.hypot(lv["u"], lv["v"])
    i = int(np.argmax(speed))
    row = WeatherProfile(valid_time=vt, lat=lat, lon=lon, levels=lv,
                         max_wind=float(speed[i]), max_wind_alt=float(lv["z"][i]))
    db.add(row); db.commit(); db.refresh(row)
    return row


def bank_of_many(rows: list[WeatherProfile]):
    """One ProfileBank holding several cached profiles (index i <-> rows[i])."""
    profs = [dict(time=r.valid_time.isoformat(), z=np.array(r.levels["z"]), p=np.array(r.levels["p"]), T=np.array(r.levels["T"]),
                  u=np.array(r.levels["u"]), v=np.array(r.levels["v"])) for r in rows]
    return wx.build_grid(profs)


def bank_of(row: WeatherProfile | None):
    """trajai ProfileBank (one profile) for a cached row, or None for the standard model."""
    if row is None:
        return None
    lv = row.levels
    prof = dict(time=row.valid_time.isoformat(), z=np.array(lv["z"]), p=np.array(lv["p"]), T=np.array(lv["T"]),
                u=np.array(lv["u"]), v=np.array(lv["v"]))
    return wx.build_grid([prof])


def profile_curves(row: WeatherProfile, step_m: float = 1000.0) -> dict:
    """Altitude curves for plotting: T [K], wind u/v [m/s], density ratio vs the standard model."""
    from trajai import physics
    bank = bank_of(row)
    h = np.arange(0, 40_001, step_m)
    k = (h / wx.DH).astype(int)
    return dict(h=h.tolist(), T=bank.T[0, k].round(2).tolist(), u=bank.u[0, k].round(2).tolist(), v=bank.v[0, k].round(2).tolist(),
                rho_ratio=(bank.rho[0, k] / physics.density(h)).round(4).tolist())
