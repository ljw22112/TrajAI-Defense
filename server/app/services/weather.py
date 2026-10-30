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


def fetch_levels(day: date, hour: int, lat: float, lon: float) -> dict:
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
    return _parse(r.json()["hourly"], hour)


FETCHER = {"fn": fetch_levels}          # swappable (tests use a synthetic profile)


def get_or_fetch(db: Session, day: date, hour: int, fetcher=None) -> WeatherProfile:
    s = get_settings()
    vt = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
    row = db.scalar(select(WeatherProfile).where(WeatherProfile.valid_time == vt, WeatherProfile.lat == s.weather_lat,
                                                 WeatherProfile.lon == s.weather_lon))
    if row:
        return row
    lv = (fetcher or FETCHER["fn"])(day, hour, s.weather_lat, s.weather_lon)
    speed = np.hypot(lv["u"], lv["v"])
    i = int(np.argmax(speed))
    row = WeatherProfile(valid_time=vt, lat=s.weather_lat, lon=s.weather_lon, levels=lv,
                         max_wind=float(speed[i]), max_wind_alt=float(lv["z"][i]))
    db.add(row); db.commit(); db.refresh(row)
    return row


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
