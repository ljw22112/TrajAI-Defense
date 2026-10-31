"""Weather-impact analyses: create (runs the batched study), list, read, delete."""
from datetime import datetime, timedelta

from trajai import weather as wx

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..audit import log
from ..db import get_db
from ..models import Role, User, WeatherAnalysis
from ..schemas import CustomAtmIn
from ..schemas import AnalysisIn, AnalysisOut, AnalysisSummaryOut, LatencyIn, Page, SensitivityIn, SensorIn
from ..security import current_user, require
from ..services import analysis as asvc
from ..services import weather as wsvc

router = APIRouter(prefix="/api/analyses", tags=["weather impact analysis"])


@router.post("", response_model=AnalysisOut, status_code=201,
             summary="Fly one scenario through every profile in a time window; compare standard vs weather-aware EKF")
async def create(body: AnalysisIn, db: Session = Depends(get_db), user: User = Depends(require(Role.analyst))):
    lat, lon = wsvc.snap(body.lat, body.lon)
    rows = [wsvc.get_or_fetch(db, body.start_date + timedelta(days=d), h, lat, lon)
            for d in range(body.days) for h in body.hours]
    if len(rows) > 28:
        raise HTTPException(422, "at most 28 profiles (7 days x 4 hours)")
    bank = wsvc.bank_of_many(rows)
    p = body.model_dump(mode="json")
    res = await run_in_threadpool(asvc.weather_impact, p, bank, rows)
    a = WeatherAnalysis(kind="weather", name=body.name, created_by=user.id, lat=lat, lon=lon, n_profiles=len(rows), params=p,
                        summary=res["summary"], items=res["items"])
    db.add(a); db.commit(); db.refresh(a)
    log(db, user, "create", "analysis", a.id, {"n": len(rows), "reduction": res["summary"]["reduction"]})
    return a


def _save(db, user, kind, body, lat, lon, n, res):
    a = WeatherAnalysis(kind=kind, name=body.name, created_by=user.id, lat=lat, lon=lon, n_profiles=n,
                        params=body.model_dump(mode="json"), summary=res["summary"], items=res["items"])
    db.add(a); db.commit(); db.refresh(a)
    log(db, user, "create", "analysis", a.id, {"kind": kind, "n": n})
    return a


@router.post("/latency", response_model=AnalysisOut, status_code=201,
             summary="Weather-data latency: EKF given the profile valid 0/6/12/24 h before the flight, vs no weather data")
async def create_latency(body: LatencyIn, db: Session = Depends(get_db), user: User = Depends(require(Role.analyst))):
    lat, lon = wsvc.snap(body.lat, body.lon)
    times = [(body.start_date + timedelta(days=d), h) for d in range(body.days) for h in body.hours]
    if len(times) > 28:
        raise HTTPException(422, "at most 28 profiles (7 days x 4 hours)")
    rows, index, lag_index = [], {}, {}

    def get(day, hour):
        key = (day, hour)
        if key not in index:
            index[key] = len(rows)
            rows.append(wsvc.get_or_fetch(db, day, hour, lat, lon))
        return index[key]

    for i, (d, h) in enumerate(times):
        get(d, h)
    for i, (d, h) in enumerate(times):
        for lag in asvc.LAGS_H:
            t = datetime(d.year, d.month, d.day, h) - timedelta(hours=lag)
            lag_index[(i, lag)] = get(t.date(), t.hour)
    bank = wsvc.bank_of_many(rows)
    res = await run_in_threadpool(asvc.latency, body.model_dump(mode="json"), bank, rows[: len(times)], lag_index)
    return _save(db, user, "latency", body, lat, lon, len(times), res)


@router.post("/sensor", response_model=AnalysisOut, status_code=201,
             summary="Sensor sweep: radar noise x observed fraction, EKF matched to the chosen atmosphere")
async def create_sensor(body: SensorIn, db: Session = Depends(get_db), user: User = Depends(require(Role.analyst))):
    lat, lon = wsvc.snap(body.lat, body.lon)
    if body.atmosphere == "date":
        if not body.weather_date:
            raise HTTPException(422, "weather_date is required for atmosphere=date")
        bank = wsvc.bank_of(wsvc.get_or_fetch(db, body.weather_date, body.weather_hour, lat, lon))
    elif body.atmosphere == "custom":
        bank = wx.build_grid([wsvc._prof(wx.custom_levels((body.custom_atm or CustomAtmIn()).model_dump()), "custom")])
    else:
        bank = None
    res = await run_in_threadpool(asvc.sensor_sweep, body.model_dump(mode="json"), bank)
    return _save(db, user, "sensor", body, lat, lon, len(res["items"]), res)


@router.post("/sensitivity", response_model=AnalysisOut, status_code=201,
             summary="Environment sensitivity: vary one user-set variable (temperature, humidity, pressure, wind, weather)")
async def create_sensitivity(body: SensitivityIn, db: Session = Depends(get_db), user: User = Depends(require(Role.analyst))):
    lat, lon = wsvc.snap(body.lat, body.lon)
    values = list(asvc.SENS_PARAMS[body.param][2])
    res = await run_in_threadpool(asvc.sensitivity, body.model_dump(mode="json"), body.base.model_dump(), body.param, values)
    return _save(db, user, "sensitivity", body, lat, lon, len(values), res)


@router.get("", response_model=Page[AnalysisSummaryOut])
def list_(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0), db: Session = Depends(get_db),
          _: User = Depends(current_user)):
    total = db.scalar(select(func.count(WeatherAnalysis.id)))
    items = db.scalars(select(WeatherAnalysis).order_by(WeatherAnalysis.id.desc()).limit(limit).offset(offset)).all()
    return Page(items=items, total=total, limit=limit, offset=offset)


@router.get("/{aid}", response_model=AnalysisOut)
def get(aid: int, db: Session = Depends(get_db), _: User = Depends(current_user)):
    a = db.get(WeatherAnalysis, aid)
    if not a:
        raise HTTPException(404, "analysis not found")
    return a


@router.delete("/{aid}", status_code=204)
def delete(aid: int, db: Session = Depends(get_db), user: User = Depends(require(Role.analyst))):
    a = db.get(WeatherAnalysis, aid)
    if not a:
        raise HTTPException(404, "analysis not found")
    if user.role != Role.admin and a.created_by != user.id:
        raise HTTPException(403, "only the owner or an admin can delete an analysis")
    db.delete(a); db.commit()
    log(db, user, "delete", "analysis", aid)
