"""Weather-impact analyses: create (runs the batched study), list, read, delete."""
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..audit import log
from ..db import get_db
from ..models import Role, User, WeatherAnalysis
from ..schemas import AnalysisIn, AnalysisOut, AnalysisSummaryOut, Page
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
    a = WeatherAnalysis(name=body.name, created_by=user.id, lat=lat, lon=lon, n_profiles=len(rows), params=p,
                        summary=res["summary"], items=res["items"])
    db.add(a); db.commit(); db.refresh(a)
    log(db, user, "create", "analysis", a.id, {"n": len(rows), "reduction": res["summary"]["reduction"]})
    return a


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
