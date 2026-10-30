from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User, WeatherProfile
from ..schemas import WeatherOut, WeatherQuery
from ..security import current_user
from ..services import weather as wsvc

router = APIRouter(prefix="/api/weather", tags=["weather"])


@router.get("", response_model=WeatherOut, summary="Profile for a date/hour (cached, fetched from Open-Meteo on a miss)")
def get_profile(day: date = Query(..., alias="date"), hour: int = Query(0, ge=0, le=23), db: Session = Depends(get_db),
                _: User = Depends(current_user)):
    q = WeatherQuery(date=day, hour=hour)
    return wsvc.get_or_fetch(db, q.date, q.hour)


@router.get("/cached", response_model=list[WeatherOut])
def cached(limit: int = 50, db: Session = Depends(get_db), _: User = Depends(current_user)):
    return db.scalars(select(WeatherProfile).order_by(WeatherProfile.valid_time.desc()).limit(min(limit, 500))).all()


@router.get("/{wid}/curves", summary="Altitude curves (T, wind, density ratio) for plotting")
def curves(wid: int, db: Session = Depends(get_db), _: User = Depends(current_user)):
    row = db.get(WeatherProfile, wid)
    if not row:
        from fastapi import HTTPException
        raise HTTPException(404, "weather profile not found")
    return wsvc.profile_curves(row)
