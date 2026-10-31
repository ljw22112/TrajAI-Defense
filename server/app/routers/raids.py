"""Raids: several scenarios as one salvo, laid out on a common clock (engagement-window timeline)."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..audit import log
from ..db import get_db
from ..models import Asset, Estimate, Raid, Role, Scenario, User
from ..schemas import RaidIn, RaidOut, RaidRow, RaidTrack
from ..security import current_user, require
from .insights import PREFERENCE

router = APIRouter(prefix="/api/raids", tags=["raids (multi-target)"])


def _timeline(db: Session, r: Raid) -> RaidOut:
    ids = [m["scenario_id"] for m in r.members]
    scn = {s.id: s for s in db.scalars(select(Scenario).where(Scenario.id.in_(ids))
                                       .options(selectinload(Scenario.estimates).selectinload(Estimate.threat))).all()}
    names = {a.id: a.name for a in db.scalars(select(Asset)).all()}
    tracks = []
    for m in r.members:
        s = scn.get(m["scenario_id"])
        if s is None:                                  # scenario deleted after the raid was saved
            continue
        latest = {}
        for e in s.estimates:
            if e.threat is not None:
                latest[e.method] = e
        best = min(latest.values(), key=lambda e: PREFERENCE.index(e.method)) if latest else None
        off = float(m["offset_s"])
        impact = (best.tot if best else s.impact_time) + off
        wins = []
        if best:
            for band, w in (best.threat.windows or {}).items():
                if w:
                    wins.append(dict(band=band, open=w["t_in"] + off, close=w["t_out"] + off))
        th = best.threat if best else None
        tracks.append(RaidTrack(scenario_id=s.id, scenario=s.name, offset_s=off, track_end=s.t_c + off, impact=impact, windows=wins,
                                level=th.level if th else None, max_prob=th.max_prob if th else None,
                                top_asset=names.get(th.top_asset_id) if th and th.top_asset_id else None,
                                method=best.method if best else None))
    # sweep line over window open/close events -> peak number of simultaneous engagements
    ev = sorted([(w["open"], 1) for t in tracks for w in t.windows] + [(w["close"], -1) for t in tracks for w in t.windows])
    cur = peak = 0; peak_at = None
    for t, d in ev:
        cur += d
        if cur > peak:
            peak, peak_at = cur, t
    return RaidOut(id=r.id, name=r.name, created_at=r.created_at, tracks=tracks, peak_simultaneous=peak, peak_at=peak_at,
                   first_impact=min((t.impact for t in tracks), default=None))


@router.post("", response_model=RaidOut, status_code=201)
def create(body: RaidIn, db: Session = Depends(get_db), user: User = Depends(require(Role.analyst))):
    ids = [m.scenario_id for m in body.members]
    found = set(db.scalars(select(Scenario.id).where(Scenario.id.in_(ids))).all())
    if missing := [i for i in ids if i not in found]:
        raise HTTPException(422, f"unknown scenario ids: {missing}")
    r = Raid(name=body.name, created_by=user.id, members=[m.model_dump() for m in body.members])
    db.add(r); db.commit(); db.refresh(r)
    log(db, user, "create", "raid", r.id, {"n": len(ids)})
    return _timeline(db, r)


@router.get("", response_model=list[RaidRow])
def list_(db: Session = Depends(get_db), _: User = Depends(current_user)):
    return [RaidRow(id=r.id, name=r.name, created_at=r.created_at, n=len(r.members))
            for r in db.scalars(select(Raid).order_by(Raid.id.desc())).all()]


@router.get("/{rid}", response_model=RaidOut)
def get(rid: int, db: Session = Depends(get_db), _: User = Depends(current_user)):
    r = db.get(Raid, rid)
    if not r:
        raise HTTPException(404, "raid not found")
    return _timeline(db, r)


@router.delete("/{rid}", status_code=204)
def delete(rid: int, db: Session = Depends(get_db), user: User = Depends(require(Role.analyst))):
    r = db.get(Raid, rid)
    if not r:
        raise HTTPException(404, "raid not found")
    if user.role != Role.admin and r.created_by != user.id:
        raise HTTPException(403, "only the owner or an admin can delete a raid")
    db.delete(r); db.commit()
    log(db, user, "delete", "raid", rid)
