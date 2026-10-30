"""Model registry, aggregate statistics, audit trail."""
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..audit import log
from ..db import get_db
from ..models import Asset, AuditLog, Estimate, Method, ModelVersion, Role, Scenario, ThreatAssessment, User, WeatherProfile
from ..schemas import MethodStat, ModelVersionOut, Stats, ThreatBoardRow
from ..security import current_user, require

router = APIRouter(prefix="/api", tags=["models & statistics"])


# most accurate first, by the paired comparisons of §9.4 (weather-aware filters < PINN < standard EKF)
PREFERENCE = [Method.ekf_wx, Method.ukf_wx, Method.pinn, Method.ekf_std]


@router.get("/threats", response_model=list[ThreatBoardRow],
            summary="Engagement priority: one row per scenario (its most accurate latest estimate), most urgent first")
def threat_board(db: Session = Depends(get_db), _: User = Depends(current_user)):
    rows = db.scalars(select(Estimate).join(ThreatAssessment).order_by(Estimate.id)).all()
    latest: dict[tuple[int, Method], Estimate] = {}
    for e in rows:
        latest[(e.scenario_id, e.method)] = e
    best: dict[int, Estimate] = {}
    for (sid, m), e in latest.items():
        cur = best.get(sid)
        if cur is None or PREFERENCE.index(m) < PREFERENCE.index(cur.method):
            best[sid] = e
    names = {a.id: a.name for a in db.scalars(select(Asset)).all()}
    out = []
    for e in best.values():
        th, t_c = e.threat, e.scenario.t_c
        wins = [w for w in (th.windows or {}).values() if w]
        out.append(ThreatBoardRow(
            scenario_id=e.scenario_id, scenario=e.scenario.name, estimate_id=e.id, method=e.method, model=e.model_name,
            level=th.level, max_prob=th.max_prob, top_asset=names.get(th.top_asset_id) if th.top_asset_id else None,
            impact_time=e.tot, time_to_impact=e.tot - t_c,
            first_window_open=min(w["t_in"] for w in wins) - t_c if wins else None,
            first_window_close=max(w["t_out"] for w in wins) - t_c if wins else None, miss_m=e.miss_m))
    out.sort(key=lambda r: (-r.level, -r.max_prob, r.time_to_impact))
    return out


@router.get("/models", response_model=list[ModelVersionOut])
def models(db: Session = Depends(get_db), _: User = Depends(current_user)):
    return db.scalars(select(ModelVersion).order_by(ModelVersion.id)).all()


@router.post("/models/{mid}/activate", response_model=ModelVersionOut)
def activate(mid: int, db: Session = Depends(get_db), admin: User = Depends(require(Role.admin))):
    m = db.get(ModelVersion, mid)
    if not m:
        raise HTTPException(404, "model not found")
    for other in db.scalars(select(ModelVersion)).all():
        other.is_active = other.id == mid
    db.commit(); db.refresh(m)
    log(db, admin, "activate", "model", mid)
    return m


@router.get("/stats", response_model=Stats)
def stats(db: Session = Depends(get_db), _: User = Depends(current_user)):
    by = []
    for meth in Method:
        rows = db.execute(select(Estimate.miss_m, Estimate.inside_cep, Estimate.latency_ms, ModelVersion.name)
                          .outerjoin(ModelVersion, Estimate.model_version_id == ModelVersion.id).where(Estimate.method == meth)).all()
        groups: dict[str | None, list] = {}
        for r in rows:
            groups.setdefault(r[3], []).append(r[:3])
        if not groups:
            by.append(MethodStat(method=meth, n=0, median_miss_m=None, coverage=None, median_latency_ms=None))
        for model, g in sorted(groups.items(), key=lambda kv: kv[0] or ""):
            a = np.array([[m, float(c), lat] for m, c, lat in g])
            by.append(MethodStat(method=meth, model=model, n=len(g), median_miss_m=float(np.median(a[:, 0])),
                                 coverage=float(a[:, 1].mean()), median_latency_ms=float(np.median(a[:, 2]))))
    levels = {str(lv): n for lv, n in db.execute(select(ThreatAssessment.level, func.count()).group_by(ThreatAssessment.level))}
    return Stats(scenarios=db.scalar(select(func.count(Scenario.id))), estimates=db.scalar(select(func.count(Estimate.id))),
                 weather_profiles=db.scalar(select(func.count(WeatherProfile.id))), assets=db.scalar(select(func.count(Asset.id))),
                 by_method=by, threat_levels=levels)


@router.get("/audit")
def audit(limit: int = 50, db: Session = Depends(get_db), _: User = Depends(require(Role.admin))):
    rows = db.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(min(limit, 500))).all()
    return [dict(id=r.id, user_id=r.user_id, action=r.action, entity=r.entity, entity_id=r.entity_id, detail=r.detail, at=r.at) for r in rows]
