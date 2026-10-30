"""Model registry, aggregate statistics, audit trail."""
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..audit import log
from ..db import get_db
from ..models import Asset, AuditLog, Estimate, Method, ModelVersion, Role, Scenario, ThreatAssessment, User, WeatherProfile
from ..schemas import MethodStat, ModelVersionOut, Stats
from ..security import current_user, require

router = APIRouter(prefix="/api", tags=["models & statistics"])


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
