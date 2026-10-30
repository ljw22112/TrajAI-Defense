import secrets

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from ..audit import log
from ..db import get_db
from ..models import Asset, Estimate, Measurement, Method, ModelVersion, Role, Scenario, ThreatAssessment, User
from ..schemas import EstimateIn, EstimateOut, Page, ScenarioIn, ScenarioOut, ScenarioSummary
from ..security import current_user, require
from ..services import estimation, simulation, threat
from ..services import weather as wsvc

router = APIRouter(prefix="/api/scenarios", tags=["scenarios & estimates"])


def _summary(s: Scenario, n: int, best: float | None) -> ScenarioSummary:
    out = ScenarioSummary.model_validate(s)
    out.n_estimates, out.best_miss_m = n, best
    return out


@router.get("", response_model=Page[ScenarioSummary])
def list_scenarios(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0), q: str | None = None,
                   weather: bool | None = None, db: Session = Depends(get_db), _: User = Depends(current_user)):
    stmt = select(Scenario)
    if q:
        stmt = stmt.where(Scenario.name.ilike(f"%{q}%"))
    if weather is not None:
        stmt = stmt.where(Scenario.weather_id.is_not(None) if weather else Scenario.weather_id.is_(None))
    total = db.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = db.scalars(stmt.order_by(Scenario.created_at.desc(), Scenario.id.desc()).limit(limit).offset(offset)).all()
    ids = [r.id for r in rows]
    agg = {sid: (n, best) for sid, n, best in db.execute(
        select(Estimate.scenario_id, func.count(Estimate.id), func.min(Estimate.miss_m)).where(Estimate.scenario_id.in_(ids)).group_by(Estimate.scenario_id))}
    return Page(items=[_summary(r, *agg.get(r.id, (0, None))) for r in rows], total=total, limit=limit, offset=offset)


@router.post("", response_model=ScenarioOut, status_code=201, summary="Create and simulate a scenario")
async def create_scenario(body: ScenarioIn, db: Session = Depends(get_db), user: User = Depends(require(Role.analyst))):
    wrow = wsvc.get_or_fetch(db, body.weather_date, body.weather_hour) if body.weather_date else None
    bank = wsvc.bank_of(wrow)
    seed = body.seed if body.seed is not None else secrets.randbelow(2**31)
    sim = await run_in_threadpool(simulation.simulate, body.h0, body.v0, body.gamma_deg, body.psi_deg, body.beta,
                                  body.obs_fraction, body.sigma_r, body.sigma_ang_mrad * 1e-3, seed, bank)
    s = Scenario(name=body.name, created_by=user.id, h0=body.h0, v0=body.v0, gamma_deg=body.gamma_deg, psi_deg=body.psi_deg,
                 beta=body.beta, obs_fraction=body.obs_fraction, sigma_r=body.sigma_r, sigma_ang=body.sigma_ang_mrad * 1e-3,
                 seed=seed, weather_id=wrow.id if wrow else None, t_c=sim["t_c"], impact_time=sim["impact_time"],
                 impact_x=sim["impact_x"], impact_y=sim["impact_y"], apogee=sim["apogee"], truth=sim["truth"])
    s.measurements = [Measurement(**m) for m in sim["measurements"]]
    db.add(s); db.commit()
    log(db, user, "create", "scenario", s.id, {"weather_id": s.weather_id, "seed": seed})
    return get_scenario(s.id, db, user)


@router.get("/{sid}", response_model=ScenarioOut)
def get_scenario(sid: int, db: Session = Depends(get_db), _: User = Depends(current_user)):
    s = db.scalar(select(Scenario).where(Scenario.id == sid).options(
        selectinload(Scenario.measurements), selectinload(Scenario.estimates).selectinload(Estimate.threat), selectinload(Scenario.weather)))
    if not s:
        raise HTTPException(404, "scenario not found")
    out = ScenarioOut.model_validate(s)
    out.n_estimates = len(s.estimates)
    out.best_miss_m = min((e.miss_m for e in s.estimates), default=None)
    return out


@router.delete("/{sid}", status_code=204)
def delete_scenario(sid: int, db: Session = Depends(get_db), user: User = Depends(require(Role.analyst))):
    s = db.get(Scenario, sid)
    if not s:
        raise HTTPException(404, "scenario not found")
    if user.role != Role.admin and s.created_by != user.id:
        raise HTTPException(403, "only the owner or an admin can delete a scenario")
    db.delete(s); db.commit()
    log(db, user, "delete", "scenario", sid)


@router.post("/{sid}/estimates", response_model=EstimateOut, status_code=201, summary="Run an estimator and assess the threat")
async def create_estimate(sid: int, body: EstimateIn, db: Session = Depends(get_db), user: User = Depends(require(Role.analyst))):
    s = db.scalar(select(Scenario).where(Scenario.id == sid).options(selectinload(Scenario.measurements), selectinload(Scenario.weather)))
    if not s:
        raise HTTPException(404, "scenario not found")
    if body.method in (Method.ekf_wx, Method.ukf_wx) and s.weather is None:
        raise HTTPException(422, "this scenario has no weather profile; use ekf_std or pinn")
    bank = wsvc.bank_of(s.weather)
    mv = None
    if body.method == Method.pinn:
        mv = db.scalar(select(ModelVersion).where(ModelVersion.is_active.is_(True)).order_by(ModelVersion.id.desc()))
        try:
            res = await run_in_threadpool(estimation.run_pinn, s, bank, mv)
        except FileNotFoundError as e:
            raise HTTPException(503, str(e))
    else:
        res = await run_in_threadpool(estimation.run_filter, s, bank, body.method)
    sc = estimation.score(res, s)
    e = Estimate(scenario_id=s.id, method=body.method, model_version_id=mv.id if mv else None, created_by=user.id,
                 ip_x=float(res["ip"][0]), ip_y=float(res["ip"][1]), cep=res["cep"], tot=res["tot"], tot_std=res["tot_std"],
                 beta_hat=res["beta_hat"], nis=res["nis"], latency_ms=res["latency_ms"], trajectory=res["trajectory"], **sc)
    assets = db.scalars(select(Asset)).all()
    th = threat.assess(res["ip"], res["sigma"], res["trajectory"], assets)
    e.threat = ThreatAssessment(**th)
    db.add(e); db.commit(); db.refresh(e)
    log(db, user, "estimate", "scenario", s.id, {"method": body.method.value, "miss_m": round(sc["miss_m"], 1)})
    return e


@router.get("/{sid}/estimates", response_model=list[EstimateOut])
def list_estimates(sid: int, db: Session = Depends(get_db), _: User = Depends(current_user)):
    return db.scalars(select(Estimate).where(Estimate.scenario_id == sid).options(selectinload(Estimate.threat)).order_by(Estimate.id)).all()
