"""FastAPI application factory.

Run:  uvicorn app.main:app --reload            (from server/, SQLite by default)
      TRAJAI_DATABASE_URL=postgresql+psycopg2://... alembic upgrade head && uvicorn app.main:app
Docs: /api/docs (Swagger UI), /api/redoc, /api/openapi.json
"""
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:                       # make the trajai package importable without installation
    sys.path.insert(0, str(ROOT))

from . import __version__  # noqa: E402
from .config import get_settings  # noqa: E402
from .db import Base, SessionLocal, engine  # noqa: E402
from .models import Asset, ModelVersion, Role, User  # noqa: E402
from .routers import analysis, assets, auth, insights, raids, scenarios, stream, weather  # noqa: E402
from .security import hash_password  # noqa: E402

MODELS = [  # registry seed: (name, kind, onnx file, uses weather, metrics key)
    ("wx_hybrid_date", "pinn-hybrid · weather input", "wx_hybrid_date.onnx", True, "wx_hybrid_date"),
    ("pinn_tf_hybrid_v2", "pinn-hybrid · std atm · v2", "pinn_tf_hybrid_v2.onnx", False, "pinn_tf_hybrid_v2"),
    ("pinn_tf_hybrid", "pinn-hybrid · std atm · v1", "pinn_tf_hybrid.onnx", False, "pinn_tf_hybrid"),
    ("ens_hybrid_v2", "pinn-hybrid · ensemble ×3",
     "pinn_tf_hybrid_v2.onnx;pinn_tf_hybrid_v2_s1.onnx;pinn_tf_hybrid_v2_s2.onnx", False, "ens_hybrid_v2"),
    ("pinn_tf_v2", "pinn · track only (β prior)", "pinn_tf_v2.onnx", False, "pinn_tf_v2"),
    ("pinn_tf_v2_rts", "pinn-rts · RTS track features", "pinn_tf_v2_rts.onnx", False, "pinn_tf_v2_rts"),
]


def seed(db) -> None:
    s = get_settings()
    if not db.scalar(select(User).where(User.email == s.admin_email)):
        db.add(User(email=s.admin_email, full_name="Administrator", hashed_password=hash_password(s.admin_password), role=Role.admin))
    if not db.scalar(select(Asset.id).limit(1)):
        from trajai.threat import ASSETS
        for i, a in enumerate(ASSETS):
            db.add(Asset(name=a["name"], x_km=a["x"], y_km=a["y"], radius_km=a["radius"], priority=5 - i % 3))
    metrics_path = ROOT / "results" / "metrics.json"
    allm = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    fb_path = ROOT / "results" / "feedback_rts.json"
    cal_path = ROOT / "results" / "calibration.json"
    for k, v in (json.loads(cal_path.read_text()) if cal_path.exists() else {}).items():
        if k in allm:
            allm[k] = {**allm[k], "cep_scale": v["cep_scale"], "cep_scale_fitted_on": v["fitted_on"],
                       **({"cep90_scale": v["cep90_scale"]} if "cep90_scale" in v else {})}
    have_active = db.scalar(select(ModelVersion.id).where(ModelVersion.is_active.is_(True)))
    for name, kind, fname, uses_wx, key in MODELS:
        paths = [s.model_dir / f for f in fname.split(";")]
        path = paths[0]
        row = db.scalar(select(ModelVersion).where(ModelVersion.name == name))
        if row is None and all(p.exists() for p in paths):
            m = {k: v for k, v in allm.get(key, {}).items() if k != "args"}
            if name == "pinn_tf_v2" and fb_path.exists():
                m["prior_sd"] = json.loads(fb_path.read_text())["feedback"]["prior_sd"]
            row = ModelVersion(name=name, kind=kind, file_path=";".join(str(p) for p in paths), uses_weather=uses_wx, metrics=m)
            db.add(row)
        elif row is not None and key in allm:
            m = {k: v for k, v in allm[key].items() if k != "args"}           # refresh after re-evaluation
            if name == "pinn_tf_v2" and fb_path.exists():
                m["prior_sd"] = json.loads(fb_path.read_text())["feedback"]["prior_sd"]
            row.metrics = m
        if row is not None and not have_active and path.exists() and kind.startswith(("pinn-hybrid", "pinn-rts")):
            row.is_active = True
            have_active = True
    db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if get_settings().database_url.startswith("sqlite"):
        Base.metadata.create_all(engine)            # dev convenience; PostgreSQL uses Alembic migrations
    with SessionLocal() as db:
        seed(db)
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title="TrajAI-Defense API", version=__version__, lifespan=lifespan,
                  docs_url="/api/docs", redoc_url="/api/redoc", openapi_url="/api/openapi.json",
                  description="Weather-aware ballistic trajectory estimation: simulation, EKF/UKF, PINN (ONNX), threat assessment.")
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        resp.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        return resp

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/api/health", tags=["meta"])
    def health():
        return {"status": "ok", "version": __version__}

    for r in (auth.router, assets.router, weather.router, scenarios.router, analysis.router, raids.router, insights.router, stream.router):
        app.include_router(r)
    dist = ROOT / "web" / "dist"
    if dist.exists():                                # single-container deployment: serve the built SPA at /
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app


app = create_app()
