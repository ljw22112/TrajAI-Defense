"""Live track replay over WebSocket: measurements arrive at (scaled) real time and a streaming EKF updates per step.

ws://host/api/ws/scenarios/{id}/live?token=<JWT>&speed=4&weather=1
messages:  {"type":"meas", ...}  {"type":"state", ...}  {"type":"done", "n": ...}  {"type":"error", "detail": ...}
"""
import asyncio

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from trajai.filters import StreamingEKF

from .. import alerts
from ..config import get_settings
from ..db import SessionLocal
from ..models import Scenario, User
from ..security import decode_token
from ..services import weather as wsvc

router = APIRouter(tags=["live stream"])


PRED_MIN, PRED_EVERY = 12, 4


def _pinn_model(db, has_weather: bool):
    from sqlalchemy import select as _sel
    from ..models import ModelVersion
    mv = db.scalar(_sel(ModelVersion).where(ModelVersion.is_active.is_(True)))
    if mv is not None and mv.uses_weather and not has_weather:
        mv = db.scalar(_sel(ModelVersion).where(ModelVersion.uses_weather.is_(False), ModelVersion.kind.like("pinn-hybrid%"))
                       .order_by(ModelVersion.id.desc())) or mv
    return mv


def _predict_prefix(s, n: int, bank, mv):
    """Run the served PINN on the first n measurements, as if the track ended there."""
    from types import SimpleNamespace
    from ..services import estimation
    part = SimpleNamespace(measurements=s.measurements[:n], t_c=s.measurements[n - 1].t, sigma_r=s.sigma_r,
                           sigma_ang=s.sigma_ang, seed=s.seed)
    try:
        r = estimation.run_pinn(part, bank, mv)
    except FileNotFoundError:
        return None
    miss = float(np.hypot(r["ip"][0] - s.impact_x, r["ip"][1] - s.impact_y))
    return {"type": "pred", "t": part.t_c, "n": n, "ip_x": float(r["ip"][0]), "ip_y": float(r["ip"][1]), "cep": r["cep"],
            "miss": miss, "latency_ms": r["latency_ms"], "model": mv.name}


@router.websocket("/api/ws/alerts")
async def alert_feed(ws: WebSocket, token: str = ""):
    """Pushes {"type": "alert", ...} whenever any user's estimate reaches threat level >= 4."""
    await ws.accept()
    db = SessionLocal()
    try:
        uid = decode_token(token)
        user = db.get(User, uid)
        if not user or not user.is_active:
            raise ValueError
    except Exception:
        await ws.send_json({"type": "error", "detail": "invalid token"}); await ws.close(code=4401); db.close(); return
    db.close()
    q = alerts.subscribe()
    try:
        await ws.send_json({"type": "ready"})
        while True:
            await ws.send_json(await q.get())
    except WebSocketDisconnect:
        pass
    finally:
        alerts.unsubscribe(q)


@router.websocket("/api/ws/scenarios/{sid}/live")
async def live(ws: WebSocket, sid: int, token: str = "", speed: float = 0.0, weather: int = 1):
    await ws.accept()
    db = SessionLocal()
    try:
        try:
            uid = decode_token(token)
        except Exception:
            await ws.send_json({"type": "error", "detail": "invalid token"}); await ws.close(code=4401); return
        user = db.get(User, uid)
        if not user or not user.is_active:
            await ws.send_json({"type": "error", "detail": "user inactive"}); await ws.close(code=4401); return
        s = db.scalar(select(Scenario).where(Scenario.id == sid).options(selectinload(Scenario.measurements), selectinload(Scenario.weather)))
        if not s:
            await ws.send_json({"type": "error", "detail": "scenario not found"}); await ws.close(code=4404); return
        bank = wsvc.bank_of(s.weather) if weather else None
        ekf = StreamingEKF(0.5, s.sigma_r, s.sigma_ang, atm=bank[np.zeros(1, int)] if bank is not None else None)
        speed = speed or get_settings().stream_speed
        mv = _pinn_model(db, bank is not None)
        for i, m in enumerate(s.measurements):
            await ws.send_json({"type": "meas", "k": m.k, "t": m.t, "x": m.x, "y": m.y, "z": m.z})
            st = await asyncio.to_thread(ekf.update, [m.range_m, m.az, m.el])
            if st:
                await ws.send_json({"type": "state", "t": m.t, **st})
            # PINN impact prediction from the track so far (every PRED_EVERY measurements once the track is long enough)
            last = i == len(s.measurements) - 1
            if mv is not None and i + 1 >= PRED_MIN and ((i + 1) % PRED_EVERY == 0 or last):
                pred = await asyncio.to_thread(_predict_prefix, s, i + 1, bank, mv)
                if pred:
                    await ws.send_json(pred)
            if speed < 1e3:
                await asyncio.sleep(0.5 / speed)
        await ws.send_json({"type": "done", "n": len(s.measurements), "atmosphere": "weather" if bank is not None else "standard"})
        await ws.close()
    except WebSocketDisconnect:
        pass
    finally:
        db.close()
