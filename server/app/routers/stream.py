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

from ..config import get_settings
from ..db import SessionLocal
from ..models import Scenario, User
from ..security import decode_token
from ..services import weather as wsvc

router = APIRouter(tags=["live stream"])


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
        for m in s.measurements:
            await ws.send_json({"type": "meas", "k": m.k, "t": m.t, "x": m.x, "y": m.y, "z": m.z})
            st = await asyncio.to_thread(ekf.update, [m.range_m, m.az, m.el])
            if st:
                await ws.send_json({"type": "state", "t": m.t, **st})
            if speed < 1e3:
                await asyncio.sleep(0.5 / speed)
        await ws.send_json({"type": "done", "n": len(s.measurements), "atmosphere": "weather" if bank is not None else "standard"})
        await ws.close()
    except WebSocketDisconnect:
        pass
    finally:
        db.close()
