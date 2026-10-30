"""Export a read-only JSON snapshot of a running server for the static demo (GitHub Pages, /app/demo/).

File names follow web/src/api/client.ts demoGet(): "/api/scenarios/3" -> "scenarios_3.json".
Usage: python scripts/export_demo.py [--base http://127.0.0.1:8000] [--out ../app/demo]
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import asyncio

import httpx
import websockets


async def record_live(base: str, tok: str, sid: int, weather: int) -> list:
    url = base.replace("http", "ws", 1) + f"/api/ws/scenarios/{sid}/live?token={tok}&speed=100000&weather={weather}"
    msgs = []
    async with websockets.connect(url, max_size=None) as ws:
        async for m in ws:
            msgs.append(json.loads(m))
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "app" / "demo"))
    ap.add_argument("--password", default="admin1234")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    c = httpx.Client(base_url=a.base, timeout=60)
    tok = c.post("/api/auth/login", data={"username": "admin@trajai.dev", "password": a.password}).json()["access_token"]
    c.headers["Authorization"] = f"Bearer {tok}"

    def save(name, obj):
        (out / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))

    for path, name in [("/api/stats", "stats"), ("/api/assets", "assets"), ("/api/models", "models"), ("/api/weather/cached", "weather_cached"),
                       ("/api/threats", "threats")]:
        save(name, c.get(path).json())
    an = c.get("/api/analyses", params={"limit": 100}).json()
    save("analyses", an)
    for x in an["items"]:
        save(f"analyses_{x['id']}", c.get(f"/api/analyses/{x['id']}").json())
    page = c.get("/api/scenarios", params={"limit": 100}).json()
    save("scenarios", page)
    for s in page["items"]:
        save(f"scenarios_{s['id']}", c.get(f"/api/scenarios/{s['id']}").json())
        # record the live replay (streaming EKF) so the static demo can play it back without a server
        for w in ((0, 1) if s.get("weather_id") else (0,)):
            save(f"live_{s['id']}_{w}", asyncio.run(record_live(a.base, tok, s["id"], w)))
    for w in c.get("/api/weather/cached").json():
        utc = datetime.fromisoformat(w["valid_time"]).astimezone(__import__("datetime").timezone.utc)
        save(f"weather_{utc:%Y-%m-%dT%H}", w)
        save(f"weather_{w['id']}_curves", c.get(f"/api/weather/{w['id']}/curves").json())
    (out / "estimates.csv").write_text(c.get("/api/scenarios/export.csv").text)
    print(f"exported {len(list(out.glob('*.json')))} files to {out}")


if __name__ == "__main__":
    main()
