"""Populate a running server through its public API (realistic demo data).

Usage: python scripts/seed_demo.py [--base http://127.0.0.1:8000] [--password admin1234]
"""
import argparse
import time

import httpx

SCENARIOS = [
    dict(name="10/03 00Z 중거리", weather_date="2026-10-03", weather_hour=0, seed=301),
    dict(name="10/08 12Z 장거리 고β", weather_date="2026-10-08", weather_hour=12, seed=302, h0=22000, v0=1900, gamma_deg=50, beta=8000),
    dict(name="10/15 00Z 제트기류", weather_date="2026-10-15", weather_hour=0, seed=303),
    dict(name="10/21 06Z 단거리 저β", weather_date="2026-10-21", weather_hour=6, seed=304, h0=10000, v0=1000, gamma_deg=40, beta=900),
    dict(name="10/27 18Z 짧은 관측 5%", weather_date="2026-10-27", weather_hour=18, seed=305, obs_fraction=0.05),
    dict(name="표준 대기 기준 시나리오", weather_date=None, seed=306),
    dict(name="표준 대기 고잡음", weather_date=None, seed=307, sigma_r=50, sigma_ang_mrad=1.5),
    # outside the training envelope (v0 > 1900 m/s, beta > 10000): the console's OOD guard hands the PINN request to the EKF
    dict(name="학습 범위 밖 표적 (고속 · 고β)", weather_date=None, seed=911, v0=2150, beta=14000),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--password", default="admin1234")
    a = ap.parse_args()
    c = httpx.Client(base_url=a.base, timeout=120)
    tok = c.post("/api/auth/login", data={"username": "admin@trajai.dev", "password": a.password}).json()["access_token"]
    c.headers["Authorization"] = f"Bearer {tok}"
    for email, role in [("analyst@trajai.dev", "analyst"), ("viewer@trajai.dev", "viewer")]:
        c.post("/api/users", json={"email": email, "password": "password123", "role": role, "full_name": role.title()})
    existing = {x["name"] for x in c.get("/api/scenarios", params={"limit": 100}).json()["items"]}
    for body in SCENARIOS:
        if body["name"] in existing:
            continue
        for attempt in range(6):                    # weather provider rate limit -> server answers 503
            r = c.post("/api/scenarios", json=body)
            if r.status_code not in (502, 503):
                break
            print("  weather provider busy, retrying in 65 s:", r.json()["detail"])
            time.sleep(65)
        r.raise_for_status()
        s = r.json()
        methods = ["ekf_std", "pinn"] + (["ekf_wx", "ukf_wx"] if s["weather"] else [])
        line = []
        for m in methods:
            e = c.post(f"/api/scenarios/{s['id']}/estimates", json={"method": m})
            line.append(f"{m}={e.json()['miss_m']:.0f}m" if e.status_code == 201 else f"{m}=HTTP{e.status_code}")
        print(f"#{s['id']:<3} {s['name']:<22} range {s['impact_x']/1000:6.1f} km  " + "  ".join(line))


if __name__ == "__main__":
    main()
