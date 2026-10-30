import pytest

from conftest import CALLS


def test_health_and_openapi(client):
    assert client.get("/api/health").json()["status"] == "ok"
    spec = client.get("/api/openapi.json").json()
    assert "/api/scenarios/{sid}/estimates" in spec["paths"]


def test_login_failures_and_me(client, admin):
    assert client.post("/api/auth/login", data={"username": "admin@trajai.dev", "password": "nope"}).status_code == 401
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401
    assert client.get("/api/auth/me", headers=admin).json()["role"] == "admin"


def test_rbac_on_assets(client, admin, users):
    body = {"name": "TEST-RBAC", "x_km": 10, "y_km": 1, "radius_km": 2, "priority": 4}
    assert client.post("/api/assets", json=body, headers=users["viewer"]).status_code == 403
    r = client.post("/api/assets", json=body, headers=users["analyst"])
    assert r.status_code == 201
    aid = r.json()["id"]
    assert client.post("/api/assets", json=body, headers=users["analyst"]).status_code == 409        # unique name
    assert client.delete(f"/api/assets/{aid}", headers=users["analyst"]).status_code == 403          # delete = admin
    assert client.delete(f"/api/assets/{aid}", headers=admin).status_code == 204
    assert client.get("/api/users", headers=users["analyst"]).status_code == 403


def test_asset_validation(client, users):
    bad = {"name": "", "x_km": 10, "y_km": 1, "radius_km": -2, "priority": 9}
    r = client.post("/api/assets", json=bad, headers=users["analyst"])
    assert r.status_code == 422
    fields = {e["loc"][-1] for e in r.json()["detail"]}
    assert {"name", "radius_km", "priority"} <= fields


def test_weather_is_cached(client, users):
    n0 = CALLS["n"]
    a = client.get("/api/weather", params={"date": "2026-10-15", "hour": 6}, headers=users["viewer"]).json()
    b = client.get("/api/weather", params={"date": "2026-10-15", "hour": 6}, headers=users["viewer"]).json()
    assert a["id"] == b["id"] and CALLS["n"] == n0 + 1
    assert 45 < a["max_wind"] < 55
    curves = client.get(f"/api/weather/{a['id']}/curves", headers=users["viewer"]).json()
    assert len(curves["h"]) == len(curves["T"]) == 41
    assert client.get("/api/weather", params={"date": "2019-01-01"}, headers=users["viewer"]).status_code == 422


@pytest.fixture(scope="module")
def scenarios(client, users):
    std = client.post("/api/scenarios", json={"name": "std", "seed": 11}, headers=users["analyst"])
    wx = client.post("/api/scenarios", json={"name": "jet", "seed": 12, "weather_date": "2026-10-15", "weather_hour": 0}, headers=users["analyst"])
    assert std.status_code == 201 and wx.status_code == 201, (std.text, wx.text)
    return std.json(), wx.json()


def test_scenario_simulation(scenarios, client, users):
    std, wx = scenarios
    assert std["weather"] is None and wx["weather"] is not None
    assert len(std["measurements"]) >= 13 and std["t_c"] < std["impact_time"]
    assert abs(std["truth"]["x"][-1] - std["impact_x"]) < 0.1          # stored rounded to 0.1 m
    assert abs(wx["impact_x"] - std["impact_x"]) > 100            # the jet moves the impact point
    page = client.get("/api/scenarios", params={"limit": 1}, headers=users["viewer"]).json()
    assert page["total"] >= 2 and len(page["items"]) == 1
    only_wx = client.get("/api/scenarios", params={"weather": True}, headers=users["viewer"]).json()
    assert all(s["weather_id"] for s in only_wx["items"])


def test_viewer_cannot_create_scenario(client, users):
    assert client.post("/api/scenarios", json={"name": "x"}, headers=users["viewer"]).status_code == 403


def test_estimates_and_threat(scenarios, client, users):
    std, wx = scenarios
    h = users["analyst"]
    r = client.post(f"/api/scenarios/{std['id']}/estimates", json={"method": "ekf_std"}, headers=h)
    assert r.status_code == 201, r.text
    e = r.json()
    assert e["miss_m"] < 3_000 and e["cep"] > 0 and 1 <= e["threat"]["level"] <= 5
    assert client.post(f"/api/scenarios/{std['id']}/estimates", json={"method": "ekf_wx"}, headers=h).status_code == 422
    miss = {}
    for m in ("ekf_std", "ekf_wx"):
        r = client.post(f"/api/scenarios/{wx['id']}/estimates", json={"method": m}, headers=h)
        assert r.status_code == 201, r.text
        miss[m] = r.json()["miss_m"]
    assert miss["ekf_wx"] < miss["ekf_std"]                          # knowing the weather helps
    r = client.post(f"/api/scenarios/{std['id']}/estimates", json={"method": "pinn"}, headers=h)
    assert r.status_code in (201, 503)
    detail = client.get(f"/api/scenarios/{wx['id']}", headers=users["viewer"]).json()
    assert detail["n_estimates"] == 2 and detail["best_miss_m"] == pytest.approx(min(miss.values()))


def test_stats(client, users, scenarios):
    s = client.get("/api/stats", headers=users["viewer"]).json()
    assert s["scenarios"] >= 2 and s["estimates"] >= 3
    by = {m["method"]: m for m in s["by_method"]}
    assert by["ekf_std"]["n"] >= 2


def test_live_stream(client, users, scenarios):
    std, _ = scenarios
    tok = users["viewer"]["Authorization"].split()[1]
    msgs = []
    with client.websocket_connect(f"/api/ws/scenarios/{std['id']}/live?token={tok}&speed=1e6") as ws:
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m["type"] in ("done", "error"):
                break
    kinds = [m["type"] for m in msgs]
    assert kinds[-1] == "done" and kinds.count("meas") == len(std["measurements"])
    last = [m for m in msgs if m["type"] == "state"][-1]
    assert abs(last["beta"] - 2500) / 2500 < 0.3


def test_live_stream_rejects_bad_token(client, scenarios):
    std, _ = scenarios
    with client.websocket_connect(f"/api/ws/scenarios/{std['id']}/live?token=bad") as ws:
        assert ws.receive_json()["type"] == "error"


def test_delete_scenario_owner_only(client, users, admin):
    sid = client.post("/api/scenarios", json={"name": "tmp", "seed": 3}, headers=admin).json()["id"]
    assert client.delete(f"/api/scenarios/{sid}", headers=users["analyst"]).status_code == 403
    assert client.delete(f"/api/scenarios/{sid}", headers=admin).status_code == 204
    assert client.get(f"/api/scenarios/{sid}", headers=admin).status_code == 404


def test_audit_trail(client, admin):
    rows = client.get("/api/audit", headers=admin).json()
    assert any(r["action"] == "estimate" for r in rows)


def test_registry_seed_fits_schema():
    """PostgreSQL enforces VARCHAR lengths (SQLite does not), so check the seed list statically."""
    from app.main import MODELS
    from app.models import ModelVersion
    lim = {c.name: getattr(c.type, "length", None) for c in ModelVersion.__table__.columns}
    for name, kind, *_ in MODELS:
        assert len(name) <= lim["name"] and len(kind) <= lim["kind"]


def test_login_throttling(client):
    from app import ratelimit
    ratelimit.reset()
    bad = {"username": "nobody@trajai.dev", "password": "wrong"}
    for _ in range(ratelimit.MAX_FAILURES):
        assert client.post("/api/auth/login", data=bad).status_code == 401
    r = client.post("/api/auth/login", data=bad)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0
    # other accounts are unaffected, and the window expires
    assert client.post("/api/auth/login", data={"username": "admin@trajai.dev", "password": "admin1234"}).status_code == 200
    key = ("nobody@trajai.dev", "testclient")
    import time
    assert ratelimit.retry_after(key, now=time.monotonic() + ratelimit.WINDOW_S + 1) == 0
    ratelimit.reset()


def test_estimates_csv_export(scenarios, client, users):
    std, _ = scenarios
    client.post(f"/api/scenarios/{std['id']}/estimates", json={"method": "ekf_std"}, headers=users["analyst"])
    r = client.get("/api/scenarios/export.csv", headers=users["viewer"])
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    import csv, io
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert rows and {"method", "miss_m", "threat_level"} <= set(rows[0])
    assert any(row["scenario_id"] == str(std["id"]) and row["method"] == "ekf_std" for row in rows)
    assert client.get("/api/scenarios/export.csv").status_code == 401


def test_estimate_and_stats_expose_model(scenarios, client, users):
    std, _ = scenarios
    e = client.post(f"/api/scenarios/{std['id']}/estimates", json={"method": "ekf_std"}, headers=users["analyst"]).json()
    assert "model_name" in e and e["model_name"] is None          # filters have no model version
    st = client.get("/api/stats", headers=users["viewer"]).json()
    assert all("model" in m for m in st["by_method"])
    assert {m["method"] for m in st["by_method"]} >= {"ekf_std", "ekf_wx", "ukf_wx", "pinn"}
