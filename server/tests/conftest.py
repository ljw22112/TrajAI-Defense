"""Test fixtures. Uses a temporary SQLite file by default; set TRAJAI_TEST_DATABASE_URL to run against PostgreSQL."""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "server")]
_tmp = tempfile.mkdtemp()
os.environ["TRAJAI_DATABASE_URL"] = os.environ.get("TRAJAI_TEST_DATABASE_URL", f"sqlite:///{_tmp}/test.db")
os.environ["TRAJAI_WEATHER_OFFLINE"] = "1"
os.environ["TRAJAI_JWT_SECRET"] = "test-secret"

from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.services import weather as wsvc  # noqa: E402

CALLS = {"n": 0}


def synthetic_levels(day, hour, lat, lon):
    """Deterministic stand-in for Open-Meteo: standard-like temperature, 50 m/s westerly jet near 11 km."""
    CALLS["n"] += 1
    from trajai import physics, weather
    z = np.array([110, 760, 1460, 3010, 4200, 5570, 7180, 9160, 10360, 11790, 13600, 16200, 18450, 20600, 23850, 26500, 31000], float)
    T = physics.temperature(z) + (hour % 3)
    u = 50.0 * np.exp(-0.5 * ((z - 11_000) / 3_000) ** 2)
    return dict(p=[l * 100.0 for l in weather.LEVELS], z=z.tolist(), T=T.tolist(), u=u.tolist(), v=[0.0] * len(z))


@pytest.fixture(scope="session")
def client():
    if not os.environ["TRAJAI_DATABASE_URL"].startswith("sqlite"):
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
    wsvc.FETCHER["fn"] = synthetic_levels
    with TestClient(app) as c:
        yield c


def token(client, email="admin@trajai.dev", password="admin1234"):
    r = client.post("/api/auth/login", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


@pytest.fixture(scope="session")
def admin(client):
    return {"Authorization": f"Bearer {token(client)}"}


@pytest.fixture(scope="session")
def users(client, admin):
    out = {}
    for role in ("analyst", "viewer"):
        email = f"{role}@trajai.dev"
        r = client.post("/api/users", json={"email": email, "password": "password123", "role": role, "full_name": role}, headers=admin)
        assert r.status_code in (201, 409), r.text
        out[role] = {"Authorization": f"Bearer {token(client, email, 'password123')}"}
    return out
