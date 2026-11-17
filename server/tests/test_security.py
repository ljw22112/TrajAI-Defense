"""Security controls: response headers, production configuration guard, token claims, WebSocket authentication,
password policy and password-spraying limits."""
import jwt
import pytest
from pydantic import ValidationError

from app import ratelimit
from app.config import DEV_JWT_SECRET, Settings


def test_security_headers(client):
    r = client.get("/api/health")
    h = r.headers
    assert h["X-Content-Type-Options"] == "nosniff"
    assert h["X-Frame-Options"] == "DENY"
    assert h["Referrer-Policy"] == "no-referrer"
    assert h["Cache-Control"] == "no-store"
    csp = h["Content-Security-Policy"]
    assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp and "object-src 'none'" in csp
    assert "Strict-Transport-Security" not in h            # only in production (behind TLS)


def test_cors_is_restricted(client):
    r = client.options("/api/health", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") != "https://evil.example"
    r = client.options("/api/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "PUT"})
    assert "PUT" not in r.headers.get("access-control-allow-methods", "")


@pytest.mark.parametrize("override, message", [
    ({"jwt_secret": DEV_JWT_SECRET}, "TRAJAI_JWT_SECRET"),
    ({"jwt_secret": "x" * 40, "admin_password": "admin1234"}, "TRAJAI_ADMIN_PASSWORD"),
    ({"jwt_secret": "x" * 40, "admin_password": "a-long-admin-pass-1", "database_url": "sqlite:///x.db"}, "PostgreSQL"),
    ({"jwt_secret": "short", "admin_password": "a-long-admin-pass-1", "database_url": "postgresql+psycopg2://u:p@h/db"}, "32"),
])
def test_production_refuses_insecure_defaults(override, message):
    with pytest.raises(ValidationError, match=message):
        Settings(env="prod", **override)


def test_production_accepts_explicit_secrets():
    s = Settings(env="prod", jwt_secret="x" * 64, admin_password="a-long-admin-pass-1",
                 database_url="postgresql+psycopg2://u:p@h/db")
    assert s.is_prod


def test_token_claims_and_audience(client, admin):
    tok = admin["Authorization"][7:]
    claims = jwt.decode(tok, options={"verify_signature": False})
    assert {"sub", "role", "iat", "nbf", "exp", "iss", "aud", "jti"} <= set(claims)
    # a token signed with the right key but for another audience is rejected
    from app.config import get_settings
    s = get_settings()
    forged = jwt.encode({**claims, "aud": "someone-else"}, s.jwt_secret, algorithm=s.jwt_algorithm)
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401
    # an unsigned token is rejected
    none_tok = jwt.encode(claims, key=None, algorithm="none")
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {none_tok}"}).status_code == 401


def test_websocket_first_message_auth(client, admin):
    tok = admin["Authorization"][7:]
    with client.websocket_connect("/api/ws/alerts") as ws:
        ws.send_json({"type": "auth", "token": tok})
        assert ws.receive_json()["type"] == "ready"
    with client.websocket_connect("/api/ws/alerts") as ws:
        ws.send_json({"type": "auth", "token": "bad"})
        assert ws.receive_json() == {"type": "error", "detail": "invalid token"}


def test_password_policy(client, admin):
    for pw in ("short1", "onlyletterslong", "1234567890123", "Password123"):
        r = client.post("/api/users", json={"email": "weak@trajai.dev", "password": pw}, headers=admin)
        assert r.status_code == 422, pw


def test_password_spraying_is_limited_per_address(client):
    ratelimit.reset()
    for i in range(ratelimit.MAX_ADDR_FAILURES):          # one failure each on many accounts
        assert client.post("/api/auth/login", data={"username": f"u{i}@trajai.dev", "password": "wrong"}).status_code == 401
    r = client.post("/api/auth/login", data={"username": "fresh@trajai.dev", "password": "wrong"})
    assert r.status_code == 429
    ratelimit.reset()


def test_failed_logins_are_audited(client, admin):
    ratelimit.reset()
    client.post("/api/auth/login", data={"username": "admin@trajai.dev", "password": "wrong-password"})
    rows = client.get("/api/audit", headers=admin).json()
    items = rows["items"] if isinstance(rows, dict) else rows
    assert any(r["action"] == "login_failed" for r in items)
    ratelimit.reset()
