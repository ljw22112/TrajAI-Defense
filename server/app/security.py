"""Password hashing (bcrypt), JWT access tokens, role-based access dependencies."""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_db
from .models import Role, User

oauth2 = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)
ROLE_RANK = {Role.viewer: 0, Role.analyst: 1, Role.admin: 2}


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode()[:72], bcrypt.gensalt(rounds=12)).decode()


# compared against when the e-mail is unknown, so a failed login costs the same bcrypt time either way
# (response time does not reveal which accounts exist)
_DUMMY_HASH = bcrypt.hashpw(b"timing-equaliser", bcrypt.gensalt(rounds=12)).decode()


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode()[:72], hashed.encode())
    except ValueError:
        return False


def burn_password_check(pw: str) -> None:
    verify_password(pw, _DUMMY_HASH)


def create_token(user: User) -> tuple[str, int]:
    s = get_settings()
    now = datetime.now(timezone.utc)
    claims = {"sub": str(user.id), "role": user.role.value, "iat": now, "nbf": now,
              "exp": now + timedelta(minutes=s.jwt_expire_minutes), "iss": s.jwt_issuer, "aud": s.jwt_audience,
              "jti": secrets.token_hex(16)}
    tok = jwt.encode(claims, s.jwt_secret, algorithm=s.jwt_algorithm)
    return tok, s.jwt_expire_minutes * 60


def decode_token(token: str) -> int:
    s = get_settings()
    try:
        payload = jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm], issuer=s.jwt_issuer,
                             audience=s.jwt_audience, options={"require": ["exp", "iat", "sub", "iss", "aud"]})
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token", headers={"WWW-Authenticate": "Bearer"})


def current_user(token: str | None = Depends(oauth2), db: Session = Depends(get_db)) -> User:
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated", headers={"WWW-Authenticate": "Bearer"})
    user = db.get(User, decode_token(token))
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user not found or inactive")
    return user


def require(role: Role):
    """Dependency factory: the caller must have at least `role`."""
    def dep(user: User = Depends(current_user)) -> User:
        if ROLE_RANK[user.role] < ROLE_RANK[role]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"requires role '{role.value}' or higher")
        return user
    return dep
