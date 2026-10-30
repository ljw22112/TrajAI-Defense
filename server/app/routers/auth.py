from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..audit import log
from ..db import get_db
from ..models import Role, User
from ..schemas import Page, Token, UserCreate, UserOut, UserUpdate
from ..security import create_token, current_user, hash_password, require, verify_password

router = APIRouter(prefix="/api", tags=["auth & users"])


@router.post("/auth/login", response_model=Token, summary="OAuth2 password login (username = email)")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == form.username.lower()))
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "incorrect email or password")
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "account disabled")
    tok, exp = create_token(user)
    log(db, user, "login", "user", user.id)
    return Token(access_token=tok, expires_in=exp)


@router.get("/auth/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user


@router.get("/users", response_model=Page[UserOut])
def list_users(limit: int = 50, offset: int = 0, db: Session = Depends(get_db), _: User = Depends(require(Role.admin))):
    total = db.scalar(select(func.count(User.id)))
    items = db.scalars(select(User).order_by(User.id).limit(min(limit, 200)).offset(offset)).all()
    return Page(items=items, total=total, limit=limit, offset=offset)


@router.post("/users", response_model=UserOut, status_code=201)
def create_user(body: UserCreate, db: Session = Depends(get_db), admin: User = Depends(require(Role.admin))):
    if db.scalar(select(User).where(User.email == body.email.lower())):
        raise HTTPException(409, "email already registered")
    u = User(email=body.email.lower(), full_name=body.full_name, hashed_password=hash_password(body.password), role=body.role)
    db.add(u); db.commit(); db.refresh(u)
    log(db, admin, "create", "user", u.id, {"role": u.role.value})
    return u


@router.patch("/users/{uid}", response_model=UserOut)
def update_user(uid: int, body: UserUpdate, db: Session = Depends(get_db), admin: User = Depends(require(Role.admin))):
    u = db.get(User, uid)
    if not u:
        raise HTTPException(404, "user not found")
    if u.id == admin.id and (body.role not in (None, Role.admin) or body.is_active is False):
        raise HTTPException(400, "admins cannot demote or disable themselves")
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(u, k, v)
    db.commit(); db.refresh(u)
    log(db, admin, "update", "user", u.id, body.model_dump(exclude_none=True, mode="json"))
    return u
