from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..audit import log
from ..db import get_db
from ..models import Asset, Role, User
from ..schemas import AssetIn, AssetOut
from ..security import current_user, require

router = APIRouter(prefix="/api/assets", tags=["assets"])


@router.get("", response_model=list[AssetOut])
def list_assets(db: Session = Depends(get_db), _: User = Depends(current_user)):
    return db.scalars(select(Asset).order_by(Asset.priority.desc(), Asset.name)).all()


@router.post("", response_model=AssetOut, status_code=201)
def create_asset(body: AssetIn, db: Session = Depends(get_db), user: User = Depends(require(Role.analyst))):
    a = Asset(**body.model_dump(), created_by=user.id)
    db.add(a)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "asset name already exists")
    db.refresh(a)
    log(db, user, "create", "asset", a.id, body.model_dump())
    return a


@router.put("/{aid}", response_model=AssetOut)
def update_asset(aid: int, body: AssetIn, db: Session = Depends(get_db), user: User = Depends(require(Role.analyst))):
    a = db.get(Asset, aid)
    if not a:
        raise HTTPException(404, "asset not found")
    for k, v in body.model_dump().items():
        setattr(a, k, v)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "asset name already exists")
    db.refresh(a)
    log(db, user, "update", "asset", a.id, body.model_dump())
    return a


@router.delete("/{aid}", status_code=204)
def delete_asset(aid: int, db: Session = Depends(get_db), user: User = Depends(require(Role.admin))):
    a = db.get(Asset, aid)
    if not a:
        raise HTTPException(404, "asset not found")
    db.delete(a); db.commit()
    log(db, user, "delete", "asset", aid)
