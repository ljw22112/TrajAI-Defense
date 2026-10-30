"""Append-only audit trail of state-changing actions."""
from sqlalchemy.orm import Session

from .models import AuditLog, User


def log(db: Session, user: User | None, action: str, entity: str, entity_id: int | None = None, detail: dict | None = None) -> None:
    db.add(AuditLog(user_id=user.id if user else None, action=action, entity=entity, entity_id=entity_id, detail=detail or {}))
    db.commit()
