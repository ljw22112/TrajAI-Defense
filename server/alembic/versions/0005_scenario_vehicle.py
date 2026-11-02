"""scenario vehicle class (label and preset for the flight replay)

Revision ID: 0005
Revises: 0004
"""
from alembic import op
import sqlalchemy as sa

revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('scenarios', sa.Column('vehicle', sa.String(length=16), server_default='generic', nullable=False))


def downgrade() -> None:
    op.drop_column('scenarios', 'vehicle')
