"""raids and the ekf_pinn estimator

Revision ID: 0004
Revises: 0003
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None

JSONT = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql')


def upgrade() -> None:
    op.create_table('raids',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=120), nullable=False),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('members', JSONT, nullable=False),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'))
    op.create_index(op.f('ix_raids_created_by'), 'raids', ['created_by'], unique=False)
    if op.get_bind().dialect.name == 'postgresql':
        with op.get_context().autocommit_block():
            op.execute("ALTER TYPE method ADD VALUE IF NOT EXISTS 'ekf_pinn'")


def downgrade() -> None:
    op.drop_index(op.f('ix_raids_created_by'), table_name='raids')
    op.drop_table('raids')
    if op.get_bind().dialect.name == 'postgresql':
        # PostgreSQL cannot drop an enum value: rebuild the type without it
        op.execute("DELETE FROM estimates WHERE method = 'ekf_pinn'")
        op.execute("ALTER TYPE method RENAME TO method_old")
        op.execute("CREATE TYPE method AS ENUM ('ekf_std', 'ekf_wx', 'ukf_wx', 'pinn')")
        op.execute("ALTER TABLE estimates ALTER COLUMN method TYPE method USING method::text::method")
        op.execute("DROP TYPE method_old")
