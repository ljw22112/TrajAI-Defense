"""analysis kinds (latency, sensor sweep, environment sensitivity)

Revision ID: 0003
Revises: 0002
"""
from alembic import op
import sqlalchemy as sa

revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('weather_analyses', sa.Column('kind', sa.String(length=20), server_default='weather', nullable=False))
    op.create_index(op.f('ix_weather_analyses_kind'), 'weather_analyses', ['kind'], unique=False)
    op.create_check_constraint('ck_analysis_kind', 'weather_analyses', "kind IN ('weather', 'latency', 'sensor', 'sensitivity')")


def downgrade() -> None:
    op.drop_constraint('ck_analysis_kind', 'weather_analyses', type_='check')
    op.drop_index(op.f('ix_weather_analyses_kind'), table_name='weather_analyses')
    op.drop_column('weather_analyses', 'kind')
