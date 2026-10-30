"""Alembic environment: uses the app's metadata and TRAJAI_DATABASE_URL."""
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "server")]

from app.config import get_settings  # noqa: E402
from app.db import Base  # noqa: E402
import app.models  # noqa: E402,F401  (register tables)

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name)
config.set_main_option("sqlalchemy.url", get_settings().database_url)
target_metadata = Base.metadata


def run_offline():
    context.configure(url=config.get_main_option("sqlalchemy.url"), target_metadata=target_metadata, literal_binds=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online():
    eng = engine_from_config(config.get_section(config.config_ini_section), prefix="sqlalchemy.", poolclass=pool.NullPool)
    with eng.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata, compare_type=True, render_as_batch=conn.dialect.name == "sqlite")
        with context.begin_transaction():
            context.run_migrations()


run_offline() if context.is_offline_mode() else run_online()
