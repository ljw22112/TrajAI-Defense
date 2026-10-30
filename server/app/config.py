"""Runtime configuration (environment variables, prefix TRAJAI_)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRAJAI_", env_file=".env", extra="ignore")

    database_url: str = f"sqlite:///{ROOT / 'server' / 'trajai.db'}"
    jwt_secret: str = "change-me-in-production"          # override in deployment
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 120
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173", "https://ljw22112.github.io"]
    model_dir: Path = ROOT / "export"
    weather_lat: float = 36.5
    weather_lon: float = 127.5
    weather_offline: bool = False                         # tests: never call the network
    weather_archive_dir: Path = ROOT / "data" / "weather"   # monthly provider downloads, read before the network
    admin_email: str = "admin@trajai.dev"
    admin_password: str = "admin1234"                     # seeded on first start; change after login
    stream_speed: float = 4.0                             # live replay: simulated seconds per wall second


@lru_cache
def get_settings() -> Settings:
    return Settings()
