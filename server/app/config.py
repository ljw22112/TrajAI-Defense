"""Runtime configuration (environment variables, prefix TRAJAI_)."""
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
DEV_JWT_SECRET = "change-me-in-production"
DEV_ADMIN_PASSWORD = "admin1234"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRAJAI_", env_file=".env", extra="ignore")

    env: str = "dev"                                      # dev | prod. prod refuses insecure defaults (see _check_prod)
    database_url: str = f"sqlite:///{ROOT / 'server' / 'trajai.db'}"
    jwt_secret: str = DEV_JWT_SECRET                      # prod: at least 32 characters, e.g. openssl rand -hex 32
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 120
    jwt_issuer: str = "trajai-defense"
    jwt_audience: str = "trajai-console"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    model_dir: Path = ROOT / "export"
    weather_lat: float = 36.5
    weather_lon: float = 127.5
    weather_offline: bool = False                         # tests: never call the network
    weather_archive_dir: Path = ROOT / "data" / "weather"   # monthly provider downloads, read before the network
    admin_email: str = "admin@trajai.dev"
    admin_password: str = DEV_ADMIN_PASSWORD              # seeded on first start; prod: at least 12 characters
    stream_speed: float = 4.0                             # live replay: simulated seconds per wall second

    @property
    def is_prod(self) -> bool:
        return self.env.lower() == "prod"

    @model_validator(mode="after")
    def _check_prod(self):
        """In production the server must not start with the published development defaults."""
        if not self.is_prod:
            return self
        bad = []
        if self.jwt_secret == DEV_JWT_SECRET or len(self.jwt_secret) < 32:
            bad.append("TRAJAI_JWT_SECRET must be set to a random value of at least 32 characters")
        if self.admin_password == DEV_ADMIN_PASSWORD or len(self.admin_password) < 12:
            bad.append("TRAJAI_ADMIN_PASSWORD must be set (at least 12 characters, not the development default)")
        if self.database_url.startswith("sqlite"):
            bad.append("TRAJAI_DATABASE_URL must point to PostgreSQL")
        if any(o == "*" for o in self.cors_origins):
            bad.append("TRAJAI_CORS_ORIGINS must list explicit origins")
        if bad:
            raise ValueError("insecure production configuration: " + "; ".join(bad))
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
