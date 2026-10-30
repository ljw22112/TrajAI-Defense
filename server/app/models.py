"""ORM schema (see docs/erd.md).

users ─┬─< scenarios ─┬─< measurements
       │              ├─< estimates ──1 threat_assessments >── assets
       │              └── weather_profiles
       ├─< assets
       └─< audit_log
model_versions ──< estimates
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, CheckConstraint, DateTime, Enum, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base

JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, enum.Enum):
    admin = "admin"          # users, assets, models
    analyst = "analyst"      # scenarios, estimates, assets
    viewer = "viewer"        # read only


class Method(str, enum.Enum):
    ekf_std = "ekf_std"      # EKF, standard atmosphere (exponential density, no wind)
    ekf_wx = "ekf_wx"        # EKF with the scenario's weather profile
    ukf_wx = "ukf_wx"        # UKF with the scenario's weather profile
    pinn = "pinn"            # PINN hybrid (ONNX Runtime), EKF features as input


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120), default="")
    hashed_password: Mapped[str] = mapped_column(String(100))
    role: Mapped[Role] = mapped_column(Enum(Role, name="role"), default=Role.viewer)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Asset(Base):
    """Protected asset in the local ENU frame (notional)."""
    __tablename__ = "assets"
    __table_args__ = (CheckConstraint("radius_km > 0", name="ck_asset_radius"),
                      CheckConstraint("priority BETWEEN 1 AND 5", name="ck_asset_priority"))
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    x_km: Mapped[float] = mapped_column(Float)
    y_km: Mapped[float] = mapped_column(Float)
    radius_km: Mapped[float] = mapped_column(Float)
    priority: Mapped[int] = mapped_column(Integer, default=3)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class WeatherProfile(Base):
    """Pressure-level profile cached from Open-Meteo (one valid time at one site)."""
    __tablename__ = "weather_profiles"
    __table_args__ = (UniqueConstraint("valid_time", "lat", "lon", name="uq_weather_time_site"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    valid_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(40), default="open-meteo")
    levels: Mapped[dict] = mapped_column(JSONType)          # {p, z, T, u, v} arrays, SI units
    max_wind: Mapped[float] = mapped_column(Float)
    max_wind_alt: Mapped[float] = mapped_column(Float)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Scenario(Base):
    """A simulated engagement: burnout state, ballistic coefficient, sensor settings and atmosphere."""
    __tablename__ = "scenarios"
    __table_args__ = (CheckConstraint("obs_fraction > 0 AND obs_fraction < 1", name="ck_obs_fraction"),
                      CheckConstraint("beta > 0", name="ck_beta"))
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    h0: Mapped[float] = mapped_column(Float)
    v0: Mapped[float] = mapped_column(Float)
    gamma_deg: Mapped[float] = mapped_column(Float)
    psi_deg: Mapped[float] = mapped_column(Float)
    beta: Mapped[float] = mapped_column(Float)
    obs_fraction: Mapped[float] = mapped_column(Float)
    sigma_r: Mapped[float] = mapped_column(Float)
    sigma_ang: Mapped[float] = mapped_column(Float)
    seed: Mapped[int] = mapped_column(Integer)
    weather_id: Mapped[int | None] = mapped_column(ForeignKey("weather_profiles.id", ondelete="SET NULL"))
    # simulation outputs
    t_c: Mapped[float] = mapped_column(Float)
    impact_time: Mapped[float] = mapped_column(Float)
    impact_x: Mapped[float] = mapped_column(Float)
    impact_y: Mapped[float] = mapped_column(Float)
    apogee: Mapped[float] = mapped_column(Float)
    truth: Mapped[dict] = mapped_column(JSONType)            # downsampled true trajectory {t, x, y, z}

    weather: Mapped[WeatherProfile | None] = relationship()
    measurements: Mapped[list[Measurement]] = relationship(back_populates="scenario", cascade="all, delete-orphan", order_by="Measurement.k")
    estimates: Mapped[list[Estimate]] = relationship(back_populates="scenario", cascade="all, delete-orphan", order_by="Estimate.id")


class Measurement(Base):
    __tablename__ = "measurements"
    __table_args__ = (UniqueConstraint("scenario_id", "k", name="uq_meas_k"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    scenario_id: Mapped[int] = mapped_column(ForeignKey("scenarios.id", ondelete="CASCADE"), index=True)
    k: Mapped[int] = mapped_column(Integer)
    t: Mapped[float] = mapped_column(Float)
    range_m: Mapped[float] = mapped_column(Float)
    az: Mapped[float] = mapped_column(Float)
    el: Mapped[float] = mapped_column(Float)
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    z: Mapped[float] = mapped_column(Float)
    scenario: Mapped[Scenario] = relationship(back_populates="measurements")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    kind: Mapped[str] = mapped_column(String(40))             # pinn-hybrid, ...
    file_path: Mapped[str] = mapped_column(String(300))
    uses_weather: Mapped[bool] = mapped_column(Boolean, default=False)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Estimate(Base):
    __tablename__ = "estimates"
    __table_args__ = (Index("ix_estimates_scenario_method", "scenario_id", "method"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    scenario_id: Mapped[int] = mapped_column(ForeignKey("scenarios.id", ondelete="CASCADE"))
    method: Mapped[Method] = mapped_column(Enum(Method, name="method"))
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id", ondelete="SET NULL"))
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ip_x: Mapped[float] = mapped_column(Float)
    ip_y: Mapped[float] = mapped_column(Float)
    cep: Mapped[float] = mapped_column(Float)
    tot: Mapped[float] = mapped_column(Float)
    tot_std: Mapped[float | None] = mapped_column(Float)
    beta_hat: Mapped[float | None] = mapped_column(Float)
    nis: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)
    miss_m: Mapped[float] = mapped_column(Float)              # vs simulated truth
    inside_cep: Mapped[bool] = mapped_column(Boolean)
    trajectory: Mapped[dict] = mapped_column(JSONType)       # {x, z} predicted descent (downsampled)
    scenario: Mapped[Scenario] = relationship(back_populates="estimates")
    threat: Mapped[ThreatAssessment | None] = relationship(back_populates="estimate", cascade="all, delete-orphan", uselist=False)
    model_version: Mapped[ModelVersion | None] = relationship(lazy="joined")

    @property
    def model_name(self) -> str | None:
        return self.model_version.name if self.model_version else None


class ThreatAssessment(Base):
    __tablename__ = "threat_assessments"
    __table_args__ = (CheckConstraint("level BETWEEN 1 AND 5", name="ck_threat_level"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    estimate_id: Mapped[int] = mapped_column(ForeignKey("estimates.id", ondelete="CASCADE"), unique=True)
    level: Mapped[int] = mapped_column(Integer)
    max_prob: Mapped[float] = mapped_column(Float)
    top_asset_id: Mapped[int | None] = mapped_column(ForeignKey("assets.id", ondelete="SET NULL"))
    probabilities: Mapped[dict] = mapped_column(JSONType)    # {asset name: P(hit)}
    windows: Mapped[dict] = mapped_column(JSONType)          # engagement altitude windows
    estimate: Mapped[Estimate] = relationship(back_populates="threat")
    top_asset: Mapped[Asset | None] = relationship()


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), index=True)
    action: Mapped[str] = mapped_column(String(40))
    entity: Mapped[str] = mapped_column(String(40))
    entity_id: Mapped[int | None] = mapped_column(Integer)
    detail: Mapped[dict] = mapped_column(JSONType, default=dict)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
