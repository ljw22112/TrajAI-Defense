"""Pydantic request / response models (the OpenAPI contract consumed by the web client)."""
from __future__ import annotations

from datetime import date, datetime
from typing import Generic, TypeVar

from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from .models import Method, Role

T = TypeVar("T")


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------- auth / users
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserOut(ORM):
    id: int
    email: str
    full_name: str
    role: Role
    is_active: bool
    created_at: datetime


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str = Field("", max_length=120)
    password: str = Field(min_length=8, max_length=128)
    role: Role = Role.viewer


class UserUpdate(BaseModel):
    full_name: str | None = Field(None, max_length=120)
    role: Role | None = None
    is_active: bool | None = None


# ---------------------------------------------------------------- assets
class AssetIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    x_km: float = Field(ge=-1000, le=1000)
    y_km: float = Field(ge=-1000, le=1000)
    radius_km: float = Field(gt=0, le=50)
    priority: int = Field(3, ge=1, le=5)


class AssetOut(ORM, AssetIn):
    id: int
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------- weather
class WeatherOut(ORM):
    id: int
    valid_time: datetime
    lat: float
    lon: float
    source: str
    max_wind: float
    max_wind_alt: float
    levels: dict


class CustomAtmIn(BaseModel):
    """User-defined atmosphere. Ranges cover observed surface climates; the profile shape follows USSA-1976."""
    t_sfc_c: float = Field(15.0, ge=-50, le=50, description="surface temperature [degC]")
    p_sfc_hpa: float = Field(1013.25, ge=900, le=1060, description="surface pressure [hPa]")
    rh_pct: float = Field(50.0, ge=0, le=100, description="surface relative humidity [%]")
    lapse_k_per_km: float = Field(6.5, ge=3, le=9.8, description="tropospheric lapse rate [K/km]")
    tropopause_km: float = Field(11.0, ge=8, le=18)
    wind_sfc_ms: float = Field(5.0, ge=0, le=40)
    wind_sfc_dir: float = Field(270.0, ge=0, le=360, description="direction the wind blows FROM [deg]")
    jet_ms: float = Field(40.0, ge=0, le=120)
    jet_km: float = Field(11.0, ge=4, le=20)
    jet_width_km: float = Field(3.0, ge=0.5, le=8)
    jet_dir: float = Field(270.0, ge=0, le=360)
    condition: Literal["clear", "cloudy", "fog", "rain", "heavy_rain", "snow"] = "clear"


class CustomAtmPreview(BaseModel):
    levels: dict
    curves: dict
    noise_factor: float
    max_wind: float
    max_wind_alt: float
    surface_density: float
    density_vs_standard_pct: float


class WeatherQuery(BaseModel):
    date: date
    hour: int = Field(0, ge=0, le=23)
    lat: float | None = Field(None, ge=-80, le=80)
    lon: float | None = Field(None, ge=-180, le=180)

    @field_validator("date")
    @classmethod
    def _range(cls, v: date) -> date:
        if v < date(2022, 1, 1):
            raise ValueError("weather archive starts on 2022-01-01")
        return v


# ---------------------------------------------------------------- scenarios
class ScenarioIn(BaseModel):
    name: str = Field("scenario", min_length=1, max_length=120)
    h0: float = Field(15_000.0, ge=5_000, le=40_000, description="burnout altitude [m]")
    v0: float = Field(1_400.0, ge=500, le=2_500, description="burnout speed [m/s]")
    gamma_deg: float = Field(45.0, ge=15, le=75, description="flight-path angle [deg]")
    psi_deg: float = Field(0.0, ge=-30, le=30, description="azimuth from +x (east) [deg]")
    beta: float = Field(2_500.0, ge=300, le=20_000, description="ballistic coefficient [kg/m^2]")
    obs_fraction: float = Field(0.12, ge=0.04, le=0.6)
    sigma_r: float = Field(15.0, ge=1, le=200, description="range noise 1-sigma [m]")
    sigma_ang_mrad: float = Field(0.4, ge=0.05, le=5, description="angle noise 1-sigma [mrad]")
    weather_date: date | None = Field(None, description="fly through this date's atmosphere (None = standard)")
    weather_hour: int = Field(0, ge=0, le=23)
    weather_lat: float | None = Field(None, ge=-80, le=80, description="weather location (engagement area); default from settings")
    weather_lon: float | None = Field(None, ge=-180, le=180)
    custom_atm: CustomAtmIn | None = Field(None, description="user-defined atmosphere (takes precedence over weather_date)")
    seed: int | None = None


class FlightIn(BaseModel):
    """Flight and sensor of a study (same ranges as a scenario)."""
    name: str = Field("analysis", min_length=1, max_length=120)
    h0: float = Field(15_000.0, ge=5_000, le=40_000)
    v0: float = Field(1_400.0, ge=500, le=2_500)
    gamma_deg: float = Field(45.0, ge=15, le=75)
    psi_deg: float = Field(0.0, ge=-30, le=30)
    beta: float = Field(2_500.0, ge=300, le=20_000)
    obs_fraction: float = Field(0.12, ge=0.04, le=0.6)
    sigma_r: float = Field(15.0, ge=1, le=200)
    sigma_ang_mrad: float = Field(0.4, ge=0.05, le=5)
    seed: int = Field(7, ge=0)
    lat: float | None = Field(None, ge=-80, le=80)
    lon: float | None = Field(None, ge=-180, le=180)


class AnalysisIn(FlightIn):
    """Weather-impact study: the flight/sensor of a scenario, flown through every profile in a time window."""
    name: str = Field("weather impact", min_length=1, max_length=120)
    start_date: date
    days: int = Field(3, ge=1, le=7)
    hours: list[int] = Field(default_factory=lambda: [0, 6, 12, 18], min_length=1, max_length=4)

    @field_validator("hours")
    @classmethod
    def _hours(cls, v: list[int]) -> list[int]:
        if not set(v) <= {0, 6, 12, 18}:
            raise ValueError("hours must be a subset of 0, 6, 12, 18 UTC")
        return sorted(set(v))

    @field_validator("start_date")
    @classmethod
    def _start(cls, v: date) -> date:
        if v < date(2022, 1, 1):
            raise ValueError("weather archive starts on 2022-01-01")
        return v


class LatencyIn(AnalysisIn):
    """Weather-data latency: the EKF uses the profile valid 0/6/12/24 h before the flight."""
    name: str = Field("weather-data latency", min_length=1, max_length=120)


class SensorIn(FlightIn):
    """Sensor sweep: radar noise x observed fraction, in one atmosphere."""
    name: str = Field("sensor sweep", min_length=1, max_length=120)
    atmosphere: Literal["standard", "date", "custom"] = "standard"
    weather_date: date | None = None
    weather_hour: int = Field(0, ge=0, le=23)
    custom_atm: CustomAtmIn | None = None


class SensitivityIn(FlightIn):
    """Vary one user-set variable of a custom atmosphere."""
    name: str = Field("environment sensitivity", min_length=1, max_length=120)
    base: CustomAtmIn = Field(default_factory=CustomAtmIn)
    param: Literal["t_sfc_c", "rh_pct", "p_sfc_hpa", "jet_ms", "jet_dir", "wind_sfc_ms", "lapse_k_per_km", "condition"] = "t_sfc_c"


class AnalysisSummaryOut(ORM):
    id: int
    kind: str
    name: str
    created_by: int | None
    created_at: datetime
    lat: float
    lon: float
    n_profiles: int
    summary: dict


class AnalysisOut(AnalysisSummaryOut):
    params: dict
    items: list


class MeasurementOut(ORM):
    k: int
    t: float
    range_m: float
    az: float
    el: float
    x: float
    y: float
    z: float


class ThreatOut(ORM):
    level: int
    max_prob: float
    probabilities: dict
    windows: dict
    top_asset_id: int | None


class EstimateOut(ORM):
    id: int
    scenario_id: int
    method: Method
    model_version_id: int | None
    model_name: str | None = None
    created_at: datetime
    ip_x: float
    ip_y: float
    cep: float
    tot: float
    tot_std: float | None
    beta_hat: float | None
    nis: float | None
    latency_ms: float
    miss_m: float
    inside_cep: bool
    trajectory: dict
    threat: ThreatOut | None


class ScenarioSummary(ORM):
    id: int
    name: str
    created_at: datetime
    created_by: int | None
    beta: float
    obs_fraction: float
    weather_id: int | None
    t_c: float
    impact_time: float
    impact_x: float
    impact_y: float
    apogee: float
    n_estimates: int = 0
    best_miss_m: float | None = None


class ScenarioOut(ScenarioSummary):
    h0: float
    v0: float
    gamma_deg: float
    psi_deg: float
    sigma_r: float
    sigma_ang: float
    seed: int
    truth: dict
    weather: WeatherOut | None
    measurements: list[MeasurementOut]
    estimates: list[EstimateOut]


class EstimateIn(BaseModel):
    method: Method


# ---------------------------------------------------------------- models / stats
class ModelVersionOut(ORM):
    id: int
    name: str
    kind: str
    uses_weather: bool
    metrics: dict
    is_active: bool
    created_at: datetime


class MethodStat(BaseModel):
    method: Method
    model: str | None = None          # PINN rows are split by model version
    n: int
    median_miss_m: float | None
    coverage: float | None
    median_latency_ms: float | None


class ThreatBoardRow(BaseModel):
    scenario_id: int
    scenario: str
    estimate_id: int
    method: Method
    model: str | None
    level: int
    max_prob: float
    top_asset: str | None
    impact_time: float            # predicted ToT [s after burnout]
    time_to_impact: float         # from the end of the observed track [s]
    first_window_open: float | None   # earliest engagement window opening, seconds after the end of the track (<= 0: already open)
    first_window_close: float | None
    miss_m: float


class RaidMemberIn(BaseModel):
    scenario_id: int
    offset_s: float = Field(0.0, ge=0, le=600, description="launch time after the raid start [s]")


class RaidIn(BaseModel):
    name: str = Field("raid", min_length=1, max_length=120)
    members: list[RaidMemberIn] = Field(..., min_length=2, max_length=12)

    @field_validator("members")
    @classmethod
    def _unique(cls, v):
        if len({m.scenario_id for m in v}) != len(v):
            raise ValueError("each scenario can appear once")
        return v


class RaidTrack(BaseModel):
    scenario_id: int
    scenario: str
    offset_s: float
    track_end: float          # end of radar track (raid clock) [s]
    impact: float             # predicted impact (raid clock) [s]
    windows: list[dict]       # [{band, open, close}] on the raid clock
    level: int | None
    max_prob: float | None
    top_asset: str | None
    method: Method | None


class RaidOut(BaseModel):
    id: int
    name: str
    created_at: datetime
    tracks: list[RaidTrack]
    peak_simultaneous: int            # most engagement windows open at once (interceptor channels needed)
    peak_at: float | None
    first_impact: float | None


class RaidRow(BaseModel):
    id: int
    name: str
    created_at: datetime
    n: int


class Stats(BaseModel):
    scenarios: int
    estimates: int
    weather_profiles: int
    assets: int
    by_method: list[MethodStat]
    threat_levels: dict[str, int]
