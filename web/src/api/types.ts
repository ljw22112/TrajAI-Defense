// Mirrors server/app/schemas.py (the OpenAPI contract).
export type Role = "admin" | "analyst" | "viewer";
export type Method = "ekf_std" | "ekf_wx" | "ukf_wx" | "pinn";

export interface Page<T> { items: T[]; total: number; limit: number; offset: number }
export interface Token { access_token: string; token_type: string; expires_in: number }

export interface User { id: number; email: string; full_name: string; role: Role; is_active: boolean; created_at: string }

export interface AssetIn { name: string; x_km: number; y_km: number; radius_km: number; priority: number }
export interface Asset extends AssetIn { id: number; created_at: string; updated_at: string }

export interface WeatherLevels { p: number[]; z: number[]; T: number[]; u: number[]; v: number[] }
export interface Weather {
  id: number; valid_time: string; lat: number; lon: number; source: string;
  max_wind: number; max_wind_alt: number; levels: WeatherLevels;
}
export interface WeatherCurves { h: number[]; T: number[]; u: number[]; v: number[]; rho_ratio: number[] }

export interface Measurement { k: number; t: number; range_m: number; az: number; el: number; x: number; y: number; z: number }
export interface Trajectory { t: number[]; x: number[]; z: number[] }
export interface Window { t_in: number; t_out: number; t_pip: number; h_pip: number }
export interface Threat {
  level: number; max_prob: number; probabilities: Record<string, number>;
  windows: Record<string, Window | null>; top_asset_id: number | null;
}
export interface Estimate {
  id: number; scenario_id: number; method: Method; model_version_id: number | null; model_name?: string | null; created_at: string;
  ip_x: number; ip_y: number; cep: number; tot: number; tot_std: number | null; beta_hat: number | null;
  nis: number | null; latency_ms: number; miss_m: number; inside_cep: boolean; trajectory: Trajectory; threat: Threat | null;
}
export interface ScenarioSummary {
  id: number; name: string; created_at: string; created_by: number | null; beta: number; obs_fraction: number;
  weather_id: number | null; t_c: number; impact_time: number; impact_x: number; impact_y: number; apogee: number;
  n_estimates: number; best_miss_m: number | null;
}
export interface Scenario extends ScenarioSummary {
  h0: number; v0: number; gamma_deg: number; psi_deg: number; sigma_r: number; sigma_ang: number; seed: number;
  truth: { t: number[]; x: number[]; y: number[]; z: number[] };
  weather: Weather | null; measurements: Measurement[]; estimates: Estimate[];
}
export interface ScenarioIn {
  name: string; h0: number; v0: number; gamma_deg: number; psi_deg: number; beta: number; obs_fraction: number;
  sigma_r: number; sigma_ang_mrad: number; weather_date: string | null; weather_hour: number; seed?: number | null;
}
export interface ModelVersion { id: number; name: string; kind: string; uses_weather: boolean; metrics: Record<string, any>; is_active: boolean; created_at: string }
export interface MethodStat { method: Method; model?: string | null; n: number; median_miss_m: number | null; coverage: number | null; median_latency_ms: number | null }
export interface Stats { scenarios: number; estimates: number; weather_profiles: number; assets: number; by_method: MethodStat[]; threat_levels: Record<string, number> }
export interface AuditRow { id: number; user_id: number | null; action: string; entity: string; entity_id: number | null; detail: Record<string, unknown>; at: string }

export type LiveMsg =
  | { type: "meas"; k: number; t: number; x: number; y: number; z: number }
  | { type: "state"; t: number; k: number; r: number[]; v: number[]; beta: number; sigma_pos: number; sigma_lnb: number; nis: number | null }
  | { type: "done"; n: number; atmosphere: string }
  | { type: "error"; detail: string };

export const METHOD_LABEL: Record<Method, string> = {
  ekf_std: "EKF · 표준 대기",
  ekf_wx: "EKF · 일자 기상",
  ukf_wx: "UKF · 일자 기상",
  pinn: "PINN 하이브리드",
};

/** Short, human-readable names for registered PINN versions. */
export const MODEL_SHORT: Record<string, string> = {
  wx_hybrid_date: "기상 입력",
  pinn_tf_hybrid_v2: "표준 대기 v2",
  pinn_tf_hybrid: "v1 (구버전)",
};
export const modelShort = (m: string | null | undefined) => (m ? MODEL_SHORT[m] ?? m : null);

/** Glossary used for hover definitions across the console. */
export const TERMS: Record<string, string> = {
  "IP 오차": "예측 탄착점(Impact Point)과 실제 탄착점 사이의 수평 거리",
  CEP: "원형 공산 오차(CEP50): 실제 탄착점이 50% 확률로 들어갈 것으로 예측한 원의 반경",
  포함: "실제 탄착점이 예측 CEP 원 안에 들어갔는지. 잘 보정된 추정기라면 약 절반이 '포함'",
  ToT: "탄착 시각(Time on Target), 발사 기준 초",
  "β̂": "추정 탄도계수 β = m / (C_D·A) [kg/m²]. 클수록 공기 저항의 영향을 덜 받음",
  NIS: "정규화 혁신 제곱. 3 근처면 필터가 측정 잡음과 일관됨(χ²₃)",
  지연: "추정 한 번에 걸린 서버 계산 시간",
  "CEP 포함률": "실제 탄착점이 예측 CEP50 원 안에 든 비율. 보정된 추정기는 0.50",
};
