// Thin typed client for the FastAPI server, with a read-only static "demo" adapter for GitHub Pages.
import type {
  Alert, Analysis, AnalysisIn, AnalysisRow, CustomAtm, CustomPreview, Raid, RaidRow, ThreatRow,
  Asset, AssetIn, AuditRow, Estimate, Method, ModelVersion, Page, Scenario, ScenarioIn, ScenarioSummary, Stats, Token, User,
  Weather, WeatherCurves,
} from "./types";

export const DEMO = import.meta.env.VITE_DEMO === "1";
const BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ?? "";
const TOKEN_KEY = "trajai.token";

export class ApiError extends Error {
  constructor(public status: number, public detail: string, public fields: Record<string, string> = {}) {
    super(detail);
  }
}

export const tokenStore = {
  get(): string | null { try { return localStorage.getItem(TOKEN_KEY); } catch { return null; } },
  set(t: string | null) { try { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); } catch { /* storage blocked */ } },
};

function parseError(status: number, body: any): ApiError {
  const d = body?.detail;
  if (Array.isArray(d)) {                      // FastAPI validation errors
    const fields: Record<string, string> = {};
    d.forEach((e: any) => { fields[String(e.loc?.[e.loc.length - 1])] = e.msg; });
    return new ApiError(status, "입력값을 확인하세요.", fields);
  }
  return new ApiError(status, typeof d === "string" ? d : `요청이 실패했습니다 (HTTP ${status}).`);
}

// ------------------------------------------------------------------ demo adapter (static JSON snapshot)
const DEMO_ROOT = `${import.meta.env.BASE_URL}demo/`;
export const demoFile = (name: string) => `${DEMO_ROOT}${name}`;
async function demoGet<T>(path: string): Promise<T> {
  const p = path.replace(/^\/api\//, "").replace(/\?.*$/, "");
  const file = p.replace(/\//g, "_") || "index";
  const r = await fetch(`${DEMO_ROOT}${file}.json`);
  if (!r.ok) throw new ApiError(404, "데모 자료에 없는 항목입니다.");
  return r.json();
}

async function request<T>(method: string, path: string, body?: unknown, form = false): Promise<T> {
  if (DEMO) {
    if (method !== "GET") throw new ApiError(403, "데모 모드는 읽기 전용입니다. 로컬 서버를 실행하면 모든 기능을 쓸 수 있습니다.");
    return demoGet<T>(path);
  }
  const headers: Record<string, string> = {};
  const tok = tokenStore.get();
  if (tok) headers.Authorization = `Bearer ${tok}`;
  let payload: BodyInit | undefined;
  if (form) { payload = new URLSearchParams(body as Record<string, string>); headers["Content-Type"] = "application/x-www-form-urlencoded"; }
  else if (body !== undefined) { payload = JSON.stringify(body); headers["Content-Type"] = "application/json"; }
  let r: Response;
  try { r = await fetch(`${BASE}${path}`, { method, headers, body: payload }); }
  catch { throw new ApiError(0, "서버에 연결할 수 없습니다. API 서버가 실행 중인지 확인하세요."); }
  if (r.status === 204) return undefined as T;
  const data = await r.json().catch(() => null);
  if (!r.ok) {
    if (r.status === 401) tokenStore.set(null);
    throw parseError(r.status, data);
  }
  return data as T;
}

const qs = (o: Record<string, unknown>) =>
  "?" + Object.entries(o).filter(([, v]) => v !== undefined && v !== null && v !== "").map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`).join("&");

/** Download all estimates as CSV (server: /api/scenarios/export.csv; demo: the exported snapshot). */
export async function downloadEstimatesCsv(): Promise<void> {
  let r: Response;
  try {
    r = DEMO ? await fetch(`${DEMO_ROOT}estimates.csv`)
      : await fetch(`${BASE}/api/scenarios/export.csv`, { headers: tokenStore.get() ? { Authorization: `Bearer ${tokenStore.get()}` } : {} });
  } catch { throw new ApiError(0, "서버에 연결할 수 없습니다."); }
  if (!r.ok) throw parseError(r.status, await r.json().catch(() => null));
  const url = URL.createObjectURL(await r.blob());
  const a = Object.assign(document.createElement("a"), { href: url, download: "trajai_estimates.csv" });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const api = {
  login: (email: string, password: string) => request<Token>("POST", "/api/auth/login", { username: email, password }, true),
  me: () => (DEMO ? Promise.resolve<User>({ id: 0, email: "demo@trajai.dev", full_name: "Demo viewer", role: "viewer", is_active: true, created_at: "" }) : request<User>("GET", "/api/auth/me")),
  users: () => request<Page<User>>("GET", "/api/users"),
  createUser: (b: { email: string; password: string; role: string; full_name: string }) => request<User>("POST", "/api/users", b),
  updateUser: (id: number, b: Partial<Pick<User, "role" | "is_active" | "full_name">>) => request<User>("PATCH", `/api/users/${id}`, b),

  assets: () => request<Asset[]>("GET", "/api/assets"),
  createAsset: (b: AssetIn) => request<Asset>("POST", "/api/assets", b),
  updateAsset: (id: number, b: AssetIn) => request<Asset>("PUT", `/api/assets/${id}`, b),
  deleteAsset: (id: number) => request<void>("DELETE", `/api/assets/${id}`),

  weather: (date: string, hour: number, lat?: number, lon?: number) =>
    request<Weather>("GET", DEMO ? `/api/weather/${date}T${String(hour).padStart(2, "0")}` : `/api/weather${qs({ date, hour, lat, lon })}`),
  weatherCached: () => request<Weather[]>("GET", "/api/weather/cached"),
  weatherCurves: (id: number) => request<WeatherCurves>("GET", `/api/weather/${id}/curves`),

  scenarios: (p: { limit?: number; offset?: number; q?: string; weather?: boolean }) =>
    request<Page<ScenarioSummary>>("GET", DEMO ? "/api/scenarios" : `/api/scenarios${qs(p)}`),
  scenario: (id: number) => request<Scenario>("GET", `/api/scenarios/${id}`),
  createScenario: (b: ScenarioIn) => request<Scenario>("POST", "/api/scenarios", b),
  deleteScenario: (id: number) => request<void>("DELETE", `/api/scenarios/${id}`),
  estimate: (id: number, method: Method) => request<Estimate>("POST", `/api/scenarios/${id}/estimates`, { method }),

  models: () => request<ModelVersion[]>("GET", "/api/models"),
  activateModel: (id: number) => request<ModelVersion>("POST", `/api/models/${id}/activate`),
  stats: () => request<Stats>("GET", "/api/stats"),
  threats: () => request<ThreatRow[]>("GET", "/api/threats"),
  analyses: () => request<Page<AnalysisRow>>("GET", "/api/analyses?limit=50"),
  analysis: (id: number) => request<Analysis>("GET", `/api/analyses/${id}`),
  createAnalysis: (b: AnalysisIn) => request<Analysis>("POST", "/api/analyses", b),
  createLatency: (b: AnalysisIn) => request<Analysis>("POST", "/api/analyses/latency", b),
  createSensor: (b: Record<string, unknown>) => request<Analysis>("POST", "/api/analyses/sensor", b),
  createSensitivity: (b: Record<string, unknown>) => request<Analysis>("POST", "/api/analyses/sensitivity", b),
  customPreview: (b: CustomAtm) => request<CustomPreview>("POST", "/api/weather/custom/preview", b),
  raids: () => request<RaidRow[]>("GET", "/api/raids"),
  raid: (id: number) => request<Raid>("GET", `/api/raids/${id}`),
  createRaid: (b: { name: string; members: { scenario_id: number; offset_s: number }[] }) => request<Raid>("POST", "/api/raids", b),
  deleteRaid: (id: number) => request<void>("DELETE", `/api/raids/${id}`),
  alerts: () => request<Alert[]>("GET", "/api/alerts"),
  deleteAnalysis: (id: number) => request<void>("DELETE", `/api/analyses/${id}`),
  audit: () => request<AuditRow[]>("GET", "/api/audit"),
};

export function liveUrl(id: number, speed: number, weather: boolean): string {
  const base = BASE || window.location.origin;
  const u = new URL(`${base}/api/ws/scenarios/${id}/live`);
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  u.searchParams.set("token", tokenStore.get() ?? "");
  u.searchParams.set("speed", String(speed));
  u.searchParams.set("weather", weather ? "1" : "0");
  return u.toString();
}

export function alertsUrl(): string {
  const base = BASE || window.location.origin;
  const u = new URL(`${base}/api/ws/alerts`);
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  u.searchParams.set("token", tokenStore.get() ?? "");
  return u.toString();
}
