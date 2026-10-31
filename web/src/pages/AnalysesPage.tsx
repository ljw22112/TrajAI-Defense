import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, ApiError, DEMO } from "../api/client";
import { CUSTOM_DEFAULT, type AnalysisIn, type AnalysisKind, type CustomAtm } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { CustomAtmForm } from "../components/CustomAtmForm";
import { LocationPicker } from "../components/LocationPicker";
import { Card, Empty, ErrorBox, PageHead, Spinner, fmt } from "../components/ui";

export const KIND_LABEL: Record<AnalysisKind, string> = {
  weather: "기상 영향 (기간)", latency: "기상 자료 지연", sensor: "센서 조건", sensitivity: "환경 민감도",
};
const KIND_HELP: Record<AnalysisKind, string> = {
  weather: "정한 위치·기간의 실제 기상 속으로 같은 비행을 여러 번 날려, 기상이 궤적과 예측 오차를 얼마나 바꾸는지 시각별로 봅니다.",
  latency: "추정기가 비행 시각의 기상 대신 6·12·24시간 전 자료를 받았을 때 오차가 얼마나 늘어나는지 봅니다. \"기상 자료가 얼마나 늦어도 되는가\"에 답합니다.",
  sensor: "레이더 잡음(0.5–4배)과 관측 비율(5–20%)을 바꿔 가며, 기상을 아는 EKF의 IP 오차가 어떻게 변하는지 격자로 봅니다. 센서를 얼마나 개선해야 하는지 판단합니다.",
  sensitivity: "직접 설정한 대기에서 변수 하나(기온, 습도, 기압, 제트기류, 지상풍, 기온감률, 날씨)만 바꿔, 탄착점 이동과 추정 오차를 봅니다.",
};
const PARAMS: Record<string, string> = {
  t_sfc_c: "지상 기온", rh_pct: "상대습도", p_sfc_hpa: "지상 기압", jet_ms: "제트기류 최대풍속", jet_dir: "제트기류 방향",
  wind_sfc_ms: "지상풍", lapse_k_per_km: "기온감률", condition: "날씨 (비·눈·안개)",
};
const FLIGHTS: Record<string, Partial<AnalysisIn>> = {
  "중거리 (기본)": { h0: 15000, v0: 1400, gamma_deg: 45, beta: 2500 },
  "단거리 · 저β": { h0: 10000, v0: 1000, gamma_deg: 40, beta: 900 },
  "장거리 · 고β": { h0: 22000, v0: 1900, gamma_deg: 50, beta: 8000 },
};
const DEFAULT: AnalysisIn = {
  name: "", h0: 15000, v0: 1400, gamma_deg: 45, psi_deg: 0, beta: 2500, obs_fraction: 0.12, sigma_r: 15, sigma_ang_mrad: 0.4,
  seed: 7, start_date: "2026-10-13", days: 3, hours: [0, 6, 12, 18], lat: 36.5, lon: 127.5,
};

export function AnalysesPage() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const { can } = useAuth();
  const list = useQuery({ queryKey: ["analyses"], queryFn: api.analyses });
  const [kind, setKind] = useState<AnalysisKind>("weather");
  const [f, setF] = useState<AnalysisIn>(DEFAULT);
  const [flight, setFlight] = useState("중거리 (기본)");
  const [atmMode, setAtmMode] = useState<"standard" | "date" | "custom">("custom");
  const [atm, setAtm] = useState<CustomAtm>(CUSTOM_DEFAULT);
  const [param, setParam] = useState("t_sfc_c");
  const [filter, setFilter] = useState<AnalysisKind | "all">("all");
  const create = useMutation({
    mutationFn: (): Promise<{ id: number }> => {
      const name = f.name.trim() || `${KIND_LABEL[kind]} · ${flight}`;
      const body = { ...f, name };
      if (kind === "weather") return api.createAnalysis(body);
      if (kind === "latency") return api.createLatency(body);
      const flightOnly = { name, h0: f.h0, v0: f.v0, gamma_deg: f.gamma_deg, psi_deg: f.psi_deg, beta: f.beta, obs_fraction: f.obs_fraction,
        sigma_r: f.sigma_r, sigma_ang_mrad: f.sigma_ang_mrad, seed: f.seed, lat: f.lat, lon: f.lon };
      if (kind === "sensor") return api.createSensor({ ...flightOnly, atmosphere: atmMode, weather_date: f.start_date, weather_hour: f.hours[0] ?? 0, custom_atm: atm });
      return api.createSensitivity({ ...flightOnly, base: atm, param });
    },
    onSuccess: a => { qc.invalidateQueries({ queryKey: ["analyses"] }); nav(`/analyses/${a.id}`); },
  });
  const n = f.days * f.hours.length;
  const fieldErr = (k: string) => (create.error instanceof ApiError ? create.error.fields[k] : undefined);
  const toggleHour = (h: number) => setF(p => ({ ...p, hours: p.hours.includes(h) ? p.hours.filter(x => x !== h) : [...p.hours, h].sort((a, b) => a - b) }));
  const timeWindow = kind === "weather" || kind === "latency";
  const items = (list.data?.items ?? []).filter(a => filter === "all" || a.kind === filter);

  function submit(e: FormEvent) { e.preventDefault(); create.mutate(); }

  return (
    <div className="page">
      <PageHead kicker="Environment & sensor" title="환경 · 센서 분석">
        같은 비행을 여러 환경과 센서 조건에서 반복해, 무엇이 탄착 예측 오차를 만드는지 분리합니다. 기온·습도·기압·바람·날씨는 직접 정할 수도 있고, 정한 위치·일자의 실제 기상(Open-Meteo)을 쓸 수도 있습니다.
      </PageHead>
      {can("analyst") && !DEMO && (
        <Card title="새 분석">
          <div className="toolbar" role="tablist" aria-label="분석 종류" style={{ marginBottom: 8 }}>
            {(Object.keys(KIND_LABEL) as AnalysisKind[]).map(k => (
              <button key={k} type="button" role="tab" aria-selected={kind === k} className={`btn sm ${kind === k ? "" : "ghost"}`} onClick={() => setKind(k)}>{KIND_LABEL[k]}</button>
            ))}
          </div>
          <p className="ph-sub" style={{ marginTop: 0 }}>{KIND_HELP[kind]}</p>
          <form className="page" style={{ gap: 12 }} onSubmit={submit} noValidate>
            <div className="form">
              <div className="field" style={{ gridColumn: "1 / -1" }}>
                <label htmlFor="an-name">이름 (선택)</label>
                <input id="an-name" value={f.name} maxLength={120} placeholder={`예: ${KIND_LABEL[kind]} · 10월`} onChange={e => setF(p => ({ ...p, name: e.target.value }))} />
              </div>
              <div className="field">
                <label htmlFor="an-flight">비행 조건</label>
                <select id="an-flight" value={flight} onChange={e => { setFlight(e.target.value); setF(p => ({ ...p, ...FLIGHTS[e.target.value] })); }}>
                  {Object.keys(FLIGHTS).map(k => <option key={k}>{k}</option>)}
                </select>
                <small>연소 종료 {fmt(f.h0 / 1000)} km · {fmt(f.v0)} m/s · β {fmt(f.beta)} kg/m²</small>
              </div>
              <div className="field">
                <label htmlFor="an-obs">관측 비율</label>
                <input id="an-obs" type="number" min={0.04} max={0.6} step={0.01} value={f.obs_fraction} onChange={e => setF(p => ({ ...p, obs_fraction: Number(e.target.value) }))} aria-invalid={!!fieldErr("obs_fraction")} />
                {fieldErr("obs_fraction") ? <span className="err">{fieldErr("obs_fraction")}</span> : <small>{kind === "sensor" ? "센서 분석은 5–20%를 모두 계산" : "레이더가 본 비행시간 비율"}</small>}
              </div>
              <div className="field">
                <label htmlFor="an-sr">레이더 잡음 (거리 m / 각도 mrad)</label>
                <div style={{ display: "flex", gap: 6 }}>
                  <input id="an-sr" type="number" min={1} max={200} value={f.sigma_r} aria-label="거리 잡음" onChange={e => setF(p => ({ ...p, sigma_r: Number(e.target.value) }))} />
                  <input type="number" min={0.05} max={5} step={0.05} value={f.sigma_ang_mrad} aria-label="각도 잡음" onChange={e => setF(p => ({ ...p, sigma_ang_mrad: Number(e.target.value) }))} />
                </div>
                <small>{kind === "sensor" ? "기준값. ×0.5, ×1, ×2, ×4를 계산" : "1σ"}</small>
              </div>
              {(timeWindow || (kind === "sensor" && atmMode === "date")) && (
                <LocationPicker idPrefix="an" lat={f.lat ?? 36.5} lon={f.lon ?? 127.5} onChange={(lat, lon) => setF(p => ({ ...p, lat, lon }))} />
              )}
              {timeWindow && (
                <>
                  <div className="field">
                    <label htmlFor="an-start">시작일</label>
                    <input id="an-start" type="date" min="2022-01-01" value={f.start_date} onChange={e => setF(p => ({ ...p, start_date: e.target.value }))} aria-invalid={!!fieldErr("start_date")} />
                    {fieldErr("start_date") && <span className="err">{fieldErr("start_date")}</span>}
                  </div>
                  <div className="field">
                    <label htmlFor="an-days">기간 (일)</label>
                    <input id="an-days" type="number" min={1} max={7} value={f.days} onChange={e => setF(p => ({ ...p, days: Number(e.target.value) }))} aria-invalid={!!fieldErr("days")} />
                    {fieldErr("days") ? <span className="err">{fieldErr("days")}</span> : <small>1–7일</small>}
                  </div>
                  <div className="field" style={{ gridColumn: "1 / -1" }}>
                    <span className="label">시각 (UTC)</span>
                    <div className="toolbar">
                      {[0, 6, 12, 18].map(h => (
                        <label key={h} className="chk"><input type="checkbox" checked={f.hours.includes(h)} onChange={() => toggleHour(h)} /> {String(h).padStart(2, "0")}Z</label>
                      ))}
                    </div>
                  </div>
                </>
              )}
              {kind === "sensor" && (
                <div className="field" style={{ gridColumn: "1 / -1" }}>
                  <span className="label">대기</span>
                  <div className="toolbar" role="radiogroup" aria-label="센서 분석 대기">
                    {([["standard", "표준 대기"], ["date", "일자 기상 (API)"], ["custom", "직접 설정"]] as const).map(([k, l]) => (
                      <label key={k} className="chk"><input type="radio" name="sensor-atm" checked={atmMode === k} onChange={() => setAtmMode(k)} /> {l}</label>
                    ))}
                    {atmMode === "date" && <input type="date" aria-label="기상 일자" min="2022-01-01" value={f.start_date} style={{ maxWidth: 170 }} onChange={e => setF(p => ({ ...p, start_date: e.target.value }))} />}
                  </div>
                </div>
              )}
              {kind === "sensitivity" && (
                <div className="field">
                  <label htmlFor="an-param">바꿀 변수</label>
                  <select id="an-param" value={param} onChange={e => setParam(e.target.value)}>
                    {Object.entries(PARAMS).map(([k, l]) => <option key={k} value={k}>{l}</option>)}
                  </select>
                  <small>나머지는 아래 설정값으로 고정</small>
                </div>
              )}
            </div>
            {(kind === "sensitivity" || (kind === "sensor" && atmMode === "custom")) && (
              <fieldset className="fs"><legend>{kind === "sensitivity" ? "기준 대기 (직접 설정)" : "대기 (직접 설정)"}</legend>
                <CustomAtmForm value={atm} onChange={setAtm} idPrefix="an-atm" />
              </fieldset>
            )}
            {create.error && !(create.error instanceof ApiError && Object.keys(create.error.fields).length) && <ErrorBox error={create.error} />}
            <div className="toolbar">
              <button className="btn" disabled={create.isPending || (timeWindow && n === 0)}>
                {create.isPending ? "계산 중…" : timeWindow ? `분석 실행 (프로파일 ${n}개)` : "분석 실행"}
              </button>
              <small className="label">{kind === "sensor" ? "16개 조건 × 잡음 표본 16개, 수십 초" : kind === "sensitivity" ? "변수 값 4–6개 × 잡음 표본 16개" : "같은 레이더 잡음으로 추정기를 비교합니다. 기상 조회 포함 수 초–수십 초"}</small>
            </div>
          </form>
        </Card>
      )}
      <Card title={`저장된 분석 (${list.data?.total ?? 0})`} aside={
        <select aria-label="분석 종류로 거르기" value={filter} onChange={e => setFilter(e.target.value as AnalysisKind | "all")} style={{ maxWidth: 180 }}>
          <option value="all">전체</option>
          {(Object.keys(KIND_LABEL) as AnalysisKind[]).map(k => <option key={k} value={k}>{KIND_LABEL[k]}</option>)}
        </select>
      }>
        <ErrorBox error={list.error} />
        {list.isLoading ? <Spinner /> : !items.length ? <Empty>아직 분석이 없습니다.</Empty> : (
          <div className="tw">
            <table>
              <thead><tr><th>종류</th><th>이름</th><th>핵심 결과</th><th>만든 시각</th></tr></thead>
              <tbody>
                {items.map(a => (
                  <tr key={a.id} className="clickable" onClick={() => nav(`/analyses/${a.id}`)}>
                    <td className="nowrap"><span className="chip">{KIND_LABEL[a.kind]}</span></td>
                    <td><Link to={`/analyses/${a.id}`}>{a.name}</Link></td>
                    <td>{headline(a)}</td>
                    <td className="mono nowrap">{new Date(a.created_at).toLocaleDateString("ko-KR")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

function headline(a: { kind: AnalysisKind; summary: Record<string, any> }): string {
  const s = a.summary;
  if (a.kind === "weather") return `IP 오차 ${fmt(s.median_miss_std)} → ${fmt(s.median_miss_wx)} m (기상 반영 ${s.reduction === null ? "—" : fmt(s.reduction * 100) + "%"} 감소)`;
  if (a.kind === "latency") return `최신 ${fmt(s.fresh_miss)} m · 자료 없음 ${fmt(s.std_miss)} m · 이득 절반 유지 ${s.max_lag_half_benefit_h === null ? "—" : s.max_lag_half_benefit_h + "h"}`;
  if (a.kind === "sensor") return `기준 ${fmt(s.baseline?.median_miss)} m${s.halve_error_with ? ` · 오차 절반: 잡음 ×${s.halve_error_with.noise_mult}, 관측 ${fmt(s.halve_error_with.obs_fraction * 100)}%` : ""}`;
  return `${s.label}: 탄착 이동 최대 ${fmt(s.max_shift_m)} m · 표준 EKF 오차 폭 ${fmt(s.range_miss_std)} m`;
}
