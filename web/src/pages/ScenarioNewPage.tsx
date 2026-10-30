import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { ScenarioIn } from "../api/types";
import { ProfileChart } from "../components/ProfileChart";
import { Card, ErrorBox, PageHead, Spinner, fmt } from "../components/ui";

type Num = Exclude<keyof ScenarioIn, "name" | "weather_date" | "seed">;
const FIELDS: { key: Num; label: string; unit: string; min: number; max: number; step: number; help?: string }[] = [
  { key: "h0", label: "연소 종료 고도", unit: "m", min: 5000, max: 40000, step: 500 },
  { key: "v0", label: "연소 종료 속도", unit: "m/s", min: 500, max: 2500, step: 50 },
  { key: "gamma_deg", label: "경로각", unit: "°", min: 15, max: 75, step: 1 },
  { key: "psi_deg", label: "방위각 (+x 기준)", unit: "°", min: -30, max: 30, step: 1 },
  { key: "beta", label: "탄도계수 β", unit: "kg/m²", min: 300, max: 20000, step: 100, help: "질량 / (C_D,ref · A)" },
  { key: "obs_fraction", label: "관측 비율", unit: "", min: 0.04, max: 0.6, step: 0.01, help: "전체 비행시간 중 레이더 관측 구간" },
  { key: "sigma_r", label: "거리 잡음 1σ", unit: "m", min: 1, max: 200, step: 1 },
  { key: "sigma_ang_mrad", label: "각도 잡음 1σ", unit: "mrad", min: 0.05, max: 5, step: 0.05 },
];
const PRESETS: Record<string, Partial<ScenarioIn>> = {
  "단거리 · 저탄도계수": { h0: 10000, v0: 1000, gamma_deg: 40, beta: 900 },
  "중거리 (기본)": { h0: 15000, v0: 1400, gamma_deg: 45, beta: 2500 },
  "장거리 · 고탄도계수": { h0: 22000, v0: 1900, gamma_deg: 50, beta: 8000 },
};
const DEFAULT: ScenarioIn = { name: "", h0: 15000, v0: 1400, gamma_deg: 45, psi_deg: 0, beta: 2500, obs_fraction: 0.12, sigma_r: 15, sigma_ang_mrad: 0.4, weather_date: "2026-10-15", weather_hour: 0, seed: null };

export function ScenarioNewPage() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const [f, setF] = useState<ScenarioIn>(DEFAULT);
  const [useWx, setUseWx] = useState(true);
  const wx = useQuery({
    queryKey: ["weather", f.weather_date, f.weather_hour],
    queryFn: () => api.weather(f.weather_date!, f.weather_hour),
    enabled: useWx && !!f.weather_date,
    retry: 0,
  });
  const create = useMutation({
    mutationFn: (b: ScenarioIn) => api.createScenario(b),
    onSuccess: s => { qc.invalidateQueries({ queryKey: ["scenarios"] }); qc.invalidateQueries({ queryKey: ["stats"] }); nav(`/scenarios/${s.id}`); },
  });
  const fieldErr = (k: string) => (create.error instanceof ApiError ? create.error.fields[k] : undefined);
  const set = (k: keyof ScenarioIn, v: unknown) => setF(p => ({ ...p, [k]: v }));

  function submit(e: FormEvent) {
    e.preventDefault();
    create.mutate({ ...f, name: f.name.trim() || `scenario ${new Date().toISOString().slice(0, 16)}`, weather_date: useWx ? f.weather_date : null });
  }

  return (
    <div className="page">
      <PageHead kicker="New scenario" title="새 시나리오">
        연소 종료 상태와 센서 조건을 정하고, 대기를 고르면 서버가 궤적을 적분하고 레이더 측정을 생성해 데이터베이스에 저장합니다.
      </PageHead>
      <form onSubmit={submit} className="page" noValidate>
        <Card title="비행체와 센서" aside={
          <div className="toolbar">{Object.entries(PRESETS).map(([k, v]) => <button type="button" key={k} className="btn ghost sm" onClick={() => setF(p => ({ ...p, ...v }))}>{k}</button>)}</div>
        }>
          <div className="form">
            <div className="field" style={{ gridColumn: "1 / -1" }}>
              <label htmlFor="name">이름</label>
              <input id="name" value={f.name} maxLength={120} placeholder="예: 10월 15일 제트기류 시험" onChange={e => set("name", e.target.value)} />
            </div>
            {FIELDS.map(d => (
              <div className="field" key={d.key}>
                <label htmlFor={d.key}>{d.label}{d.unit && <> ({d.unit})</>}</label>
                <input id={d.key} type="number" min={d.min} max={d.max} step={d.step} value={f[d.key] as number}
                  onChange={e => set(d.key, e.target.value === "" ? "" : Number(e.target.value))} aria-invalid={!!fieldErr(d.key)} />
                {fieldErr(d.key) ? <span className="err">{fieldErr(d.key)}</span> : <small>{d.help ?? `${d.min} – ${d.max}`}</small>}
              </div>
            ))}
            <div className="field">
              <label htmlFor="seed">난수 시드 (선택)</label>
              <input id="seed" type="number" value={f.seed ?? ""} onChange={e => set("seed", e.target.value === "" ? null : Number(e.target.value))} />
              <small>같은 시드 = 같은 측정 잡음</small>
            </div>
          </div>
        </Card>
        <Card title="대기" aside={<label className="chk"><input type="checkbox" checked={useWx} onChange={e => setUseWx(e.target.checked)} /> 선택 일자 기상 사용</label>}>
          {useWx ? (
            <>
              <div className="form">
                <div className="field"><label htmlFor="wdate">기상 일자</label><input id="wdate" type="date" min="2022-01-01" value={f.weather_date ?? ""} onChange={e => set("weather_date", e.target.value)} /></div>
                <div className="field"><label htmlFor="whour">시각 (UTC)</label>
                  <select id="whour" value={f.weather_hour} onChange={e => set("weather_hour", Number(e.target.value))}>
                    {[0, 6, 12, 18].map(h => <option key={h} value={h}>{String(h).padStart(2, "0")}Z</option>)}
                  </select></div>
              </div>
              {wx.isLoading ? <Spinner label="Open-Meteo에서 기압면 자료를 받는 중" /> : <ErrorBox error={wx.error} />}
              {wx.data && (
                <>
                  <p className="ph-sub">최대풍 <b className="mono">{fmt(wx.data.max_wind)} m/s</b> (고도 {fmt(wx.data.max_wind_alt / 1000, 1)} km). 기압면 1000–10 hPa, 약 0–31 km. 그 위는 표준 대기로 이어 붙이고 바람은 0으로 줄입니다.</p>
                  <ProfileChart levels={wx.data.levels} height={240} />
                </>
              )}
            </>
          ) : <p className="ph-sub">표준 모델(지수 밀도, USSA-1976 음속, 무풍)로 비행합니다.</p>}
        </Card>
        <ErrorBox error={create.error} />
        <div className="toolbar">
          <button className="btn" disabled={create.isPending || (useWx && !wx.data)}>{create.isPending ? "시뮬레이션 중…" : "시뮬레이션 실행"}</button>
          <button type="button" className="btn ghost" onClick={() => setF(DEFAULT)}>초기화</button>
        </div>
      </form>
    </div>
  );
}
