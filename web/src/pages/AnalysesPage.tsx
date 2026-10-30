import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, ApiError, DEMO } from "../api/client";
import type { AnalysisIn } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { LocationPicker } from "../components/LocationPicker";
import { Card, Empty, ErrorBox, PageHead, Spinner, Term, fmt } from "../components/ui";

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
  const [f, setF] = useState<AnalysisIn>(DEFAULT);
  const [flight, setFlight] = useState("중거리 (기본)");
  const create = useMutation({
    mutationFn: api.createAnalysis,
    onSuccess: a => { qc.invalidateQueries({ queryKey: ["analyses"] }); nav(`/analyses/${a.id}`); },
  });
  const n = f.days * f.hours.length;
  const fieldErr = (k: string) => (create.error instanceof ApiError ? create.error.fields[k] : undefined);

  function submit(e: FormEvent) {
    e.preventDefault();
    create.mutate({ ...f, name: f.name.trim() || `${f.start_date} 부터 ${f.days}일 · ${flight}` });
  }
  const toggleHour = (h: number) => setF(p => ({ ...p, hours: p.hours.includes(h) ? p.hours.filter(x => x !== h) : [...p.hours, h].sort((a, b) => a - b) }));

  return (
    <div className="page">
      <PageHead kicker="Weather impact" title="기상 영향 분석">
        같은 비행을 정한 위치·기간의 실제 기상 속으로 여러 번 날려, 그날의 기온·바람이 궤적을 얼마나 움직이는지, 그리고 기상을 반영한 추정이 예측 오차를 얼마나 줄이는지 시각별로 비교합니다. 운용 측면에서는 "언제 기상 자료를 꼭 받아 써야 하는가"를 판단하는 도구입니다.
      </PageHead>
      {can("analyst") && !DEMO && (
        <Card title="새 분석">
          <form className="form" onSubmit={submit} noValidate>
            <div className="field" style={{ gridColumn: "1 / -1" }}>
              <label htmlFor="an-name">이름 (선택)</label>
              <input id="an-name" value={f.name} maxLength={120} placeholder="예: 10월 셋째 주 · 중부" onChange={e => setF(p => ({ ...p, name: e.target.value }))} />
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
              {fieldErr("obs_fraction") ? <span className="err">{fieldErr("obs_fraction")}</span> : <small>레이더가 본 비행시간 비율</small>}
            </div>
            <LocationPicker idPrefix="an" lat={f.lat ?? 36.5} lon={f.lon ?? 127.5} onChange={(lat, lon) => setF(p => ({ ...p, lat, lon }))} />
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
              {fieldErr("hours") && <span className="err">{fieldErr("hours")}</span>}
            </div>
            <div className="toolbar" style={{ gridColumn: "1 / -1" }}>
              <button className="btn" disabled={create.isPending || n === 0}>{create.isPending ? `프로파일 ${n}개 분석 중…` : `분석 실행 (프로파일 ${n}개)`}</button>
              <small className="label">같은 레이더 잡음으로 표준 대기 EKF와 기상 반영 EKF를 각각 실행합니다. 기상 자료 조회 포함 수 초–수십 초.</small>
            </div>
            {create.error && !(create.error instanceof ApiError && Object.keys(create.error.fields).length) && <div style={{ gridColumn: "1 / -1" }}><ErrorBox error={create.error} /></div>}
          </form>
        </Card>
      )}
      <Card title={`저장된 분석 (${list.data?.total ?? 0})`}>
        <ErrorBox error={list.error} />
        {list.isLoading ? <Spinner /> : !list.data?.items.length ? <Empty>아직 분석이 없습니다.</Empty> : (
          <div className="tw">
            <table>
              <thead><tr><th>이름</th><th>위치</th><th className="n">프로파일</th><th className="n"><Term k="탄착 이동">탄착 이동 중앙값</Term></th>
                <th className="n">IP 오차 · 표준 → 기상</th><th className="n"><Term k="기상 반영 이득" /></th></tr></thead>
              <tbody>
                {list.data.items.map(a => (
                  <tr key={a.id} className="clickable" onClick={() => nav(`/analyses/${a.id}`)}>
                    <td><Link to={`/analyses/${a.id}`}>{a.name}</Link></td>
                    <td className="mono">{a.lat.toFixed(2)}°N {a.lon.toFixed(2)}°E</td>
                    <td className="n">{a.n_profiles}</td>
                    <td className="n">{fmt(a.summary.median_shift_m)} m</td>
                    <td className="n">{fmt(a.summary.median_miss_std)} → <b>{fmt(a.summary.median_miss_wx)}</b> m</td>
                    <td className="n">{a.summary.reduction === null ? "—" : `${fmt(a.summary.reduction * 100)}%`}</td>
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
