import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../api/client";
import type { AnalysisItem } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { Legend, Plot, type Series } from "../components/Plot";
import { PrintButton } from "../components/PrintButton";
import { Card, ErrorBox, PageHead, Spinner, Stat, Term, fmt } from "../components/ui";
import { OtherAnalysis } from "./AnalysisKinds";

const utc = (iso: string) => new Date(iso).toISOString();
const label = (iso: string) => { const s = utc(iso); return `${s.slice(5, 7)}/${s.slice(8, 10)} ${s.slice(11, 13)}Z`; };

export function AnalysisDetailPage() {
  const id = Number(useParams().id);
  const nav = useNavigate();
  const qc = useQueryClient();
  const { can, user } = useAuth();
  const q = useQuery({ queryKey: ["analysis", id], queryFn: () => api.analysis(id) });
  const del = useMutation({ mutationFn: () => api.deleteAnalysis(id), onSuccess: () => { qc.invalidateQueries({ queryKey: ["analyses"] }); nav("/analyses"); } });
  const a = q.data;

  const t0 = a ? new Date(a.items[0].valid_time).getTime() : 0;
  // x axis in days so that "nice" ticks land on whole days (00Z)
  const hrs = (it: AnalysisItem) => (new Date(it.valid_time).getTime() - t0) / 8.64e7;
  const fmtT = (v: number) => label(new Date(t0 + Math.round(v * 4) / 4 * 8.64e7).toISOString());

  const errSeries = useMemo<Series[]>(() => !a ? [] : [
    { kind: "line", x: a.items.map(hrs), y: a.items.map(i => i.miss_std), color: "--truth", width: 2, dash: [6, 4] },
    { kind: "points", x: a.items.map(hrs), y: a.items.map(i => i.miss_std), color: "--truth", size: 2.5 },
    { kind: "line", x: a.items.map(hrs), y: a.items.map(i => i.miss_wx), color: "--ekf", width: 2.2 },
    { kind: "points", x: a.items.map(hrs), y: a.items.map(i => i.miss_wx), color: "--ekf", size: 2.5 },
  ], [a]);  // eslint-disable-line react-hooks/exhaustive-deps
  const windSeries = useMemo<Series[]>(() => !a ? [] : [
    { kind: "line", x: a.items.map(hrs), y: a.items.map(i => i.max_wind), color: "--pred", width: 2 },
    { kind: "points", x: a.items.map(hrs), y: a.items.map(i => i.max_wind), color: "--pred", size: 2.5 },
  ], [a]);  // eslint-disable-line react-hooks/exhaustive-deps
  const impactSeries = useMemo<Series[]>(() => !a ? [] : [
    { kind: "cross", x: [0], y: [0], color: "--ink", size: 6 },
    { kind: "points", x: a.items.map(i => i.dx_m / 1000), y: a.items.map(i => i.dy_m / 1000), color: "--pred", size: 3.5 },
  ], [a]);

  if (q.isLoading) return <Spinner />;
  if (q.error || !a) return <ErrorBox error={q.error ?? new Error("분석을 찾을 수 없습니다.")} />;
  const canDel = can("admin") || (can("analyst") && a.created_by === user?.id);
  if (a.kind !== "weather") return <OtherAnalysis a={a} canDelete={canDel} onDelete={() => del.mutate()} deleting={del.isPending} error={del.error} />;
  const s = a.summary, p = a.params;
  const rho = s.spearman_wind_vs_std_error;
  const canDelete = can("admin") || (can("analyst") && a.created_by === user?.id);

  return (
    <div className="page">
      <PageHead kicker={`Analysis #${a.id}`} title={a.name}
        actions={<div className="toolbar"><PrintButton />{canDelete && <button className="btn danger no-print" disabled={del.isPending} onClick={() => del.mutate()}>삭제</button>}</div>}>
        {a.lat.toFixed(2)}°N {a.lon.toFixed(2)}°E · {label(a.items[0].valid_time)} – {label(a.items[a.items.length - 1].valid_time)} · 프로파일 {a.n_profiles}개 ·
        연소 종료 {fmt(p.h0 / 1000)} km, {fmt(p.v0)} m/s, 경로각 {fmt(p.gamma_deg)}°, β {fmt(p.beta)} kg/m² · 관측 {fmt(p.obs_fraction * 100)}%
      </PageHead>
      <ErrorBox error={del.error} />
      <div className="grid g4">
        <Stat label="IP 오차 중앙값 · 표준 대기 EKF" value={fmt(s.median_miss_std)} unit="m" />
        <Stat label="IP 오차 중앙값 · 기상 반영 EKF" value={fmt(s.median_miss_wx)} unit="m" tone="ok" />
        <Stat label="기상 반영 이득" value={s.reduction === null ? "—" : `${fmt(s.reduction * 100)}%`} unit={`· ${fmt(s.wx_better_frac * 100)}% 시각에서 개선`} tone="acc" />
        <Stat label="탄착 이동 (중앙값 / 최대)" value={fmt(s.median_shift_m)} unit={`/ ${fmt(s.max_shift_m)} m`} />
      </div>
      <Card title="해석">
        <div className="prose" style={{ maxWidth: "none" }}>
          <p>이 기간의 기온·바람은 같은 비행의 탄착점을 표준 대기(USSA-1976, 무풍) 대비 중앙값 <b>{fmt(s.median_shift_m)} m</b>, 최대 <b>{fmt(s.max_shift_m)} m</b> 옮겼습니다. 기간 안에서도 탄착점은 하향거리로 {fmt(s.downrange_range_m)} m, 횡방향으로 {fmt(s.crossrange_range_m)} m 범위에서 움직였으므로(<Term k="일간 변동" /> 중앙값 {fmt(s.spread_p50_m)} m), 하루 전 기상이나 기후값으로는 이 차이를 메울 수 없습니다.</p>
          <p>같은 레이더 측정에서 표준 대기 EKF의 IP 오차는 중앙값 {fmt(s.median_miss_std)} m(CEP 포함률 {fmt(s.coverage_std, 2)}), 그 시각 기상을 반영한 EKF는 {fmt(s.median_miss_wx)} m(포함률 {fmt(s.coverage_wx, 2)})였습니다.
            {rho !== null && <> 최대 풍속과 표준 대기 EKF 오차의 순위 상관은 ρ = {fmt(rho, 2)}로, {Math.abs(rho) >= 0.5 ? "바람이 강한 시각일수록 기상 자료 없이 추정한 오차가 큽니다" : "오차는 최대 풍속만으로 설명되지 않으며 기온(밀도) 분포의 영향이 큽니다"}.</>}
            {" "}기상 반영의 이득이 가장 컸던 시각은 <b>{label(s.largest_gain_time)}</b>({fmt(s.largest_gain_m)} m 감소)입니다.</p>
          <p className="label">주의: 이 분석에서 기상 반영 EKF는 비행에 쓰인 것과 같은 프로파일을 받으므로 분석장 오차가 없는 낙관적 조건입니다. 분석장 오차를 넣은 평가는 기술 문서 §9.3에 있습니다.</p>
        </div>
      </Card>
      <div className="grid g2">
        <Card title="시각별 IP 오차">
          <Plot series={errSeries} xLabel="유효 시각 (UTC)" yLabel="IP 오차 (m)" height={280} logY fmtX={fmtT} ariaLabel="시각별 표준 대기와 기상 반영 EKF의 IP 오차" />
          <Legend items={[{ label: "표준 대기 EKF", color: "--truth", dash: true }, { label: "기상 반영 EKF", color: "--ekf" }]} />
        </Card>
        <Card title="최대 풍속 (기압면 1000–10 hPa)">
          <Plot series={windSeries} xLabel="유효 시각 (UTC)" yLabel="최대 풍속 (m/s)" height={280} fmtX={fmtT} ariaLabel="시각별 최대 풍속" />
          <Legend items={[{ label: "최대 풍속", color: "--pred" }]} />
        </Card>
      </div>
      <div className="grid g2">
        <Card title="탄착점 이동 (표준 대기 기준, 지면 투영)">
          <Plot series={impactSeries} xLabel="Δx 하향거리 (km)" yLabel="Δy 횡방향 (km)" height={300} equal ariaLabel="표준 대기 대비 시각별 탄착점 이동" />
          <small className="label">+ = 표준 대기(USSA-1976, 무풍) 탄착점 · 점 = 각 시각의 실제 탄착점</small>
        </Card>
        <Card title={`시각별 결과 (${a.items.length})`}>
          <div className="tw" style={{ maxHeight: 360, overflowY: "auto" }}>
            <table>
              <thead><tr><th>시각</th><th className="n">최대풍</th><th className="n"><Term k="탄착 이동" /></th><th className="n">표준 EKF</th><th className="n">기상 EKF</th></tr></thead>
              <tbody>
                {a.items.map(i => (
                  <tr key={i.valid_time}>
                    <td className="mono nowrap">{label(i.valid_time)}</td>
                    <td className="n">{fmt(i.max_wind)} m/s</td>
                    <td className="n">{fmt(i.shift_m)} m</td>
                    <td className={`n ${i.inside_std ? "" : "bad"}`}>{fmt(i.miss_std)} m</td>
                    <td className={`n ${i.inside_wx ? "ok" : "bad"}`}>{fmt(i.miss_wx)} m</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <small className="label">빨강 = 실제 탄착점이 예측 CEP 밖</small>
        </Card>
      </div>
    </div>
  );
}
