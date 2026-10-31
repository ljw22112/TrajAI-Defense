import { useMemo } from "react";
import type { Analysis } from "../api/types";
import { CONDITION_LABEL, type Condition } from "../api/types";
import { Legend, Plot, type Series } from "../components/Plot";
import { PrintButton } from "../components/PrintButton";
import { Card, ErrorBox, PageHead, Stat, fmt } from "../components/ui";
import { KIND_LABEL } from "./AnalysesPage";

type Props = { a: Analysis; canDelete: boolean; onDelete: () => void; deleting: boolean; error: unknown };

/** Detail views for the latency, sensor-sweep and environment-sensitivity studies. */
export function OtherAnalysis({ a, canDelete, onDelete, deleting, error }: Props) {
  const p = a.params;
  return (
    <div className="page">
      <PageHead kicker={`${KIND_LABEL[a.kind]} · #${a.id}`} title={a.name}
        actions={<div className="toolbar"><PrintButton />{canDelete && <button className="btn danger no-print" disabled={deleting} onClick={onDelete}>삭제</button>}</div>}>
        연소 종료 {fmt(p.h0 / 1000)} km, {fmt(p.v0)} m/s, 경로각 {fmt(p.gamma_deg)}°, β {fmt(p.beta)} kg/m² · 관측 {fmt(p.obs_fraction * 100)}% ·
        레이더 {fmt(p.sigma_r)} m / {fmt(p.sigma_ang_mrad, 2)} mrad
      </PageHead>
      <ErrorBox error={error} />
      {a.kind === "latency" && <Latency a={a} />}
      {a.kind === "sensor" && <Sensor a={a} />}
      {a.kind === "sensitivity" && <Sensitivity a={a} />}
    </div>
  );
}

function Latency({ a }: { a: Analysis }) {
  const s = a.summary as any;
  const fresh = s.curve.filter((c: any) => c.lag_h !== null);
  const series = useMemo<Series[]>(() => [
    { kind: "line", x: fresh.map((c: any) => c.lag_h), y: fresh.map((c: any) => c.median_miss), color: "--ekf", width: 2.4 },
    { kind: "points", x: fresh.map((c: any) => c.lag_h), y: fresh.map((c: any) => c.median_miss), color: "--ekf", size: 3.5 },
    { kind: "line", x: [0, 24], y: [s.std_miss, s.std_miss], color: "--truth", width: 1.8, dash: [6, 4] },
    ...(s.ussa_miss ? [{ kind: "line" as const, x: [0, 24], y: [s.ussa_miss, s.ussa_miss], color: "--pred", width: 1.8, dash: [3, 3] }] : []),
  ], [a]);  // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <>
      <div className="grid g4">
        <Stat label="최신 기상 (지연 0 h)" value={fmt(s.fresh_miss)} unit="m" tone="ok" />
        <Stat label="24 h 지난 기상" value={fmt(fresh[fresh.length - 1].median_miss)} unit="m" />
        <Stat label="일자 자료 없음 · USSA-1976" value={fmt(s.ussa_miss ?? s.std_miss)} unit={`m (지수 대기 ${fmt(s.std_miss)} m)`} />
        <Stat label="이득 절반 유지 지연" value={s.max_lag_half_benefit_h === null ? "—" : String(s.max_lag_half_benefit_h)} unit="h" tone="acc" />
      </div>
      <Card title="해석">
        <div className="prose" style={{ maxWidth: "none" }}>
          <p>비행 시각의 기상을 그대로 쓰면 IP 오차 중앙값은 <b>{fmt(s.fresh_miss)} m</b>입니다. 자료가 6·12·24시간 늦으면 각각 {fresh.slice(1).map((c: any) => `${fmt(c.median_miss)} m`).join(", ")}가 됩니다.
            일자 자료 없이 USSA-1976 표준 대기를 쓰면 {fmt(s.ussa_miss)} m, 지수 대기(표준 모델)를 쓰면 {fmt(s.std_miss)} m입니다.
            {s.max_lag_half_benefit_h !== null ? <> USSA-1976 대비 일자 기상의 이득은 자료가 <b>{s.max_lag_half_benefit_h}시간</b> 늦어도 절반 이상 남습니다.</> : " USSA-1976 대비 일자 기상의 이득은 6시간만 늦어도 절반 이하로 줄어듭니다."}
            {" "}오차의 대부분은 지수 대기 모델 자체에서 오므로, 먼저 현실적인 대기 모델을 쓰고 그다음 자료의 신선도를 챙기는 순서가 맞습니다.</p>
          <p className="label">지연된 자료는 그 시각의 분석장으로 근사했습니다(같은 시각의 실제 예보 오차와는 다를 수 있음). 같은 레이더 잡음으로 모든 지연을 비교합니다.</p>
        </div>
      </Card>
      <div className="grid g2">
        <Card title="자료 지연에 따른 IP 오차 (중앙값)">
          <Plot series={series} xLabel="기상 자료 지연 (h)" yLabel="IP 오차 (m)" height={280} logY ariaLabel="기상 자료 지연에 따른 IP 오차" />
          <Legend items={[{ label: "지연된 기상 반영 EKF", color: "--ekf" }, { label: "USSA-1976 (일자 자료 없음)", color: "--pred", dash: true }, { label: "지수 대기 (표준 모델)", color: "--truth", dash: true }]} />
        </Card>
        <Card title="지연별 결과">
          <div className="tw"><table>
            <thead><tr><th>지연</th><th className="n">IP 오차 중앙값</th><th className="n">p90</th><th className="n">CEP 포함률</th></tr></thead>
            <tbody>{s.curve.map((c: any) => (
              <tr key={String(c.lag_h ?? c.ref)}><td>{c.lag_h !== null ? `${c.lag_h} h` : c.ref === "ussa" ? "USSA-1976 (자료 없음)" : "지수 대기 (표준 모델)"}</td><td className="n">{fmt(c.median_miss)} m</td>
                <td className="n">{fmt(c.p90_miss)} m</td><td className="n">{fmt(c.coverage, 2)}</td></tr>))}
            </tbody>
          </table></div>
        </Card>
      </div>
    </>
  );
}

function Sensor({ a }: { a: Analysis }) {
  const s = a.summary as any;
  const cells = a.items as any[];
  const noise = [...new Set(cells.map(c => c.noise_mult))].sort((x, y) => x - y);
  const obs = [...new Set(cells.map(c => c.obs_fraction))].sort((x, y) => x - y);
  const cell = (m: number, f: number) => cells.find(c => c.noise_mult === m && c.obs_fraction === f);
  const lo = Math.log(Math.min(...cells.map(c => c.median_miss))), hi = Math.log(Math.max(...cells.map(c => c.median_miss)));
  const shade = (v: number) => `color-mix(in srgb, var(--pred) ${Math.round(8 + 72 * (Math.log(v) - lo) / Math.max(hi - lo, 1e-9))}%, transparent)`;
  const series = useMemo<Series[]>(() => noise.map((m, i) => ({
    kind: "line" as const, x: obs.map(f => f * 100), y: obs.map(f => cell(m, f).median_miss), color: ["--ok", "--ekf", "--pred", "--danger"][i % 4], width: 2.2,
  })), [a]);  // eslint-disable-line react-hooks/exhaustive-deps
  const atm = a.params.atmosphere === "custom" ? "직접 설정한 대기" : a.params.atmosphere === "date" ? `${a.params.weather_date} 기상` : "표준 대기";
  return (
    <>
      <div className="grid g4">
        <Stat label="기준 조건 IP 오차" value={fmt(s.baseline.median_miss)} unit="m" />
        <Stat label="최선 조건" value={fmt(Math.min(...cells.map(c => c.median_miss)))} unit="m" tone="ok" />
        <Stat label="최악 조건" value={fmt(Math.max(...cells.map(c => c.median_miss)))} unit="m" />
        <Stat label="오차 절반이 되는 가장 싼 조건" value={s.halve_error_with ? `×${s.halve_error_with.noise_mult} · ${fmt(s.halve_error_with.obs_fraction * 100)}%` : "없음"} tone="acc" />
      </div>
      <Card title="해석">
        <div className="prose" style={{ maxWidth: "none" }}>
          <p>{atm}에서 대기를 아는 EKF의 IP 오차(중앙값, 잡음 표본 {s.reps}개)입니다. 기준 조건(잡음 ×1, 관측 {fmt(s.baseline.obs_fraction * 100)}%)은 {fmt(s.baseline.median_miss)} m입니다.
            {s.halve_error_with ? <> 오차를 절반으로 줄이는 가장 적은 관측 비율 조건은 잡음 ×{s.halve_error_with.noise_mult}, 관측 {fmt(s.halve_error_with.obs_fraction * 100)}%({fmt(s.halve_error_with.median_miss)} m)입니다.</> : " 이 격자 안에서는 오차를 절반으로 줄이는 조건이 없습니다."}
            {" "}관측 비율이 짧을수록 β를 관측할 시간이 부족해 오차가 급격히 커지는 것이 일반적입니다.</p>
        </div>
      </Card>
      <div className="grid g2">
        <Card title="IP 오차 중앙값 [m] (행 = 레이더 잡음 배수, 열 = 관측 비율)">
          <div className="tw"><table className="heat">
            <thead><tr><th>잡음</th>{obs.map(f => <th key={f} className="n">{fmt(f * 100)}%</th>)}</tr></thead>
            <tbody>{noise.map(m => (
              <tr key={m}><td className="nowrap">×{m} <small className="muted">({fmt(cell(m, obs[0]).sigma_r)} m)</small></td>
                {obs.map(f => { const c = cell(m, f); return <td key={f} className="n" style={{ background: shade(c.median_miss) }} title={`CEP 포함률 ${fmt(c.coverage, 2)} · 관측 ${fmt(c.track_s, 1)} s`}>{fmt(c.median_miss)}</td>; })}
              </tr>))}
            </tbody>
          </table></div>
          <small className="label">진할수록 오차가 큼 (로그 척도). 칸에 마우스를 올리면 CEP 포함률과 관측 시간이 나옵니다.</small>
        </Card>
        <Card title="관측 비율에 따른 IP 오차">
          <Plot series={series} xLabel="관측 비율 (%)" yLabel="IP 오차 (m)" height={280} logY ariaLabel="잡음 배수별 관측 비율과 IP 오차" />
          <Legend items={noise.map((m, i) => ({ label: `잡음 ×${m}`, color: ["--ok", "--ekf", "--pred", "--danger"][i % 4] }))} />
        </Card>
      </div>
    </>
  );
}

function Sensitivity({ a }: { a: Analysis }) {
  const s = a.summary as any;
  const items = a.items as any[];
  const isCond = s.param === "condition";
  const lbl = (v: any) => isCond ? CONDITION_LABEL[v as Condition] ?? v : `${fmt(v, Number.isInteger(v) ? 0 : 1)}${s.unit ? " " + s.unit : ""}`;
  const xs = items.map((it, i) => (isCond ? i : it.value));
  const err = useMemo<Series[]>(() => [
    { kind: "line", x: xs, y: items.map(i => i.miss_std), color: "--truth", width: 2, dash: [6, 4] },
    { kind: "points", x: xs, y: items.map(i => i.miss_std), color: "--truth", size: 3 },
    { kind: "line", x: xs, y: items.map(i => i.miss_wx), color: "--ekf", width: 2.4 },
    { kind: "points", x: xs, y: items.map(i => i.miss_wx), color: "--ekf", size: 3 },
  ], [a]);  // eslint-disable-line react-hooks/exhaustive-deps
  const shift = useMemo<Series[]>(() => [
    { kind: "line", x: xs, y: items.map(i => i.dx_m), color: "--pred", width: 2.4 },
    { kind: "points", x: xs, y: items.map(i => i.dx_m), color: "--pred", size: 3 },
    { kind: "line", x: xs, y: items.map(i => i.dy_m), color: "--obs", width: 2, dash: [5, 4] },
  ], [a]);  // eslint-disable-line react-hooks/exhaustive-deps
  const fmtX = isCond ? (v: number) => lbl(items[Math.round(v)]?.value ?? "") : undefined;
  const big = items.reduce((m, i) => (i.shift_m > m.shift_m ? i : m), items[0]);
  return (
    <>
      <div className="grid g4">
        <Stat label="바꾼 변수" value={s.label} />
        <Stat label="탄착점 이동 최대" value={fmt(s.max_shift_m)} unit={`m @ ${lbl(big.value)}`} tone="acc" />
        <Stat label="표준 대기 EKF 오차 폭" value={fmt(s.range_miss_std)} unit="m" />
        <Stat label="대기를 아는 EKF 오차 폭" value={fmt(s.range_miss_wx)} unit="m" tone="ok" />
      </div>
      <Card title="해석">
        <div className="prose" style={{ maxWidth: "none" }}>
          {isCond
            ? <p>날씨는 비행 자체(항력)를 거의 바꾸지 않으므로 탄착점은 그대로이고, 레이더 잡음만 커집니다(최대 ×{fmt(Math.max(...items.map(i => i.noise_factor)), 1)}). 대기를 아는 EKF의 오차는 {lbl(items[0].value)} {fmt(items[0].miss_wx)} m에서 {lbl(items[items.length - 1].value)} {fmt(items[items.length - 1].miss_wx)} m로 변합니다. 강수 중에는 센서 성능 저하가 지배적인 오차 원인입니다.</p>
            : <p>{s.label}을(를) 바꾸면 같은 비행의 탄착점이 기준 대기 대비 최대 <b>{fmt(s.max_shift_m)} m</b>({lbl(big.value)}) 움직입니다. 이 변화를 모르는 표준 대기 EKF의 오차는 {fmt(Math.min(...items.map(i => i.miss_std)))}–{fmt(Math.max(...items.map(i => i.miss_std)))} m 범위로 흔들리지만, 대기를 아는 EKF는 {fmt(Math.min(...items.map(i => i.miss_wx)))}–{fmt(Math.max(...items.map(i => i.miss_wx)))} m입니다. 이 변수를 측정해 반영하는 것이 얼마나 중요한지를 보여 줍니다.</p>}
          {(s.param === "t_sfc_c" || s.param === "lapse_k_per_km") && <p>이 모델에서는 지상 기온이나 기온감률을 바꾸면 대류권계면(190–230 K로 제한)과 성층권 기온도 함께 바뀝니다. 차가운 공기 기둥은 고도가 올라갈수록 밀도가 빨리 줄어 상층 항력이 작아지므로 사거리가 늘어납니다. 극단값에서의 이동 거리는 실제 대기보다 클 수 있습니다.</p>}
          <p className="label">나머지 변수는 기준 대기로 고정했습니다. 잡음 표본 {s.reps}개의 중앙값입니다.</p>
        </div>
      </Card>
      <div className="grid g2">
        <Card title="IP 오차 (중앙값)">
          <Plot series={err} xLabel={isCond ? "날씨" : `${s.label}${s.unit ? ` (${s.unit})` : ""}`} yLabel="IP 오차 (m)" height={280} logY fmtX={fmtX} ariaLabel="변수 값별 IP 오차" />
          <Legend items={[{ label: "표준 대기 EKF", color: "--truth", dash: true }, { label: "대기를 아는 EKF", color: "--ekf" }]} />
        </Card>
        <Card title="탄착점 이동 (기준 대기 대비)">
          <Plot series={shift} xLabel={isCond ? "날씨" : `${s.label}${s.unit ? ` (${s.unit})` : ""}`} yLabel="이동 (m)" height={280} fmtX={fmtX} ariaLabel="변수 값별 탄착점 이동" />
          <Legend items={[{ label: "Δx 하향거리", color: "--pred" }, { label: "Δy 횡방향", color: "--obs", dash: true }]} />
        </Card>
      </div>
      <Card title="값별 결과">
        <div className="tw"><table>
          <thead><tr><th>{s.label}</th><th className="n">지상 밀도</th><th className="n">레이더 잡음</th><th className="n">탄착 이동</th><th className="n">비행시간</th><th className="n">표준 EKF</th><th className="n">대기 반영 EKF</th><th className="n">CEP 포함률</th></tr></thead>
          <tbody>{items.map(i => (
            <tr key={String(i.value)}><td>{lbl(i.value)}</td><td className="n">{fmt(i.rho_sfc, 3)} kg/m³</td><td className="n">×{fmt(i.noise_factor, 2)}</td>
              <td className="n">{fmt(i.shift_m)} m</td><td className="n">{fmt(i.impact_time, 1)} s</td><td className="n">{fmt(i.miss_std)} m</td>
              <td className="n">{fmt(i.miss_wx)} m</td><td className="n">{fmt(i.coverage_wx, 2)}</td></tr>))}
          </tbody>
        </table></div>
      </Card>
    </>
  );
}
