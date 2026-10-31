import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Fragment, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api/client";
import { CONDITION_LABEL, METHOD_LABEL, modelShort, type Condition, type Estimate, type Method } from "../api/types";
import { PrintButton } from "../components/PrintButton";
import { useAuth } from "../auth/AuthContext";
import { Legend, Plot, type Series } from "../components/Plot";
import { ProfileChart } from "../components/ProfileChart";
import { Card, Empty, ErrorBox, PageHead, Spinner, Stat, Term, ThreatBadge, fmt, km } from "../components/ui";
import { useLiveTrack } from "../hooks/useLiveTrack";

const METHOD_COLOR: Record<Method, string> = { ekf_std: "--truth", ekf_wx: "--ekf", ukf_wx: "--ukf", pinn: "--pred", ekf_pinn: "--obs" };

export function ScenarioDetailPage() {
  const id = Number(useParams().id);
  const nav = useNavigate();
  const qc = useQueryClient();
  const { can, user } = useAuth();
  const q = useQuery({ queryKey: ["scenario", id], queryFn: () => api.scenario(id), enabled: Number.isFinite(id) });
  const assets = useQuery({ queryKey: ["assets"], queryFn: api.assets });
  const run = useMutation({
    mutationFn: (m: Method) => api.estimate(id, m),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["scenario", id] }); qc.invalidateQueries({ queryKey: ["stats"] }); },
  });
  const del = useMutation({ mutationFn: () => api.deleteScenario(id), onSuccess: () => { qc.invalidateQueries({ queryKey: ["scenarios"] }); nav("/scenarios"); } });
  const live = useLiveTrack(id);
  const [speed, setSpeed] = useState(8);
  const [liveWx, setLiveWx] = useState(true);
  const s = q.data;

  // latest estimate per method
  const latest = useMemo(() => {
    const m = new Map<Method, Estimate>();
    s?.estimates.forEach(e => m.set(e.method, e));
    return [...m.values()];
  }, [s]);
  const latestIds = useMemo(() => new Set(latest.map(e => e.id)), [latest]);

  const sideSeries = useMemo<Series[]>(() => {
    if (!s) return [];
    const out: Series[] = [
      { kind: "line", x: s.truth.x.map(v => v / 1000), y: s.truth.z.map(v => v / 1000), color: "--truth", width: 2, dash: [3, 5] },
      { kind: "points", x: s.measurements.map(m => m.x / 1000), y: s.measurements.map(m => m.z / 1000), color: "--obs", size: 2.2 },
    ];
    latest.forEach(e => out.push({ kind: "line", x: e.trajectory.x.map(v => v / 1000), y: e.trajectory.z.map(v => v / 1000), color: METHOD_COLOR[e.method], width: 2.4, dash: [10, 6] }));
    if (live.meas.length) out.push({ kind: "points", x: live.meas.map(m => m.x / 1000), y: live.meas.map(m => m.z / 1000), color: "--pred", size: 3 });
    return out;
  }, [s, latest, live.meas]);

  const footSeries = useMemo<Series[]>(() => {
    if (!s) return [];
    const out: Series[] = [];
    (assets.data ?? []).forEach(a => out.push({ kind: "circle", x: [a.x_km], y: [a.y_km], r: [a.radius_km], color: "--ink-3", width: 1.2 }));
    latest.forEach(e => {
      out.push({ kind: "circle", x: [e.ip_x / 1000], y: [e.ip_y / 1000], r: [e.cep / 1000], color: METHOD_COLOR[e.method], width: 2 });
      out.push({ kind: "points", x: [e.ip_x / 1000], y: [e.ip_y / 1000], color: METHOD_COLOR[e.method], size: 3.5 });
    });
    out.push({ kind: "cross", x: [s.impact_x / 1000], y: [s.impact_y / 1000], color: "--ink", width: 2, size: 8 });
    return out;
  }, [s, latest, assets.data]);

  const footDomain = useMemo<{ x: [number, number]; y: [number, number] } | undefined>(() => {
    if (!s) return undefined;
    const pts = latest.flatMap(e => [[e.ip_x, e.ip_y, e.cep]]).concat([[s.impact_x, s.impact_y, 1500]]);
    const ext = Math.max(4000, ...pts.map(([x, y, r]) => Math.max(Math.abs(x - s.impact_x), Math.abs(y - s.impact_y)) + r)) * 1.25 / 1000;
    return { x: [s.impact_x / 1000 - ext, s.impact_x / 1000 + ext], y: [s.impact_y / 1000 - ext, s.impact_y / 1000 + ext] };
  }, [s, latest]);

  const liveSeries = useMemo<Series[]>(() => {
    if (!live.states.length) return [];
    const t = live.states.map(x => x.t), b = live.states.map(x => x.beta), sl = live.states.map(x => x.sigma_lnb);
    return [
      { kind: "band", x: t, y: b.map((v, i) => v * Math.exp(-2 * sl[i])), y2: b.map((v, i) => v * Math.exp(2 * sl[i])), color: "--pred-a" },
      { kind: "line", x: t, y: b, color: "--pred", width: 2.2 },
      ...(s ? [{ kind: "line" as const, x: [t[0], t[t.length - 1]], y: [s.beta, s.beta], color: "--ink", width: 1.4, dash: [6, 4] }] : []),
    ];
  }, [live.states, s]);

  const lastPred = live.preds[live.preds.length - 1];
  const predSeries = useMemo<Series[]>(() => !live.preds.length ? [] : [
    { kind: "line", x: live.preds.map(p => p.t), y: live.preds.map(p => Math.max(p.miss, 1)), color: "--pred", width: 2.2 },
    { kind: "points", x: live.preds.map(p => p.t), y: live.preds.map(p => Math.max(p.miss, 1)), color: "--pred", size: 2.5 },
    { kind: "line", x: live.preds.map(p => p.t), y: live.preds.map(p => p.cep), color: "--ekf", width: 1.6, dash: [6, 4] },
  ], [live.preds]);

  if (q.isLoading) return <Spinner />;
  if (q.error || !s) return <ErrorBox error={q.error ?? new Error("시나리오를 찾을 수 없습니다.")} />;
  const methods: Method[] = s.weather ? ["ekf_std", "ekf_wx", "ukf_wx", "ekf_pinn", "pinn"] : ["ekf_std", "ekf_pinn", "pinn"];
  const lastState = live.states[live.states.length - 1];

  return (
    <div className="page">
      <PageHead kicker={`Scenario #${s.id}`} title={s.name} actions={
        <>
          {can("analyst") && methods.map(m => (
            <button key={m} className="btn ghost" disabled={run.isPending} onClick={() => run.mutate(m)}>{METHOD_LABEL[m]} 실행</button>
          ))}
          <Link className="btn ghost no-print" to={`/compare?a=${s.id}`}>비교</Link>
          <PrintButton />
          {can("analyst") && (user?.role === "admin" || s.created_by === user?.id) && (
            <button className="btn danger" disabled={del.isPending} onClick={() => del.mutate()}>삭제</button>
          )}
        </>
      }>
        {s.weather ? (s.weather.source === "custom"
          ? <>대기: <span className="chip on">직접 설정</span> 지상 {fmt((s.weather.levels as any).custom?.t_sfc_c)}°C · 습도 {fmt((s.weather.levels as any).custom?.rh_pct)}% · {CONDITION_LABEL[(s.weather.levels as any).custom?.condition as Condition] ?? ""} · 최대풍 {fmt(s.weather.max_wind)} m/s</>
          : <>대기: <span className="chip on">{new Date(s.weather.valid_time).toISOString().slice(0, 13).replace("T", " ")}Z 기상</span> 최대풍 {fmt(s.weather.max_wind)} m/s</>) : <>대기: 표준 모델 (무풍)</>}
        {" · "}seed {s.seed}
      </PageHead>
      <ErrorBox error={run.error || del.error} />
      {run.isPending && <Spinner label="추정 실행 중" />}
      <div className="grid g4">
        <Stat label="관측 종료" value={`T+${fmt(s.t_c, 1)}`} unit={`s · 측정 ${s.measurements.length}개`} />
        <Stat label="실제 ToT" value={fmt(s.impact_time, 1)} unit="s" />
        <Stat label="실제 탄착점" value={`${km(s.impact_x)}, ${km(s.impact_y)}`} unit="km" />
        <Stat label="최선 IP 오차" value={s.best_miss_m === null ? "—" : fmt(s.best_miss_m)} unit="m" tone="acc" />
      </div>
      <div className="grid g2">
        <Card title="측면 궤적 (고도–하향거리)" className="span2">
          <Plot series={sideSeries} xLabel="하향거리 x (km)" yLabel="고도 h (km)" height={340} ariaLabel="측면 궤적: 실제, 측정, 추정" />
          <Legend items={[{ label: "실제 궤적", color: "--truth", dash: true }, { label: "레이더 측정", color: "--obs" },
            ...latest.map(e => ({ label: METHOD_LABEL[e.method], color: METHOD_COLOR[e.method], dash: true })),
            ...(live.meas.length ? [{ label: "실시간 수신", color: "--pred" }] : [])]} />
        </Card>
        <Card title="탄착 영역 (지면 투영)">
          {latest.length === 0 ? <Empty>추정을 실행하면 예측 탄착점과 CEP 원이 표시됩니다.</Empty> : (
            <Plot series={footSeries} xLabel="x (km, 동쪽)" yLabel="y (km, 북쪽)" height={320} equal xDomain={footDomain?.x} yDomain={footDomain?.y} ariaLabel="실제 탄착점과 추정 CEP" />
          )}
          <small className="label">+ 실제 탄착점 · 원 = CEP50 · 회색 원 = 보호 자산</small>
        </Card>
        <Card title="추정 결과">
          {s.estimates.length === 0 ? <Empty>아직 추정이 없습니다.</Empty> : (
            <div className="tw">
              <table>
                <thead><tr><th>추정기</th><th className="n"><Term k="IP 오차" /></th><th className="n"><Term k="CEP" /></th><th><Term k="포함" /></th><th className="n"><Term k="ToT" /></th><th className="n"><Term k="β̂" /></th><th className="n"><Term k="지연" /></th><th>위협</th></tr></thead>
                <tbody>
                  {[...s.estimates].reverse().map(e => (
                    <tr key={e.id} className={latestIds.has(e.id) ? "" : "muted-row"} title={latestIds.has(e.id) ? undefined : "같은 추정기의 이전 실행"}>
                      <td className="nowrap">{METHOD_LABEL[e.method]}{modelShort(e.model_name) && <span className="chip model">{modelShort(e.model_name)}</span>}
                        {e.trajectory.guard?.fallback && <span className="chip acc model"><Term k="분포 밖">분포 밖 → EKF</Term></span>}</td>
                      <td className="n">{fmt(e.miss_m)} m</td><td className="n">{fmt(e.cep)} m</td>
                      <td className={`nowrap ${e.inside_cep ? "ok" : "bad"}`}>{e.inside_cep ? "✓ 포함" : "✗ 밖"}</td>
                      <td className="n">{fmt(e.tot, 1)} s</td><td className="n">{fmt(e.beta_hat)}</td>
                      <td className="n">{fmt(e.latency_ms, 1)} ms</td>
                      <td>{e.threat && <ThreatBadge level={e.threat.level} />}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
      {latest.some(e => e.threat) && (
        <Card title="위협 평가 (최근 추정)">
          <div className="grid g3">
            {latest.filter(e => e.threat).map(e => {
              const t = e.threat!;
              const top = Object.entries(t.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 3);
              return (
                <div key={e.id} className="card" style={{ padding: 12 }}>
                  <div className="card-h"><span>{METHOD_LABEL[e.method]}{modelShort(e.model_name) && <span className="chip model">{modelShort(e.model_name)}</span>}</span><ThreatBadge level={t.level} /></div>
                  <dl className="kv">
                    {top.map(([n, p]) => <Fragment key={n}><dt>{n}</dt><dd>{fmt(p * 100, 1)}%</dd></Fragment>)}
                    {Object.entries(t.windows).map(([b, w]) => <Fragment key={b}><dt>{b === "upper" ? "상층" : "하층"} 교전 창</dt>
                      <dd>{w ? `T+${fmt(w.t_in)}–${fmt(w.t_out)} s · PIP ${fmt(w.h_pip / 1000, 1)} km` : "—"}</dd></Fragment>)}
                  </dl>
                </div>
              );
            })}
          </div>
        </Card>
      )}
      <Card title="실시간 추적 재생 (WebSocket · 스트리밍 EKF)" aside={
        <div className="toolbar">
          <label className="chk">속도 <select value={speed} onChange={e => setSpeed(Number(e.target.value))} style={{ width: 80 }}>{[1, 4, 8, 20].map(v => <option key={v} value={v}>×{v}</option>)}</select></label>
          {s.weather && <label className="chk"><input type="checkbox" checked={liveWx} onChange={e => setLiveWx(e.target.checked)} /> 기상 반영</label>}
          {live.status === "running" ? <button className="btn ghost sm" onClick={live.stop}>정지</button> : <button className="btn sm" onClick={() => live.start(speed, liveWx && !!s.weather)}>재생</button>}
        </div>
      }>
        {live.error && <div className="error">{live.error}</div>}
        <div className="live-bar" aria-label="재생 진행률"><span style={{ width: `${(live.meas.length / s.measurements.length) * 100}%` }} /></div>
        {live.states.length > 1 ? (
          <div className="grid g2">
            <div>
              <Plot series={liveSeries} xLabel="시간 t (s)" yLabel="β (kg/m²)" height={240} logY ariaLabel="탄도계수 추정 수렴" fmtY={v => (v >= 1000 ? `${v / 1000}k` : String(v))} />
              <Legend items={[{ label: "β̂ ± 2σ", color: "--pred" }, { label: "실제 β", color: "--ink", dash: true }]} />
            </div>
            <dl className="kv" style={{ alignContent: "start" }}>
              <dt>상태</dt><dd>{live.status === "running" ? "추적 중" : live.status === "done" ? "완료" : live.status}</dd>
              <dt>측정 수신</dt><dd>{live.meas.length} / {s.measurements.length}</dd>
              <dt>β̂</dt><dd>{fmt(lastState?.beta)} kg/m²</dd>
              <dt>위치 불확실성 (1σ)</dt><dd>{fmt(lastState?.sigma_pos)} m</dd>
              <dt>NIS (마지막)</dt><dd>{fmt(lastState?.nis, 2)}</dd>
              <dt>추정 위치 고도</dt><dd>{lastState ? `${fmt(lastState.r[2] / 1000, 2)} km` : "—"}</dd>
            </dl>
          </div>
        ) : <Empty>재생을 누르면 측정이 시간 순서대로 도착하고 서버의 EKF가 단계마다 갱신됩니다.</Empty>}
        {live.preds.length > 0 && (
          <div className="grid g2" style={{ marginTop: 12 }}>
            <div>
              <Plot series={predSeries} xLabel="관측 종료 시각 t (s)" yLabel="IP 오차 (m)" height={220} logY ariaLabel="트랙 길이에 따른 PINN 예측 IP 오차" />
              <Legend items={[{ label: "PINN 예측 IP 오차", color: "--pred" }, { label: "CEP50", color: "--ekf", dash: true }]} />
            </div>
            <dl className="kv" style={{ alignContent: "start" }}>
              <dt>PINN 예측 (마지막)</dt><dd>{fmt(lastPred!.ip_x / 1000, 2)}, {fmt(lastPred!.ip_y / 1000, 2)} km</dd>
              <dt>IP 오차 · CEP</dt><dd>{fmt(lastPred!.miss)} m · {fmt(lastPred!.cep)} m</dd>
              <dt>예측 갱신</dt><dd>{live.preds.length}회 · 회당 {fmt(lastPred!.latency_ms, 1)} ms</dd>
              <dt>모델</dt><dd>{modelShort(lastPred!.model) ?? lastPred!.model}</dd>
              <dt className="label" style={{ gridColumn: "1 / -1" }}>측정 {"4"}개마다 그때까지의 트랙으로 PINN을 다시 실행합니다. 오차는 사후 검증용(실제 탄착점 기준)입니다.</dt>
            </dl>
          </div>
        )}
      </Card>
      {s.weather && (
        <Card title="이 시나리오의 대기 프로파일">
          <ProfileChart levels={s.weather.levels} height={220} />
        </Card>
      )}
      <Card title="시나리오 파라미터">
        <dl className="kv" style={{ maxWidth: 520 }}>
          <dt>연소 종료 고도 / 속도</dt><dd>{km(s.h0)} km / {fmt(s.v0)} m/s</dd>
          <dt>경로각 / 방위각</dt><dd>{fmt(s.gamma_deg)}° / {fmt(s.psi_deg)}°</dd>
          <dt>탄도계수 β</dt><dd>{fmt(s.beta)} kg/m²</dd>
          <dt>관측 비율</dt><dd>{fmt(s.obs_fraction * 100)}%</dd>
          <dt>레이더 잡음</dt><dd>{fmt(s.sigma_r)} m, {fmt(s.sigma_ang * 1000, 2)} mrad</dd>
          <dt>최고 고도</dt><dd>{km(s.apogee)} km</dd>
        </dl>
      </Card>
    </div>
  );
}
