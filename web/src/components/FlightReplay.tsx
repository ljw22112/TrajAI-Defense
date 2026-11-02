import { useEffect, useMemo, useRef, useState } from "react";
import type { Flight, FlightEvent } from "../api/types";
import { fmt } from "./ui";

/** Animated side view of the true flight with the atmosphere it passes through, and a timeline of the
 *  moments where the environment changes (layer boundaries, jet stream, Mach regime, max-q, radar end, ...). */

const KIND_COLOR: Record<FlightEvent["kind"], string> = {
  start: "var(--ink-2)", layer: "var(--ekf)", wind: "var(--ukf)", mach: "var(--pred)", apogee: "var(--ink)",
  maxq: "var(--danger)", radar: "var(--obs)", weather: "var(--ekf)", impact: "var(--danger)",
};
const KIND_LABEL: Record<FlightEvent["kind"], string> = {
  start: "시작", layer: "대기층", wind: "바람", mach: "속도 구간", apogee: "정점", maxq: "공기력", radar: "센서", weather: "날씨", impact: "탄착",
};
const SPEEDS = [5, 10, 20, 40];
const W = 1000, H = 400, PL = 54, PR = 16, PT = 14, PB = 34;

function idxAt(t: number[], v: number) {
  let lo = 0, hi = t.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (t[m] <= v) lo = m; else hi = m; }
  return lo;
}
function lerp(f: Flight, key: keyof Flight, i: number, a: number) {
  const arr = f[key] as number[];
  return arr[i] + (arr[Math.min(i + 1, arr.length - 1)] - arr[i]) * a;
}

/** An estimator's forecast made at the end of radar tracking (t, downrange x, altitude z). */
export interface Forecast { label: string; color: string; t: number[]; x: number[]; z: number[] }

function at(ts: number[], vs: number[], v: number) {
  if (v <= ts[0]) return vs[0];
  if (v >= ts[ts.length - 1]) return vs[vs.length - 1];
  const k = idxAt(ts, v), a = (v - ts[k]) / (ts[k + 1] - ts[k] || 1);
  return vs[k] + (vs[k + 1] - vs[k]) * a;
}

export function FlightReplay({ f, forecasts = [] }: { f: Flight; forecasts?: Forecast[] }) {
  const T = f.t[f.t.length - 1];
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(10);
  const raf = useRef<number | null>(null);
  const last = useRef<number | null>(null);

  useEffect(() => {
    if (!playing) { last.current = null; return; }
    const step = (now: number) => {
      const dt = last.current === null ? 0 : (now - last.current) / 1000;
      last.current = now;
      setT(prev => {
        const nt = prev + dt * speed;
        if (nt >= T) { setPlaying(false); return T; }
        return nt;
      });
      raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
  }, [playing, speed, T]);

  // geometry: downrange along the launch azimuth, altitude up
  const along = useMemo(() => {
    const dx = f.x[f.x.length - 1], dy = f.y[f.y.length - 1], n = Math.hypot(dx, dy) || 1;
    return f.x.map((x, i) => (x * dx + f.y[i] * dy) / n);
  }, [f]);
  const dx0 = f.x[f.x.length - 1], dy0 = f.y[f.y.length - 1], cosA = dx0 / (Math.hypot(dx0, dy0) || 1) || 1;
  const fc = forecasts.filter(e => e.t.length > 1).map(e => ({ ...e, a: e.x.map(x => x / cosA) }));
  const xmax = Math.max(...along, ...fc.map(e => Math.max(...e.a))) * 1.04, zmax = Math.max(f.apogee * 1.12, (f.layers.tropopause ?? 11000) * 1.3);
  const sx = (v: number) => PL + (v / xmax) * (W - PL - PR);
  const sz = (v: number) => H - PB - (v / zmax) * (H - PT - PB);

  const i = idxAt(f.t, t), a = f.t[i + 1] !== undefined ? (t - f.t[i]) / (f.t[i + 1] - f.t[i] || 1) : 0;
  const cur = {
    x: along[i] + ((along[i + 1] ?? along[i]) - along[i]) * a,
    z: lerp(f, "z", i, a), speed: lerp(f, "speed", i, a), mach: lerp(f, "mach", i, a), temp: lerp(f, "temp_c", i, a),
    rho: lerp(f, "rho_ratio", i, a), wind: lerp(f, "wind", i, a), dir: f.wind_dir[i], q: lerp(f, "q_kpa", i, a), drag: lerp(f, "drag_g", i, a),
  };
  const i0 = Math.max(0, Math.min(i, f.t.length - 2)), j = i0 + 1;
  const ang = Math.atan2(-(sz(f.z[j]) - sz(f.z[i0])), sx(along[j]) - sx(along[i0]) || 1e-6) * 180 / Math.PI;
  const L = f.layers;
  const layer = cur.z >= (L.thin_air ?? Infinity) ? "희박 대기 (항력 거의 없음)" : cur.z >= L.tropopause ? "성층권" : "대류권";
  const inJet = L.jet && cur.z >= L.jet.bottom && cur.z <= L.jet.top;
  const past = f.events.filter(e => e.t <= t + 1e-6);
  const recent = past.length ? past[past.length - 1] : null;
  const next = f.events.find(e => e.t > t + 1e-6);
  const path = (k0: number, k1: number) => along.slice(k0, k1 + 1).map((x, k) => `${k ? "L" : "M"}${sx(x).toFixed(1)},${sz(f.z[k0 + k]).toFixed(1)}`).join("");
  const kc = idxAt(f.t, f.t_c);
  // wind barbs along the left: component along the flight direction (tail + / head -)
  const dx = f.x[f.x.length - 1], dy = f.y[f.y.length - 1], nn = Math.hypot(dx, dy) || 1;
  const barbs = f.column.z.map((z, k) => ({ z, w: (f.column.u[k] * dx + f.column.v[k] * dy) / nn, s: Math.hypot(f.column.u[k], f.column.v[k]) }))
    .filter((b, k) => k % 2 === 0 && b.z <= zmax && b.z > 0);
  const showFc = t >= f.t_c && fc.length > 0;
  const fcNow = fc.map(e => {
    const ax = at(e.t, e.a, t), az = Math.max(0, at(e.t, e.z, t));
    return { ...e, ax, az, err: Math.hypot(ax - cur.x, az - cur.z) };
  });
  const D = f.drift;
  const ticksZ = Array.from({ length: Math.floor(zmax / 10000) + 1 }, (_, k) => k * 10000);
  const tickStep = xmax > 300000 ? 100000 : xmax > 120000 ? 50000 : 20000;
  const ticksX = Array.from({ length: Math.floor(xmax / tickStep) + 1 }, (_, k) => k * tickStep);

  return (
    <div className="fr">
      <div className="fr-main">
        <div className="fr-pan"><svg viewBox={`0 0 ${W} ${H}`} className="fr-svg" role="img" aria-label={`비행 재생: T+${fmt(t, 0)} s, 고도 ${fmt(cur.z / 1000, 1)} km`}>
          {/* atmosphere bands */}
          <rect x={PL} y={sz(zmax)} width={W - PL - PR} height={sz(0) - sz(zmax)} className="fr-sky-thin" />
          {L.thin_air !== null && <rect x={PL} y={sz(Math.min(L.thin_air, zmax))} width={W - PL - PR} height={sz(L.tropopause) - sz(Math.min(L.thin_air, zmax))} className="fr-sky-strat" />}
          <rect x={PL} y={sz(L.tropopause)} width={W - PL - PR} height={sz(0) - sz(L.tropopause)} className="fr-sky-trop" />
          {L.jet && <rect x={PL} y={sz(L.jet.top)} width={W - PL - PR} height={sz(L.jet.bottom) - sz(L.jet.top)} className="fr-jet" />}
          {f.condition && !["clear", "cloudy"].includes(f.condition) && <rect x={PL} y={sz(3000)} width={W - PL - PR} height={sz(0) - sz(3000)} className="fr-precip" />}
          <text x={W - PR - 6} y={sz(L.tropopause) - 5} className="fr-lbl" textAnchor="end">대류권계면 {fmt(L.tropopause / 1000, 1)} km</text>
          <line x1={PL} x2={W - PR} y1={sz(L.tropopause)} y2={sz(L.tropopause)} className="fr-bound" />
          {L.thin_air !== null && L.thin_air < zmax && <>
            <line x1={PL} x2={W - PR} y1={sz(L.thin_air)} y2={sz(L.thin_air)} className="fr-bound" />
            <text x={W - PR - 6} y={sz(L.thin_air) - 5} className="fr-lbl" textAnchor="end">희박 대기 (밀도 &lt; 1%) {fmt(L.thin_air / 1000, 0)} km</text></>}
          {L.jet && <text x={W - PR - 6} y={sz(L.jet.bottom) - 5} className="fr-lbl jet" textAnchor="end">제트기류 {fmt(L.jet.bottom / 1000, 0)}–{fmt(L.jet.top / 1000, 0)} km · 최대 {fmt(L.jet.peak)} m/s</text>}
          {/* axes */}
          {ticksZ.map(z => <g key={z}><line x1={PL - 4} x2={PL} y1={sz(z)} y2={sz(z)} className="fr-axis" /><text x={PL - 8} y={sz(z) + 4} textAnchor="end" className="fr-tick">{z / 1000}</text></g>)}
          {ticksX.map(x => <g key={x}><line x1={sx(x)} x2={sx(x)} y1={sz(0)} y2={sz(0) + 4} className="fr-axis" /><text x={sx(x)} y={sz(0) + 17} textAnchor="middle" className="fr-tick">{x / 1000}</text></g>)}
          <text x={14} y={PT + 6} className="fr-tick" transform={`rotate(-90 14 ${PT + 6})`} textAnchor="end">고도 (km)</text>
          <text x={W - PR} y={H - 4} className="fr-tick" textAnchor="end">하향거리 (km)</text>
          <line x1={PL} x2={W - PR} y1={sz(0)} y2={sz(0)} className="fr-ground" />
          {/* wind along the flight direction */}
          {barbs.map(b => {
            const len = Math.max(-34, Math.min(34, b.w * 0.8));
            return <g key={b.z} className="fr-barb" opacity={Math.min(1, 0.3 + b.s / 40)}>
              <line x1={PL + 40} x2={PL + 40 + len} y1={sz(b.z)} y2={sz(b.z)} />
              {Math.abs(len) > 3 && <path d={`M${PL + 40 + len},${sz(b.z)} l${len > 0 ? -5 : 5},-3.5 v7 z`} />}
            </g>;
          })}
          <text x={PL + 6} y={sz(0) - 6} className="fr-lbl">→ 순풍 / ← 맞바람</text>
          {/* trajectory: full (faint), radar-observed (teal), flown (solid) */}
          <path d={path(0, along.length - 1)} className="fr-path-all" />
          <path d={path(0, Math.min(kc, i))} className="fr-path-obs" />
          {i > kc && <path d={path(kc, i)} className="fr-path-flown" />}
          {f.events.filter(e => e.kind !== "start").map((e, k) => {
            const ii = idxAt(f.t, e.t);
            return <circle key={k} cx={sx(along[ii])} cy={sz(f.z[ii])} r={e.t <= t ? 4.5 : 3.5} style={{ fill: e.t <= t ? KIND_COLOR[e.kind] : "var(--sheet)", stroke: KIND_COLOR[e.kind] }} className="fr-ev">
              <title>{`T+${fmt(e.t, 1)} s · ${e.title}`}</title></circle>;
          })}
          {/* estimator forecasts made at the end of radar tracking */}
          {showFc && fcNow.map(e => (
            <g key={e.label} className="fr-fc" style={{ color: `var(${e.color})` }}>
              <path d={e.a.map((x, k) => `${k ? "L" : "M"}${sx(x).toFixed(1)},${sz(Math.max(0, e.z[k])).toFixed(1)}`).join("")} />
              <circle cx={sx(e.ax)} cy={sz(e.az)} r={4} />
            </g>
          ))}
          {/* vehicle */}
          <g transform={`translate(${sx(cur.x)},${sz(cur.z)}) rotate(${-ang})`} className="fr-veh">
            <path d="M12,0 L-8,-5 L-5,0 L-8,5 Z" />
          </g>
          {t >= T && <g className="fr-impact"><line x1={sx(along[along.length - 1]) - 6} x2={sx(along[along.length - 1]) + 6} y1={sz(0) - 6} y2={sz(0) + 6} /><line x1={sx(along[along.length - 1]) - 6} x2={sx(along[along.length - 1]) + 6} y1={sz(0) + 6} y2={sz(0) - 6} /></g>}
        </svg></div>
        {D && <DriftView f={f} along={along} t={t} i={i} a={a} />}
        <div className="fr-ctl">
          <button type="button" className="btn sm" onClick={() => { if (t >= T) setT(0); setPlaying(p => !p); }}>{playing ? "일시정지" : t >= T ? "다시 재생" : "재생"}</button>
          <div className="fr-track">
            <input type="range" min={0} max={T} step={0.5} value={t} aria-label="비행 시각" onChange={e => { setPlaying(false); setT(Number(e.target.value)); }} />
            <div className="fr-ticks" aria-hidden>
              {f.events.map((e, k) => <span key={k} style={{ left: `${(e.t / T) * 100}%`, background: KIND_COLOR[e.kind] }} title={e.title} />)}
              <span className="obs" style={{ left: 0, width: `${(f.t_c / T) * 100}%` }} />
            </div>
          </div>
          <select value={speed} onChange={e => setSpeed(Number(e.target.value))} aria-label="재생 속도">
            {SPEEDS.map(s => <option key={s} value={s}>×{s}</option>)}
          </select>
          <span className="mono fr-clock">T+{fmt(t, 1)} s</span>
        </div>
      </div>
      <aside className="fr-side">
        <div className="fr-now">
          <span className="label">지금 · {f.vehicle_label}</span>
          <dl className="kv">
            <dt>고도</dt><dd>{fmt(cur.z / 1000, 1)} km</dd>
            <dt>속도</dt><dd>{fmt(cur.speed)} m/s · 마하 {fmt(cur.mach, 2)}</dd>
            <dt>대기층</dt><dd>{layer}{inJet ? " · 제트기류" : ""}</dd>
            <dt>기온</dt><dd>{fmt(cur.temp, 1)} °C</dd>
            <dt>공기 밀도</dt><dd>지상의 {cur.rho >= 0.01 ? fmt(cur.rho * 100, 1) : fmt(cur.rho * 100, 3)}%</dd>
            <dt>바람</dt><dd>{f.atmosphere === "standard" ? "무풍 (표준 대기)" : `${fmt(cur.wind)} m/s · ${fmt(cur.dir)}°에서`}</dd>
            <dt>동압</dt><dd>{fmt(cur.q, 1)} kPa</dd>
            <dt>항력 감속</dt><dd>{fmt(cur.drag, 2)} g</dd>
            {D && <><dt>바람 누적 표류</dt><dd>{fmt(D.offset_m[i])} m</dd></>}
          </dl>
          {fc.length > 0 && (
            <div className="fr-fcnow">
              <span className="label">예측과 실제의 차이 (지금)</span>
              {showFc ? fcNow.map(e => (
                <div key={e.label}><i style={{ background: `var(${e.color})` }} />{e.label}<b className="mono">{fmt(e.err)} m</b></div>
              )) : <small>레이더 관측이 끝나는 T+{fmt(f.t_c, 1)} s부터 추정기 예측 경로가 겹쳐집니다.</small>}
            </div>
          )}
        </div>
        {recent && (
          <div className="fr-callout" style={{ borderColor: KIND_COLOR[recent.kind] }}>
            <span className="label">T+{fmt(recent.t, 1)} s · {KIND_LABEL[recent.kind]}</span>
            <b>{recent.title}</b>
            <p>{recent.detail}</p>
            {next && <small>다음: T+{fmt(next.t, 0)} s {next.title}</small>}
          </div>
        )}
      </aside>
      <ol className="fr-events">
        {f.events.map((e, k) => (
          <li key={k} className={e.t <= t ? "done" : ""}>
            <button type="button" onClick={() => { setPlaying(false); setT(e.t); }}>
              <span className="dot" style={{ background: KIND_COLOR[e.kind] }} />
              <span className="mono t">T+{fmt(e.t, 1)} s</span>
              <span className="mono h">{fmt(e.h / 1000, 1)} km</span>
              <span className="ti">{e.title}</span>
              <span className="de">{e.detail}</span>
            </button>
          </li>
        ))}
      </ol>
    </div>
  );
}

/** Top-down view: the flight with and without wind (same temperature and density), crossrange exaggerated. */
function DriftView({ f, along, t, i, a }: { f: Flight; along: number[]; t: number; i: number; a: number }) {
  const D = f.drift!;
  const h = 130, pl = PL, pr = PR, pt = 18, pb = 22;
  const xmax = Math.max(...along, D.impact_calm[0]) * 1.04;
  const ymax = Math.max(200, ...f.y.map(Math.abs), ...D.y.map(Math.abs)) * 1.3;
  const X = (v: number) => pl + (v / xmax) * (W - pl - pr);
  const Y = (v: number) => pt + (h - pt - pb) / 2 - (v / ymax) * ((h - pt - pb) / 2);
  const line = (xs: number[], ys: number[], k1 = xs.length - 1) => xs.slice(0, k1 + 1).map((x, k) => `${k ? "L" : "M"}${X(x).toFixed(1)},${Y(ys[k]).toFixed(1)}`).join("");
  const cx = f.x[i] + ((f.x[i + 1] ?? f.x[i]) - f.x[i]) * a, cy = f.y[i] + ((f.y[i + 1] ?? f.y[i]) - f.y[i]) * a;
  const kx = D.x[i] + ((D.x[i + 1] ?? D.x[i]) - D.x[i]) * a, ky = D.y[i] + ((D.y[i + 1] ?? D.y[i]) - D.y[i]) * a;
  const T = f.t[f.t.length - 1];
  return (
    <svg viewBox={`0 0 ${W} ${h}`} className="fr-svg fr-top" role="img"
      aria-label={`위에서 본 궤적: 바람 때문에 탄착점이 ${fmt(D.shift_m)} m 이동`}>
      <line x1={pl} x2={W - pr} y1={Y(0)} y2={Y(0)} className="fr-bound" />
      <text x={pl} y={12} className="fr-lbl">위에서 본 궤적 (횡거리 과장, ±{fmt(ymax)} m)</text>
      <text x={W - pr} y={12} className="fr-lbl" textAnchor="end">바람에 의한 탄착 이동 {fmt(D.shift_m)} m (하향 {D.shift_down_m >= 0 ? "+" : ""}{fmt(D.shift_down_m)} · 횡 {D.shift_cross_m >= 0 ? "+" : ""}{fmt(D.shift_cross_m)})</text>
      <path d={line(D.x, D.y)} className="fr-calm" />
      <path d={line(f.x, f.y)} className="fr-top-all" />
      <path d={line(f.x, f.y, i)} className="fr-path-flown" />
      <circle cx={X(kx)} cy={Y(ky)} r={3.5} className="fr-calm-dot" />
      <circle cx={X(cx)} cy={Y(cy)} r={4.5} className="fr-now-dot" />
      {t >= T && <>
        <text x={X(D.impact_calm[0])} y={Y(D.impact_calm[1]) + 4} textAnchor="middle" className="fr-lbl">○</text>
        <text x={X(f.x[f.x.length - 1])} y={Y(f.y[f.y.length - 1]) + 4} textAnchor="middle" className="fr-imp-x">×</text></>}
      <text x={pl} y={h - 6} className="fr-tick">주황 = 실제(바람 있음) · 파란 점선 = 무풍이었다면 · 점 = 같은 시각의 위치</text>
    </svg>
  );
}
