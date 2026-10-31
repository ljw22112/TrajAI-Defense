import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { METHOD_LABEL, modelShort, type Estimate, type Method, type Scenario } from "../api/types";
import { Legend, Plot, type Series } from "../components/Plot";
import { PrintButton } from "../components/PrintButton";
import { Card, Empty, ErrorBox, PageHead, Spinner, Term, ThreatBadge, fmt } from "../components/ui";

const SIDE: [string, string] = ["--ekf", "--pred"];
const ORDER: Method[] = ["ekf_std", "ekf_wx", "ukf_wx", "ekf_pinn", "pinn"];

function latestByMethod(s: Scenario): Map<Method, Estimate> {
  const m = new Map<Method, Estimate>();
  s.estimates.forEach(e => m.set(e.method, e));
  return m;
}

/** Two scenarios side by side: parameters, atmosphere, and the latest estimate of each method. */
export function ComparePage() {
  const [params, setParams] = useSearchParams();
  const a = Number(params.get("a")) || 0, b = Number(params.get("b")) || 0;
  const list = useQuery({ queryKey: ["scenarios", { limit: 100 }], queryFn: () => api.scenarios({ limit: 100 }) });
  const qa = useQuery({ queryKey: ["scenario", a], queryFn: () => api.scenario(a), enabled: a > 0 });
  const qb = useQuery({ queryKey: ["scenario", b], queryFn: () => api.scenario(b), enabled: b > 0 });
  const pick = (k: "a" | "b", v: string) => { const p = new URLSearchParams(params); p.set(k, v); setParams(p, { replace: true }); };
  const both = qa.data && qb.data ? [qa.data, qb.data] as const : null;

  const side = useMemo<Series[]>(() => !both ? [] : both.flatMap((s, i) => [
    { kind: "line" as const, x: s.truth.x.map(v => v / 1000), y: s.truth.z.map(v => v / 1000), color: SIDE[i], width: 2.2 },
    { kind: "points" as const, x: s.measurements.map(m => m.x / 1000), y: s.measurements.map(m => m.z / 1000), color: SIDE[i], size: 1.8 },
  ]), [both]);

  const methods = both ? ORDER.filter(m => both.some(s => latestByMethod(s).has(m))) : [];
  const atm = (s: Scenario) => !s.weather ? "표준 대기" : s.weather.source === "custom" ? "직접 설정" : `${new Date(s.weather.valid_time).toISOString().slice(0, 13).replace("T", " ")}Z 기상`;
  const rows: [string, (s: Scenario) => string][] = [
    ["대기", atm],
    ["연소 종료 고도 / 속도", s => `${fmt(s.h0 / 1000, 1)} km / ${fmt(s.v0)} m/s`],
    ["경로각 / 방위각", s => `${fmt(s.gamma_deg)}° / ${fmt(s.psi_deg)}°`],
    ["탄도계수 β", s => `${fmt(s.beta)} kg/m²`],
    ["관측 비율 / 측정 수", s => `${fmt(s.obs_fraction * 100)}% / ${s.measurements.length}`],
    ["레이더 잡음", s => `${fmt(s.sigma_r, 1)} m, ${fmt(s.sigma_ang * 1000, 2)} mrad`],
    ["실제 탄착 (x, y)", s => `${fmt(s.impact_x / 1000, 2)}, ${fmt(s.impact_y / 1000, 2)} km`],
    ["비행시간 / 최고 고도", s => `${fmt(s.impact_time, 1)} s / ${fmt(s.apogee / 1000, 1)} km`],
  ];

  return (
    <div className="page">
      <PageHead kicker="Compare" title="시나리오 비교" actions={<PrintButton />}>
        두 시나리오의 조건과 추정 결과를 나란히 봅니다. 같은 비행을 다른 대기나 센서 조건에서 만든 두 시나리오를 고르면 환경의 영향이 그대로 드러납니다.
      </PageHead>
      <Card title="선택">
        <div className="form">
          {(["a", "b"] as const).map((k, i) => (
            <div className="field" key={k}>
              <label htmlFor={`cmp-${k}`}><span className="chip" style={{ borderColor: `var(${SIDE[i]})`, color: `var(${SIDE[i]})` }}>{k.toUpperCase()}</span> 시나리오</label>
              <select id={`cmp-${k}`} value={k === "a" ? a || "" : b || ""} onChange={e => pick(k, e.target.value)}>
                <option value="">선택…</option>
                {list.data?.items.map(s => <option key={s.id} value={s.id}>#{s.id} {s.name}</option>)}
              </select>
            </div>
          ))}
        </div>
      </Card>
      <ErrorBox error={qa.error || qb.error} />
      {!both ? ((qa.isLoading || qb.isLoading) ? <Spinner /> : <Empty>비교할 두 시나리오를 고르세요.</Empty>) : (
        <>
          <Card title="측면 궤적 (실선 = 실제 궤적, 점 = 레이더 측정)">
            <Plot series={side} xLabel="하향거리 x (km)" yLabel="고도 h (km)" height={300} ariaLabel="두 시나리오의 실제 궤적" />
            <Legend items={both.map((s, i) => ({ label: `${"AB"[i]} · ${s.name}`, color: SIDE[i] }))} />
          </Card>
          <Card title="조건">
            <div className="tw"><table>
              <thead><tr><th>항목</th>{both.map((s, i) => <th key={s.id}><Link to={`/scenarios/${s.id}`}>{"AB"[i]} · {s.name}</Link></th>)}</tr></thead>
              <tbody>{rows.map(([l, f]) => <tr key={l}><td>{l}</td>{both.map(s => <td key={s.id} className="mono">{f(s)}</td>)}</tr>)}</tbody>
            </table></div>
          </Card>
          <Card title="추정 결과 (추정기별 최신)">
            {methods.length === 0 ? <Empty>두 시나리오 모두 추정이 없습니다.</Empty> : (
              <div className="tw"><table>
                <thead><tr><th>추정기</th>{both.map((s, i) => <th key={s.id} className="n">{"AB"[i]} · <Term k="IP 오차" /></th>)}<th className="n">차이 (B − A)</th>{both.map((s, i) => <th key={`t${s.id}`}>{"AB"[i]} · 위협</th>)}</tr></thead>
                <tbody>{methods.map(m => {
                  const ea = latestByMethod(both[0]).get(m), eb = latestByMethod(both[1]).get(m);
                  return (
                    <tr key={m}>
                      <td className="nowrap">{METHOD_LABEL[m]}{[ea, eb].map(e => modelShort(e?.model_name)).find(Boolean) && <span className="chip model">{[ea, eb].map(e => modelShort(e?.model_name)).find(Boolean)}</span>}</td>
                      <td className="n">{ea ? `${fmt(ea.miss_m)} m` : "—"}</td><td className="n">{eb ? `${fmt(eb.miss_m)} m` : "—"}</td>
                      <td className={`n ${ea && eb ? (eb.miss_m < ea.miss_m ? "ok" : "bad") : ""}`}>{ea && eb ? `${eb.miss_m - ea.miss_m >= 0 ? "+" : ""}${fmt(eb.miss_m - ea.miss_m)} m` : "—"}</td>
                      <td>{ea?.threat ? <ThreatBadge level={ea.threat.level} /> : "—"}</td><td>{eb?.threat ? <ThreatBadge level={eb.threat.level} /> : "—"}</td>
                    </tr>);
                })}</tbody>
              </table></div>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
