import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, ApiError, DEMO } from "../api/client";
import { METHOD_LABEL, type Raid } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { PrintButton } from "../components/PrintButton";
import { Card, Empty, ErrorBox, PageHead, Spinner, Stat, Term, ThreatBadge, fmt } from "../components/ui";

export function RaidsPage() {
  const nav = useNavigate();
  const qc = useQueryClient();
  const { can } = useAuth();
  const list = useQuery({ queryKey: ["raids"], queryFn: api.raids });
  const scn = useQuery({ queryKey: ["scenarios", { limit: 100 }], queryFn: () => api.scenarios({ limit: 100 }) });
  const [name, setName] = useState("");
  const [sel, setSel] = useState<Record<number, number>>({});       // scenario id -> launch offset [s]
  const create = useMutation({
    mutationFn: api.createRaid,
    onSuccess: r => { qc.invalidateQueries({ queryKey: ["raids"] }); nav(`/raids/${r.id}`); },
  });
  const ids = Object.keys(sel).map(Number);

  function submit(e: FormEvent) {
    e.preventDefault();
    create.mutate({ name: name.trim() || `동시 공격 ${ids.length}발`, members: ids.map(i => ({ scenario_id: i, offset_s: sel[i] })) });
  }

  return (
    <div className="page">
      <PageHead kicker="Multi-target" title="다표적 교전 타임라인">
        여러 시나리오를 한 번의 동시 공격으로 묶어 같은 시계 위에 놓습니다. 각 표적의 관측 구간, 요격 고도 대역을 지나는 교전 창, 예측 탄착 시각을 한 화면에서 보고, 동시에 열리는 교전 창의 최대 수(필요한 요격 채널 수)를 계산합니다.
      </PageHead>
      {can("analyst") && !DEMO && (
        <Card title="새 동시 공격">
          <form onSubmit={submit} className="page" style={{ gap: 10 }}>
            <div className="field" style={{ maxWidth: 420 }}><label htmlFor="raid-name">이름</label>
              <input id="raid-name" value={name} maxLength={120} placeholder="예: 3발 동시 공격" onChange={e => setName(e.target.value)} /></div>
            {scn.isLoading ? <Spinner /> : (
              <div className="tw">
                <table>
                  <thead><tr><th>포함</th><th>시나리오</th><th className="n">사거리</th><th className="n">발사 지연 (s)</th></tr></thead>
                  <tbody>
                    {scn.data?.items.map(s => (
                      <tr key={s.id}>
                        <td><input type="checkbox" aria-label={`${s.name} 포함`} checked={s.id in sel}
                          onChange={e => setSel(p => { const n = { ...p }; if (e.target.checked) n[s.id] = 0; else delete n[s.id]; return n; })} /></td>
                        <td>{s.name}</td>
                        <td className="n">{fmt(s.impact_x / 1000, 1)} km</td>
                        <td className="n"><input type="number" min={0} max={600} step={5} style={{ width: 90 }} disabled={!(s.id in sel)} value={sel[s.id] ?? 0}
                          aria-label={`${s.name} 발사 지연`} onChange={e => setSel(p => ({ ...p, [s.id]: Number(e.target.value) }))} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <ErrorBox error={create.error instanceof ApiError && Object.keys(create.error.fields).length ? new Error("2–12개 시나리오를 고르세요.") : create.error} />
            <div className="toolbar"><button className="btn" disabled={ids.length < 2 || create.isPending}>타임라인 만들기 ({ids.length}발)</button></div>
          </form>
        </Card>
      )}
      <Card title="저장된 동시 공격">
        <ErrorBox error={list.error} />
        {list.isLoading ? <Spinner /> : !list.data?.length ? <Empty>아직 없습니다.</Empty> : (
          <div className="tw"><table>
            <thead><tr><th>이름</th><th className="n">표적 수</th><th>만든 시각</th></tr></thead>
            <tbody>{list.data.map(r => (
              <tr key={r.id} className="clickable" onClick={() => nav(`/raids/${r.id}`)}>
                <td><Link to={`/raids/${r.id}`}>{r.name}</Link></td><td className="n">{r.n}</td>
                <td className="mono">{new Date(r.created_at).toLocaleString("ko-KR")}</td>
              </tr>))}
            </tbody>
          </table></div>
        )}
      </Card>
    </div>
  );
}

function Timeline({ raid }: { raid: Raid }) {
  const tmax = Math.max(...raid.tracks.map(t => t.impact)) * 1.04;
  const pct = (t: number) => `${(t / tmax) * 100}%`;
  const ticks = Array.from({ length: Math.floor(tmax / 60) + 1 }, (_, i) => i * 60);
  return (
    <div className="tl" role="img" aria-label="표적별 관측 구간, 교전 창, 탄착 시각 타임라인">
      {raid.tracks.map(t => (
        <div className="tl-row" key={t.scenario_id}>
          <span className="nm" title={t.scenario}>{t.level !== null && <ThreatBadge level={t.level} />} {t.scenario}</span>
          <div className="tl-track">
            <span className="fl" style={{ left: pct(t.offset_s), width: pct(t.impact - t.offset_s) }} />
            <span className="obs" style={{ left: pct(t.offset_s), width: pct(t.track_end - t.offset_s) }} title={`관측 T+${fmt(t.offset_s)}–${fmt(t.track_end)} s`} />
            {t.windows.map(w => <span key={w.band} className={`win ${w.band}`} style={{ left: pct(w.open), width: pct(w.close - w.open) }}
              title={`${w.band === "upper" ? "상층" : "하층"} 교전 창 T+${fmt(w.open)}–${fmt(w.close)} s`} />)}
            <span className="imp" style={{ left: pct(t.impact) }} title={`예측 탄착 T+${fmt(t.impact)} s`} />
          </div>
        </div>
      ))}
      {raid.peak_at !== null && <div className="tl-peak" style={{ left: `calc(190px + (100% - 190px) * ${raid.peak_at / tmax})` }} aria-hidden />}
      <div className="tl-axis"><span /><div>{ticks.map(s => <span key={s} style={{ left: pct(s) }}>{s}s</span>)}</div></div>
    </div>
  );
}

export function RaidDetailPage() {
  const id = Number(useParams().id);
  const nav = useNavigate();
  const qc = useQueryClient();
  const { can } = useAuth();
  const q = useQuery({ queryKey: ["raid", id], queryFn: () => api.raid(id) });
  const del = useMutation({ mutationFn: () => api.deleteRaid(id), onSuccess: () => { qc.invalidateQueries({ queryKey: ["raids"] }); nav("/raids"); } });
  if (q.isLoading) return <Spinner />;
  if (q.error || !q.data) return <ErrorBox error={q.error ?? new Error("찾을 수 없습니다.")} />;
  const r = q.data;
  const order = [...r.tracks].sort((a, b) => (b.level ?? 0) - (a.level ?? 0) || a.impact - b.impact);
  return (
    <div className="page">
      <PageHead kicker={`Raid #${r.id}`} title={r.name} actions={<div className="toolbar"><PrintButton />
        {can("analyst") && !DEMO && <button className="btn danger" onClick={() => del.mutate()}>삭제</button>}</div>}>
        시계는 동시 공격 시작(T+0) 기준입니다. 각 표적은 가장 정확한 최신 추정(기상 반영 EKF → UKF → PINN β 사전 EKF → PINN → 표준 EKF)으로 그립니다.
      </PageHead>
      <ErrorBox error={del.error} />
      <div className="grid g4">
        <Stat label="표적" value={fmt(r.tracks.length)} />
        <Stat label="동시 교전 최대 (요격 채널)" value={fmt(r.peak_simultaneous)} unit={r.peak_at !== null ? `@ T+${fmt(r.peak_at)} s` : ""} tone="acc" />
        <Stat label="첫 탄착" value={r.first_impact === null ? "—" : `T+${fmt(r.first_impact)}`} unit="s" />
        <Stat label="LEVEL 4 이상" value={fmt(r.tracks.filter(t => (t.level ?? 0) >= 4).length)} />
      </div>
      <Card title="타임라인">
        <Timeline raid={r} />
        <small className="label">회색 = 비행 · 초록 = 레이더 관측 · 파랑 = 상층 교전 창 · 주황 = 하층 교전 창 · 빨강 막대 = 예측 탄착 · 세로선 = 동시 교전 최대 시점</small>
      </Card>
      <Card title="교전 순서 (위협 등급 → 탄착 시각)">
        <div className="tw"><table>
          <thead><tr><th>#</th><th>등급</th><th>표적</th><th>위협 자산</th><th className="n">발사</th><th className="n"><Term k="교전 창">첫 교전 창</Term></th><th className="n">예측 탄착</th><th>근거 추정기</th></tr></thead>
          <tbody>{order.map((t, i) => {
            const w = t.windows.length ? Math.min(...t.windows.map(x => x.open)) : null;
            return (
              <tr key={t.scenario_id} className="clickable" onClick={() => nav(`/scenarios/${t.scenario_id}`)}>
                <td className="n">{i + 1}</td><td>{t.level !== null ? <ThreatBadge level={t.level} /> : "—"}</td><td>{t.scenario}</td>
                <td>{t.top_asset && (t.max_prob ?? 0) > 0.001 ? `${t.top_asset} ${fmt((t.max_prob ?? 0) * 100, 0)}%` : "—"}</td>
                <td className="n">T+{fmt(t.offset_s)} s</td><td className="n">{w === null ? "—" : `T+${fmt(w)} s`}</td>
                <td className="n">T+{fmt(t.impact)} s</td><td className="nowrap">{t.method ? METHOD_LABEL[t.method] : "추정 없음"}</td>
              </tr>);
          })}</tbody>
        </table></div>
      </Card>
    </div>
  );
}
