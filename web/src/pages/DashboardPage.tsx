import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { METHOD_LABEL } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { Card, Empty, ErrorBox, PageHead, Spinner, Stat, fmt, km } from "../components/ui";

export function DashboardPage() {
  const nav = useNavigate();
  const { can } = useAuth();
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const recent = useQuery({ queryKey: ["scenarios", { limit: 8 }], queryFn: () => api.scenarios({ limit: 8 }) });
  const s = stats.data;
  const best = s?.by_method.filter(m => m.n > 0 && m.median_miss_m !== null).sort((a, b) => a.median_miss_m! - b.median_miss_m!)[0];
  const maxMiss = Math.max(1, ...(s?.by_method.map(m => m.median_miss_m ?? 0) ?? [1]));

  return (
    <div className="page">
      <PageHead kicker="Overview" title="대시보드" actions={can("analyst") && <Link className="btn" to="/scenarios/new">새 시나리오</Link>}>
        시뮬레이션한 교전 시나리오와 추정기별 성능, 위협 등급 분포를 한눈에 봅니다.
      </PageHead>
      <ErrorBox error={stats.error || recent.error} />
      {stats.isLoading ? <Spinner /> : s && (
        <>
          <div className="grid g4">
            <Stat label="시나리오" value={fmt(s.scenarios)} />
            <Stat label="추정 실행" value={fmt(s.estimates)} />
            <Stat label="최선 추정기 (IP 오차 중앙값)" value={best ? fmt(best.median_miss_m) : "—"} unit={best ? `m · ${METHOD_LABEL[best.method]}` : ""} tone="acc" />
            <Stat label="캐시된 기상 프로파일" value={fmt(s.weather_profiles)} />
          </div>
          <div className="grid g2">
            <Card title="추정기별 성능 (전체 추정 실행 기준)">
              <div className="tw">
                <table>
                  <thead><tr><th>추정기</th><th className="n">실행</th><th>IP 오차 중앙값</th><th className="n">CEP 포함률</th><th className="n">지연 중앙값</th></tr></thead>
                  <tbody>
                    {s.by_method.map(m => (
                      <tr key={m.method}>
                        <td>{METHOD_LABEL[m.method]}</td>
                        <td className="n">{m.n}</td>
                        <td>
                          {m.median_miss_m === null ? "—" : (
                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                              <div style={{ flex: 1, height: 8, background: "var(--grid)", borderRadius: 2 }}>
                                <div style={{ width: `${(m.median_miss_m / maxMiss) * 100}%`, height: "100%", background: "var(--pred)", borderRadius: 2 }} />
                              </div>
                              <span className="mono">{fmt(m.median_miss_m)} m</span>
                            </div>
                          )}
                        </td>
                        <td className="n">{m.coverage === null ? "—" : fmt(m.coverage, 2)}</td>
                        <td className="n">{m.median_latency_ms === null ? "—" : `${fmt(m.median_latency_ms, 1)} ms`}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <small className="label">CEP 포함률: 실제 탄착점이 예측 CEP50 원 안에 든 비율, 보정된 추정기는 0.50</small>
            </Card>
            <Card title="위협 등급 분포">
              {Object.keys(s.threat_levels).length === 0 ? <Empty>아직 위협 평가가 없습니다.</Empty> : (
                <div style={{ display: "flex", alignItems: "flex-end", gap: 12, height: 170, padding: "8px 4px 0" }}>
                  {[1, 2, 3, 4, 5].map(l => {
                    const n = s.threat_levels[String(l)] ?? 0;
                    const max = Math.max(...Object.values(s.threat_levels));
                    return (
                      <div key={l} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
                        <span className="mono">{n}</span>
                        <div className={`lv lv${l}`} style={{ width: "100%", height: `${(n / max) * 120 + 2}px`, padding: 0 }} />
                        <span className="label">L{l}</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </Card>
          </div>
        </>
      )}
      <Card title="최근 시나리오" aside={<Link to="/scenarios">전체 보기</Link>}>
        {recent.isLoading ? <Spinner /> : recent.data && recent.data.items.length === 0 ? (
          <Empty>시나리오가 없습니다.{can("analyst") && <> <Link to="/scenarios/new">첫 시나리오를 만들어 보세요.</Link></>}</Empty>
        ) : (
          <div className="tw">
            <table>
              <thead><tr><th>#</th><th>이름</th><th>대기</th><th className="n">β (kg/m²)</th><th className="n">관측</th><th className="n">사거리</th><th className="n">추정</th><th className="n">최선 IP 오차</th></tr></thead>
              <tbody>
                {recent.data?.items.map(r => (
                  <tr key={r.id} className="clickable" onClick={() => nav(`/scenarios/${r.id}`)}>
                    <td className="mono">{r.id}</td><td>{r.name}</td>
                    <td>{r.weather_id ? <span className="chip on">일자 기상</span> : <span className="chip">표준</span>}</td>
                    <td className="n">{fmt(r.beta)}</td><td className="n">{fmt(r.obs_fraction * 100)}%</td>
                    <td className="n">{km(r.impact_x)} km</td><td className="n">{r.n_estimates}</td>
                    <td className="n">{r.best_miss_m === null ? "—" : `${fmt(r.best_miss_m)} m`}</td>
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
