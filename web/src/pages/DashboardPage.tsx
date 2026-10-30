import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { METHOD_LABEL, modelShort, type MethodStat } from "../api/types";
import { useState } from "react";
import { useAuth } from "../auth/AuthContext";
import { Card, Empty, ErrorBox, PageHead, Spinner, Stat, Term, fmt, km } from "../components/ui";

const label = (m: MethodStat) => METHOD_LABEL[m.method] + (modelShort(m.model) ? ` · ${modelShort(m.model)}` : "");
const GUIDE_KEY = "trajai.guide.hidden";
const readHidden = () => { try { return localStorage.getItem(GUIDE_KEY) === "1"; } catch { return false; } };

export function DashboardPage() {
  const nav = useNavigate();
  const { can } = useAuth();
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const recent = useQuery({ queryKey: ["scenarios", { limit: 8 }], queryFn: () => api.scenarios({ limit: 8 }) });
  const s = stats.data;
  const best = s?.by_method.filter(m => m.n > 0 && m.median_miss_m !== null).sort((a, b) => a.median_miss_m! - b.median_miss_m!)[0];
  const [hideGuide, setHideGuide] = useState(readHidden);
  const closeGuide = () => { setHideGuide(true); try { localStorage.setItem(GUIDE_KEY, "1"); } catch { /* storage blocked */ } };
  const maxMiss = Math.max(1, ...(s?.by_method.map(m => m.median_miss_m ?? 0) ?? [1]));

  return (
    <div className="page">
      <PageHead kicker="Overview" title="대시보드" actions={can("analyst") && <Link className="btn" to="/scenarios/new">새 시나리오</Link>}>
        시뮬레이션한 교전 시나리오와 추정기별 성능, 위협 등급 분포를 한눈에 봅니다.
      </PageHead>
      {!hideGuide && (
        <Card title="처음이라면 이렇게 둘러보세요" aside={<button className="btn ghost sm" onClick={closeGuide} aria-label="안내 닫기">닫기</button>}>
          <ol className="guide" style={{ margin: 0, padding: 0 }}>
            <li><b>시나리오 열기</b><p>아래 표에서 "10/15 00Z 제트기류"처럼 기상이 반영된 시나리오를 고릅니다. 실제 궤적과 레이더 측정이 그려져 있습니다.</p></li>
            <li><b>추정기 비교</b><p>측면 궤적·탄착 영역에서 EKF, UKF, PINN의 예측을 겹쳐 봅니다. 표의 용어에 마우스를 올리면 뜻이 나옵니다. 원이 CEP(50% 확률 반경)입니다.</p></li>
            <li><b>실시간 재생</b><p>"재생"을 누르면 측정이 시간 순서대로 들어오며 필터가 탄도계수 β를 좁혀 가는 과정을 볼 수 있습니다. 기상 반영을 끄고 다시 재생해 비교해 보세요.</p></li>
          </ol>
        </Card>
      )}
      <ErrorBox error={stats.error || recent.error} />
      {stats.isLoading ? <Spinner /> : s && (
        <>
          <div className="grid g4">
            <Stat label="시나리오" value={fmt(s.scenarios)} />
            <Stat label="추정 실행" value={fmt(s.estimates)} />
            <Stat label="최선 추정기 (IP 오차 중앙값)" value={best ? fmt(best.median_miss_m) : "—"} unit={best ? `m · ${label(best)}` : ""} tone="acc" />
            <Stat label="캐시된 기상 프로파일" value={fmt(s.weather_profiles)} />
          </div>
          <div className="grid g2">
            <Card title="추정기별 성능 (전체 추정 실행 기준)">
              <div className="tw">
                <table>
                  <thead><tr><th>추정기</th><th className="n">실행</th><th><Term k="IP 오차">IP 오차 중앙값</Term></th><th className="n"><Term k="CEP 포함률" /></th><th className="n"><Term k="지연">지연 중앙값</Term></th></tr></thead>
                  <tbody>
                    {s.by_method.map(m => (
                      <tr key={`${m.method}:${m.model ?? ""}`}>
                        <td>{METHOD_LABEL[m.method]}{modelShort(m.model) && <span className="chip model">{modelShort(m.model)}</span>}</td>
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
