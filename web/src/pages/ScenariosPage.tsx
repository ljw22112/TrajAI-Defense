import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, DEMO } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { Card, Empty, ErrorBox, PageHead, Spinner, fmt, km } from "../components/ui";

const PAGE = 15;

export function ScenariosPage() {
  const nav = useNavigate();
  const { can } = useAuth();
  const [q, setQ] = useState("");
  const [wx, setWx] = useState<"all" | "wx" | "std">("all");
  const [offset, setOffset] = useState(0);
  const params = { limit: PAGE, offset, q: q || undefined, weather: wx === "all" ? undefined : wx === "wx" };
  const { data, error, isLoading, isFetching } = useQuery({ queryKey: ["scenarios", params], queryFn: () => api.scenarios(params), placeholderData: keepPreviousData });
  const items = DEMO && data ? data.items.filter(r => (!q || r.name.includes(q)) && (wx === "all" || (wx === "wx") === !!r.weather_id)) : data?.items;

  return (
    <div className="page">
      <PageHead kicker="Scenarios" title="시나리오" actions={can("analyst") && <Link className="btn" to="/scenarios/new">새 시나리오</Link>}>
        각 시나리오는 연소 종료 상태, 탄도계수, 레이더 잡음, 대기(표준 또는 선택 일자 기상)로 정의된 모의 교전입니다.
      </PageHead>
      <Card>
        <div className="toolbar">
          <input aria-label="이름 검색" placeholder="이름 검색" value={q} onChange={e => { setQ(e.target.value); setOffset(0); }} />
          <select aria-label="대기 필터" value={wx} onChange={e => { setWx(e.target.value as typeof wx); setOffset(0); }} style={{ maxWidth: 180 }}>
            <option value="all">전체 대기</option><option value="wx">일자 기상</option><option value="std">표준 대기</option>
          </select>
          {isFetching && <span className="label">갱신 중</span>}
        </div>
        <ErrorBox error={error} />
        {isLoading ? <Spinner /> : !items?.length ? <Empty>조건에 맞는 시나리오가 없습니다.</Empty> : (
          <div className="tw">
            <table>
              <thead><tr><th>#</th><th>이름</th><th>생성</th><th>대기</th><th className="n">β</th><th className="n">관측 종료</th><th className="n">ToT</th><th className="n">사거리</th><th className="n">최고 고도</th><th className="n">추정</th><th className="n">최선 IP 오차</th></tr></thead>
              <tbody>
                {items.map(r => (
                  <tr key={r.id} className="clickable" onClick={() => nav(`/scenarios/${r.id}`)}>
                    <td className="mono">{r.id}</td><td>{r.name}</td>
                    <td className="mono">{new Date(r.created_at).toLocaleString("ko-KR", { dateStyle: "short", timeStyle: "short" })}</td>
                    <td>{r.weather_id ? <span className="chip on">일자 기상</span> : <span className="chip">표준</span>}</td>
                    <td className="n">{fmt(r.beta)}</td><td className="n">T+{fmt(r.t_c, 1)} s</td><td className="n">{fmt(r.impact_time, 1)} s</td>
                    <td className="n">{km(r.impact_x)} km</td><td className="n">{km(r.apogee)} km</td><td className="n">{r.n_estimates}</td>
                    <td className="n">{r.best_miss_m === null ? "—" : `${fmt(r.best_miss_m)} m`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {data && !DEMO && (
          <div className="toolbar" style={{ justifyContent: "space-between" }}>
            <span className="label">{data.total === 0 ? 0 : offset + 1}–{Math.min(offset + PAGE, data.total)} / {data.total}</span>
            <div className="toolbar">
              <button className="btn ghost sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>이전</button>
              <button className="btn ghost sm" disabled={offset + PAGE >= data.total} onClick={() => setOffset(offset + PAGE)}>다음</button>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
