import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { Card, Empty, ErrorBox, PageHead, Spinner, fmt } from "../components/ui";

const SPLITS: [string, string][] = [["test_id_f10", "ID 10%"], ["test_id_f15", "ID 15%"], ["test_id_noise", "고잡음"], ["test_ood", "OOD"], ["test_wx", "기상"]];

export function ModelsPage() {
  const qc = useQueryClient();
  const { can } = useAuth();
  const q = useQuery({ queryKey: ["models"], queryFn: api.models });
  const act = useMutation({ mutationFn: api.activateModel, onSuccess: () => qc.invalidateQueries({ queryKey: ["models"] }) });

  return (
    <div className="page">
      <PageHead kicker="Model registry" title="모델 레지스트리">
        ONNX로 배포된 PINN 모델과 오프라인 평가 지표입니다. 활성 모델이 "PINN 하이브리드" 추정에 쓰입니다. 지표는 각 시험 분할 1,000개 궤적의 IP 오차 중앙값(m)과 CEP 포함률입니다.
      </PageHead>
      <ErrorBox error={q.error || act.error} />
      {q.isLoading ? <Spinner /> : !q.data?.length ? <Empty>등록된 모델이 없습니다. scripts/export_onnx.py로 모델을 내보내세요.</Empty> : q.data.map(m => (
        <Card key={m.id} title={m.name} aside={
          <div className="toolbar">
            {m.is_active ? <span className="chip on">ACTIVE</span> : can("admin") && <button className="btn ghost sm" onClick={() => act.mutate(m.id)}>활성화</button>}
            {m.uses_weather && <span className="chip acc">기상 입력</span>}
          </div>
        }>
          <p className="ph-sub" style={{ margin: 0 }}>{m.kind} · 등록 {new Date(m.created_at).toLocaleDateString("ko-KR")}</p>
          <div className="tw">
            <table>
              <thead><tr><th>지표</th>{SPLITS.map(([, l]) => <th key={l} className="n">{l}</th>)}</tr></thead>
              <tbody>
                <tr><td>IP 오차 중앙값</td>{SPLITS.map(([k]) => <td key={k} className="n">{m.metrics[k] ? `${fmt(m.metrics[k].miss.median)} m` : "—"}</td>)}</tr>
                <tr><td>ADE 중앙값</td>{SPLITS.map(([k]) => <td key={k} className="n">{m.metrics[k] ? `${fmt(m.metrics[k].ade.median)} m` : "—"}</td>)}</tr>
                <tr><td>CEP 포함률</td>{SPLITS.map(([k]) => <td key={k} className="n">{m.metrics[k] ? fmt(m.metrics[k].coverage, 2) : "—"}</td>)}</tr>
                <tr><td>물리 잔차 RMS</td>{SPLITS.map(([k]) => <td key={k} className="n">{m.metrics[k]?.phys_rms_g ? `${fmt(m.metrics[k].phys_rms_g.median, 3)} g` : "—"}</td>)}</tr>
              </tbody>
            </table>
          </div>
          {m.metrics.latency && <small className="label">추론 지연 (PyTorch, 1 thread) p50 {fmt(m.metrics.latency.p50_ms, 2)} ms · p99 {fmt(m.metrics.latency.p99_ms, 2)} ms</small>}
        </Card>
      ))}
    </div>
  );
}
