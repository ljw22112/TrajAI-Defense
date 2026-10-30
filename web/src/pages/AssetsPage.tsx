import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState, type FormEvent } from "react";
import { api, ApiError } from "../api/client";
import type { Asset, AssetIn } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { Plot, type Series } from "../components/Plot";
import { Card, Empty, ErrorBox, PageHead, Spinner, fmt } from "../components/ui";

const BLANK: AssetIn = { name: "", x_km: 100, y_km: 0, radius_km: 3, priority: 3 };

export function AssetsPage() {
  const qc = useQueryClient();
  const { can } = useAuth();
  const q = useQuery({ queryKey: ["assets"], queryFn: api.assets });
  const [edit, setEdit] = useState<Asset | null>(null);
  const [form, setForm] = useState<AssetIn>(BLANK);
  const save = useMutation({
    mutationFn: (b: AssetIn) => (edit ? api.updateAsset(edit.id, b) : api.createAsset(b)),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["assets"] }); setEdit(null); setForm(BLANK); },
  });
  const del = useMutation({ mutationFn: (id: number) => api.deleteAsset(id), onSuccess: () => qc.invalidateQueries({ queryKey: ["assets"] }) });
  const err = (k: string) => (save.error instanceof ApiError ? save.error.fields[k] : undefined);

  const series = useMemo<Series[]>(() => (q.data ?? []).flatMap(a => [
    { kind: "circle" as const, x: [a.x_km], y: [a.y_km], r: [a.radius_km], color: a.priority >= 4 ? "--pred" : "--ekf", width: 2 },
    { kind: "points" as const, x: [a.x_km], y: [a.y_km], color: "--ink", size: 2 },
  ]), [q.data]);

  function submit(e: FormEvent) { e.preventDefault(); save.mutate(form); }
  const num = (k: keyof AssetIn) => (e: React.ChangeEvent<HTMLInputElement>) => setForm(f => ({ ...f, [k]: e.target.value === "" ? "" : Number(e.target.value) }));

  return (
    <div className="page">
      <PageHead kicker="Assets" title="보호 자산">
        위협 평가는 예측 탄착 분포가 각 자산의 반경 안에 들어갈 확률로 등급(1–5)을 매깁니다. 좌표는 국소 ENU(km)이며 모두 가상 자산입니다.
      </PageHead>
      <ErrorBox error={q.error || del.error} />
      <div className="grid g2">
        <Card title="배치도">
          {q.isLoading ? <Spinner /> : <Plot series={series} xLabel="x (km, 동쪽)" yLabel="y (km, 북쪽)" height={300} equal ariaLabel="보호 자산 배치도" />}
          <small className="label">주황 = 우선순위 4–5</small>
        </Card>
        {can("analyst") ? (
          <Card title={edit ? `자산 수정 · ${edit.name}` : "자산 추가"}>
            <form className="form" onSubmit={submit} noValidate>
              <div className="field" style={{ gridColumn: "1 / -1" }}><label htmlFor="an">이름</label>
                <input id="an" value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} aria-invalid={!!err("name")} />
                {err("name") && <span className="err">{err("name")}</span>}</div>
              {(["x_km", "y_km", "radius_km", "priority"] as const).map(k => (
                <div className="field" key={k}><label htmlFor={k}>{{ x_km: "x (km)", y_km: "y (km)", radius_km: "반경 (km)", priority: "우선순위 (1–5)" }[k]}</label>
                  <input id={k} type="number" step={k === "priority" ? 1 : 0.5} value={form[k]} onChange={num(k)} aria-invalid={!!err(k)} />
                  {err(k) && <span className="err">{err(k)}</span>}</div>
              ))}
              <div className="toolbar" style={{ gridColumn: "1 / -1" }}>
                <button className="btn" disabled={save.isPending}>{edit ? "저장" : "추가"}</button>
                {edit && <button type="button" className="btn ghost" onClick={() => { setEdit(null); setForm(BLANK); }}>취소</button>}
              </div>
              {save.error && !(save.error instanceof ApiError && Object.keys(save.error.fields).length) && <div style={{ gridColumn: "1 / -1" }}><ErrorBox error={save.error} /></div>}
            </form>
          </Card>
        ) : <Card title="권한"><Empty>자산 추가·수정은 analyst 이상 권한이 필요합니다.</Empty></Card>}
      </div>
      <Card title={`자산 목록 (${q.data?.length ?? 0})`}>
        <div className="tw">
          <table>
            <thead><tr><th>이름</th><th className="n">x</th><th className="n">y</th><th className="n">반경</th><th className="n">우선순위</th><th>수정</th><th /></tr></thead>
            <tbody>
              {q.data?.map(a => (
                <tr key={a.id}>
                  <td>{a.name}</td><td className="n">{fmt(a.x_km, 1)} km</td><td className="n">{fmt(a.y_km, 1)} km</td>
                  <td className="n">{fmt(a.radius_km, 1)} km</td><td className="n">{a.priority}</td>
                  <td className="mono">{new Date(a.updated_at).toLocaleDateString("ko-KR")}</td>
                  <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                    {can("analyst") && <button className="btn ghost sm" onClick={() => { setEdit(a); setForm({ name: a.name, x_km: a.x_km, y_km: a.y_km, radius_km: a.radius_km, priority: a.priority }); }}>수정</button>}{" "}
                    {can("admin") && <button className="btn danger sm" disabled={del.isPending} onClick={() => del.mutate(a.id)}>삭제</button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
