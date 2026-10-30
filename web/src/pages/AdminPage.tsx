import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { api, ApiError } from "../api/client";
import type { Role } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { Card, ErrorBox, PageHead, Spinner } from "../components/ui";

export function AdminPage() {
  const qc = useQueryClient();
  const { user: me } = useAuth();
  const users = useQuery({ queryKey: ["users"], queryFn: api.users });
  const audit = useQuery({ queryKey: ["audit"], queryFn: api.audit });
  const [nu, setNu] = useState({ email: "", full_name: "", password: "", role: "analyst" as Role });
  const create = useMutation({ mutationFn: api.createUser, onSuccess: () => { qc.invalidateQueries({ queryKey: ["users"] }); setNu({ email: "", full_name: "", password: "", role: "analyst" }); } });
  const upd = useMutation({ mutationFn: ({ id, b }: { id: number; b: { role?: Role; is_active?: boolean } }) => api.updateUser(id, b), onSuccess: () => qc.invalidateQueries({ queryKey: ["users"] }) });
  const err = (k: string) => (create.error instanceof ApiError ? create.error.fields[k] : undefined);

  function submit(e: FormEvent) { e.preventDefault(); create.mutate(nu); }

  return (
    <div className="page">
      <PageHead kicker="Administration" title="사용자 · 감사 로그">역할: viewer(조회) · analyst(시나리오·추정·자산) · admin(사용자·모델·삭제).</PageHead>
      <ErrorBox error={users.error || upd.error} />
      <div className="grid g2">
        <Card title="사용자">
          {users.isLoading ? <Spinner /> : (
            <div className="tw">
              <table>
                <thead><tr><th>이메일</th><th>이름</th><th>역할</th><th>상태</th></tr></thead>
                <tbody>
                  {users.data?.items.map(u => (
                    <tr key={u.id}>
                      <td>{u.email}</td><td>{u.full_name}</td>
                      <td>
                        <select aria-label={`${u.email} 역할`} value={u.role} disabled={u.id === me?.id} onChange={e => upd.mutate({ id: u.id, b: { role: e.target.value as Role } })} style={{ width: 110 }}>
                          <option value="viewer">viewer</option><option value="analyst">analyst</option><option value="admin">admin</option>
                        </select>
                      </td>
                      <td><label className="chk"><input type="checkbox" checked={u.is_active} disabled={u.id === me?.id} onChange={e => upd.mutate({ id: u.id, b: { is_active: e.target.checked } })} />{u.is_active ? "활성" : "중지"}</label></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
        <Card title="사용자 추가">
          <form className="form" onSubmit={submit} noValidate>
            <div className="field"><label htmlFor="ue">이메일</label><input id="ue" type="email" value={nu.email} onChange={e => setNu({ ...nu, email: e.target.value })} />{err("email") && <span className="err">{err("email")}</span>}</div>
            <div className="field"><label htmlFor="un">이름</label><input id="un" value={nu.full_name} onChange={e => setNu({ ...nu, full_name: e.target.value })} /></div>
            <div className="field"><label htmlFor="up">초기 비밀번호</label><input id="up" type="password" value={nu.password} onChange={e => setNu({ ...nu, password: e.target.value })} />{err("password") ? <span className="err">{err("password")}</span> : <small>8자 이상</small>}</div>
            <div className="field"><label htmlFor="ur">역할</label><select id="ur" value={nu.role} onChange={e => setNu({ ...nu, role: e.target.value as Role })}><option value="viewer">viewer</option><option value="analyst">analyst</option><option value="admin">admin</option></select></div>
            <div style={{ gridColumn: "1 / -1" }} className="toolbar"><button className="btn" disabled={create.isPending}>추가</button></div>
            {create.error && !(create.error instanceof ApiError && Object.keys(create.error.fields).length) && <div style={{ gridColumn: "1 / -1" }}><ErrorBox error={create.error} /></div>}
          </form>
        </Card>
      </div>
      <Card title="감사 로그 (최근 50건)">
        <ErrorBox error={audit.error} />
        <div className="tw">
          <table>
            <thead><tr><th>시각</th><th className="n">사용자</th><th>동작</th><th>대상</th><th>상세</th></tr></thead>
            <tbody>
              {audit.data?.map(r => (
                <tr key={r.id}>
                  <td className="mono">{new Date(r.at).toLocaleString("ko-KR")}</td><td className="n">{r.user_id ?? "—"}</td>
                  <td>{r.action}</td><td className="mono">{r.entity}{r.entity_id ? ` #${r.entity_id}` : ""}</td>
                  <td className="mono" style={{ fontSize: 12, color: "var(--ink-2)" }}>{JSON.stringify(r.detail)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
