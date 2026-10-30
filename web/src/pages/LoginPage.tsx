import { useState, type FormEvent } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { ErrorBox } from "../components/ui";

export function LoginPage() {
  const { login } = useAuth();
  const nav = useNavigate();
  const loc = useLocation() as { state?: { from?: string } };
  const [email, setEmail] = useState("admin@trajai.dev");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true); setError(null);
    try { await login(email, password); nav(loc.state?.from ?? "/", { replace: true }); }
    catch (err) { setError(err); }
    finally { setBusy(false); }
  }

  return (
    <div className="login">
      <form onSubmit={submit} aria-label="로그인">
        <span className="label">TrajAI-Defense</span>
        <h1>운용 콘솔 로그인</h1>
        <div className="field"><label htmlFor="email">이메일</label><input id="email" type="email" autoComplete="username" value={email} onChange={e => setEmail(e.target.value)} required /></div>
        <div className="field"><label htmlFor="pw">비밀번호</label><input id="pw" type="password" autoComplete="current-password" value={password} onChange={e => setPassword(e.target.value)} required /></div>
        <ErrorBox error={error} />
        <button className="btn" disabled={busy}>{busy ? "확인 중…" : "로그인"}</button>
        <p className="hint">로컬 초기 계정: admin@trajai.dev / admin1234 (첫 로그인 후 변경 권장)</p>
      </form>
    </div>
  );
}
