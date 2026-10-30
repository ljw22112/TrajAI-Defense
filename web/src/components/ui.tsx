import type { ReactNode } from "react";
import { ApiError } from "../api/client";
import { TERMS } from "../api/types";

export const fmt = (v: number | null | undefined, d = 0) =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
export const km = (m: number, d = 1) => fmt(m / 1000, d);

export function PageHead({ kicker, title, actions, children }: { kicker: string; title: string; actions?: ReactNode; children?: ReactNode }) {
  return (
    <header className="page-head">
      <div className="ph-main">
        <span className="label">{kicker}</span>
        <h1>{title}</h1>
        {children && <p className="ph-sub">{children}</p>}
      </div>
      {actions && <div className="ph-actions">{actions}</div>}
    </header>
  );
}

export function Card({ title, aside, children, className = "" }: { title?: string; aside?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card ${className}`}>
      {(title || aside) && <div className="card-h"><span className="label">{title}</span>{aside}</div>}
      {children}
    </section>
  );
}

export function Term({ k, children }: { k: string; children?: ReactNode }) {
  const def = TERMS[k];
  return def ? <abbr className="term" title={def} tabIndex={0}>{children ?? k}</abbr> : <>{children ?? k}</>;
}

export function Stat({ label, value, unit, tone }: { label: string; value: string; unit?: string; tone?: "acc" | "ok" }) {
  return (
    <div className="stat">
      <span className="label">{label}</span>
      <span className={`stat-v ${tone ?? ""}`}>{value}{unit && <small> {unit}</small>}</span>
    </div>
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  if (!error) return null;
  const msg = error instanceof ApiError ? error.detail : error instanceof Error ? error.message : String(error);
  return <div className="error" role="alert">{msg}</div>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function ThreatBadge({ level }: { level: number }) {
  return <span className={`lv lv${level}`}>LEVEL {level}</span>;
}

export function Spinner({ label = "불러오는 중" }: { label?: string }) {
  return <div className="spinner" role="status"><span className="dot" />{label}…</div>;
}
