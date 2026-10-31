import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { alertsUrl, api, DEMO } from "../api/client";
import type { Alert } from "../api/types";
import { METHOD_LABEL } from "../api/types";
import { fmt } from "./ui";

const SEEN_KEY = "trajai.alerts.seen";
const readSeen = () => { try { return Number(localStorage.getItem(SEEN_KEY) ?? 0); } catch { return 0; } };

/** Threat alerts: recent level-4/5 estimates on load, live pushes over WebSocket (server mode). */
export function AlertCenter() {
  const qc = useQueryClient();
  const recent = useQuery({ queryKey: ["alerts"], queryFn: api.alerts, refetchInterval: DEMO ? false : 60_000 });
  const [live, setLive] = useState<Alert[]>([]);
  const [seen, setSeen] = useState(readSeen);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (DEMO) return;
    let ws: WebSocket | null = null, retry: number | undefined, closed = false;
    const connect = () => {
      ws = new WebSocket(alertsUrl());
      ws.onmessage = ev => {
        const m = JSON.parse(ev.data);
        if (m.type === "alert") {
          setLive(p => [m as Alert, ...p].slice(0, 20));
          qc.invalidateQueries({ queryKey: ["threats"] });
        }
      };
      ws.onclose = () => { if (!closed) retry = window.setTimeout(connect, 5000); };
    };
    connect();
    return () => { closed = true; clearTimeout(retry); ws?.close(); };
  }, [qc]);

  const all = [...live, ...(recent.data ?? [])].filter((a, i, arr) => arr.findIndex(b => b.estimate_id === a.estimate_id) === i);
  const unseen = all.filter(a => a.estimate_id > seen);
  const banner = live.find(a => a.estimate_id > seen);
  const markSeen = () => {
    const top = Math.max(seen, ...all.map(a => a.estimate_id));
    setSeen(top); try { localStorage.setItem(SEEN_KEY, String(top)); } catch { /* storage blocked */ }
  };

  return (
    <>
      {banner && (
        <div className={`alert-banner lv${banner.level}`} role="alert">
          <b>위협 LEVEL {banner.level}</b> · {banner.scenario} → {banner.top_asset ?? "보호 자산"} {fmt(banner.max_prob * 100, 0)}% · 탄착까지 {fmt(banner.time_to_impact)} s
          <Link to={`/scenarios/${banner.scenario_id}`} onClick={markSeen}>열기</Link>
          <button className="btn ghost sm" onClick={markSeen} aria-label="알림 닫기">닫기</button>
        </div>
      )}
      <div className="bell">
        <button className="btn ghost sm" onClick={() => setOpen(o => !o)} aria-expanded={open} aria-label={`위협 알림 ${unseen.length}건`}>
          위협 알림{unseen.length > 0 && <span className="badge">{unseen.length}</span>}
        </button>
        {open && (
          <div className="bell-pop" role="dialog" aria-label="위협 알림 목록">
            {all.length === 0 ? <p className="label">LEVEL 4 이상 위협이 없습니다.</p> : all.slice(0, 10).map(a => (
              <Link key={a.estimate_id} to={`/scenarios/${a.scenario_id}`} className={a.estimate_id > seen ? "new" : ""} onClick={() => setOpen(false)}>
                <span className={`lv lv${a.level}`}>L{a.level}</span> {a.scenario}
                <small>{a.top_asset ?? "—"} {fmt(a.max_prob * 100, 0)}% · {METHOD_LABEL[a.method]}</small>
              </Link>
            ))}
            {all.length > 0 && <button className="btn ghost sm" onClick={() => { markSeen(); setOpen(false); }}>모두 읽음</button>}
          </div>
        )}
      </div>
    </>
  );
}
