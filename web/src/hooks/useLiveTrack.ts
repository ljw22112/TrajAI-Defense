import { useCallback, useEffect, useRef, useState } from "react";
import { DEMO, demoFile, liveUrl } from "../api/client";
import type { LiveMsg } from "../api/types";

type LiveState = Extract<LiveMsg, { type: "state" }>;
type LiveMeas = Extract<LiveMsg, { type: "meas" }>;

export interface LiveTrack {
  status: "idle" | "running" | "done" | "error";
  meas: LiveMeas[];
  states: LiveState[];
  error: string | null;
  start: (speed: number, weather: boolean) => void;
  stop: () => void;
}

/** WebSocket replay of a scenario's radar track with a streaming EKF on the server. */
export function useLiveTrack(scenarioId: number): LiveTrack {
  const ws = useRef<WebSocket | null>(null);
  const timers = useRef<number[]>([]);
  const [status, setStatus] = useState<LiveTrack["status"]>("idle");
  const [meas, setMeas] = useState<LiveMeas[]>([]);
  const [states, setStates] = useState<LiveState[]>([]);
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => {
    ws.current?.close(); ws.current = null;
    timers.current.forEach(clearTimeout); timers.current = [];
  }, []);

  const handle = useCallback((m: LiveMsg) => {
    if (m.type === "meas") setMeas(p => [...p, m]);
    else if (m.type === "state") setStates(p => [...p, m]);
    else if (m.type === "done") setStatus("done");
    else if (m.type === "error") { setStatus("error"); setError(m.detail); }
  }, []);

  const start = useCallback((speed: number, weather: boolean) => {
    stop();
    setMeas([]); setStates([]); setError(null);
    setStatus("running");
    if (DEMO) {
      // static demo: play back a stream recorded from the real server at the same pacing (0.5 s per measurement / speed)
      fetch(demoFile(`live_${scenarioId}_${weather ? 1 : 0}.json`))
        .then(r => (r.ok ? r.json() : Promise.reject(new Error("녹화된 재생 자료가 없습니다."))))
        .then((msgs: LiveMsg[]) => {
          let k = 0;
          msgs.forEach(m => {
            if (m.type === "meas") k += 1;
            timers.current.push(window.setTimeout(() => handle(m), (k * 500) / speed));
          });
        })
        .catch(e => { setStatus("error"); setError(String(e.message ?? e)); });
      return;
    }
    const sock = new WebSocket(liveUrl(scenarioId, speed, weather));
    ws.current = sock;
    sock.onmessage = ev => handle(JSON.parse(ev.data) as LiveMsg);
    sock.onerror = () => { setStatus("error"); setError("WebSocket 연결 오류"); };
    sock.onclose = () => setStatus(s => (s === "running" ? "done" : s));
  }, [scenarioId, stop, handle]);

  useEffect(() => stop, [stop]);
  return { status, meas, states, error, start, stop };
}
