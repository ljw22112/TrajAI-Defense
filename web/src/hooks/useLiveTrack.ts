import { useCallback, useEffect, useRef, useState } from "react";
import { DEMO, liveUrl } from "../api/client";
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
  const [status, setStatus] = useState<LiveTrack["status"]>("idle");
  const [meas, setMeas] = useState<LiveMeas[]>([]);
  const [states, setStates] = useState<LiveState[]>([]);
  const [error, setError] = useState<string | null>(null);

  const stop = useCallback(() => { ws.current?.close(); ws.current = null; }, []);

  const start = useCallback((speed: number, weather: boolean) => {
    stop();
    setMeas([]); setStates([]); setError(null);
    if (DEMO) { setStatus("error"); setError("실시간 재생은 서버 연결이 필요합니다 (데모 모드)."); return; }
    setStatus("running");
    const sock = new WebSocket(liveUrl(scenarioId, speed, weather));
    ws.current = sock;
    sock.onmessage = ev => {
      const m = JSON.parse(ev.data) as LiveMsg;
      if (m.type === "meas") setMeas(p => [...p, m]);
      else if (m.type === "state") setStates(p => [...p, m]);
      else if (m.type === "done") setStatus("done");
      else if (m.type === "error") { setStatus("error"); setError(m.detail); }
    };
    sock.onerror = () => { setStatus("error"); setError("WebSocket 연결 오류"); };
    sock.onclose = () => setStatus(s => (s === "running" ? "done" : s));
  }, [scenarioId, stop]);

  useEffect(() => stop, [stop]);
  return { status, meas, states, error, start, stop };
}
