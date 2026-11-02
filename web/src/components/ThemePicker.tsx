import { useEffect, useState } from "react";

export type Theme = "system" | "light" | "dark" | "military";
const KEY = "trajai.theme";
const OPTIONS: [Theme, string][] = [["system", "시스템"], ["light", "라이트"], ["dark", "다크"], ["military", "밀리터리"]];

export function readTheme(): Theme {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" || v === "military" ? v : "system";
  } catch { return "system"; }
}

/** Applies a theme to <html> and tells canvas plots to redraw with the new colours. */
export function applyTheme(t: Theme) {
  const el = document.documentElement;
  if (t === "system") el.removeAttribute("data-theme"); else el.setAttribute("data-theme", t);
  window.dispatchEvent(new Event("trajai-theme"));
}

export function ThemePicker() {
  const [t, setT] = useState<Theme>(readTheme);
  useEffect(() => {
    applyTheme(t);
    try { localStorage.setItem(KEY, t); } catch { /* private mode: theme still applies for this visit */ }
  }, [t]);
  return (
    <label className="theme-pick">
      <span className="label">화면</span>
      <select value={t} onChange={e => setT(e.target.value as Theme)} aria-label="화면 테마">
        {OPTIONS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
      </select>
    </label>
  );
}
