import { useEffect, useRef } from "react";

export interface Series {
  kind: "line" | "points" | "circle" | "cross" | "band";
  x: number[];
  y: number[];
  y2?: number[];            // band: upper edge (y = lower)
  r?: number[];             // circle: radius in data units (x scale)
  color: string;            // CSS custom property (e.g. "--pred") or literal colour
  width?: number;
  dash?: number[];
  size?: number;
  alpha?: number;
  label?: string;
}

interface Props {
  series: Series[];
  xLabel: string;
  yLabel: string;
  height?: number;
  equal?: boolean;          // same data scale on both axes (map views)
  xDomain?: [number, number];
  yDomain?: [number, number];
  logY?: boolean;
  fmtX?: (v: number) => string;
  fmtY?: (v: number) => string;
  ariaLabel: string;
}

function niceStep(range: number, count: number) {
  const raw = range / count, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}
const cssVar = (c: string) => (c.startsWith("--") ? getComputedStyle(document.documentElement).getPropertyValue(c).trim() || "#888" : c);

export function Plot({ series, xLabel, yLabel, height = 320, equal, xDomain, yDomain, logY, fmtX, fmtY, ariaLabel }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const draw = () => {
      const W = cv.clientWidth, H = height, dpr = Math.min(2, window.devicePixelRatio || 1);
      cv.width = W * dpr; cv.height = H * dpr;
      const g = cv.getContext("2d");
      if (!g || W === 0) return;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      const ink = cssVar("--ink"), ink2 = cssVar("--ink-2"), grid = cssVar("--grid"), sheet = cssVar("--sheet");
      g.fillStyle = sheet; g.fillRect(0, 0, W, H);
      const m = { l: 62, r: 14, t: 12, b: 42 };
      const xs = series.flatMap(s => s.x.concat(s.kind === "circle" && s.r ? s.x.map((v, i) => v + s.r![i]).concat(s.x.map((v, i) => v - s.r![i])) : []));
      const ys = series.flatMap(s => s.y.concat(s.y2 ?? []).concat(s.kind === "circle" && s.r ? s.y.map((v, i) => v + s.r![i]).concat(s.y.map((v, i) => v - s.r![i])) : []));
      const finite = (a: number[]) => a.filter(Number.isFinite);
      let [x0, x1] = xDomain ?? [Math.min(...finite(xs)), Math.max(...finite(xs))];
      let [y0, y1] = yDomain ?? [Math.min(...finite(ys)), Math.max(...finite(ys))];
      if (!Number.isFinite(x0)) { x0 = 0; x1 = 1; }
      if (!Number.isFinite(y0)) { y0 = 0; y1 = 1; }
      if (x1 - x0 < 1e-9) { x0 -= 1; x1 += 1; }
      if (y1 - y0 < 1e-9) { y0 -= 1; y1 += 1; }
      const padX = (x1 - x0) * 0.04, padY = (y1 - y0) * 0.06;
      if (!xDomain) { x0 -= padX; x1 += padX; }
      if (!yDomain && !logY) { y0 -= padY; y1 += padY; }
      const pw = W - m.l - m.r, ph = H - m.t - m.b;
      if (equal) {                                          // expand the shorter range so 1 unit is the same on both axes
        const sx = pw / (x1 - x0), sy = ph / (y1 - y0);
        if (sx > sy) { const c = (x0 + x1) / 2, h = pw / sy / 2; x0 = c - h; x1 = c + h; }
        else { const c = (y0 + y1) / 2, h = ph / sx / 2; y0 = c - h; y1 = c + h; }
      }
      const ly = (v: number) => (logY ? Math.log10(Math.max(v, 1e-12)) : v);
      const Y0 = ly(y0), Y1 = ly(y1);
      const X = (v: number) => m.l + ((v - x0) / (x1 - x0)) * pw;
      const Y = (v: number) => m.t + ph - ((ly(v) - Y0) / (Y1 - Y0)) * ph;
      g.font = "11px 'IBM Plex Mono', ui-monospace, monospace";
      g.lineWidth = 1;
      const xt = niceStep(x1 - x0, Math.max(3, Math.floor(pw / 90)));
      for (let v = Math.ceil(x0 / xt) * xt; v <= x1; v += xt) {
        g.strokeStyle = grid; g.beginPath(); g.moveTo(X(v), m.t); g.lineTo(X(v), m.t + ph); g.stroke();
        g.fillStyle = ink2; g.textAlign = "center"; g.fillText(fmtX ? fmtX(v) : String(+v.toFixed(6)), X(v), m.t + ph + 16);
      }
      if (logY) {
        for (let e = Math.ceil(Y0); e <= Y1; e++) {
          const v = Math.pow(10, e);
          g.strokeStyle = grid; g.beginPath(); g.moveTo(m.l, Y(v)); g.lineTo(m.l + pw, Y(v)); g.stroke();
          g.fillStyle = ink2; g.textAlign = "right"; g.fillText(fmtY ? fmtY(v) : String(v), m.l - 8, Y(v) + 4);
        }
      } else {
        const yt = niceStep(y1 - y0, Math.max(3, Math.floor(ph / 60)));
        for (let v = Math.ceil(y0 / yt) * yt; v <= y1; v += yt) {
          g.strokeStyle = grid; g.beginPath(); g.moveTo(m.l, Y(v)); g.lineTo(m.l + pw, Y(v)); g.stroke();
          g.fillStyle = ink2; g.textAlign = "right"; g.fillText(fmtY ? fmtY(v) : String(+v.toFixed(6)), m.l - 8, Y(v) + 4);
        }
      }
      g.strokeStyle = ink; g.lineWidth = 1.5; g.beginPath(); g.moveTo(m.l, m.t); g.lineTo(m.l, m.t + ph); g.lineTo(m.l + pw, m.t + ph); g.stroke();
      g.font = "12px 'IBM Plex Sans KR', system-ui, sans-serif"; g.fillStyle = ink2;
      g.textAlign = "center"; g.fillText(xLabel, m.l + pw / 2, H - 6);
      g.save(); g.translate(13, m.t + ph / 2); g.rotate(-Math.PI / 2); g.fillText(yLabel, 0, 0); g.restore();
      g.save(); g.beginPath(); g.rect(m.l, m.t, pw, ph); g.clip();
      for (const s of series) {
        const col = cssVar(s.color);
        g.globalAlpha = s.alpha ?? 1; g.strokeStyle = col; g.fillStyle = col; g.lineWidth = s.width ?? 2; g.setLineDash(s.dash ?? []);
        if (s.kind === "line") {
          g.beginPath(); s.x.forEach((v, i) => (i ? g.lineTo(X(v), Y(s.y[i])) : g.moveTo(X(v), Y(s.y[i])))); g.stroke();
        } else if (s.kind === "band" && s.y2) {
          g.beginPath(); s.x.forEach((v, i) => (i ? g.lineTo(X(v), Y(s.y2![i])) : g.moveTo(X(v), Y(s.y2![i]))));
          for (let i = s.x.length - 1; i >= 0; i--) g.lineTo(X(s.x[i]), Y(s.y[i]));
          g.closePath(); g.fill();
        } else if (s.kind === "points") {
          const r = s.size ?? 2.5; s.x.forEach((v, i) => { g.beginPath(); g.arc(X(v), Y(s.y[i]), r, 0, 7); g.fill(); });
        } else if (s.kind === "circle" && s.r) {
          s.x.forEach((v, i) => { g.beginPath(); g.arc(X(v), Y(s.y[i]), Math.max(2, (s.r![i] / (x1 - x0)) * pw), 0, 7); g.stroke(); });
        } else if (s.kind === "cross") {
          const r = s.size ?? 6;
          s.x.forEach((v, i) => { const cx = X(v), cy = Y(s.y[i]); g.beginPath(); g.moveTo(cx - r, cy); g.lineTo(cx + r, cy); g.moveTo(cx, cy - r); g.lineTo(cx, cy + r); g.stroke(); });
        }
      }
      g.restore(); g.globalAlpha = 1; g.setLineDash([]);
    };
    draw();
    const ro = new ResizeObserver(draw); ro.observe(cv);
    const mq = window.matchMedia("(prefers-color-scheme: dark)"); mq.addEventListener("change", draw);
    return () => { ro.disconnect(); mq.removeEventListener("change", draw); };
  }, [series, xLabel, yLabel, height, equal, xDomain, yDomain, logY, fmtX, fmtY]);

  return <canvas ref={ref} className="plot" style={{ height }} role="img" aria-label={ariaLabel} />;
}

export function Legend({ items }: { items: { label: string; color: string; dash?: boolean }[] }) {
  return (
    <div className="legend">
      {items.map(i => (
        <span key={i.label}><i className="sw" style={{ borderTopColor: `var(${i.color})`, borderTopStyle: i.dash ? "dashed" : "solid" }} />{i.label}</span>
      ))}
    </div>
  );
}
