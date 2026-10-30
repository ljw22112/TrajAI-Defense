import { useMemo } from "react";
import type { WeatherLevels } from "../api/types";
import { Legend, Plot, type Series } from "./Plot";

/** Temperature and wind components vs altitude, from raw pressure levels. */
export function ProfileChart({ levels, height = 280 }: { levels: WeatherLevels; height?: number }) {
  const zk = levels.z.map(z => z / 1000);
  const temp = useMemo<Series[]>(() => [{ kind: "line", x: levels.T.map(t => t - 273.15), y: zk, color: "--pred", width: 2.2 }], [levels]);
  const wind = useMemo<Series[]>(() => [
    { kind: "line", x: levels.u, y: zk, color: "--ekf", width: 2.2 },
    { kind: "line", x: levels.v, y: zk, color: "--obs", width: 2, dash: [5, 4] },
    { kind: "line", x: levels.u.map((u, i) => Math.hypot(u, levels.v[i])), y: zk, color: "--ink", width: 1.2, alpha: 0.6 },
  ], [levels]);
  return (
    <div className="grid g2">
      <div>
        <Plot series={temp} xLabel="기온 (°C)" yLabel="고도 (km)" height={height} ariaLabel="고도별 기온" />
      </div>
      <div>
        <Plot series={wind} xLabel="바람 (m/s)" yLabel="고도 (km)" height={height} ariaLabel="고도별 바람 성분" />
        <Legend items={[{ label: "u (동쪽 +)", color: "--ekf" }, { label: "v (북쪽 +)", color: "--obs", dash: true }, { label: "풍속", color: "--ink" }]} />
      </div>
    </div>
  );
}
