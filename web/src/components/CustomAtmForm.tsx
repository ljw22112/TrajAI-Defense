import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api, DEMO } from "../api/client";
import { CONDITION_LABEL, type Condition, type CustomAtm } from "../api/types";
import { ProfileChart } from "./ProfileChart";
import { ErrorBox, Spinner, fmt } from "./ui";

type NumKey = Exclude<keyof CustomAtm, "condition">;
const GROUPS: { title: string; fields: { k: NumKey; label: string; unit: string; min: number; max: number; step: number; help?: string }[] }[] = [
  { title: "기온 · 기압 · 습도", fields: [
    { k: "t_sfc_c", label: "지상 기온", unit: "°C", min: -50, max: 50, step: 1 },
    { k: "p_sfc_hpa", label: "지상 기압", unit: "hPa", min: 900, max: 1060, step: 1 },
    { k: "rh_pct", label: "상대습도", unit: "%", min: 0, max: 100, step: 5, help: "가상온도로 밀도에 반영" },
    { k: "lapse_k_per_km", label: "기온감률", unit: "K/km", min: 3, max: 9.8, step: 0.1, help: "표준 6.5" },
    { k: "tropopause_km", label: "대류권계면", unit: "km", min: 8, max: 18, step: 0.5 },
  ] },
  { title: "바람", fields: [
    { k: "wind_sfc_ms", label: "지상풍", unit: "m/s", min: 0, max: 40, step: 1 },
    { k: "wind_sfc_dir", label: "지상풍 풍향", unit: "°", min: 0, max: 360, step: 5, help: "불어오는 쪽, 270 = 서풍" },
    { k: "jet_ms", label: "제트기류 최대풍속", unit: "m/s", min: 0, max: 120, step: 5 },
    { k: "jet_km", label: "제트기류 고도", unit: "km", min: 4, max: 20, step: 0.5 },
    { k: "jet_width_km", label: "제트기류 두께 (1σ)", unit: "km", min: 0.5, max: 8, step: 0.5 },
    { k: "jet_dir", label: "제트기류 풍향", unit: "°", min: 0, max: 360, step: 5 },
  ] },
];
const PRESETS: Record<string, Partial<CustomAtm>> = {
  "표준 (USSA)": { t_sfc_c: 15, p_sfc_hpa: 1013.25, rh_pct: 0, lapse_k_per_km: 6.5, tropopause_km: 11, wind_sfc_ms: 0, jet_ms: 0, condition: "clear" },
  "한여름 장마": { t_sfc_c: 31, p_sfc_hpa: 1004, rh_pct: 90, lapse_k_per_km: 5.5, tropopause_km: 16, wind_sfc_ms: 6, wind_sfc_dir: 200, jet_ms: 25, jet_km: 13, jet_dir: 250, condition: "heavy_rain" },
  "한겨울 제트기류": { t_sfc_c: -8, p_sfc_hpa: 1030, rh_pct: 40, lapse_k_per_km: 6.8, tropopause_km: 10, wind_sfc_ms: 8, wind_sfc_dir: 315, jet_ms: 85, jet_km: 10.5, jet_dir: 270, condition: "clear" },
  "강한 맞바람": { jet_ms: 70, jet_dir: 90, wind_sfc_ms: 12, wind_sfc_dir: 90 },
};

/** Editor for a user-defined atmosphere with a debounced server-side preview (density, wind, radar noise factor). */
export function CustomAtmForm({ value, onChange, idPrefix = "atm", preview = true }: {
  value: CustomAtm; onChange: (v: CustomAtm) => void; idPrefix?: string; preview?: boolean;
}) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => { const t = setTimeout(() => setDebounced(value), 350); return () => clearTimeout(t); }, [value]);
  const pv = useQuery({ queryKey: ["custom-atm", debounced], queryFn: () => api.customPreview(debounced), enabled: preview && !DEMO, retry: 0 });
  const set = (k: keyof CustomAtm, v: number | string) => onChange({ ...value, [k]: v });

  return (
    <div className="page" style={{ gap: 12 }}>
      <div className="toolbar">
        <span className="label">프리셋</span>
        {Object.entries(PRESETS).map(([k, p]) => <button type="button" key={k} className="btn ghost sm" onClick={() => onChange({ ...value, ...p })}>{k}</button>)}
      </div>
      {GROUPS.map(g => (
        <fieldset key={g.title} className="fs">
          <legend>{g.title}</legend>
          <div className="form">
            {g.fields.map(f => (
              <div className="field" key={f.k}>
                <label htmlFor={`${idPrefix}-${f.k}`}>{f.label} ({f.unit}) <b className="mono">{fmt(value[f.k] as number, f.step < 1 ? 1 : 0)}</b></label>
                <input id={`${idPrefix}-${f.k}`} type="range" min={f.min} max={f.max} step={f.step} value={value[f.k] as number}
                  onChange={e => set(f.k, Number(e.target.value))} />
                <small>{f.help ?? `${f.min} – ${f.max}`}</small>
              </div>
            ))}
          </div>
        </fieldset>
      ))}
      <fieldset className="fs">
        <legend>날씨 (레이더 영향)</legend>
        <div className="toolbar" role="radiogroup" aria-label="날씨">
          {(Object.keys(CONDITION_LABEL) as Condition[]).map(c => (
            <label key={c} className="chk"><input type="radio" name={`${idPrefix}-cond`} checked={value.condition === c} onChange={() => set("condition", c)} /> {CONDITION_LABEL[c]}</label>
          ))}
        </div>
        <small className="label">강수·안개는 탄도체 항력에 거의 영향이 없고(공기 밀도의 0.1% 미만), 레이더 신호를 약하게 만들어 측정 잡음을 키웁니다. 비 ×1.3, 폭우 ×2.0, 눈 ×1.4, 안개 ×1.05로 근사합니다.</small>
      </fieldset>
      {preview && !DEMO && (
        pv.isLoading ? <Spinner label="대기 프로파일 계산 중" /> : pv.error ? <ErrorBox error={pv.error} /> : pv.data && (
          <div>
            <div className="toolbar" style={{ marginBottom: 8 }}>
              <span className="chip">지상 밀도 {fmt(pv.data.surface_density, 3)} kg/m³ ({pv.data.density_vs_standard_pct >= 0 ? "+" : ""}{fmt(pv.data.density_vs_standard_pct, 1)}% vs 표준)</span>
              <span className="chip acc">최대풍 {fmt(pv.data.max_wind)} m/s @ {fmt(pv.data.max_wind_alt / 1000, 1)} km</span>
              <span className={`chip ${pv.data.noise_factor > 1 ? "acc" : ""}`}>레이더 잡음 ×{fmt(pv.data.noise_factor, 2)}</span>
            </div>
            <ProfileChart levels={pv.data.levels} height={220} />
          </div>
        )
      )}
    </div>
  );
}
