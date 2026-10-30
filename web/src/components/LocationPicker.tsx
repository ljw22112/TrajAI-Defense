import { LOCATIONS } from "../api/types";

/** Engagement-area picker: preset regions or custom coordinates (snapped by the server to the 0.25 deg weather grid). */
export function LocationPicker({ lat, lon, onChange, idPrefix = "loc" }: {
  lat: number; lon: number; onChange: (lat: number, lon: number) => void; idPrefix?: string;
}) {
  const preset = LOCATIONS.findIndex(l => l.lat === lat && l.lon === lon);
  return (
    <>
      <div className="field">
        <label htmlFor={`${idPrefix}-preset`}>기상 위치 (교전 해역)</label>
        <select id={`${idPrefix}-preset`} value={preset} onChange={e => { const l = LOCATIONS[Number(e.target.value)]; if (l) onChange(l.lat, l.lon); }}>
          {LOCATIONS.map((l, i) => <option key={l.label} value={i}>{l.label} · {l.lat.toFixed(2)}°N {l.lon.toFixed(2)}°E</option>)}
          <option value={-1}>직접 입력</option>
        </select>
        <small>Open-Meteo 격자(0.25°)로 맞춰 조회합니다</small>
      </div>
      <div className="field">
        <label htmlFor={`${idPrefix}-lat`}>위도 / 경도 (°)</label>
        <div style={{ display: "flex", gap: 6 }}>
          <input id={`${idPrefix}-lat`} type="number" step={0.25} min={-80} max={80} value={lat} aria-label="위도" onChange={e => onChange(Number(e.target.value), lon)} />
          <input id={`${idPrefix}-lon`} type="number" step={0.25} min={-180} max={180} value={lon} aria-label="경도" onChange={e => onChange(lat, Number(e.target.value))} />
        </div>
      </div>
    </>
  );
}
