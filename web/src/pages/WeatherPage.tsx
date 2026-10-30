import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api, DEMO } from "../api/client";
import { LocationPicker } from "../components/LocationPicker";
import { ProfileChart } from "../components/ProfileChart";
import { Card, Empty, ErrorBox, PageHead, Spinner, fmt } from "../components/ui";

export function WeatherPage() {
  const cached = useQuery({ queryKey: ["weather-cached"], queryFn: api.weatherCached });
  const [date, setDate] = useState("2026-10-15");
  const [hour, setHour] = useState(0);
  const [loc, setLoc] = useState({ lat: 36.5, lon: 127.5 });
  const [sel, setSel] = useState<{ date: string; hour: number; lat?: number; lon?: number } | null>(null);
  const prof = useQuery({ queryKey: ["weather", sel?.date, sel?.hour, sel?.lat, sel?.lon], queryFn: () => api.weather(sel!.date, sel!.hour, sel!.lat, sel!.lon), enabled: !!sel, retry: 0 });
  const shown = prof.data ?? (sel ? undefined : cached.data?.[0]);

  return (
    <div className="page">
      <PageHead kicker="Weather" title="기상 프로파일">
        Open-Meteo 기압면 자료(1000–10 hPa)를 일자·시각별로 받아 데이터베이스에 캐시합니다. 같은 시각을 다시 요청하면 외부 호출 없이 캐시를 씁니다.
      </PageHead>
      {!DEMO && (
        <Card title="일자 조회">
          <div className="toolbar">
            <input type="date" aria-label="일자" min="2022-01-01" value={date} onChange={e => setDate(e.target.value)} style={{ maxWidth: 180 }} />
            <select aria-label="시각" value={hour} onChange={e => setHour(Number(e.target.value))} style={{ maxWidth: 110 }}>
              {[0, 6, 12, 18].map(h => <option key={h} value={h}>{String(h).padStart(2, "0")}Z</option>)}
            </select>
            <button className="btn" onClick={() => setSel({ date, hour, ...loc })}>불러오기</button>
          </div>
          <div className="form" style={{ marginTop: 10 }}>
            <LocationPicker idPrefix="wx" lat={loc.lat} lon={loc.lon} onChange={(lat, lon) => setLoc({ lat, lon })} />
          </div>
          {prof.isLoading && <Spinner label="기상 자료를 받는 중" />}
          <ErrorBox error={prof.error} />
        </Card>
      )}
      {shown && (
        <Card title={`${new Date(shown.valid_time).toISOString().slice(0, 13).replace("T", " ")}Z · ${fmt(shown.lat, 1)}°N ${fmt(shown.lon, 1)}°E`}
          aside={<span className="chip acc">최대풍 {fmt(shown.max_wind)} m/s @ {fmt(shown.max_wind_alt / 1000, 1)} km</span>}>
          <ProfileChart levels={shown.levels} />
        </Card>
      )}
      <Card title="캐시된 프로파일">
        <ErrorBox error={cached.error} />
        {cached.isLoading ? <Spinner /> : !cached.data?.length ? <Empty>캐시된 프로파일이 없습니다.</Empty> : (
          <div className="tw">
            <table>
              <thead><tr><th>유효 시각 (UTC)</th><th>위치</th><th>출처</th><th className="n">최대풍</th><th className="n">최대풍 고도</th><th className="n">최상층 고도</th><th /></tr></thead>
              <tbody>
                {cached.data.map(w => (
                  <tr key={w.id}>
                    <td className="mono">{new Date(w.valid_time).toISOString().slice(0, 16).replace("T", " ")}</td>
                    <td className="mono">{w.lat.toFixed(2)}, {w.lon.toFixed(2)}</td><td>{w.source}</td><td className="n">{fmt(w.max_wind, 1)} m/s</td><td className="n">{fmt(w.max_wind_alt / 1000, 1)} km</td>
                    <td className="n">{fmt(w.levels.z[w.levels.z.length - 1] / 1000, 1)} km</td>
                    <td><button className="btn ghost sm" onClick={() => { const d = new Date(w.valid_time); setSel({ date: d.toISOString().slice(0, 10), hour: d.getUTCHours(), lat: w.lat, lon: w.lon }); }}>보기</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
