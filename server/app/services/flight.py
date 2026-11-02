"""Flight replay with the environment along the way: what the vehicle meets and when.

The truth trajectory is re-flown (deterministic: same state, beta and atmosphere as the stored scenario) at 0.5 s and,
for every sample, the air it flies through is looked up: temperature, density, wind, speed of sound. From these:
Mach number (air-relative), dynamic pressure, drag deceleration. Events mark where the environment changes in a
way that matters to the flight or to the estimators (layer boundaries, jet stream, Mach regime, max-q, ...).
"""
from __future__ import annotations

import numpy as np

from trajai import physics
from trajai import weather as wx

DT = 0.5
G0 = 9.80665
RHO0 = 1.225
JET_MS = 30.0           # wind speed that counts as jet stream [m/s]
THIN_AIR = 0.01         # density ratio below which drag is negligible (about 31 km and higher)

VEHICLES = {
    "srbm": "단거리 탄도 미사일급",
    "mrbm": "중거리 탄도 미사일급",
    "lrbm": "장거리 탄도 미사일급",
    "sounding": "관측 로켓 (과학 탐사)",
    "debris": "로켓 상단 낙하 잔해",
    "generic": "일반 탄도체",
}


class _Std:
    """Standard atmosphere (USSA-1976, no wind) with the same interface as a weather grid."""
    def rho(self, h): return physics.density(h)
    def sos(self, h): return physics.speed_of_sound(h)
    def temp(self, h): return physics.temperature(h)
    def wind(self, h): return np.zeros(np.shape(h) + (3,))


class _Wx:
    def __init__(self, bank):
        self.g = bank[np.zeros(1, int)]
        self.bank = bank

    def rho(self, h): return self.g.rho(h)
    def sos(self, h): return self.g.sos(h)
    def wind(self, h): return self.g.wind(h)

    def temp(self, h):
        x = np.clip(h, 0.0, wx.H_GRID[-1] - 1e-6) / wx.DH
        i = x.astype(int); f = x - i
        return self.bank.T[0, i] * (1 - f) + self.bank.T[0, i + 1] * f


def _crossings(t, y, level):
    """Times where y crosses `level` (linear interpolation), with direction +1 up / -1 down."""
    d = y - level
    out = []
    for k in np.nonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]:
        f = d[k] / (d[k] - d[k + 1])
        out.append((float(t[k] + f * (t[k + 1] - t[k])), 1 if d[k + 1] > d[k] else -1))
    return out


def flight_profile(s, bank, condition: str | None = None, vehicle: str | None = None) -> dict:
    g, psi = np.radians(s.gamma_deg), np.radians(s.psi_deg)
    vh = s.v0 * np.cos(g)
    s0 = np.array([[0.0, 0.0, s.h0, vh * np.cos(psi), vh * np.sin(psi), s.v0 * np.sin(g)]])
    atm = bank[np.zeros(1, int)] if bank is not None else None
    times, pos, vel, T, r_imp = physics.simulate_ballistic(s0, np.array([s.beta]), dt=0.1, out_dt=DT, atm=atm)
    ok = ~np.isnan(pos[0, :, 0])
    t, p, v = times[ok], pos[0][ok], vel[0][ok]
    t = np.append(t, T[0]); p = np.vstack([p, [r_imp[0, 0], r_imp[0, 1], 0.0]]); v = np.vstack([v, v[-1]])
    A = _Wx(bank) if bank is not None else _Std()
    h = np.maximum(p[:, 2], 0.0)
    w = A.wind(h)
    rel = v - w
    vr = np.linalg.norm(rel, axis=1)
    sos = A.sos(h); rho = A.rho(h)
    mach = vr / sos
    q = 0.5 * rho * vr ** 2
    drag = rho * vr ** 2 * physics.cd_ratio(mach) / (2.0 * s.beta)
    wspd = np.hypot(w[:, 0], w[:, 1])
    wdir = (np.degrees(np.arctan2(-w[:, 0], -w[:, 1])) + 360) % 360          # meteorological: blowing FROM
    temp_c = A.temp(h) - 273.15
    # vertical structure for the background (every 1 km up to above the apogee)
    zt = np.arange(0.0, max(float(h.max()) * 1.08, 12_000.0), 1000.0)
    wz = A.wind(zt)
    tz = A.temp(zt)
    i_tp = int(np.argmin(np.where(zt <= 20_000, tz, np.inf)))
    tropo = float(zt[i_tp]) if bank is not None else 11_000.0
    spd_z = np.hypot(wz[:, 0], wz[:, 1])
    low = zt <= 20_000
    i_max = int(np.argmax(np.where(low, spd_z, -1)))
    jet = None
    if spd_z[i_max] >= JET_MS:
        lo = hi = i_max
        while lo > 0 and spd_z[lo - 1] >= JET_MS: lo -= 1
        while hi < len(zt) - 1 and spd_z[hi + 1] >= JET_MS: hi += 1
        jet = dict(bottom=float(zt[lo]), top=float(zt[hi]), peak=float(spd_z[i_max]), peak_alt=float(zt[i_max]))

    # thin-air altitude from the density profile
    rz = A.rho(zt) / RHO0
    thin = float(zt[np.argmax(rz < THIN_AIR)]) if (rz < THIN_AIR).any() else None

    ev = []

    def add(tt, kind, title, detail):
        k = int(np.clip(np.searchsorted(t, tt), 0, len(t) - 1))
        ev.append(dict(t=round(float(tt), 1), h=round(float(h[k]), 0), kind=kind, title=title, detail=detail))

    in_jet = jet is not None and jet["bottom"] <= s.h0 <= jet["top"]
    add(0.0, "start", "연소 종료 · 탄도 비행 시작", f"고도 {s.h0 / 1000:.1f} km, 속도 {s.v0:,.0f} m/s, 경로각 {s.gamma_deg:.0f}°"
        + (f". 출발 고도가 이미 제트기류 안(풍속 {wspd[0]:.0f} m/s)입니다." if in_jet else "")
        + (". 출발 고도가 이미 희박 대기(밀도가 지상의 1% 미만)라 처음에는 항력이 거의 없습니다." if thin is not None and s.h0 >= thin else ""))
    for tt, d in _crossings(t, h, tropo):
        add(tt, "layer", "대류권계면 통과 " + ("(성층권 진입)" if d > 0 else "(대류권 재진입)"),
            f"고도 약 {tropo / 1000:.1f} km. 위로는 기온이 더 내려가지 않고 공기가 건조·안정해집니다." if d > 0 else
            f"고도 약 {tropo / 1000:.1f} km. 아래로 기온이 오르고 날씨(구름·강수)가 있는 층입니다.")
    if jet:
        for edge, name in ((jet["bottom"], "아래"), (jet["top"], "위")):
            for tt, d in _crossings(t, h, edge):
                inside = (d > 0) == (name == "아래")
                add(tt, "wind", "제트기류 " + ("진입" if inside else "이탈"),
                    f"풍속 {JET_MS:.0f} m/s 이상 구간({jet['bottom'] / 1000:.0f}–{jet['top'] / 1000:.0f} km, 최대 {jet['peak']:.0f} m/s). "
                    + ("강한 바람이 공기 상대속도를 바꿔 항력 방향이 틀어집니다." if inside else "바람의 영향이 줄어듭니다."))
    if thin:
        for tt, d in _crossings(t, h, thin):
            add(tt, "layer", "희박 대기 " + ("진입" if d > 0 else "이탈 (재진입)"),
                f"고도 약 {thin / 1000:.0f} km, 공기 밀도가 지상의 1% 미만. " + ("항력이 거의 사라져 포물선에 가깝게 납니다." if d > 0 else
                                                                   "항력이 다시 커지기 시작합니다. 탄도계수 β가 궤적을 좌우하는 구간입니다."))
    for lvl, name, det in ((1.2, "초음속 → 천음속", "항력 계수가 정점(마하 약 1.2)으로 올라가는 구간. 감속이 커집니다."),
                           (0.8, "천음속 → 아음속", "항력 계수가 낮아지고 속도가 안정됩니다.")):
        for tt, d in _crossings(t, mach, lvl):
            if d < 0:
                add(tt, "mach", f"{name} (마하 {lvl})", det)
    k_ap = int(np.argmax(h))
    add(t[k_ap], "apogee", "정점", f"최고 고도 {h[k_ap] / 1000:.1f} km. 이후 하강하며 다시 짙은 공기로 들어갑니다.")
    desc = np.arange(len(t)) > k_ap
    if desc.any():
        k_q = k_ap + 1 + int(np.argmax(q[desc]))
        add(t[k_q], "maxq", "최대 동압 (재진입 중)", f"{q[k_q] / 1000:,.0f} kPa, 감속 {drag[k_q] / G0:.1f} g. 공기력이 가장 큰 순간입니다.")
    if desc.any() and wspd[desc].max() > 0:
        k_w = k_ap + 1 + int(np.argmax(wspd[desc]))
        add(t[k_w], "wind", "하강 중 최대 풍속", f"{wspd[k_w]:.0f} m/s, {wdir[k_w]:.0f}°에서 부는 바람. 탄착점을 옆으로 미는 주된 원인입니다.")
    add(s.t_c, "radar", "레이더 관측 종료", "이 시점까지의 측정으로 추정기가 탄착점을 예측합니다. 이후의 환경 변화는 예측에 들어 있지 않습니다.")
    if condition and condition not in ("clear", "cloudy"):
        k = int(np.argmax(h < 3000) if (h < 3000).any() else len(h) - 1)
        if k > k_ap:
            add(t[k], "weather", "강수·안개층 (지상 3 km 이하)",
                {"rain": "비", "heavy_rain": "폭우", "snow": "눈", "fog": "안개"}.get(condition, condition)
                + " 구간. 비행에는 거의 영향이 없지만 레이더 측정 잡음을 키웁니다(관측 정확도 저하).")
    add(T[0], "impact", "탄착", f"하향거리 {r_imp[0, 0] / 1000:.1f} km, 횡거리 {r_imp[0, 1] / 1000:+.1f} km. 비행 시간 {T[0]:.0f} s.")
    ev.sort(key=lambda e: e["t"])

    r = lambda a, n=1: np.round(a, n).tolist()
    return dict(
        vehicle=vehicle or "generic", vehicle_label=VEHICLES.get(vehicle or "generic", VEHICLES["generic"]),
        atmosphere="standard" if bank is None else "weather", condition=condition,
        t=r(t), x=r(p[:, 0], 0), y=r(p[:, 1], 0), z=r(h, 0), speed=r(np.linalg.norm(v, axis=1)), mach=r(mach, 2),
        temp_c=r(temp_c), rho_ratio=np.round(rho / RHO0, 5).tolist(), wind=r(wspd), wind_dir=r(wdir, 0),
        q_kpa=r(q / 1000, 2), drag_g=r(drag / G0, 3),
        layers=dict(tropopause=tropo, jet=jet, thin_air=thin),
        column=dict(z=r(zt, 0), u=r(wz[:, 0]), v=r(wz[:, 1]), temp_c=r(tz - 273.15)),
        t_c=s.t_c, impact_time=float(T[0]), apogee=float(h.max()), events=ev, drift=_drift(s0, s.beta, bank, t, p))


def _drift(s0, beta, bank, t, p):
    """Same flight with the wind switched off (temperature and density kept): how far the wind moved the vehicle.
    None for the standard atmosphere (no wind)."""
    if bank is None:
        return None
    calm = wx.ProfileBank(bank.T, bank.rho, bank.sos, bank.u * 0.0, bank.v * 0.0, bank.times)
    times, pos, _, T, r_imp = physics.simulate_ballistic(s0, np.array([beta]), dt=0.1, out_dt=DT, atm=calm[np.zeros(1, int)])
    ok = ~np.isnan(pos[0, :, 0])
    tc, pc = np.append(times[ok], T[0]), np.vstack([pos[0][ok], [r_imp[0, 0], r_imp[0, 1], 0.0]])
    xs, ys = np.interp(t, tc, pc[:, 0]), np.interp(t, tc, pc[:, 1])       # calm position at the same times
    imp = np.array([p[-1, 0], p[-1, 1]])
    d = imp - r_imp[0, :2]
    return dict(x=np.round(xs, 0).tolist(), y=np.round(ys, 0).tolist(), impact_calm=[float(r_imp[0, 0]), float(r_imp[0, 1])],
                shift_m=float(np.hypot(*d)), shift_down_m=float(d[0]), shift_cross_m=float(d[1]),
                offset_m=np.round(np.hypot(p[:, 0] - xs, p[:, 1] - ys), 0).tolist(), impact_time_calm=float(T[0]))
