"""Weather-impact analysis: how much does the day's atmosphere move the trajectory, and how much does
weather-aware estimation reduce the prediction error?

For one fixed flight (burnout state, beta, sensor) the truth is flown through each selected profile.
Each track is then estimated twice from the same radar measurements:
  * EKF with the standard model (exponential density, no wind)  -> the error an operator gets without weather data
  * EKF with that profile                                         -> the error with weather-aware estimation
The trajectory shift is measured against USSA-1976 without wind, so it isolates the day's temperature and wind.
Everything is batched over profiles (one simulator call, two filter calls, two Monte Carlo calls).

Further studies in this module: weather-data latency (stale profiles), sensor sweep (noise x observed fraction)
and sensitivity to one user-set environment variable of a custom atmosphere.
"""
from __future__ import annotations

import numpy as np

from trajai import filters, metrics, physics, radar
from trajai import weather as wx

from .simulation import DT_MEAS

N_PARTICLES = 150


def _flight(p, n):
    g, psi = np.radians(p["gamma_deg"]), np.radians(p["psi_deg"])
    vh = p["v0"] * np.cos(g)
    s0 = np.array([[0.0, 0.0, p["h0"], vh * np.cos(psi), vh * np.sin(psi), p["v0"] * np.sin(g)]])
    return np.repeat(s0, n, 0), np.full(n, p["beta"])


def weather_impact(p: dict, bank, rows) -> dict:
    """p: flight/sensor parameters (ScenarioIn fields); bank: ProfileBank with len(rows) profiles."""
    K = len(rows)
    s0, beta = _flight(p, K)
    atm = bank[np.arange(K)]
    times, pos, _, T, r_imp = physics.simulate_ballistic(s0, beta, dt=0.1, out_dt=DT_MEAS, atm=atm)
    # reference: USSA-1976, no wind -> the shift isolates what the day's temperature and wind do
    _, _, _, T_std, r_std = physics.simulate_ballistic(s0[:1], beta[:1], dt=0.1, out_dt=DT_MEAS, atm=wx.ussa_bank()[np.zeros(1, int)])

    # radar track over the same observed fraction; the same noise draw for every profile (paired comparison)
    k_c = np.maximum(12, np.floor(p["obs_fraction"] * T / DT_MEAS).astype(int))
    kmax = int(k_c.max()) + 1
    sr, sa = p["sigma_r"], p["sigma_ang_mrad"] * 1e-3
    pol = np.zeros((K, kmax, 3)); car = np.zeros((K, kmax, 3))
    for i in range(K):
        zp, zc = radar.measure(pos[i, : k_c[i] + 1], np.random.default_rng(p.get("seed") or 0), sigma_r=sr, sigma_ang=sa)
        pol[i, : k_c[i] + 1], car[i, : k_c[i] + 1] = zp, zc
    n = k_c + 1
    t_c = k_c * DT_MEAS

    out = {}
    for key, a in (("std", None), ("wx", atm)):
        x, P, nis, _ = filters.ekf(pol, car, n, DT_MEAS, sr, sa, atm=a)
        mc = filters.monte_carlo_impact(x, P, t_c, n_particles=N_PARTICLES, seed=1, atm=a)
        miss = metrics.ip_miss(mc["mpi"], r_imp)
        out[key] = dict(miss=miss, cep=mc["cep"], inside=miss <= mc["cep"], tot_err=np.abs(mc["tot"] - T))

    shift = np.hypot(r_imp[:, 0] - r_std[0, 0], r_imp[:, 1] - r_std[0, 1])
    centre = r_imp[:, :2].mean(0)
    spread = np.hypot(r_imp[:, 0] - centre[0], r_imp[:, 1] - centre[1])     # day-to-day variability inside the window
    max_wind = np.array([r.max_wind for r in rows])
    items = []
    for i, r in enumerate(rows):
        items.append(dict(
            weather_id=r.id, valid_time=r.valid_time.isoformat(), max_wind=float(r.max_wind), max_wind_alt=float(r.max_wind_alt),
            impact_x=float(r_imp[i, 0]), impact_y=float(r_imp[i, 1]), impact_time=float(T[i]),
            shift_m=float(shift[i]), spread_m=float(spread[i]), dx_m=float(r_imp[i, 0] - r_std[0, 0]), dy_m=float(r_imp[i, 1] - r_std[0, 1]),
            dtot_s=float(T[i] - T_std[0]),
            miss_std=float(out["std"]["miss"][i]), miss_wx=float(out["wx"]["miss"][i]),
            cep_std=float(out["std"]["cep"][i]), cep_wx=float(out["wx"]["cep"][i]),
            inside_std=bool(out["std"]["inside"][i]), inside_wx=bool(out["wx"]["inside"][i]),
        ))
    ms, mw = out["std"]["miss"], out["wx"]["miss"]
    rank = lambda v: np.argsort(np.argsort(v))
    rho = float(np.corrcoef(rank(max_wind), rank(ms))[0, 1]) if K >= 3 and np.ptp(max_wind) > 0 else None
    worst = int(np.argmax(ms - mw))
    summary = dict(
        n=K, std_impact_x=float(r_std[0, 0]), std_impact_time=float(T_std[0]),
        median_shift_m=float(np.median(shift)), max_shift_m=float(shift.max()),
        spread_p50_m=float(np.median(spread)), spread_max_m=float(spread.max()),
        downrange_range_m=float(np.ptp(r_imp[:, 0])), crossrange_range_m=float(np.ptp(r_imp[:, 1])),
        median_miss_std=float(np.median(ms)), median_miss_wx=float(np.median(mw)),
        reduction=float(1 - np.median(mw) / np.median(ms)) if np.median(ms) > 0 else None,
        wx_better_frac=float((mw < ms).mean()),
        coverage_std=float(out["std"]["inside"].mean()), coverage_wx=float(out["wx"]["inside"].mean()),
        spearman_wind_vs_std_error=rho, largest_gain_time=rows[worst].valid_time.isoformat(),
        largest_gain_m=float(ms[worst] - mw[worst]),
    )
    return dict(summary=summary, items=items)


# ---------------------------------------------------------------- shared helpers
def _tracks(p, pos, T, K, sr, sa, seeds):
    """Radar tracks for K truths; seeds[i] picks the noise draw of track i."""
    k_c = np.maximum(12, np.floor(p["obs_fraction"] * T / DT_MEAS).astype(int))
    kmax = int(k_c.max()) + 1
    pol = np.zeros((K, kmax, 3)); car = np.zeros((K, kmax, 3))
    for i in range(K):
        zp, zc = radar.measure(pos[i, : k_c[i] + 1], np.random.default_rng(int(seeds[i])), sigma_r=sr, sigma_ang=sa)
        pol[i, : k_c[i] + 1], car[i, : k_c[i] + 1] = zp, zc
    return pol, car, k_c + 1, k_c * DT_MEAS


def _estimate(pol, car, n, t_c, sr, sa, atm, r_imp):
    x, P, _, _ = filters.ekf(pol, car, n, DT_MEAS, sr, sa, atm=atm)
    mc = filters.monte_carlo_impact(x, P, t_c, n_particles=N_PARTICLES, seed=1, atm=atm)
    miss = metrics.ip_miss(mc["mpi"], r_imp)
    return miss, mc["cep"], miss <= mc["cep"]


# ---------------------------------------------------------------- 1. weather-data latency
LAGS_H = (0, 6, 12, 24)


def latency(p: dict, bank, rows, lag_index: dict) -> dict:
    """bank holds the target profiles (first K) and the lagged ones; lag_index[(i, lag)] -> bank index.

    The truth flies through profile i; the EKF is given the profile valid `lag` hours earlier, i.e. the newest data
    an operator would hold if the feed were that stale. lag = None means no weather data (standard model).
    """
    K = len(rows)
    s0, beta = _flight(p, K)
    atm = bank[np.arange(K)]
    _, pos, _, T, r_imp = physics.simulate_ballistic(s0, beta, dt=0.1, out_dt=DT_MEAS, atm=atm)
    sr, sa = p["sigma_r"], p["sigma_ang_mrad"] * 1e-3
    pol, car, n, t_c = _tracks(p, pos, T, K, sr, sa, np.full(K, p.get("seed") or 0))
    res = {}
    for lag in (*LAGS_H, None):
        a = None if lag is None else bank[np.array([lag_index[(i, lag)] for i in range(K)])]
        res["std" if lag is None else str(lag)] = _estimate(pol, car, n, t_c, sr, sa, a, r_imp)
    # a good standard atmosphere without any dated data: separates "use a realistic atmosphere" from "use fresh data"
    res["ussa"] = _estimate(pol, car, n, t_c, sr, sa, wx.ussa_bank()[np.zeros(K, int)], r_imp)
    keys = [str(l) for l in LAGS_H] + ["ussa", "std"]
    curve = [dict(lag_h=(None if k in ("std", "ussa") else int(k)), ref=(k if k in ("std", "ussa") else None),
                  median_miss=float(np.median(res[k][0])), p90_miss=float(np.percentile(res[k][0], 90)),
                  coverage=float(res[k][2].mean())) for k in keys]
    items = [dict(valid_time=r.valid_time.isoformat(), max_wind=float(r.max_wind),
                  **{f"miss_{k}": float(res[k][0][i]) for k in keys}) for i, r in enumerate(rows)]
    fresh, std, ussa = curve[0]["median_miss"], curve[-1]["median_miss"], curve[-2]["median_miss"]
    # longest lag whose data still keeps at least half of the benefit of fresh data over USSA-1976 (no dated data)
    lagged = [c for c in curve if c["lag_h"] is not None]
    half = [c["lag_h"] for c in lagged if c["median_miss"] <= fresh + 0.5 * (ussa - fresh)] if ussa > fresh else []
    summary = dict(n=K, curve=curve, fresh_miss=fresh, std_miss=std, ussa_miss=ussa,
                   max_lag_half_benefit_h=max(half) if half else None)
    return dict(summary=summary, items=items)


# ---------------------------------------------------------------- 2. sensor sweep
NOISE_MULT = (0.5, 1.0, 2.0, 4.0)
OBS_FRAC = (0.05, 0.08, 0.12, 0.2)
N_REP = 16


def sensor_sweep(p: dict, atm_bank) -> dict:
    """Median IP miss and CEP coverage of a matched-atmosphere EKF over radar noise x observed fraction.

    One truth (in the chosen atmosphere), N_REP independent noise draws per cell; the filter knows the true noise level
    and the atmosphere, so the grid isolates what the sensor itself contributes.
    """
    s0, beta = _flight(p, 1)
    atm1 = atm_bank[np.zeros(1, int)] if atm_bank is not None else None
    _, pos, _, T, r_imp = physics.simulate_ballistic(s0, beta, dt=0.1, out_dt=DT_MEAS, atm=atm1)
    cells = []
    for m in NOISE_MULT:
        sr, sa = p["sigma_r"] * m, p["sigma_ang_mrad"] * 1e-3 * m
        for f in OBS_FRAC:
            q = {**p, "obs_fraction": f}
            K = N_REP
            posK, TK, rK = np.repeat(pos, K, 0), np.repeat(T, K), np.repeat(r_imp, K, 0)
            pol, car, n, t_c = _tracks(q, posK, TK, K, sr, sa, np.arange(K) + 1000 * int(m * 10) + int(f * 100))
            atmK = atm_bank[np.zeros(K, int)] if atm_bank is not None else None
            miss, cep, inside = _estimate(pol, car, n, t_c, sr, sa, atmK, rK)
            cells.append(dict(noise_mult=m, sigma_r=sr, sigma_ang_mrad=sa * 1e3, obs_fraction=f, track_s=float(t_c[0]),
                              median_miss=float(np.median(miss)), p90_miss=float(np.percentile(miss, 90)),
                              median_cep=float(np.median(cep)), coverage=float(inside.mean())))
    base = next(c for c in cells if c["noise_mult"] == 1.0 and abs(c["obs_fraction"] - min(OBS_FRAC, key=lambda f: abs(f - p["obs_fraction"]))) < 1e-9)
    target = 0.5 * base["median_miss"]
    hits = [c for c in cells if c["median_miss"] <= target]
    cheapest = min(hits, key=lambda c: (c["obs_fraction"], -c["noise_mult"])) if hits else None
    summary = dict(n=len(cells), reps=N_REP, impact_x=float(r_imp[0, 0]), flight_time=float(T[0]),
                   baseline=base, halve_error_with=cheapest)
    return dict(summary=summary, items=cells)


# ---------------------------------------------------------------- environment sensitivity (user-set variables)
SENS_PARAMS = {
    "t_sfc_c": ("지상 기온", "°C", (-30, -15, 0, 15, 30, 45)),
    "rh_pct": ("상대습도", "%", (0, 25, 50, 75, 100)),
    "p_sfc_hpa": ("지상 기압", "hPa", (960, 990, 1013.25, 1030, 1050)),
    "jet_ms": ("제트기류 최대풍속", "m/s", (0, 20, 40, 60, 80, 100)),
    "jet_dir": ("제트기류 방향(불어오는 쪽)", "°", (0, 90, 180, 270)),
    "wind_sfc_ms": ("지상풍", "m/s", (0, 10, 20, 30)),
    "lapse_k_per_km": ("기온감률", "K/km", (4.5, 5.5, 6.5, 7.5, 8.5)),
    "condition": ("날씨", "", ("clear", "fog", "rain", "snow", "heavy_rain")),
}


def sensitivity(p: dict, base: dict, param: str, values) -> dict:
    """Vary one variable of a custom atmosphere; everything else fixed at `base`.

    For each value: where the round lands (vs the base atmosphere) and the EKF error with and without that atmosphere.
    'condition' changes only the radar noise (precipitation barely changes drag), so it shows the sensor side of weather.
    """
    from . import weather as wsvc                 # local import: avoid a cycle at module load
    K = len(values)
    profs = [wsvc._prof(wx.custom_levels({**base, param: v}), str(v)) for v in values]
    bank = wx.build_grid(profs + [wsvc._prof(wx.custom_levels(base), "base")])
    s0, beta = _flight(p, K + 1)
    atm = bank[np.arange(K + 1)]
    _, pos, _, T, r_imp = physics.simulate_ballistic(s0, beta, dt=0.1, out_dt=DT_MEAS, atm=atm)
    items = []
    for i, v in enumerate(values):
        k = wx.CONDITION_NOISE[v if param == "condition" else base.get("condition", "clear")]
        sr, sa = p["sigma_r"] * k, p["sigma_ang_mrad"] * 1e-3 * k
        pol, car, n, t_c = _tracks(p, pos[i:i + 1].repeat(N_REP, 0), np.repeat(T[i], N_REP), N_REP, sr, sa, np.arange(N_REP) + 50)
        rK = np.repeat(r_imp[i:i + 1], N_REP, 0)
        m_std, _, in_std = _estimate(pol, car, n, t_c, sr, sa, None, rK)
        m_wx, _, in_wx = _estimate(pol, car, n, t_c, sr, sa, bank[np.full(N_REP, i)], rK)
        items.append(dict(value=v, noise_factor=k, rho_sfc=float(bank.rho[i, 0]), impact_x=float(r_imp[i, 0]), impact_y=float(r_imp[i, 1]),
                          dx_m=float(r_imp[i, 0] - r_imp[K, 0]), dy_m=float(r_imp[i, 1] - r_imp[K, 1]),
                          shift_m=float(np.hypot(r_imp[i, 0] - r_imp[K, 0], r_imp[i, 1] - r_imp[K, 1])),
                          impact_time=float(T[i]), miss_std=float(np.median(m_std)), miss_wx=float(np.median(m_wx)),
                          coverage_wx=float(in_wx.mean())))
    label, unit, _ = SENS_PARAMS[param]
    shifts = [it["shift_m"] for it in items]
    summary = dict(n=K, param=param, label=label, unit=unit, base=base, reps=N_REP,
                   max_shift_m=float(max(shifts)), range_downrange_m=float(np.ptp([it["impact_x"] for it in items])),
                   range_miss_wx=float(np.ptp([it["miss_wx"] for it in items])),
                   range_miss_std=float(np.ptp([it["miss_std"] for it in items])))
    return dict(summary=summary, items=items)
