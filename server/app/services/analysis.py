"""Weather-impact analysis: how much does the day's atmosphere move the trajectory, and how much does
weather-aware estimation reduce the prediction error?

For one fixed flight (burnout state, beta, sensor) the truth is flown through each selected profile.
Each track is then estimated twice from the same radar measurements:
  * EKF with the standard model (exponential density, no wind)  -> the error an operator gets without weather data
The trajectory shift is measured against USSA-1976 without wind, so it isolates the day's temperature and wind.
  * EKF with that profile                                         -> the error with weather-aware estimation
Everything is batched over profiles (one simulator call, two filter calls, two Monte Carlo calls).
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
