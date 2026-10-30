"""Scenario simulation: truth trajectory through the chosen atmosphere + polar radar measurements."""
from __future__ import annotations

import numpy as np

from trajai import physics, radar

DT_MEAS = 0.5


def simulate(h0, v0, gamma_deg, psi_deg, beta, obs_fraction, sigma_r, sigma_ang, seed, bank=None) -> dict:
    g, psi = np.radians(gamma_deg), np.radians(psi_deg)
    vh = v0 * np.cos(g)
    s0 = np.array([[0.0, 0.0, h0, vh * np.cos(psi), vh * np.sin(psi), v0 * np.sin(g)]])
    atm = bank[np.zeros(1, int)] if bank is not None else None
    times, pos, vel, T, r_imp = physics.simulate_ballistic(s0, np.array([beta]), dt=0.1, out_dt=DT_MEAS, atm=atm)
    pos = pos[0]
    valid = ~np.isnan(pos[:, 0])
    k_c = max(12, int(np.floor(obs_fraction * T[0] / DT_MEAS)))
    k_c = min(k_c, int(valid.sum()) - 2)
    rng = np.random.default_rng(seed)
    zp, zc = radar.measure(pos[: k_c + 1], rng, sigma_r=sigma_r, sigma_ang=sigma_ang)
    meas = [dict(k=k, t=k * DT_MEAS, range_m=float(zp[k, 0]), az=float(zp[k, 1]), el=float(zp[k, 2]),
                 x=float(zc[k, 0]), y=float(zc[k, 1]), z=float(zc[k, 2])) for k in range(k_c + 1)]
    idx = np.nonzero(valid)[0][::4]                       # truth every 2 s
    truth = dict(t=times[idx].round(2).tolist(), x=pos[idx, 0].round(1).tolist(), y=pos[idx, 1].round(1).tolist(),
                 z=pos[idx, 2].round(1).tolist())
    truth["t"].append(round(float(T[0]), 2)); truth["x"].append(round(float(r_imp[0, 0]), 1))
    truth["y"].append(round(float(r_imp[0, 1]), 1)); truth["z"].append(0.0)
    return dict(measurements=meas, t_c=k_c * DT_MEAS, impact_time=float(T[0]), impact_x=float(r_imp[0, 0]),
                impact_y=float(r_imp[0, 1]), apogee=float(np.nanmax(pos[:, 2])), truth=truth)
