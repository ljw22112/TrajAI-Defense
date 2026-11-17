"""Qualitative check of the drag / gravity / atmosphere model against a publicly observed re-entry (§9.16).

Data: Hayabusa sample return capsule (SRC), 13 June 2010, positions from photographic observations, Table 3 of
J. Borovička et al., "Photographic and radiometric observations of the HAYABUSA re-entry", PASJ 63 (2011) 1003, arXiv:1108.6006.
Stated precision of the coordinates ~100 m; times are relative; the last point is marked less reliable.

This is not a ballistic-missile track (12 km/s super-orbital entry of a blunt capsule, unknown drag coefficient), so it
tests the physics, not the estimators: the state at the first observed point and ln(beta) are fitted by least squares
to the first n_fit points, then the remaining points are predicted. For a 12 km/s arc over ~200 km the flat local frame
of the simulator is not adequate, so this check integrates the same drag law (physics.density, physics.cd_ratio,
ballistic coefficient beta) in an Earth-centred rotating frame (WGS84, spherical gravity, Coriolis and centrifugal).
Baselines: the same fit without drag (vacuum) and a constant-velocity extrapolation.
Writes results/realdata_hayabusa.json.  Usage: python scripts/realdata_hayabusa.py
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajai import physics  # noqa: E402

# Table 3 (SRC): t [s, relative], longitude [deg E], latitude [deg], height [km]; the first row ('beg') has no time
SRC = [(16.0, 133.0768, -29.6545, 64.71), (19.0, 133.4016, -29.7532, 59.60), (22.0, 133.7122, -29.8466, 54.81),
       (25.0, 134.0014, -29.9328, 50.43), (28.0, 134.2593, -30.0092, 46.58), (32.0, 134.5429, -30.0921, 42.39),
       (34.8, 134.6962, -30.1367, 40.09)]
A_WGS, F_WGS = 6378137.0, 1 / 298.257223563
E2 = F_WGS * (2 - F_WGS)
OMEGA = np.array([0.0, 0.0, 7.2921159e-5])
MU = 3.986004418e14


def geo2ecef(lon, lat, h):
    lon, lat = np.radians(lon), np.radians(lat)
    N = A_WGS / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.array([(N + h) * np.cos(lat) * np.cos(lon), (N + h) * np.cos(lat) * np.sin(lon), (N * (1 - E2) + h) * np.sin(lat)])


def height(r):
    """Geodetic height of an ECEF point (Bowring, two iterations)."""
    x, y, z = r
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2))
    for _ in range(3):
        N = A_WGS / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - E2 * N / (N + h)))
    return h


def deriv(s, beta, drag=True):
    r, v = s[:3], s[3:]
    acc = -MU * r / np.linalg.norm(r) ** 3 - 2 * np.cross(OMEGA, v) - np.cross(OMEGA, np.cross(OMEGA, r))
    if drag:
        h = height(r)
        V = np.linalg.norm(v)                       # air at rest in the rotating frame
        k = physics.density(h) * V * physics.cd_ratio(V / physics.speed_of_sound(h)) / (2 * beta)
        acc = acc - k * v
    return np.r_[v, acc]


def propagate(s0, beta, times, drag=True, dt=0.01):
    out, s, t = [], s0.copy(), times[0]
    for tt in times:
        while t < tt - 1e-9:
            h_ = min(dt, tt - t)
            k1 = deriv(s, beta, drag); k2 = deriv(s + 0.5 * h_ * k1, beta, drag)
            k3 = deriv(s + 0.5 * h_ * k2, beta, drag); k4 = deriv(s + h_ * k3, beta, drag)
            s = s + h_ / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h_
        out.append(s[:3].copy())
    return np.array(out)


def fit_and_predict(t, R, n_fit, drag=True):
    v0 = (R[1] - R[0]) / (t[1] - t[0])
    x0 = np.r_[R[0], v0, np.log(200.0)] if drag else np.r_[R[0], v0]

    def resid(x):
        beta = np.exp(x[6]) if drag else 1.0
        return (propagate(x[:6], beta, t[:n_fit], drag) - R[:n_fit]).ravel() / 100.0      # in units of the 100 m precision

    sol = least_squares(resid, x0, x_scale="jac", max_nfev=200)
    beta = float(np.exp(sol.x[6])) if drag else None
    pred = propagate(sol.x[:6], beta or 1.0, t, drag)
    err = np.linalg.norm(pred - R, axis=1)
    speed = np.linalg.norm(sol.x[3:6])
    return dict(beta=beta, fit_rms_m=float(np.sqrt(np.mean(err[:n_fit] ** 2))), pred_err_m=err[n_fit:].tolist(),
                speed_at_first_point_ms=float(speed))


def main():
    t = np.array([p[0] for p in SRC])
    R = np.array([geo2ecef(p[1], p[2], p[3] * 1000) for p in SRC])
    out = {"source": "Borovicka et al. 2011, PASJ 63, 1003 (arXiv:1108.6006), Table 3", "precision_m": 100,
           "times_s": t.tolist(), "heights_km": [p[3] for p in SRC], "fits": {}}
    for n_fit in (4, 5):
        drag = fit_and_predict(t, R, n_fit, True)
        vac = fit_and_predict(t, R, n_fit, False)
        v_cv = (R[n_fit - 1] - R[n_fit - 2]) / (t[n_fit - 1] - t[n_fit - 2])
        cv = np.linalg.norm(R[n_fit - 1] + v_cv * (t[n_fit:, None] - t[n_fit - 1]) - R[n_fit:], axis=1)
        out["fits"][str(n_fit)] = {"drag_model": drag, "vacuum": vac, "constant_velocity_err_m": cv.tolist(),
                                   "predicted_times_s": t[n_fit:].tolist()}
        print(f"fit on {n_fit} points (to t = {t[n_fit - 1]:.0f} s): beta {drag['beta']:.0f} kg/m2, fit RMS {drag['fit_rms_m']:.0f} m")
        for i, tt in enumerate(t[n_fit:]):
            print(f"   t = {tt:4.1f} s  drag model {drag['pred_err_m'][i]:7.0f} m   vacuum {vac['pred_err_m'][i]:7.0f} m   "
                  f"constant velocity {cv[i]:7.0f} m")
    json.dump(out, open("results/realdata_hayabusa.json", "w"), indent=1)


if __name__ == "__main__":
    main()
