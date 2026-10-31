"""#2 + #1  Manoeuvre detection and a manoeuvre-aware filter (IMM), for tracks observed past apogee.

§9.9 cut every track at 12 % of the flight, before apogee, so a lift manoeuvre that starts after apogee was never in
the measurements and no estimator could see it. Here tracking continues to 55 % of the flight (capped at the 100 s
radar window, see dataset.K_MAX), so for most tracks the descent - and the lift - is partly observed.

Splits (truth only; every estimator keeps the nominal model), 600 tracks each, nominal radar noise:
  long_nom   nominal flight       long_lift  weak lift after apogee       long_cd  Cd(M) shape error
Estimators: EKF; IMM (ballistic + manoeuvre mode, trajai.filters.imm); "switch" = EKF unless the innovation test flags
the track, then IMM.  Detection: windowed NIS of the last 20 updates; threshold = 95th percentile on a separate set of
nominal tracks (long_nom_cal), so the false-alarm rate is fixed in advance.

Usage: python scripts/maneuver.py  -> results/maneuver.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from trajai import dataset, filters, metrics, radar  # noqa: E402

N, FRAC, W, Q = 600, 0.55, 20, 0.95
NZ = radar.NOISE_NOMINAL
SPLITS = {"long_nom_cal": (7170, None), "long_nom": (7171, None), "long_lift": (7172, dict(cd=False, lift=True)),
          "long_cd": (7173, dict(cd=True, lift=False))}
CACHE = "data/maneuver"


def run(split):
    path = f"{CACHE}/{split}.npz"
    if os.path.exists(path):
        return dict(np.load(path))
    seed, mm = SPLITS[split]
    d = dataset.generate(n=N, seed=seed, frac=FRAC, mismatch=mm)
    args = (d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5, NZ["sigma_r"], NZ["sigma_ang"])
    out = dict(t_c=d["t_c"], T=d["T"], past_apogee=d["state_c"][:, 5] < 0, n_meas=d["n_meas"])
    x, P, _, _ = filters.ekf(*args)
    mc = filters.monte_carlo_impact(x, P, d["t_c"], n_particles=100, seed=1)
    out.update(miss_ekf=metrics.ip_miss(mc["mpi"], d["impact"]), cep_ekf=mc["cep"])
    x, P, mu, nis = filters.imm(*args)
    mc = filters.monte_carlo_impact(x, P, d["t_c"], n_particles=100, seed=1)
    out.update(miss_imm=metrics.ip_miss(mc["mpi"], d["impact"]), cep_imm=mc["cep"])
    last = lambda h: np.array([np.nanmean(r[~np.isnan(r)][-W:]) for r in h])
    out.update(nis_w=last(nis), mu_w=last(mu))
    np.savez_compressed(path, **out)
    print(split, "EKF", round(np.median(out["miss_ekf"])), "IMM", round(np.median(out["miss_imm"])), flush=True)
    return out


def auroc(neg, pos):
    s = np.r_[neg, pos]
    r = np.argsort(np.argsort(s)) + 1
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def summ(miss, cep):
    return dict(miss=float(np.median(miss)), miss_p90=float(np.percentile(miss, 90)), coverage=float((miss <= cep).mean()),
                cep=float(np.median(cep)))


def robust_mc(x, P, t_c, ld_max, n_particles=100, seed=1):
    """EKF + MC where each particle also flies a random weak lift (L/D ~ U(0, ld_max), random sign, after apogee):
    the CEP then covers manoeuvres the measurements cannot reveal. ld_max is an assumed threat envelope."""
    from trajai import mismatch as mm
    from trajai import physics
    rng = np.random.default_rng(seed)
    n = len(x)
    Lc = np.linalg.cholesky(P + 1e-9 * np.eye(7))
    smp = x[:, None] + np.einsum("nij,npj->npi", Lc, rng.standard_normal((n, n_particles, 7)))
    smp[..., 2] = np.maximum(smp[..., 2], 1.0)
    beta = np.exp(np.clip(smp[..., 6], filters.LB_MIN, filters.LB_MAX))
    m = n * n_particles
    atm = mm.Mismatch(np.ones(m), np.ones(m), rng.uniform(0, ld_max, m), rng.choice([-1.0, 1.0], m))
    ri = physics.propagate_to_ground(smp[..., :6].reshape(-1, 6), beta.reshape(-1), dt=0.25, atm=atm)["r"].reshape(n, n_particles, 3)
    mpi = ri[..., :2].mean(1)
    return mpi, np.median(np.linalg.norm(ri[..., :2] - mpi[:, None], axis=-1), 1)


def robust():
    """Manoeuvre-hypothesis CEP on the 12 % splits of §9.9 (nominal ID 10 % and weak lift)."""
    out = {}
    for s in ("test_id_f10", "test_mm_lift", "test_mm_cd"):
        d = dict(np.load(f"data/{s}.npz"))
        x, P, _, _ = filters.ekf(d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5, NZ["sigma_r"], NZ["sigma_ang"])
        out[s] = {}
        for ld in (0.0, 0.05, 0.10):
            if ld == 0.0:
                mc = filters.monte_carlo_impact(x, P, d["t_c"], n_particles=100, seed=1); mpi, cep = mc["mpi"], mc["cep"]
            else:
                mpi, cep = robust_mc(x, P, d["t_c"], ld)
            out[s][f"{ld:.2f}"] = summ(metrics.ip_miss(mpi, d["impact"]), cep)
            print(s, ld, {k: round(v, 2) for k, v in out[s][f"{ld:.2f}"].items()}, flush=True)
    return out


def lift_accel(n=300):
    """Size of the lift acceleration inside the radar window (to the long cut) vs over the whole flight [m/s^2]."""
    from trajai import mismatch as mm
    from trajai import physics
    rng = np.random.default_rng(SPLITS["long_lift"][0])
    p = dataset.sample_params(n, rng, dataset.ID_RANGES)
    s0 = dataset.initial_state(p)
    atm = mm.sample(n, rng, cd=False, lift=True)
    _, pos, vel, T, _ = physics.simulate_ballistic(s0, p[:, 4], dt=0.1, out_dt=0.5, atm=atm)
    win, full = [], []
    for i in range(n):
        K = int(T[i] / 0.5); kc = min(int(FRAC * T[i] / 0.5), dataset.K_MAX - 1)
        dacc = np.linalg.norm(physics.ballistic_accel(pos[i, :K], vel[i, :K], p[i, 4], atm=atm[i:i + 1])
                              - physics.ballistic_accel(pos[i, :K], vel[i, :K], p[i, 4]), axis=-1)
        win.append(dacc[:kc].max()); full.append(dacc.max())
    return dict(window_median=float(np.median(win)), window_p90=float(np.percentile(win, 90)),
                flight_median=float(np.median(full)), flight_p90=float(np.percentile(full, 90)))


def main():
    os.makedirs(CACHE, exist_ok=True)
    R = {s: run(s) for s in SPLITS}
    thr = {k: float(np.quantile(R["long_nom_cal"][k], Q)) for k in ("nis_w", "mu_w")}
    res = {"n": N, "frac": FRAC, "window": W, "thresholds": thr, "splits": {}}
    for s in ("long_nom", "long_lift", "long_cd"):
        r = R[s]
        flag = r["nis_w"] > thr["nis_w"]
        sw_m = np.where(flag, r["miss_imm"], r["miss_ekf"]); sw_c = np.where(flag, r["cep_imm"], r["cep_ekf"])
        e = dict(past_apogee=float(r["past_apogee"].mean()), t_c=float(np.median(r["t_c"])), frac_obs=float(np.median(r["t_c"] / r["T"])),
                 ekf=summ(r["miss_ekf"], r["cep_ekf"]), imm=summ(r["miss_imm"], r["cep_imm"]), switch=summ(sw_m, sw_c),
                 flagged=float(flag.mean()), flagged_mu=float((r["mu_w"] > thr["mu_w"]).mean()))
        if s != "long_nom":
            e["auroc"] = {k: auroc(R["long_nom"][k], r[k]) for k in ("nis_w", "mu_w")}
            pa = r["past_apogee"]
            if pa.any():
                e["auroc_past_apogee"] = {k: auroc(R["long_nom"][k], r[k][pa]) for k in ("nis_w", "mu_w")}
        res["splits"][s] = e
        print(s, json.dumps({k: (round(v["miss"]), round(v["coverage"], 2)) if isinstance(v, dict) and "miss" in v else v
                             for k, v in e.items()}, ensure_ascii=False), flush=True)
    res["robust"] = robust()
    res["lift_accel"] = lift_accel()
    json.dump(res, open("results/maneuver.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:] == ["liftacc"]:
        r = json.load(open("results/maneuver.json")); r["lift_accel"] = lift_accel()
        json.dump(r, open("results/maneuver.json", "w"), indent=1); print(r["lift_accel"])
    else:
        main()
