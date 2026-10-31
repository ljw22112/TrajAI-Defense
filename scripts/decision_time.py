"""#4 REQ-P-03: how much earlier can an engagement decision be made?

The same 500 tracks (ID ranges, nominal noise) are cut at growing observation fractions f = 5 ... 30 %.
At each cut each estimator gives an impact point and CEP50. A track is "decidable" at the first cut where
CEP50 <= R (R = 300 / 500 / 1000 m, the impact-point precision a defended-asset decision needs).
Decision-time gain = t_decide(EKF) - t_decide(EKF + PINN beta prior); the time left to impact at decision is
also reported, with the actual miss at the decision cut (CEP is only useful if it is honest there).

Usage: python scripts/decision_time.py   -> results/decision_time.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from trajai import dataset, features, filters, metrics, radar  # noqa: E402

torch.set_num_threads(2)
FRACS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30]
RADII = [300, 500, 1000]
N, SEED = 500, 7070
NZ = radar.NOISE_NOMINAL
CACHE = "data/decision"


def run_cut(f):
    path = f"{CACHE}/cut_{int(f * 1000):03d}.npz"
    if os.path.exists(path):
        return dict(np.load(path))
    from evaluate import load_model
    from feedback_rts import ln_beta
    d = dataset.generate(n=N, seed=SEED, frac=f)            # same seed -> same flights, same noise; only the cut moves
    mp, mc_ = d["meas_pol"].astype(float), d["meas"].astype(float)
    x, P, _, _ = filters.ekf(mp, mc_, d["n_meas"], 0.5, NZ["sigma_r"], NZ["sigma_ang"])
    mc = filters.monte_carlo_impact(x, P, d["t_c"], n_particles=100, seed=1)
    feats, pad, origin = features.track_features(d["meas"], d["n_meas"])
    split = f"dec_{int(f * 1000):03d}"
    np.savez_compressed(f"data/prep_{split}.npz", feats=feats, pad=pad, g=features.global_features(x, P, origin), T=d["T"])
    tm, ta = load_model("pinn_tf_v2")
    lb = ln_beta(tm, ta, split)
    os.remove(f"data/prep_{split}.npz")
    sd = json.load(open("results/feedback_rts.json"))["feedback"]["prior_sd"]
    x2, P2, _, _ = filters.ekf(mp, mc_, d["n_meas"], 0.5, NZ["sigma_r"], NZ["sigma_ang"], lb_prior=lb, slb_prior=sd)
    mc2 = filters.monte_carlo_impact(x2, P2, d["t_c"], n_particles=100, seed=1)
    out = dict(t_c=d["t_c"], T=d["T"], cep_ekf=mc["cep"], miss_ekf=metrics.ip_miss(mc["mpi"], d["impact"]),
               cep_fb=mc2["cep"], miss_fb=metrics.ip_miss(mc2["mpi"], d["impact"]))
    np.savez_compressed(path, **out)
    print(f"cut {f:.3f}: EKF miss {np.median(out['miss_ekf']):.0f} CEP {np.median(out['cep_ekf']):.0f} | "
          f"+prior miss {np.median(out['miss_fb']):.0f} CEP {np.median(out['cep_fb']):.0f}", flush=True)
    return out


def first_cut(cep, R):
    """index of the first cut with CEP <= R (len(FRACS) = never within the cuts)"""
    ok = cep <= R
    return np.where(ok.any(0), ok.argmax(0), len(FRACS))


def main():
    os.makedirs(CACHE, exist_ok=True)
    cuts = [run_cut(f) for f in FRACS]
    tc = np.stack([c["t_c"] for c in cuts]); T = cuts[0]["T"]
    res = {"n": N, "fracs": FRACS, "per_cut": [], "radii": {}}
    for f, c in zip(FRACS, cuts):
        res["per_cut"].append({k: float(np.median(c[k])) for k in ("cep_ekf", "miss_ekf", "cep_fb", "miss_fb")}
                              | dict(frac=f, t_c=float(np.median(c["t_c"])),
                                     cov_ekf=float((c["miss_ekf"] <= c["cep_ekf"]).mean()),
                                     cov_fb=float((c["miss_fb"] <= c["cep_fb"]).mean())))
    tc_pad = np.vstack([tc, T[None]])                         # "never" = decide at impact (no warning time)
    # raw CEPs are conservative here (coverage > 0.5), which hides any accuracy gain. Calibrated variant: one scale per
    # estimator, fitted on the even-indexed tracks (pooled over cuts) so that 50 % of misses fall inside; reported on all.
    cal = np.arange(N) % 2 == 0
    res["scale"] = {k: float(np.median(np.concatenate([c[f"miss_{k}"][cal] / c[f"cep_{k}"][cal] for c in cuts])))
                    for k in ("ekf", "fb")}
    for R, mode in [(R, m) for m in ("raw", "cal") for R in RADII]:
        r = {}
        for k in ("ekf", "fb"):
            sc = res["scale"][k] if mode == "cal" else 1.0
            cep = np.stack([c[f"cep_{k}"] * sc for c in cuts]); miss = np.stack([c[f"miss_{k}"] for c in cuts])
            i = first_cut(cep, R)
            t_dec = tc_pad[i, np.arange(N)]
            got = i < len(FRACS)
            m_at = np.where(got, miss[np.minimum(i, len(FRACS) - 1), np.arange(N)], np.nan)
            r[k] = dict(decided=float(got.mean()), t_decide=t_dec, left=T - t_dec,
                        miss_at=float(np.nanmedian(m_at)) if got.any() else None,
                        honest=float(np.nanmean(m_at <= R)) if got.any() else None)
        gain = r["ekf"]["t_decide"] - r["fb"]["t_decide"]
        res["radii"].setdefault(mode, {})[str(R)] = dict(
            **{f"{k}_{q}": r[k][q] for k in ("ekf", "fb") for q in ("decided", "miss_at", "honest")},
            **{f"{k}_left_median": float(np.median(r[k]["left"])) for k in ("ekf", "fb")},
            gain_median=float(np.median(gain)), gain_mean=float(np.mean(gain)),
            gain_p90=float(np.percentile(gain, 90)), earlier=float((gain > 0).mean()), later=float((gain < 0).mean()))
        g = res["radii"][mode][str(R)]
        print(f"{mode} R={R}: decided EKF {g['ekf_decided']:.2f} / +prior {g['fb_decided']:.2f}; time left {g['ekf_left_median']:.0f} -> "
              f"{g['fb_left_median']:.0f} s; gain mean {g['gain_mean']:.1f} s (p90 {g['gain_p90']:.1f}), earlier {g['earlier']:.2f}")
    json.dump(res, open("results/decision_time.json", "w"), indent=1)


if __name__ == "__main__":
    main()
