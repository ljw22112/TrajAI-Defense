"""#2 Model-mismatch test: truth flies with a different Cd(M) shape or a weak lift manoeuvre (trajai.mismatch),
every estimator keeps the nominal model. Same flight ranges, 12 % observation, nominal radar noise as test_id_f10.

Steps (each cached):  data  -> data/test_mm_{cd,lift}.npz
                      prep  -> data/prep_test_mm_*.npz (SG + RTS features, EKF global features, targets)
                      eval  -> EKF, UKF, EKF + PINN beta prior, networks; results/mismatch.json and pred files
Usage: python scripts/mismatch_experiment.py [data|prep|eval|all]
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from trajai import dataset, features, filters, metrics, radar  # noqa: E402

torch.set_num_threads(1)
SPLITS = {"test_mm_cd": dict(cd=True, lift=False), "test_mm_lift": dict(cd=False, lift=True)}
NETS = ["data_tf_v2", "pinn_tf_v2", "pinn_tf_v2_rts", "pinn_tf_hybrid_v2"]
NZ = radar.NOISE_NOMINAL


def make_data():
    for i, (s, kw) in enumerate(SPLITS.items()):
        if not os.path.exists(f"data/{s}.npz"):
            np.savez_compressed(f"data/{s}.npz", **dataset.generate(n=1000, seed=5050 + i, frac=0.12, mismatch=kw))
            print("generated", s, flush=True)


def prep():
    for s in SPLITS:
        if os.path.exists(f"data/prep_{s}.npz"):
            continue
        d = dict(np.load(f"data/{s}.npz"))
        feats, pad, origin = features.track_features(d["meas"], d["n_meas"])
        feats_rts, _ = features.track_features_rts(d["meas_pol"], d["meas"], d["n_meas"], NZ["sigma_r"], NZ["sigma_ang"])
        x, P, _, _ = filters.ekf(d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5, NZ["sigma_r"], NZ["sigma_ang"])
        fut, dT, lb = features.targets(d, origin)
        np.savez_compressed(f"data/prep_{s}.npz", feats=feats, feats_rts=feats_rts, pad=pad, origin=origin,
                            g=features.global_features(x, P, origin), fut=fut, dT=dT, lb=lb, t_c=d["t_c"], T=d["T"],
                            impact=d["impact"], future=d["future"], beta=d["beta"])
        print("prepared", s, flush=True)


def summ(miss, cep):
    return dict(miss=float(np.median(miss)), miss_p90=float(np.percentile(miss, 90)), coverage=float((miss <= cep).mean()))


def run_filters(s, d, lb_prior=None, prior_sd=None):
    out = {}
    for name, fn in (("ekf", filters.ekf), ("ukf", filters.ukf)):
        x, P, _, _ = fn(d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5, NZ["sigma_r"], NZ["sigma_ang"])
        mc = filters.monte_carlo_impact(x, P, d["t_c"], n_particles=100, seed=1)
        out[name] = (metrics.ip_miss(mc["mpi"], d["impact"]), mc["cep"])
    if lb_prior is not None:
        x, P, _, _ = filters.ekf(d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5, NZ["sigma_r"], NZ["sigma_ang"],
                                 lb_prior=lb_prior, slb_prior=prior_sd)
        mc = filters.monte_carlo_impact(x, P, d["t_c"], n_particles=100, seed=1)
        out["ekf_pinn"] = (metrics.ip_miss(mc["mpi"], d["impact"]), mc["cep"])
    return out


def evaluate_all():
    from evaluate import load_model, predict
    from feedback_rts import ln_beta
    prior_sd = json.load(open("results/feedback_rts.json"))["feedback"]["prior_sd"]
    res = {}
    for s in SPLITS:
        d = dict(np.load(f"data/{s}.npz"))
        pd_ = dict(np.load(f"data/prep_{s}.npz"))
        r = {}
        for name in NETS:
            model, a = load_model(name)
            p = predict(model, pd_, a, s)
            miss = metrics.ip_miss(p["ip"], d["impact"])
            cep = metrics.gaussian_cep(p["sig"][:, 0], p["sig"][:, 1])
            r[name] = (miss, cep)
            np.savez_compressed(f"results/pred_{name}_{s}.npz", miss=miss, cep=cep, ip=p["ip"], sig=p["sig"],
                                ade=metrics.ade(p["grid"], pd_["future"]), phys_rms_g=p["res"])
        tm, ta = load_model("pinn_tf_v2")
        lb = ln_beta(tm, ta, s)
        r.update(run_filters(s, d, lb, prior_sd))
        for k in ("ekf", "ukf", "ekf_pinn"):
            np.savez_compressed(f"results/baseline_{k}_{s}.npz", miss=r[k][0], cep=r[k][1])
        res[s] = {k: summ(*v) for k, v in r.items()}
        ref = r["ekf"][0]
        for k, (m, _) in r.items():
            res[s][k]["win_vs_ekf"] = float((m < ref).mean())
        print(s, {k: round(v["miss"]) for k, v in res[s].items()}, flush=True)
    # matched-model reference: the same estimators on test_id_f10 (no mismatch, same 12 %-like observation)
    json.dump(res, open("results/mismatch.json", "w"), indent=1)


def make_trainset():
    """Training data with model error: every track has a random Cd(M) shape and a random weak lift (both effects),
    per-track radar noise (augmentation as in train_aug). Prep: SG + RTS features, EKF features, targets."""
    for s, n, seed in (("train_mm", 20_000, 6060), ("val_mm", 2_000, 6061)):
        if not os.path.exists(f"data/{s}.npz"):
            np.savez_compressed(f"data/{s}.npz", **dataset.generate(n=n, seed=seed, frac=(0.08, 0.20), noise="mixed",
                                                                     mismatch=dict(cd=True, lift=True)))
            print("generated", s, flush=True)
        if os.path.exists(f"data/prep_{s}.npz"):
            continue
        d = dict(np.load(f"data/{s}.npz"))
        sr, sa = d["sigma"][:, 0], d["sigma"][:, 1]
        feats, pad, origin = features.track_features(d["meas"], d["n_meas"])
        feats_rts, _ = features.track_features_rts(d["meas_pol"], d["meas"], d["n_meas"], sr, sa)
        gs = []
        for i in range(0, len(sr), 2500):
            sl = slice(i, i + 2500)
            x, P, _, _ = filters.ekf(d["meas_pol"][sl].astype(float), d["meas"][sl].astype(float), d["n_meas"][sl], 0.5, sr[sl], sa[sl])
            gs.append(features.global_features(x, P, origin[sl]))
        fut, dT, lb = features.targets(d, origin)
        np.savez_compressed(f"data/prep_{s}.npz", feats=feats, feats_rts=feats_rts, pad=pad, origin=origin, g=np.concatenate(gs),
                            fut=fut, dT=dT, lb=lb, t_c=d["t_c"], T=d["T"], impact=d["impact"], future=d["future"], beta=d["beta"])
        print("prepared", s, flush=True)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("data", "all"):
        make_data()
    if step in ("prep", "all"):
        prep()
    if step in ("eval", "all"):
        evaluate_all()
    if step == "trainset":
        make_trainset()
