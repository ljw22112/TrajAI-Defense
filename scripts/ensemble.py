"""#9 Deep ensemble of the hybrid PINN (seeds 0, 1, 2) and #7 seed variance.

Ensemble: mean of the members' trajectories and impact points; impact uncertainty is the mixture variance
  sigma^2 = mean_i(sigma_i^2) + var_i(ip_i)      (aleatoric + epistemic, Lakshminarayanan et al. 2017)
CEP50 / CEP90 are the 50 % / 90 % radii of the resulting 2-D Gaussian. The same radii are computed for the single
seed-0 model, so the 90 % coverage shows whether the ensemble repairs the heavy error tail (§9.5).

Seed variance: median IP miss of every model trained with several seeds, mean +- sd over seeds, per split.
Writes results/metrics.json["ens_hybrid_v2"] and results/ensemble.json.
Usage: python scripts/ensemble.py
"""
import json
import os

import numpy as np

from trajai import metrics

R = "results"
SPLITS = ["test_id_f05", "test_id_f10", "test_id_f15", "test_id_noise", "test_ood", "test_wx"]
MEMBERS = ["pinn_tf_hybrid_v2", "pinn_tf_hybrid_v2_s1", "pinn_tf_hybrid_v2_s2"]
SEED_GROUPS = {"PINN + EKF 하이브리드": MEMBERS, "PINN-Transformer": ["pinn_tf_v2", "pinn_tf_v2_s1"],
               "Transformer (data only)": ["data_tf_v2", "data_tf_v2_s1"]}


def gaussian_radius(sx, sy, q, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, 2))
    r = np.sqrt((e[None, :, 0] * sx[:, None]) ** 2 + (e[None, :, 1] * sy[:, None]) ** 2)
    return np.quantile(r, q, axis=1)


def summarize(a):
    return dict(median=float(np.median(a)), mean=float(np.mean(a)), p90=float(np.percentile(a, 90)))


def evaluate_split(s):
    preds = [dict(np.load(f"{R}/pred_{m}_{s}.npz")) for m in MEMBERS if os.path.exists(f"{R}/pred_{m}_{s}.npz")]
    if len(preds) < len(MEMBERS):
        return None
    d = dict(np.load(f"data/prep_{s}.npz"))
    ip_true = d["impact"][:, :2]
    ips = np.stack([p["ip"][:, :2] for p in preds])           # (M, N, 2)
    sig = np.stack([p["sig"] for p in preds])                  # (M, N, 2)
    grid = np.mean([p["grid"] for p in preds], 0)
    ip = ips.mean(0)
    s_mix = np.sqrt((sig ** 2).mean(0) + ips.var(0))
    out = {}
    for name, (ipx, sg) in {"ensemble": (ip, s_mix), "single": (ips[0], sig[0])}.items():
        miss = np.linalg.norm(ipx - ip_true, axis=1)
        r50, r90 = gaussian_radius(sg[:, 0], sg[:, 1], 0.5), gaussian_radius(sg[:, 0], sg[:, 1], 0.9, seed=1)
        out[name] = dict(miss=summarize(miss), cep=summarize(r50), coverage=float((miss <= r50).mean()),
                         coverage90=float((miss <= r90).mean()))
    ade = metrics.ade(grid, d["future"])
    tot = np.abs(np.mean([p["tot_pred"] for p in preds], 0) - d["T"])
    out["ensemble"].update(ade=summarize(ade), tot=summarize(tot))
    out["epistemic_share"] = float(np.median(ips.var(0).sum(1) / (s_mix ** 2).sum(1)))
    return out


def seed_variance():
    M = json.load(open(f"{R}/metrics.json"))
    res = {}
    for label, names in SEED_GROUPS.items():
        have = [n for n in names if n in M]
        if len(have) < 2:
            continue
        res[label] = dict(models=have, splits={s: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), values=v)
                                                for s in SPLITS[:5] if len(v := [M[n][s]["miss"]["median"] for n in have if s in M[n]]) >= 2})
    pl = {}
    for s in SPLITS[:5]:
        pairs = [(p, d) for p, d in (("pinn_tf_v2", "data_tf_v2"), ("pinn_tf_v2_s1", "data_tf_v2_s1")) if p in M and d in M]
        if len(pairs) >= 2:
            diffs = [M[p][s]["miss"]["median"] - M[d][s]["miss"]["median"] for p, d in pairs]
            pl[s] = dict(diffs=diffs, consistent=bool(all(x < 0 for x in diffs) or all(x > 0 for x in diffs)))
    return dict(groups=res, physics_loss_by_seed=pl)


if __name__ == "__main__":
    ens = {s: r for s in SPLITS if (r := evaluate_split(s))}
    sv = seed_variance()
    json.dump(dict(ensemble=ens, seeds=sv), open(f"{R}/ensemble.json", "w"), indent=1)
    if ens:
        M = json.load(open(f"{R}/metrics.json"))
        M["ens_hybrid_v2"] = {s: dict(ade=r["ensemble"]["ade"], miss=r["ensemble"]["miss"], tot=r["ensemble"]["tot"],
                                      cep=r["ensemble"]["cep"], coverage=r["ensemble"]["coverage"]) for s, r in ens.items()}
        json.dump(M, open(f"{R}/metrics.json", "w"), indent=1)
    for s, r in ens.items():
        print(f"{s:14s} miss {r['single']['miss']['median']:7.0f} -> {r['ensemble']['miss']['median']:7.0f}  "
              f"cov50 {r['single']['coverage']:.2f} -> {r['ensemble']['coverage']:.2f}  cov90 {r['single']['coverage90']:.2f} -> {r['ensemble']['coverage90']:.2f}"
              f"  epistemic {r['epistemic_share']:.2f}")
    print(json.dumps(sv, ensure_ascii=False)[:800])
