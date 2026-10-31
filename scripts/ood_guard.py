"""#1 Knowing when not to trust the network: out-of-distribution scores and an EKF fallback.

Scores (higher = less like the training data), all fitted on the validation split only:
  state   Mahalanobis distance of the EKF posterior features g (position, velocity, ln beta, uncertainties)
  latent  Mahalanobis distance of the PINN encoder context c (served RTS-input model)
  spread  disagreement of the three hybrid seeds' impact points (deep-ensemble epistemic spread)
Detection quality: AUROC for in-distribution test tracks vs each shifted split (OOD envelope, model mismatch).
Guard: when the score exceeds the 95th percentile of validation scores, use EKF + MC instead of the network.

Usage: python scripts/ood_guard.py   -> results/ood.json, results/ood_guard_params.json (served by the console)
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from evaluate import load_model  # noqa: E402

torch.set_num_threads(1)
NET = "pinn_tf_v2_rts"
SEEDS = ["pinn_tf_hybrid_v2", "pinn_tf_hybrid_v2_s1", "pinn_tf_hybrid_v2_s2"]
ID = ["test_id_f10", "test_id_f15"]
SHIFTED = ["test_ood", "test_mm_cd", "test_mm_lift"]
Q = 0.95


def contexts(model, a, split, bs=100):
    d = np.load(f"data/prep_{split}.npz")
    key = a.get("feat_key", "feats")
    G = d["g"]
    out = []
    with torch.no_grad():
        for i in range(0, len(G), bs):
            x = torch.from_numpy(d[key][i:i + bs]); pad = torch.from_numpy(d["pad"][i:i + bs])
            L = int((~pad).sum(1).max())
            c, _ = model.context(x[:, :L], pad[:, :L], torch.from_numpy(G[i:i + bs]), torch.zeros(len(x[:, 0]), 0))
            out.append(c.numpy())
    return np.concatenate(out)


class Maha:
    def __init__(self, X, ridge=1e-3):
        self.mu = X.mean(0)
        C = np.cov(X - self.mu, rowvar=False)
        self.P = np.linalg.inv(C + ridge * np.trace(C) / C.shape[0] * np.eye(C.shape[0]))

    def __call__(self, X):
        D = X - self.mu
        return np.sqrt(np.einsum("ni,ij,nj->n", D, self.P, D))


def auroc(neg, pos):
    """P(score_pos > score_neg) via ranks (Mann-Whitney)."""
    s = np.r_[neg, pos]
    r = np.argsort(np.argsort(s)) + 1
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def spread(split):
    ips = []
    for m in SEEDS:
        f = f"results/pred_{m}_{split}.npz"
        if not os.path.exists(f):
            return None
        ips.append(np.load(f)["ip"][:, :2])
    ips = np.stack(ips)
    return np.linalg.norm(ips - ips.mean(0), axis=-1).mean(0)


def miss_of(name, split):
    for f in (f"results/pred_{name}_{split}.npz", f"results/baseline_{name}_{split}.npz"):
        if os.path.exists(f):
            return np.load(f)["miss"]
    return None


def main():
    model, a = load_model(NET)
    g_val = np.load("data/prep_val_aug.npz")["g"]
    c_val = contexts(model, a, "val_aug")
    m_state, m_lat = Maha(g_val), Maha(c_val)
    thr = {"state": float(np.quantile(m_state(g_val), Q)), "latent": float(np.quantile(m_lat(c_val), Q))}
    splits = [s for s in ID + SHIFTED if os.path.exists(f"data/prep_{s}.npz")]
    scores = {}
    for s in splits:
        g = np.load(f"data/prep_{s}.npz")["g"]
        scores[s] = {"state": m_state(g), "latent": m_lat(contexts(model, a, s))}
        sp = spread(s)
        if sp is not None:
            scores[s]["spread"] = sp
    # spread threshold from the in-distribution test tracks (no validation predictions are stored for the seeds)
    if all("spread" in scores[s] for s in ID):
        thr["spread"] = float(np.quantile(np.concatenate([scores[s]["spread"] for s in ID]), Q))
    idneg = {k: np.concatenate([scores[s][k] for s in ID]) for k in thr}
    out = {"threshold_quantile": Q, "thresholds": thr, "detect": {}, "guard": {}}
    for s in SHIFTED:
        if s not in scores:
            continue
        out["detect"][s] = {k: dict(auroc=auroc(idneg[k], scores[s][k]), flagged=float((scores[s][k] > thr[k]).mean()))
                            for k in thr if k in scores[s]}
    for s in splits:
        net, ekf = miss_of(NET, s), miss_of("ekf", s)
        if net is None or ekf is None:
            continue
        g = {}
        for k in thr:
            if k not in scores[s]:
                continue
            flag = scores[s][k] > thr[k]
            mixed = np.where(flag, ekf, net)
            g[k] = dict(flagged=float(flag.mean()), miss=float(np.median(mixed)), miss_p90=float(np.percentile(mixed, 90)))
        out["guard"][s] = dict(network=float(np.median(net)), network_p90=float(np.percentile(net, 90)),
                               ekf=float(np.median(ekf)), ekf_p90=float(np.percentile(ekf, 90)), **g)
    json.dump(out, open("results/ood.json", "w"), indent=1)
    # parameters for the console guard (EKF-state score: model-agnostic, needs only the posterior the server computes)
    json.dump(dict(state=dict(mu=m_state.mu.tolist(), P=m_state.P.tolist(), threshold=thr["state"]),
                   latent=dict(model=NET, mu=m_lat.mu.tolist(), P=m_lat.P.tolist(), threshold=thr["latent"]), quantile=Q),
              open("results/ood_guard_params.json", "w"))
    for s, r in out["detect"].items():
        print(s, {k: (round(v["auroc"], 3), round(v["flagged"], 2)) for k, v in r.items()})
    for s, r in out["guard"].items():
        print(s, {k: (round(v, 0) if isinstance(v, float) else {kk: round(vv, 2) for kk, vv in v.items()}) for k, v in r.items()})


if __name__ == "__main__":
    main()
