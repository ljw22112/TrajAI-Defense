"""#8 Kalman/PINN coupling, the two halves the report had left as "planned".

(a) PINN -> EKF feedback: the track-only PINN (pinn_tf_v2) predicts ln beta from the observed track; that prediction,
    with a width fitted on the validation split, replaces the broad default prior of the EKF (ln 2450 +- 1.0).
    The EKF + Monte Carlo pipeline is otherwise unchanged. Evaluated on every test split, paired with the default EKF.
(b) RTS smoothing: velocity/acceleration error of the network's Savitzky-Golay features vs an EKF-RTS smoother,
    measured on freshly simulated tracks where the true velocity at each measurement time is known.

Writes results/feedback_rts.json.  Usage: python scripts/feedback_rts.py [model]
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
from evaluate import load_model  # noqa: E402
from trajai import features, filters, metrics, physics, radar  # noqa: E402

torch.set_num_threads(1)
DATA = "data"
W = 9                      # Savitzky-Golay window of trajai.features.track_features
SPLITS = ["test_id_f05", "test_id_f10", "test_id_f15", "test_id_noise", "test_ood"]


def ln_beta(model, a, split, bs=100):
    """ln beta from the PINN head only (no decoder / physics residual: keeps memory low next to training jobs)."""
    from evaluate import inputs_for
    d = dict(np.load(f"{DATA}/prep_{split}.npz"))
    G, WX = inputs_for(a, split, d)
    out = []
    with torch.no_grad():
        for i in range(0, len(d["T"]), bs):
            sl = slice(i, i + bs)
            x = torch.from_numpy(d["feats"][sl]); pad = torch.from_numpy(d["pad"][sl])
            L = int((~pad).sum(1).max())
            _, h = model.context(x[:, :L], pad[:, :L], torch.from_numpy(G[sl]), torch.from_numpy(WX[sl]))
            out.append(model.unpack_head(h)[0].numpy())       # ln beta (unpack_head de-normalises)
    return np.concatenate(out)


def run_ekf(d, nz, lb=None, slb=None):
    x, P, _, _ = filters.ekf(d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5,
                             nz["sigma_r"], nz["sigma_ang"], lb_prior=lb, slb_prior=slb)
    mc = filters.monte_carlo_impact(x, P, d["t_c"], n_particles=100, seed=1)
    miss = metrics.ip_miss(mc["mpi"], d["impact"])
    return dict(miss=miss, cov=(miss <= mc["cep"]), lb_err=np.abs(x[:, 6] - np.log(d["beta"])), cep=mc["cep"])


def summ(r):
    return dict(miss=float(np.median(r["miss"])), miss_p90=float(np.percentile(r["miss"], 90)), coverage=float(r["cov"].mean()),
                lb_err=float(np.median(r["lb_err"])), cep=float(np.median(r["cep"])))


def feedback(name):
    model, a = load_model(name)
    val = dict(np.load(f"{DATA}/val_aug.npz"))
    lb_val = ln_beta(model, a, "val_aug")
    resid = lb_val - np.log(val["beta"])
    slb = float(1.4826 * np.median(np.abs(resid - np.median(resid))))     # robust std of the learned prior
    import os
    part = "results/feedback_rts.partial.json"
    out = json.load(open(part)) if os.path.exists(part) else dict(model=name, prior_sd=slb, default_sd=filters.SLB0, splits={})
    rng = np.random.default_rng(0)
    for s in SPLITS:
        if s in out["splits"]:                       # resumable: the container can kill long runs
            continue
        d = dict(np.load(f"{DATA}/{s}.npz"))
        nz = radar.NOISE_HIGH if s == "test_id_noise" else radar.NOISE_NOMINAL    # same as scripts/prepare.py baselines
        lb = ln_beta(model, a, s)
        base, fb = run_ekf(d, nz), run_ekf(d, nz, lb, slb)
        diff = fb["miss"] - base["miss"]
        boot = np.array([np.median(fb["miss"][i]) - np.median(base["miss"][i]) for i in rng.integers(0, len(diff), (1000, len(diff)))])
        out["splits"][s] = dict(default=summ(base), feedback=summ(fb), win=float((diff < 0).mean()),
                                ci_diff=np.percentile(boot, [2.5, 97.5]).tolist())
        print(f"{s:14s} miss {out['splits'][s]['default']['miss']:7.0f} -> {out['splits'][s]['feedback']['miss']:7.0f}  "
              f"cov {out['splits'][s]['default']['coverage']:.2f} -> {out['splits'][s]['feedback']['coverage']:.2f}  "
              f"|dlnb| {out['splits'][s]['default']['lb_err']:.3f} -> {out['splits'][s]['feedback']['lb_err']:.3f}", flush=True)
        json.dump(out, open(part, "w"))
    return out


def rts_features(n=300, K=40, seed=3):
    rng = np.random.default_rng(seed)
    g = np.radians(rng.uniform(30, 60, n)); v0 = rng.uniform(900, 1900, n)
    s0 = np.stack([np.zeros(n), np.zeros(n), rng.uniform(8e3, 25e3, n), v0 * np.cos(g), np.zeros(n), v0 * np.sin(g)], 1)
    beta = np.exp(rng.uniform(np.log(600), np.log(10_000), n))
    _, pos, vel, _, _ = physics.simulate_ballistic(s0, beta, dt=0.1, out_dt=0.5)
    acc = np.gradient(vel, 0.5, axis=1)
    nz = radar.NOISE_NOMINAL
    pol = np.zeros((n, K, 3)); car = np.zeros((n, K, 3))
    for i in range(n):
        pol[i], car[i] = radar.measure(pos[i, :K], rng, sigma_r=nz["sigma_r"], sigma_ang=nz["sigma_ang"])
    nm = np.full(n, K)
    xs, _ = filters.rts_smooth(pol, car, nm, 0.5, nz["sigma_r"], nz["sigma_ang"])
    st = {}
    filters.ekf(pol, car, nm, 0.5, nz["sigma_r"], nz["sigma_ang"], store=st)
    k = np.arange(4, K - 4)
    # features.track_features stores velocity / acceleration normalised; recover SI from the raw SG derivative
    from scipy.signal import savgol_filter
    v_sg = np.stack([savgol_filter(car[i], W, 2, deriv=1, delta=0.5, axis=0) for i in range(n)])
    a_sg = np.stack([savgol_filter(car[i], W, 2, deriv=2, delta=0.5, axis=0) for i in range(n)])
    a_rts = np.gradient(xs[:, :, 3:6], 0.5, axis=1)
    err = lambda e: float(np.median(np.linalg.norm(e[:, k], axis=-1)))
    return dict(n=n, steps=K, vel_err=dict(sg=err(v_sg - vel[:, :K]), ekf=err(st["xf"][:, :, 3:6] - vel[:, :K]),
                                           rts=err(xs[:, :, 3:6] - vel[:, :K])),
                acc_err=dict(sg=err(a_sg - acc[:, :K]), rts=err(a_rts - acc[:, :K])))


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "pinn_tf_v2"
    res = dict(feedback=feedback(name), rts=rts_features())
    print("RTS", json.dumps(res["rts"]))
    json.dump(res, open("results/feedback_rts.json", "w"), indent=1)
