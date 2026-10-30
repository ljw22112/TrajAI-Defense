"""Evaluate trained models and EKF/UKF baselines on every test split; write results/metrics.json and per-sample arrays.

Usage: python scripts/evaluate.py pinn_tf data_tf pinn_lstm ...
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajai import metrics  # noqa: E402
from trajai.features import L_POS  # noqa: E402
from trajai.model import PINNPredictor, physics_residual  # noqa: E402

TESTS = ["test_id_f05", "test_id_f10", "test_id_f15", "test_id_noise", "test_ood"]


def load_model(name):
    ck = torch.load(f"checkpoints/{name}.pt", weights_only=False)
    a = ck["args"]
    m = PINNPredictor(a["encoder"], hybrid=a["hybrid"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m, a


def predict(model, d, bs=250):
    """Returns grid predictions aligned to true times, impact points, ToT, sigma, ln beta, physics residual RMS."""
    out = {k: [] for k in ["grid", "ip", "tot", "sig", "lb", "res"]}
    n = d["feats"].shape[0]
    K = d["fut"].shape[1]
    tau_true = np.linspace(0, 1, K)
    for i in range(0, n, bs):
        sl = slice(i, i + bs)
        x = torch.from_numpy(d["feats"][sl]); pad = torch.from_numpy(d["pad"][sl]); g = torch.from_numpy(d["g"][sl])
        L = int((~pad).sum(1).max()); x, pad = x[:, :L], pad[:, :L]
        with torch.no_grad():
            c, h = model.context(x, pad, g)
            lb, dT_hat, log_sig = model.unpack_head(h)
            dT_true = torch.from_numpy(d["T"][sl] - d["t_c"][sl]).float()
            tau = torch.from_numpy(tau_true).float()[None] * (dT_true / dT_hat)[:, None]
            r = model.decode(c, tau.clamp(max=1.0))
            ip = model.decode(c, torch.ones(c.shape[0], 1))[:, 0]
        with torch.enable_grad():
            torch.manual_seed(0)
            z0 = torch.from_numpy(d["origin"][sl, 2]).float()
            _, res = physics_residual(model, c.detach(), lb.detach(), dT_hat.detach(), z0, n_col=64)
        origin = d["origin"][sl]
        out["grid"].append(r.numpy() * L_POS + origin[:, None])
        out["ip"].append(ip.numpy() * L_POS + origin)
        out["tot"].append(d["t_c"][sl] + dT_hat.numpy())
        out["sig"].append(np.exp(log_sig.numpy()) * 1000.0)
        out["lb"].append(lb.numpy())
        out["res"].append(np.sqrt((res.numpy() ** 2).sum(-1).mean(1)))
    return {k: np.concatenate(v) for k, v in out.items()}


def latency(model, d, n=200):
    times = []
    for i in range(n):
        k = int((~d["pad"][i]).sum())
        x = torch.from_numpy(d["feats"][i:i + 1, :k]); pad = torch.from_numpy(d["pad"][i:i + 1, :k]); g = torch.from_numpy(d["g"][i:i + 1])
        t0 = time.perf_counter()
        with torch.no_grad():
            c, h = model.context(x, pad, g)
            model.decode(c, torch.linspace(0, 1, 129)[None])
        times.append(time.perf_counter() - t0)
    t = np.array(times[10:]) * 1000
    return dict(p50_ms=float(np.median(t)), p99_ms=float(np.percentile(t, 99)))


def summarize(arr):
    return dict(median=float(np.median(arr)), mean=float(np.mean(arr)), p90=float(np.percentile(arr, 90)))


def main(names):
    torch.set_num_threads(1)
    path = "results/metrics.json"
    allm = json.load(open(path)) if os.path.exists(path) else {}
    # baselines
    for b in ["ekf", "ukf"]:
        allm.setdefault(b, {})
        for s in TESTS:
            f = f"results/baseline_{b}_{s}.npz"
            if os.path.exists(f):
                r = np.load(f)
                allm[b][s] = {k: summarize(r[k]) for k in ["ade", "miss", "tot", "cep"]}
                allm[b][s]["coverage"] = float(r["cov"].mean())
                allm[b][s]["nis"] = summarize(r["nis"])
    for name in names:
        model, a = load_model(name)
        allm[name] = {"args": a}
        for s in TESTS:
            d = dict(np.load(f"data/prep_{s}.npz"))
            p = predict(model, d)
            cep = metrics.gaussian_cep(p["sig"][:, 0], p["sig"][:, 1])
            res = dict(ade=metrics.ade(p["grid"], d["future"]), miss=metrics.ip_miss(p["ip"], d["impact"]),
                       tot=metrics.tot_error(p["tot"], d["T"]), cep=cep, phys_rms_g=p["res"],
                       lb_err=np.abs(p["lb"] - np.log(d["beta"])))
            allm[name][s] = {k: summarize(v) for k, v in res.items()}
            allm[name][s]["coverage"] = float(metrics.cep_coverage(p["ip"][:, :2], cep, d["impact"]).mean())
            np.savez_compressed(f"results/pred_{name}_{s}.npz", grid=p["grid"].astype(np.float32), ip=p["ip"], tot_pred=p["tot"],
                                sig=p["sig"], **{k: v for k, v in res.items()})
            print(f"{name:18s} {s:14s} ADE {allm[name][s]['ade']['median']:7.0f}  miss {allm[name][s]['miss']['median']:7.0f}  "
                  f"ToT {allm[name][s]['tot']['median']:5.2f}  CEP {allm[name][s]['cep']['median']:6.0f}  cov {allm[name][s]['coverage']:.2f}  "
                  f"res {allm[name][s]['phys_rms_g']['median']:.3f} g", flush=True)
        allm[name]["latency"] = latency(model, dict(np.load("data/prep_test_id_f10.npz")))
        print(name, "latency", allm[name]["latency"], flush=True)
    json.dump(allm, open(path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
