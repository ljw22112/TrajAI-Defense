"""Weather (date-selected atmosphere) experiment.

Truth: each test launch flies through the Open-Meteo profile of a date in 2026-10-01..28 plus a smooth
random perturbation (analysis error). Training launches use the previous year's profiles (2025-10..2026-09).

Step 1 (filters): EKF/UKF with different atmosphere models -> results/weather_filters.json
  exp     exponential density, no wind (the model used so far)
  ussa    USSA-1976 hydrostatic standard atmosphere, no wind
  clim    climatological mean of the previous year's profiles (no date selection)
  date    the selected date's API profile (what the date picker provides)
  oracle  the perturbed truth profile itself (upper bound, not available in practice)
Step 2 (NN data): train_wx / val_wx / test_wx splits + prep files with EKF-clim and EKF-date global features
and 27-dim weather features of the selected profile.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajai import dataset, features, filters, metrics, radar, weather  # noqa: E402

DATA = "data"


def summarize(arr):
    return dict(median=float(np.median(arr)), mean=float(np.mean(arr)), p90=float(np.percentile(arr, 90)))


def atm_models(banks, wx_idx, split):
    n = len(wx_idx)
    src = banks["test"] if split == "test" else banks["train"]
    truth = banks["test_truth"] if split == "test" else banks["train_truth"]
    return {
        "exp": None,
        "ussa": weather.ussa_bank()[np.zeros(n, int)],
        "clim": banks["train"].mean()[np.zeros(n, int)],
        "date": src[wx_idx],
        "oracle": truth[wx_idx],
    }


def run_filters(banks):
    d = dict(np.load(f"{DATA}/test_wx.npz"))
    nz = radar.NOISE_NOMINAL
    out = {}
    for name, atm in atm_models(banks, d["wx_idx"], "test").items():
        for fname, fn in [("ekf", filters.ekf), ("ukf", filters.ukf)]:
            if fname == "ukf" and name not in ("exp", "date"):
                continue
            t0 = time.time()
            x, P, nis, _ = fn(d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5, nz["sigma_r"], nz["sigma_ang"], atm=atm)
            mc = filters.monte_carlo_impact(x, P, d["t_c"], n_particles=100, seed=1, atm=atm)
            grid = filters.nominal_on_grid(x, d["t_c"], d["T"], atm=atm)
            key = f"{fname}_{name}"
            out[key] = dict(ade=summarize(metrics.ade(grid, d["future"])), miss=summarize(metrics.ip_miss(mc["mpi"], d["impact"])),
                            tot=summarize(metrics.tot_error(mc["tot"], d["T"])), cep=summarize(mc["cep"]),
                            coverage=float(metrics.cep_coverage(mc["mpi"], mc["cep"], d["impact"]).mean()),
                            nis=summarize(nis), lb_err=summarize(np.abs(x[:, 6] - np.log(d["beta"]))))
            print(f"{key:12s} {time.time()-t0:5.0f}s  ADE {out[key]['ade']['median']:7.0f}  miss {out[key]['miss']['median']:7.0f}  "
                  f"CEP {out[key]['cep']['median']:6.0f}  cov {out[key]['coverage']:.2f}  NIS {out[key]['nis']['median']:.2f}  "
                  f"|dlnb| {out[key]['lb_err']['median']:.3f}", flush=True)
    json.dump(out, open("results/weather_filters.json", "w"), indent=1)


def prep(name, banks, split_kind):
    d = dict(np.load(f"{DATA}/{name}.npz"))
    if "sigma" in d and name.startswith(("train", "val")):
        nz = dict(sigma_r=d["sigma"][:, 0], sigma_ang=d["sigma"][:, 1])
    else:
        nz = radar.NOISE_NOMINAL
    feats, pad, origin = features.track_features(d["meas"], d["n_meas"])
    models = atm_models(banks, d["wx_idx"], split_kind)
    gs = {}
    n = len(d["T"])
    kinds = ("clim", "date", "exp") if split_kind == "test" else ("clim", "date")
    for k in kinds:
        parts = []
        for i in range(0, n, 2500):          # chunked to bound memory
            sl = slice(i, i + 2500)
            sr = nz["sigma_r"][sl] if np.ndim(nz["sigma_r"]) else nz["sigma_r"]
            sa = nz["sigma_ang"][sl] if np.ndim(nz["sigma_ang"]) else nz["sigma_ang"]
            atm = None if models[k] is None else models[k][sl]
            x, P, _, _ = filters.ekf(d["meas_pol"][sl].astype(float), d["meas"][sl].astype(float), d["n_meas"][sl], 0.5, sr, sa, atm=atm)
            parts.append(features.global_features(x, P, origin[sl]))
        gs[k] = np.concatenate(parts)
    src = banks["test"] if split_kind == "test" else banks["train"]
    wx = src.features(d["wx_idx"])
    fut, dT, lb = features.targets(d, origin)
    np.savez_compressed(f"{DATA}/prep_{name}.npz", feats=feats, pad=pad, origin=origin, g=gs["clim"], g_date=gs["date"], **({"g_exp": gs["exp"]} if "exp" in gs else {}), wx=wx,
                        wx_idx=d["wx_idx"], fut=fut, dT=dT, lb=lb, t_c=d["t_c"], T=d["T"], impact=d["impact"], future=d["future"], beta=d["beta"])
    print("prepared", name, flush=True)


def main(step):
    banks = weather.load_banks()
    if step in ("all", "data"):
        for name, kw in [("test_wx", dict(n=1000, seed=4040, frac=0.12, bank=banks["test_truth"])),
                         ("train_wx", dict(n=20_000, seed=4041, frac=(0.08, 0.20), noise="mixed", bank=banks["train_truth"])),
                         ("val_wx", dict(n=2_000, seed=4042, frac=(0.08, 0.20), noise="mixed", bank=banks["train_truth"]))]:
            if not os.path.exists(f"{DATA}/{name}.npz"):
                t0 = time.time()
                np.savez_compressed(f"{DATA}/{name}.npz", **dataset.generate(ranges=dataset.ID_RANGES, **kw))
                print(f"generated {name} in {time.time()-t0:.0f}s", flush=True)
    if step in ("all", "filters"):
        run_filters(banks)
    if step in ("all", "prep"):
        for name, kind in [("test_wx", "test"), ("val_wx", "train"), ("train_wx", "train")]:
            if name == "test_wx" or not os.path.exists(f"{DATA}/prep_{name}.npz"):
                prep(name, banks, kind)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
