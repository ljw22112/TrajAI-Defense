"""Paired comparison of two models on the test splits (default: Bi-LSTM data only vs PINN).

Reads results/metrics.json and results/pred_<model>_<split>.npz, writes a JSON summary: per split, impact-point
miss (median and RMSE), physics residual, CEP coverage, plus a paired bootstrap / Wilcoxon test on the miss
distance (same procedure as site_stats.compare, Holm-corrected over the splits).
Usage: python scripts/lstm_compare.py [--a pinn_lstm_v2] [--b data_lstm_v2] [--out results/lstm_vs_pinn.json]
       (the summary keys stay "data" = b and "pinn" = a for compatibility)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from site_stats import compare, holm  # noqa: E402

SPLITS = ["test_id_f05", "test_id_f10", "test_id_f15", "test_id_noise", "test_ood"]


def rmse(s, data, pinn):
    """Impact-point RMSE [m] from the per-track miss distances (needs results/pred_* and baseline_* files)."""
    out = {}
    for key, f in (("rmse_data_m", f"results/pred_{data}_{s}.npz"), ("rmse_pinn_m", f"results/pred_{pinn}_{s}.npz"),
                   ("rmse_ekf_m", f"results/baseline_ekf_{s}.npz")):
        if os.path.exists(f):
            out[key] = float(np.sqrt(np.mean(np.load(f)["miss"] ** 2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="pinn_lstm_v2")
    ap.add_argument("--b", default="data_lstm_v2")
    ap.add_argument("--out", default="results/lstm_vs_pinn.json")
    a = ap.parse_args()
    PINN, DATA = a.a, a.b
    m = json.load(open("results/metrics.json"))
    rng = np.random.default_rng(7)
    rows = [r for r in (compare(PINN, DATA, s, rng) for s in SPLITS) if r]
    holm(rows)
    out = dict(models=dict(data=DATA, pinn=PINN), splits={}, tests=rows)
    for s in SPLITS:
        if s not in m.get(DATA, {}) or s not in m.get(PINN, {}):
            continue
        d, p = m[DATA][s], m[PINN][s]
        rd, rp = d["phys_rms_g"]["median"], p["phys_rms_g"]["median"]
        out["splits"][s] = dict(
            miss_data_m=d["miss"]["median"], miss_pinn_m=p["miss"]["median"],
            phys_data_g=rd, phys_pinn_g=rp, phys_reduction=1.0 - rp / rd if rd > 0 else None,
            cov_data=d["coverage"], cov_pinn=p["coverage"],
            miss_ekf_m=m.get("ekf", {}).get(s, {}).get("miss", {}).get("median"),
            **rmse(s, DATA, PINN))
    out["latency"] = {k: m[k].get("latency") for k in (DATA, PINN) if k in m}
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"{PINN} vs {DATA}")
    print(f"{'split':14s} {'miss b':>8s} {'miss a':>8s} {'rmse b':>8s} {'rmse a':>8s} {'phys b':>7s} {'phys a':>7s}  p_holm")
    ph = {r["split"]: r["p_holm"] for r in rows}
    for s, v in out["splits"].items():
        print(f"{s:14s} {v['miss_data_m']:8.0f} {v['miss_pinn_m']:8.0f} {v.get('rmse_data_m', float('nan')):8.0f} "
              f"{v.get('rmse_pinn_m', float('nan')):8.0f} {v['phys_data_g']:7.3f} {v['phys_pinn_g']:7.3f}  {ph.get(s, float('nan')):.3g}")


if __name__ == "__main__":
    main()
