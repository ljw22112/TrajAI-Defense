"""LSTM vs PINN comparison (same LSTM encoder, physics loss off / on).

Reads results/metrics.json and results/pred_{data,pinn}_lstm_v2_*.npz, writes results/lstm_vs_pinn.json:
per split, impact-point miss and physics residual for both models, plus a paired bootstrap / Wilcoxon test
on the miss distance (same procedure as site_stats.compare, Holm-corrected over the splits).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from site_stats import compare, holm  # noqa: E402

DATA, PINN = "data_lstm_v2", "pinn_lstm_v2"
SPLITS = ["test_id_f05", "test_id_f10", "test_id_f15", "test_id_noise", "test_ood"]


def main():
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
            miss_ekf_m=m.get("ekf", {}).get(s, {}).get("miss", {}).get("median"))
    out["latency"] = {k: m[k].get("latency") for k in (DATA, PINN) if k in m}
    json.dump(out, open("results/lstm_vs_pinn.json", "w"), indent=1, ensure_ascii=False)
    print(f"{'split':14s} {'miss data':>10s} {'miss pinn':>10s} {'phys data':>10s} {'phys pinn':>10s} {'-phys':>6s}  p_holm")
    ph = {r["split"]: r["p_holm"] for r in rows}
    for s, v in out["splits"].items():
        print(f"{s:14s} {v['miss_data_m']:10.0f} {v['miss_pinn_m']:10.0f} {v['phys_data_g']:10.3f} {v['phys_pinn_g']:10.3f} "
              f"{(v['phys_reduction'] or 0) * 100:5.0f}%  {ph.get(s, float('nan')):.3g}")


if __name__ == "__main__":
    main()
