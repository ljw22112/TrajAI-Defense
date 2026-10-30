"""Export a handful of test cases (truth, measurements, EKF and PINN predictions, threat assessment)
to results/samples.json for prototype/viewer.html.

Usage: python scripts/export_samples.py pinn_tf_hybrid
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajai import threat  # noqa: E402

CASES = [("test_id_f10", [3, 41, 77]), ("test_id_f05", [12]), ("test_ood", [5, 60])]


def km(a, nd=3):
    return np.round(np.asarray(a, float) / 1000.0, nd).tolist()


def main(model):
    out = []
    for split, idxs in CASES:
        raw = np.load(f"data/{split}.npz")
        ekf = np.load(f"results/baseline_ekf_{split}.npz")
        pr = np.load(f"results/pred_{model}_{split}.npz")
        for i in idxs:
            n = int(raw["n_meas"][i])
            t_c, T = float(raw["t_c"][i]), float(raw["T"][i])
            tau = np.linspace(0, 1, raw["future"].shape[1])
            times = t_c + tau * (T - t_c)
            grid = pr["grid"][i]
            th = threat.assess(pr["ip"][i, :2], pr["sig"][i], times, grid[:, 2])
            out.append(dict(
                split=split, index=int(i), t_c=t_c, T=T, beta=float(raw["beta"][i]),
                meas=km(raw["meas"][i, :n:2]), truth=km(raw["future"][i]),
                ekf=dict(grid=km(ekf["grid"][i]), ip=km(ekf["mpi"][i]), cep=float(ekf["cep"][i]), miss=float(ekf["miss"][i])),
                pinn=dict(grid=km(grid), ip=km(pr["ip"][i, :2]), cep=float(pr["cep"][i]), miss=float(pr["miss"][i]),
                          tot=float(pr["tot_pred"][i]), sigma=np.round(pr["sig"][i], 1).tolist()),
                impact=km(raw["impact"][i, :2]),
                threat=th,
            ))
    meta = dict(model=model, assets=threat.ASSETS, bands=threat.BANDS)
    json.dump(dict(meta=meta, cases=out), open("results/samples.json", "w"), ensure_ascii=False)
    print("wrote results/samples.json with", len(out), "cases")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pinn_tf_hybrid")
