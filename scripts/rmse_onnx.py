"""Impact-point RMSE of the served ONNX models on the test splits (per-track errors are not versioned, so the
leaderboard's RMSE column is recomputed from the exported models). Writes results/rmse_onnx.json.
Usage: python scripts/rmse_onnx.py [model ...]   (needs data/prep_<split>.npz with feats_rts for the RTS model)
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajai import features  # noqa: E402

SPLITS = ["test_id_f05", "test_id_f10", "test_id_f15", "test_id_noise", "test_ood"]
FEAT = {"pinn_tf_v2_rts": "feats_rts"}


def main(models):
    out = json.load(open("results/rmse_onnx.json")) if os.path.exists("results/rmse_onnx.json") else {}
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    for m in models:
        sess = ort.InferenceSession(f"export/{m}.onnx", so, providers=["CPUExecutionProvider"])
        out[m] = {}
        for s in SPLITS:
            d = dict(np.load(f"data/prep_{s}.npz"))
            key = FEAT.get(m, "feats")
            if key not in d:
                continue
            miss = []
            for i in range(len(d["T"])):
                k = int((~d["pad"][i]).sum())
                tr = sess.run(None, {"feats": d[key][i:i + 1, :k].astype(np.float32), "global": d["g"][i:i + 1].astype(np.float32)})[0]
                ip = tr[0, -1, :2] * features.L_POS + d["origin"][i, :2]
                miss.append(float(np.hypot(*(ip - d["impact"][i, :2]))))
            miss = np.array(miss)
            out[m][s] = dict(median_m=float(np.median(miss)), rmse_m=float(np.sqrt(np.mean(miss ** 2))))
            print(f"{m:20s} {s:14s} median {out[m][s]['median_m']:8.0f}  RMSE {out[m][s]['rmse_m']:8.0f}", flush=True)
    json.dump(out, open("results/rmse_onnx.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["pinn_tf_v2_rts", "pinn_tf_v2", "pinn_tf_hybrid_v2"])
