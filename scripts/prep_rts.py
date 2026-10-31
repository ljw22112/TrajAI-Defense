"""Add RTS-smoothed track features ("feats_rts") to data/prep_<split>.npz (same layout as "feats").
Noise per track: the augmented splits store it; the high-noise test split uses NOISE_HIGH; others nominal.
Usage: python scripts/prep_rts.py [split ...]
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from trajai import radar  # noqa: E402
from trajai.features import track_features_rts  # noqa: E402

SPLITS = ["val_aug", "test_id_f05", "test_id_f10", "test_id_f15", "test_id_noise", "test_ood", "train_aug"]

for s in sys.argv[1:] or SPLITS:
    t0 = time.time()
    raw = dict(np.load(f"data/{s}.npz"))
    prep = dict(np.load(f"data/prep_{s}.npz"))
    if "feats_rts" in prep:
        print("skip", s); continue
    if "sigma" in raw and s.endswith("_aug"):
        sr, sa = raw["sigma"][:, 0], raw["sigma"][:, 1]
    else:
        nz = radar.NOISE_HIGH if s == "test_id_noise" else radar.NOISE_NOMINAL
        sr, sa = nz["sigma_r"], nz["sigma_ang"]
    f, mask = track_features_rts(raw["meas_pol"], raw["meas"], raw["n_meas"], sr, sa)
    assert (mask == prep["pad"]).all()
    prep["feats_rts"] = f
    np.savez_compressed(f"data/prep_{s}.npz", **prep)
    print(f"{s}: {time.time() - t0:.0f}s, nan={int(np.isnan(f).sum())}", flush=True)
