"""Export a trained model to ONNX and benchmark single-track latency:
PyTorch (CPU, 1 thread) vs ONNX Runtime (CPU, 1 thread) vs the EKF + Monte Carlo baseline.

Usage: python scripts/export_onnx.py pinn_tf_hybrid
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajai import filters, radar  # noqa: E402
from trajai.model import PINNPredictor  # noqa: E402


class Deployable(torch.nn.Module):
    """Single-track inference graph: features (1, K, 12), global (1, 10) [, wx (1, 27)] -> trajectory (1, 129, 3), head (1, 4),
    context (1, C). K (track length) is a dynamic dimension. The context feeds the out-of-distribution guard (§9.8)."""

    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer("tau", torch.linspace(0, 1, 129)[None])

    def forward(self, feats, g, wx=None):
        pad = torch.zeros(feats.shape[0], feats.shape[1], dtype=torch.bool)
        c, h = self.m.context(feats, pad, g, wx)
        return self.m.decode(c, self.tau), h, c


def bench(fn, n=300, warm=20):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    t = np.array(t) * 1000
    return dict(p50_ms=float(np.median(t)), p99_ms=float(np.percentile(t, 99)))


def main(name):
    torch.set_num_threads(1)
    ck = torch.load(f"checkpoints/{name}.pt", weights_only=False)
    use_wx = bool(ck["args"].get("wx"))
    m = PINNPredictor(ck["args"]["encoder"], hybrid=ck["args"]["hybrid"], n_wx=27 if use_wx else 0); m.load_state_dict(ck["state"]); m.eval()
    dep = Deployable(m).eval()
    d = np.load("data/prep_test_id_f10.npz")
    k = int((~d["pad"][0]).sum())
    feats = torch.from_numpy(d["feats"][:1, :k]); g = torch.from_numpy(d["g"][:1])
    os.makedirs("export", exist_ok=True)
    path = f"export/{name}.onnx"
    K = torch.export.Dim("K", min=4, max=400)
    args = (feats, g) + ((torch.zeros(1, 27),) if use_wx else ())
    names = ["feats", "global"] + (["wx"] if use_wx else [])
    shapes = {"feats": {1: K}, "g": None, **({"wx": None} if use_wx else {})}
    torch.onnx.export(dep, args, path, input_names=names, output_names=["trajectory", "head", "context"], dynamo=True, dynamic_shapes=shapes)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    def feed(f):
        x = {"feats": f.numpy(), "global": g.numpy()}
        if use_wx:
            x["wx"] = np.zeros((1, 27), np.float32)
        return x
    max_diff_m = 0.0
    for L in (13, 26, 46, 90, 134):        # validate several track lengths (dynamic K)
        f = torch.from_numpy(np.ascontiguousarray(np.resize(d["feats"][0, :k], (L, 12))[None]))
        with torch.no_grad():
            ref = dep(f, g, *( (torch.zeros(1, 27),) if use_wx else ()))[0].numpy()
        max_diff_m = max(max_diff_m, float(np.abs(sess.run(None, feed(f))[0] - ref).max() * 1e5))
    res = {"model": name, "onnx_file": path, "onnx_size_kb": os.path.getsize(path) / 1024, "onnx_vs_torch_max_diff_m": max_diff_m}
    extra = (torch.zeros(1, 27),) if use_wx else ()
    with torch.no_grad():
        res["torch_cpu"] = bench(lambda: dep(feats, g, *extra))
    res["onnxruntime_cpu"] = bench(lambda: sess.run(None, feed(feats)))
    # EKF + Monte Carlo baseline for the same single track
    raw = np.load("data/test_id_f10.npz")
    nz = radar.NOISE_NOMINAL
    sl = slice(0, 1)

    def ekf_mc():
        x, P, _, _ = filters.ekf(raw["meas_pol"][sl].astype(float), raw["meas"][sl].astype(float), raw["n_meas"][sl], 0.5, nz["sigma_r"], nz["sigma_ang"])
        filters.monte_carlo_impact(x, P, raw["t_c"][sl], n_particles=100)

    def ekf_only():
        filters.ekf(raw["meas_pol"][sl].astype(float), raw["meas"][sl].astype(float), raw["n_meas"][sl], 0.5, nz["sigma_r"], nz["sigma_ang"])

    res["ekf_filter_only"] = bench(ekf_only, n=30, warm=3)
    res["ekf_plus_mc100"] = bench(ekf_mc, n=15, warm=2)
    print(json.dumps(res, indent=1))
    out = "results/latency.json" if name == "pinn_tf_hybrid" or len(sys.argv) < 3 else f"results/latency_{name}.json"
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pinn_tf_hybrid")
