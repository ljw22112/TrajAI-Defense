"""Figures for the results section (figures/*.png) from results/metrics.json and results/latency.json."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

M = json.load(open("results/metrics.json"))
LAT = json.load(open("results/latency.json")) if os.path.exists("results/latency.json") else None
os.makedirs("figures", exist_ok=True)

TESTS = ["test_id_f05", "test_id_f10", "test_id_f15", "test_id_noise", "test_ood"]
TLAB = ["ID 5%", "ID 10%", "ID 15%", "ID high noise", "OOD 12%"]
MAIN = [("ekf", "EKF+MC"), ("ukf", "UKF+MC"), ("data_tf_v2", "Transformer (data only)"),
        ("pinn_tf_v2", "PINN-Transformer"), ("pinn_tf_hybrid_v2", "PINN-Transformer + EKF (hybrid)")]
COL = {"ekf": "#7d8a9c", "ukf": "#4b5b71", "data_tf": "#9fb7d8", "pinn_tf": "#cf5f14", "pinn_tf_v2": "#cf5f14", "pinn_lstm": "#e8a06a",
       "pinn_tf_hybrid": "#17806f", "pinn_tf_hybrid_v2": "#17806f", "data_tf_v2": "#9fb7d8", "data_tf_2k": "#9fb7d8", "pinn_tf_2k": "#cf5f14"}
INK, GRID = "#14233a", "#e6ebf1"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 160})


def val(m, t, k, stat="median"):
    try:
        v = M[m][t][k]
        return v[stat] if isinstance(v, dict) else v
    except KeyError:
        return np.nan


def grouped(ax, metric, methods, stat="median", scale=1.0, log=False):
    n = len(methods)
    w = 0.8 / n
    x = np.arange(len(TESTS))
    for i, (m, lab) in enumerate(methods):
        ax.bar(x + (i - (n - 1) / 2) * w, [val(m, t, metric, stat) * scale for t in TESTS], w, label=lab, color=COL[m])
    ax.set_xticks(x, TLAB)
    if log:
        ax.set_yscale("log")


avail = [(m, l) for m, l in MAIN if m in M]

# Fig A: IP miss
fig, ax = plt.subplots(figsize=(9, 3.6))
grouped(ax, "miss", avail, log=True)
ax.set_ylabel("IP miss, median [m]")
ax.legend(ncol=3, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.22))
fig.tight_layout(); fig.savefig("figures/ip_miss.png", facecolor="white"); plt.close(fig)

# Fig B: CEP coverage
fig, ax = plt.subplots(figsize=(9, 3.2))
n = len(avail); w = 0.8 / n; x = np.arange(len(TESTS))
for i, (m, lab) in enumerate(avail):
    ax.bar(x + (i - (n - 1) / 2) * w, [M[m][t]["coverage"] if t in M[m] else np.nan for t in TESTS], w, label=lab, color=COL[m])
ax.axhline(0.5, color=INK, lw=1.2, ls="--")
ax.text(len(TESTS) - 0.5, 0.52, "calibrated = 0.5", ha="right", va="bottom", fontsize=8.5, color=INK)
ax.set_xticks(x, TLAB); ax.set_ylim(0, 1); ax.set_ylabel("true IP inside CEP50")
fig.tight_layout(); fig.savefig("figures/cep_coverage.png", facecolor="white"); plt.close(fig)

# Fig C: physics residual of NN models
nn = [(m, l) for m, l in avail if m not in ("ekf", "ukf")] + [(m, l) for m, l in [("data_tf_2k", "Transformer data only (2k)"), ("pinn_tf_2k", "PINN-Transformer (2k)")] if m in M]
fig, ax = plt.subplots(figsize=(9, 3.0))
y = np.arange(len(nn))
for j, (t, lab, hatch) in enumerate([("test_id_f10", "ID 10%", ""), ("test_ood", "OOD", "///")]):
    vals = [val(m, t, "phys_rms_g") for m, _ in nn]
    ax.barh(y + (j - 0.5) * 0.38, vals, 0.36, color=[COL[m] for m, _ in nn], hatch=hatch, edgecolor="white", label=lab)
ax.set_yticks(y, [l for _, l in nn]); ax.invert_yaxis(); ax.set_xlim(left=0)
ax.set_xlabel("physics residual RMS, median [g]  (Eq. 7.2)")
ax.legend(loc="lower right", fontsize=8.5)
fig.tight_layout(); fig.savefig("figures/physics_residual.png", facecolor="white"); plt.close(fig)

# Fig D: latency
if LAT:
    items = [("PINN (PyTorch)", LAT["torch_cpu"]), ("PINN (ONNX Runtime)", LAT["onnxruntime_cpu"]),
             ("EKF filter only", LAT["ekf_filter_only"]), ("EKF + MC (100)", LAT["ekf_plus_mc100"])]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    yy = np.arange(len(items))
    ax.barh(yy, [v["p50_ms"] for _, v in items], color=["#17806f", "#17806f", "#7d8a9c", "#7d8a9c"])
    ax.errorbar([v["p50_ms"] for _, v in items], yy, xerr=[[0] * len(items), [v["p99_ms"] - v["p50_ms"] for _, v in items]],
                fmt="none", ecolor=INK, capsize=3, lw=1)
    for i, (_, v) in enumerate(items):
        ax.text(v["p99_ms"] * 1.15, i, f"{v['p50_ms']:.1f} ms", va="center", fontsize=9, color=INK)
    ax.axvline(50, color="#cf5f14", ls="--", lw=1.2); ax.text(52, len(items) - 0.6, "REQ-P-01: 50 ms", color="#cf5f14", fontsize=8.5)
    ax.set_yticks(yy, [k for k, _ in items]); ax.invert_yaxis(); ax.set_xscale("log")
    ax.set_xlabel("single-track latency, p50 (bar) / p99 (whisker) [ms], 1 CPU thread")
    fig.tight_layout(); fig.savefig("figures/latency.png", facecolor="white"); plt.close(fig)
print("figures written:", sorted(os.listdir("figures")))
