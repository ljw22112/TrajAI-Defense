"""Multi-radar fusion (§9.12): the same test tracks observed by a second, notional radar, fused in the EKF.

The primary radar's measurements are regenerated with the dataset generator's random sequence, so they are identical
to data/<split>.npz (checked: the single-radar EKF reproduces the reported baseline). A second radar at RADAR_B
measures the same true positions at the same instants with its own noise. Compared per split:
  A only (baseline) and A + B fused (sequential EKF updates). Impact point and CEP from the usual Monte Carlo.
Writes results/multiradar.json.  Usage: python scripts/multiradar.py [split ...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajai import filters, metrics, physics, radar  # noqa: E402
from trajai.dataset import DT_MEAS, ID_RANGES, K_MAX, initial_state, sample_params  # noqa: E402

SEED = 2026
SPLITS = {"test_id_f05": dict(seed=SEED + 6, frac=0.05, noise=radar.NOISE_NOMINAL),
          "test_id_f10": dict(seed=SEED + 2, frac=0.10, noise=radar.NOISE_NOMINAL),
          "test_id_noise": dict(seed=SEED + 4, frac=0.12, noise=radar.NOISE_HIGH)}
RADAR_B = np.array([90_000.0, -60_000.0, 0.0])     # notional second site, south-east of the impact area


def tracks(n, seed, frac, noise):
    """Same random sequence as trajai.dataset.generate (single chunk, no weather, fixed noise)."""
    rng = np.random.default_rng(seed)
    p = sample_params(n, rng, ID_RANGES)
    s0 = initial_state(p)
    times, pos, vel, T, r_imp = physics.simulate_ballistic(s0, p[:, 4], dt=0.1, out_dt=DT_MEAS)
    f = np.full(n, frac)
    k_c = np.maximum(np.minimum(np.floor(f * T / DT_MEAS).astype(int), K_MAX - 1), 12)
    sr, sa = noise["sigma_r"], noise["sigma_ang"]
    A_pol, A_c, B_pol = np.zeros((n, K_MAX, 3)), np.zeros((n, K_MAX, 3)), np.zeros((n, K_MAX, 3))
    for i in range(n):
        k = k_c[i] + 1
        zp, zc = radar.measure(pos[i, :k], rng, sigma_r=sr, sigma_ang=sa)
        A_pol[i, :k], A_c[i, :k] = zp, zc
    rng_b = np.random.default_rng(seed + 1000)
    for i in range(n):
        k = k_c[i] + 1
        zp, _ = radar.measure(pos[i, :k], rng_b, sigma_r=sr, sigma_ang=sa, site=RADAR_B)
        B_pol[i, :k] = zp
    return dict(A_pol=A_pol, A_c=A_c, B_pol=B_pol, n_meas=k_c + 1, t_c=k_c * DT_MEAS, T=T, impact=r_imp, sr=sr, sa=sa)


def run(name, d, x, P):
    mc = filters.monte_carlo_impact(x, P, d["t_c"], n_particles=100, seed=1)
    miss = metrics.ip_miss(mc["mpi"], d["impact"])
    cov = metrics.cep_coverage(mc["mpi"], mc["cep"], d["impact"]).mean()
    return dict(miss_median_m=float(np.median(miss)), miss_rmse_m=float(np.sqrt(np.mean(miss ** 2))),
                miss_p90_m=float(np.percentile(miss, 90)), cep_median_m=float(np.median(mc["cep"])), coverage=float(cov),
                sd_ln_beta=float(np.median(np.sqrt(P[:, 6, 6])))), miss


def main(splits):
    out = {"radar_a": radar.RADAR_SITE.tolist(), "radar_b": RADAR_B.tolist(), "splits": {}}
    for s in splits:
        cfg = SPLITS[s]
        d = tracks(1000, **cfg)
        ref = os.path.exists(f"data/{s}.npz") and np.allclose(np.load(f"data/{s}.npz")["meas_pol"][:, :5], d["A_pol"][:, :5], atol=1e-3)
        n = d["n_meas"]
        xa, Pa, _, _ = filters.ekf(d["A_pol"], d["A_c"], n, DT_MEAS, d["sr"], d["sa"])
        xf, Pf, _, _ = filters.ekf(d["A_pol"], d["A_c"], n, DT_MEAS, d["sr"], d["sa"], extra=[(d["B_pol"], RADAR_B, d["sr"], d["sa"])])
        ra, ma = run("A", d, xa, Pa)
        rf, mf = run("A+B", d, xf, Pf)
        rng = np.random.default_rng(7)
        idx = rng.integers(0, len(ma), (2000, len(ma)))
        diff = np.median(mf[idx], 1) - np.median(ma[idx], 1)
        out["splits"][s] = {"identical_to_dataset": bool(ref), "A": ra, "A+B": rf,
                            "diff_median_m": float(np.median(mf) - np.median(ma)), "ci_diff": np.percentile(diff, [2.5, 97.5]).tolist(),
                            "win": float((mf < ma).mean())}
        print(f"{s:14s} same data {ref}  A {ra['miss_median_m']:6.0f} m (RMSE {ra['miss_rmse_m']:6.0f}, cov {ra['coverage']:.2f})"
              f"  A+B {rf['miss_median_m']:6.0f} m (RMSE {rf['miss_rmse_m']:6.0f}, cov {rf['coverage']:.2f})", flush=True)
    json.dump(out, open("results/multiradar.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or list(SPLITS))
