"""Precompute network features, EKF/UKF posteriors and targets for every split (data/prep_*.npz),
and the EKF / UKF Monte Carlo baseline results for the test splits (results/baseline_*.npz)."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajai import features, filters, metrics, radar  # noqa: E402

DATA = "data"
TESTS = ["test_id_f05", "test_id_f10", "test_id_f15", "test_id_noise", "test_ood"]


def noise_of(split):
    return radar.NOISE_HIGH if split == "test_id_noise" else radar.NOISE_NOMINAL


def main():
    os.makedirs("results", exist_ok=True)
    for split in ["train", "val"] + TESTS:
        d = dict(np.load(f"{DATA}/{split}.npz"))
        nz = noise_of(split)
        t0 = time.time()
        feats, pad, origin = features.track_features(d["meas"], d["n_meas"])
        x, P, nis, _ = filters.ekf(d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5, nz["sigma_r"], nz["sigma_ang"])
        t_filter = time.time() - t0
        g = features.global_features(x, P, origin)
        fut, dT, lb = features.targets(d, origin)
        np.savez_compressed(f"{DATA}/prep_{split}.npz", feats=feats, pad=pad, origin=origin, g=g, fut=fut, dT=dT, lb=lb,
                            t_c=d["t_c"], T=d["T"], impact=d["impact"], future=d["future"], beta=d["beta"])
        print(f"{split:14s} features+EKF {t_filter:5.1f}s  median NIS {np.median(nis):.2f}", flush=True)
        if split in TESTS:
            for name, fn in [("ekf", filters.ekf), ("ukf", filters.ukf)]:
                t0 = time.time()
                if name == "ekf":
                    xf, Pf, nisf = x, P, nis
                else:
                    xf, Pf, nisf, _ = fn(d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5, nz["sigma_r"], nz["sigma_ang"])
                mc = filters.monte_carlo_impact(xf, Pf, d["t_c"], n_particles=100, seed=1)
                grid = filters.nominal_on_grid(xf, d["t_c"], d["T"])
                np.savez_compressed(f"results/baseline_{name}_{split}.npz",
                                    ade=metrics.ade(grid, d["future"]), miss=metrics.ip_miss(mc["mpi"], d["impact"]),
                                    tot=metrics.tot_error(mc["tot"], d["T"]), cep=mc["cep"],
                                    cov=metrics.cep_coverage(mc["mpi"], mc["cep"], d["impact"]), nis=nisf,
                                    lb_err=np.abs(xf[:, 6] - np.log(d["beta"])), mpi=mc["mpi"], grid=grid.astype(np.float32),
                                    tot_pred=mc["tot"])
                print(f"   {name}: {time.time()-t0:5.1f}s  ADE {np.median(metrics.ade(grid, d['future'])):7.0f} m  "
                      f"miss {np.median(metrics.ip_miss(mc['mpi'], d['impact'])):7.0f} m  CEP {np.median(mc['cep']):6.0f} m  "
                      f"cov {metrics.cep_coverage(mc['mpi'], mc['cep'], d['impact']).mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
