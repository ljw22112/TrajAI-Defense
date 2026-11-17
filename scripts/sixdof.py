"""How much does the 3-DOF assumption cost? (§9.15)  Truth from trajai.sixdof (angle of attack, coning, induced drag
and lift); every estimator keeps the 3-DOF point-mass model.

The same flights (ID envelope, sampled as in trajai.dataset) are simulated twice: with the sampled tip-off angle of
attack and with alpha0 = trim = 0 (identical to the 3-DOF model). Radar measurements use a per-track random stream, so the two
versions of a track get the same noise and are compared pairwise. Cut-off at 10 % and 15 % of the flight.
Estimators: EKF + Monte Carlo and the served RTS-input PINN (export/pinn_tf_v2_rts.onnx).
Writes results/sixdof.json.  Usage: python scripts/sixdof.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajai import features, filters, metrics, radar, sixdof  # noqa: E402
from trajai.dataset import DT_MEAS, ID_RANGES, K_MAX, initial_state, sample_params  # noqa: E402

N, SEED = 600, 9301
NZ = radar.NOISE_NOMINAL


def tracks(prm, s0, beta, frac):
    t, pos, T, r_imp = sixdof.simulate(s0, beta, prm, dt=0.02, out_dt=DT_MEAS)
    n = len(T)
    k_c = np.maximum(np.minimum(np.floor(frac * T / DT_MEAS).astype(int), K_MAX - 1), 12)
    pol, car = np.zeros((n, K_MAX, 3)), np.zeros((n, K_MAX, 3))
    for i in range(n):
        rng = np.random.default_rng(SEED * 1000 + i)            # same noise for both truths of track i
        zp, zc = radar.measure(pos[i, :k_c[i] + 1], rng, NZ["sigma_r"], NZ["sigma_ang"])
        pol[i, :k_c[i] + 1], car[i, :k_c[i] + 1] = zp, zc
    return dict(meas_pol=pol, meas=car, n_meas=k_c + 1, t_c=k_c * DT_MEAS, T=T, impact=r_imp)


def ekf_miss(d):
    x, P, _, _ = filters.ekf(d["meas_pol"], d["meas"], d["n_meas"], DT_MEAS, NZ["sigma_r"], NZ["sigma_ang"])
    mc = filters.monte_carlo_impact(x, P, d["t_c"], n_particles=100, seed=1)
    return metrics.ip_miss(mc["mpi"], d["impact"]), metrics.cep_coverage(mc["mpi"], mc["cep"], d["impact"]), x, P


def pinn_miss(d, x, P):
    import onnxruntime as ort
    path = "export/pinn_tf_v2_rts.onnx"
    if not os.path.exists(path):
        return None
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    _, pad, origin = features.track_features(d["meas"].astype(np.float32), d["n_meas"])
    f, _ = features.track_features_rts(d["meas_pol"], d["meas"], d["n_meas"], NZ["sigma_r"], NZ["sigma_ang"])
    g = features.global_features(x, P, origin)
    ip = np.zeros((len(d["T"]), 2))
    for i in range(len(ip)):
        k = int((~pad[i]).sum())
        tr = sess.run(None, {"feats": f[i:i + 1, :k].astype(np.float32), "global": g[i:i + 1].astype(np.float32)})[0]
        ip[i] = tr[0, -1, :2] * features.L_POS + origin[i, :2]
    return metrics.ip_miss(ip, d["impact"])


def paired(a, b, rng):
    idx = rng.integers(0, len(a), (2000, len(a)))
    diff = np.median(b[idx], 1) - np.median(a[idx], 1)
    return dict(median_3dof_m=float(np.median(a)), median_6dof_m=float(np.median(b)),
                rmse_3dof_m=float(np.sqrt(np.mean(a ** 2))), rmse_6dof_m=float(np.sqrt(np.mean(b ** 2))),
                diff_median_m=float(np.median(b) - np.median(a)), ci_diff=np.percentile(diff, [2.5, 97.5]).tolist())


def main():
    rng = np.random.default_rng(SEED)
    p = sample_params(N, rng, ID_RANGES)
    s0, beta = initial_state(p), p[:, 4]
    prm = sixdof.sample(N, rng)
    prm0 = {**prm, "alpha0": np.zeros(N), "trim": np.zeros(N)}
    out = {"n": N}
    _, _, T6, r6 = sixdof.simulate(s0, beta, prm, dt=0.02, out_dt=1.0)
    _, _, T3, r3 = sixdof.simulate(s0, beta, prm0, dt=0.02, out_dt=1.0)
    shift = np.linalg.norm(r6[:, :2] - r3[:, :2], axis=1)
    slow = prm["p"] < 2 * np.pi * 0.3
    out["truth_shift_slow_roll"] = dict(median_m=float(np.median(shift[slow])), p90_m=float(np.percentile(shift[slow], 90)))
    out["truth_shift"] = dict(median_m=float(np.median(shift)), p90_m=float(np.percentile(shift, 90)),
                              dT_median_s=float(np.median(T6 - T3)))
    print(f"impact shift 6-DOF vs 3-DOF: median {np.median(shift):.0f} m, p90 {np.percentile(shift, 90):.0f} m", flush=True)
    cr = np.random.default_rng(7)
    for frac in (0.10, 0.15):
        d3, d6 = tracks(prm0, s0, beta, frac), tracks(prm, s0, beta, frac)
        m3, c3, x3, P3 = ekf_miss(d3)
        m6, c6, x6, P6 = ekf_miss(d6)
        slow = prm["p"] < 2 * np.pi * 0.3                      # slowly rolling bodies: trim lift does not average out
        r = {"ekf": {**paired(m3, m6, cr), "coverage_3dof": float(c3.mean()), "coverage_6dof": float(c6.mean()),
                     "abs_change_median_m": float(np.median(np.abs(m6 - m3))),
                     "slow_roll": {**paired(m3[slow], m6[slow], cr), "n": int(slow.sum())}}}
        q3, q6 = pinn_miss(d3, x3, P3), pinn_miss(d6, x6, P6)
        if q3 is not None:
            r["pinn_rts"] = paired(q3, q6, cr)
        out[f"f{int(frac * 100):02d}"] = r
        print(f"frac {frac:.2f}  EKF {r['ekf']['median_3dof_m']:.0f} -> {r['ekf']['median_6dof_m']:.0f} m "
              f"(cov {r['ekf']['coverage_3dof']:.2f} -> {r['ekf']['coverage_6dof']:.2f})"
              + (f"  PINN-RTS {r['pinn_rts']['median_3dof_m']:.0f} -> {r['pinn_rts']['median_6dof_m']:.0f} m" if "pinn_rts" in r else ""),
              flush=True)
    json.dump(out, open("results/sixdof.json", "w"), indent=1)


if __name__ == "__main__":
    main()
