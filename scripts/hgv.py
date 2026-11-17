"""Glide-manoeuvring targets (§9.14): how wrong are ballistic estimators, and does a manoeuvre-aware filter help?

Truth: trajai.hgv (lift L/D 1.0-2.5 with bank, after apogee). Estimators keep the ballistic model.
Splits (600 tracks each, nominal radar noise, ID flight envelope):
  nom_f12 / hgv_f12    track cut at 12 % of the flight: the manoeuvre has not started when tracking ends
  nom_long / hgv_long  tracking continues to 55 % of the flight (capped at the 100 s radar window): glide partly seen
Estimators: EKF + Monte Carlo, IMM (ballistic + manoeuvre mode, trajai.filters.imm) + Monte Carlo, and on the 12 % splits
the served RTS-input PINN (export/pinn_tf_v2_rts.onnx).
Writes results/hgv.json.  Usage: python scripts/hgv.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajai import dataset, features, filters, hgv, metrics, radar  # noqa: E402

N = 600
NZ = radar.NOISE_NOMINAL
SPLITS = {"nom_f12": (8811, 0.12, False), "hgv_f12": (8812, 0.12, True),
          "nom_long": (8813, 0.55, False), "hgv_long": (8814, 0.55, True)}


def pinn_rts(d):
    import onnxruntime as ort
    path = "export/pinn_tf_v2_rts.onnx"
    if not os.path.exists(path):
        return None
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    meas, pol, n = d["meas"], d["meas_pol"].astype(float), d["n_meas"]
    _, pad, origin = features.track_features(meas, n)
    f_rts, _ = features.track_features_rts(pol, meas.astype(float), n, NZ["sigma_r"], NZ["sigma_ang"])
    x, P, _, _ = filters.ekf(pol, meas.astype(float), n, 0.5, NZ["sigma_r"], NZ["sigma_ang"])
    g = features.global_features(x, P, origin)
    ip = np.zeros((len(n), 2))
    for i in range(len(n)):
        k = int((~pad[i]).sum())
        tr = sess.run(None, {"feats": f_rts[i:i + 1, :k].astype(np.float32), "global": g[i:i + 1].astype(np.float32)})[0]
        ip[i] = tr[0, -1, :2] * features.L_POS + origin[i, :2]
    return metrics.ip_miss(ip, d["impact"])


CHI2_90 = 4.605          # chi-square(2 dof) 90 % quantile


def footprint(x, P, impact, n_particles=100, seed=3, ld=(0.0, 2.5), bank_deg=45.0):
    """Manoeuvre footprint instead of a point: sample the EKF posterior AND a glide hypothesis per particle
    (L/D ~ U(ld), bank ~ U(+-bank_deg); L/D = 0 is the ballistic case), propagate every particle to the ground and
    report the 90 % ellipse of the impacts. Returns coverage of the true impact and the ellipse area [km^2]."""
    from trajai import physics
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    Lc = np.linalg.cholesky(P + 1e-9 * np.eye(filters.NX))
    smp = x[:, None] + np.einsum("nij,npj->npi", Lc, rng.standard_normal((n, n_particles, filters.NX)))
    smp[..., 2] = np.maximum(smp[..., 2], 1.0)
    beta = np.exp(np.clip(smp[..., 6], filters.LB_MIN, filters.LB_MAX)).reshape(-1)
    g = hgv.Glide(rng.uniform(*ld, n * n_particles), np.radians(rng.uniform(-bank_deg, bank_deg, n * n_particles)))
    r = physics.propagate_to_ground(smp[..., :6].reshape(-1, 6), beta, dt=0.25, atm=g)["r"].reshape(n, n_particles, 3)[..., :2]
    mu = r.mean(1)
    d = r - mu[:, None]
    C = np.einsum("npi,npj->nij", d, d) / (n_particles - 1) + 1e-6 * np.eye(2)
    e = impact[:, :2] - mu
    m2 = np.einsum("ni,nij,nj->n", e, np.linalg.inv(C), e)
    area = np.pi * CHI2_90 * np.sqrt(np.linalg.det(C)) / 1e6
    return dict(coverage90=float(np.mean(m2 <= CHI2_90)), area_median_km2=float(np.median(area)),
                center_miss_median_m=float(np.median(np.linalg.norm(e, axis=1))))


def summary(miss, cov=None):
    out = dict(miss_median_m=float(np.median(miss)), miss_p90_m=float(np.percentile(miss, 90)),
               miss_rmse_m=float(np.sqrt(np.mean(miss ** 2))))
    if cov is not None:
        out["coverage"] = float(np.mean(cov))
    return out


def main():
    res = {}
    for name, (seed, frac, glide) in SPLITS.items():
        truth = (lambda m, r: hgv.sample(m, r)) if glide else (lambda m, r: hgv.nominal(m))
        d = dataset.generate(N, seed=seed, frac=frac, truth=truth)
        args = (d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5, NZ["sigma_r"], NZ["sigma_ang"])
        r = {"t_c_median_s": float(np.median(d["t_c"])), "flight_median_s": float(np.median(d["T"])),
             "range_median_km": float(np.median(np.linalg.norm(d["impact"][:, :2], axis=1)) / 1000)}
        x, P, _, _ = filters.ekf(*args)
        mc = filters.monte_carlo_impact(x, P, d["t_c"], n_particles=100, seed=1)
        miss = metrics.ip_miss(mc["mpi"], d["impact"])
        r["ekf"] = summary(miss, metrics.cep_coverage(mc["mpi"], mc["cep"], d["impact"]))
        r["ekf"]["cep_median_m"] = float(np.median(mc["cep"]))
        x, P, mu, _ = filters.imm(*args)
        mc = filters.monte_carlo_impact(x, P, d["t_c"], n_particles=100, seed=1)
        miss = metrics.ip_miss(mc["mpi"], d["impact"])
        r["imm"] = summary(miss, metrics.cep_coverage(mc["mpi"], mc["cep"], d["impact"]))
        r["imm"]["cep_median_m"] = float(np.median(mc["cep"]))
        x, P, _, _ = filters.ekf(*args)
        r["footprint"] = footprint(x, P, d["impact"])
        r["ekf"]["area90_median_km2"] = float(np.pi * CHI2_90 / np.log(4) * np.median(r["ekf"]["cep_median_m"]) ** 2 / 1e6)
        if frac < 0.2:
            m = pinn_rts(d)
            if m is not None:
                r["pinn_rts"] = summary(m)
        res[name] = r
        print(f"{name:9s} t_c {r['t_c_median_s']:5.1f} s  T {r['flight_median_s']:5.0f} s  "
              f"EKF {r['ekf']['miss_median_m']:8.0f} m cov {r['ekf']['coverage']:.2f}  "
              f"IMM {r['imm']['miss_median_m']:8.0f} m cov {r['imm']['coverage']:.2f}"
              + (f"  PINN-RTS {r['pinn_rts']['miss_median_m']:8.0f} m" if "pinn_rts" in r else "")
              + f"  footprint cov90 {r['footprint']['coverage90']:.2f} area {r['footprint']['area_median_km2']:.0f} km2", flush=True)
    json.dump(res, open("results/hgv.json", "w"), indent=1)


if __name__ == "__main__":
    main()
