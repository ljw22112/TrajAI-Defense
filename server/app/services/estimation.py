"""Estimators served by the API: EKF / UKF (+ Monte Carlo impact dispersion) and the PINN hybrid via ONNX Runtime."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from trajai import features, filters, metrics, physics

from ..models import Method

N_PARTICLES = 200
_sessions: dict[str, object] = {}
_lock = threading.Lock()


def _arrays(scn):
    m = scn.measurements
    pol = np.array([[[x.range_m, x.az, x.el] for x in m]], float)
    car = np.array([[[x.x, x.y, x.z] for x in m]], float)
    return pol, car, np.array([len(m)])


def _filter(scn, atm, kind="ekf"):
    pol, car, n = _arrays(scn)
    fn = filters.ekf if kind == "ekf" else filters.ukf
    x, P, nis, _ = fn(pol, car, n, 0.5, scn.sigma_r, scn.sigma_ang, atm=atm)
    return x, P, float(nis[0])


def _descent(x, t_c, atm):
    res = physics.propagate_to_ground(x[:, :6], np.exp(x[:, 6]), dt=0.25, t0=t_c, record_every=8, atm=atm)
    xs = [float(x[0, 0])] + [float(p[0, 0]) for _, p, alive in res["snaps"] if alive[0]] + [float(res["r"][0, 0])]
    zs = [float(x[0, 2])] + [float(p[0, 2]) for _, p, alive in res["snaps"] if alive[0]] + [0.0]
    ts = [t_c] + [float(t) for t, _, alive in res["snaps"] if alive[0]] + [float(res["t"][0])]
    return dict(t=ts, x=xs, z=zs)


def run_filter(scn, bank, method: Method) -> dict:
    atm = bank[np.zeros(1, int)] if (bank is not None and method in (Method.ekf_wx, Method.ukf_wx)) else None
    t0 = time.perf_counter()
    x, P, nis = _filter(scn, atm, "ukf" if method == Method.ukf_wx else "ekf")
    mc = filters.monte_carlo_impact(x, P, np.array([scn.t_c]), n_particles=N_PARTICLES, seed=scn.seed, atm=atm)
    latency = (time.perf_counter() - t0) * 1000
    traj = _descent(x, scn.t_c, atm)
    smp = mc["samples"][0]
    return dict(ip=mc["mpi"][0], cep=float(mc["cep"][0]), tot=float(mc["tot"][0]), tot_std=float(mc["tot_std"][0]),
                beta_hat=float(np.exp(x[0, 6])), nis=nis, latency_ms=latency, trajectory=traj,
                sigma=np.array([smp[:, 0].std(), smp[:, 1].std()]))


def _session(path: str):
    with _lock:
        if path not in _sessions:
            import onnxruntime as ort
            so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
            _sessions[path] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        return _sessions[path]


def _run_onnx(path, feats, g, wxf):
    sess = _session(path)
    names = {i.name for i in sess.get_inputs()}
    inputs = {k: v for k, v in (("feats", feats), ("global", g), ("wx", wxf)) if k in names}
    return sess.run(None, inputs)


def run_ekf_pinn(scn, track_model, prior_sd: float, bank=None) -> dict:
    """PINN -> EKF feedback: the track-only PINN's ln beta replaces the broad default prior of the EKF.
    Uses the scenario's atmosphere when it has one (the prior and the weather are complementary)."""
    atm = bank[np.zeros(1, int)] if bank is not None else None
    pol, car, n = _arrays(scn)
    feats, _, _ = features.track_features(car.astype(np.float32), n)
    t0 = time.perf_counter()
    _, head = _run_onnx(track_model.file_path, feats, None, None)
    lb = float(head[0, 0] * features.LB_SD + features.LB_MU)
    x, P, nis, _ = filters.ekf(pol, car, n, 0.5, scn.sigma_r, scn.sigma_ang, atm=atm, lb_prior=lb, slb_prior=prior_sd)
    mc = filters.monte_carlo_impact(x, P, np.array([scn.t_c]), n_particles=N_PARTICLES, seed=scn.seed, atm=atm)
    latency = (time.perf_counter() - t0) * 1000
    smp = mc["samples"][0]
    return dict(ip=mc["mpi"][0], cep=float(mc["cep"][0]), tot=float(mc["tot"][0]), tot_std=float(mc["tot_std"][0]),
                beta_hat=float(np.exp(x[0, 6])), nis=float(nis[0]), latency_ms=latency, trajectory=_descent(x, scn.t_c, atm),
                sigma=np.array([smp[:, 0].std(), smp[:, 1].std()]))


def run_pinn(scn, bank, model_version) -> dict:
    """PINN hybrid: track features + EKF posterior (+ weather summary if the model uses it) -> ONNX."""
    if model_version is None or not all(Path(f).exists() for f in model_version.file_path.split(";")):
        raise FileNotFoundError("no active PINN model file")
    t0 = time.perf_counter()
    pol, car, n = _arrays(scn)
    feats, pad, origin = features.track_features(car.astype(np.float32), n)
    if model_version.kind.startswith("pinn-rts"):          # trained on EKF-RTS smoothed track features (§9.6)
        feats, _ = features.track_features_rts(pol, car, n, scn.sigma_r, scn.sigma_ang)
    use_wx = model_version.uses_weather and bank is not None
    atm = bank[np.zeros(1, int)] if use_wx else None
    x, P, nis, _ = filters.ekf(pol, car, n, 0.5, scn.sigma_r, scn.sigma_ang, atm=atm)
    g = features.global_features(x, P, origin)
    wxf = bank.features(np.zeros(1, int)) if use_wx else np.zeros((1, 27), np.float32)
    # a registry entry may hold several ONNX files (deep ensemble, ';'-separated): run each member and combine
    trajs, heads = [], []
    for path in model_version.file_path.split(";"):
        traj, head = _run_onnx(path, feats, g, wxf)
        trajs.append(traj[0]); heads.append(head[0])
    latency = (time.perf_counter() - t0) * 1000
    trajs, heads = np.stack(trajs), np.stack(heads)
    pos = trajs.mean(0) * features.L_POS + origin[0]
    dT = 100.0 * float(np.exp(heads[:, 1]).mean())
    # split-conformal scale (scripts/site_stats.py): raw heteroscedastic sigmas are mis-calibrated, and the threat
    # probabilities are computed from sigma, so the scale is applied to sigma (CEP is linear in sigma).
    scale = float((model_version.metrics or {}).get("cep_scale", 1.0))
    sig2 = (np.exp(heads[:, 2:4]) * 1000.0) ** 2
    ip_m = trajs[:, -1, :2] * features.L_POS
    sig = np.sqrt(sig2.mean(0) + ip_m.var(0)) * scale          # mixture variance = mean aleatoric + spread of members
    heads = heads.mean(0, keepdims=True)
    head = heads
    tau = np.linspace(0, 1, pos.shape[0])
    return dict(ip=pos[-1, :2], cep=float(metrics.gaussian_cep(sig[:1], sig[1:])[0]), tot=scn.t_c + dT, tot_std=None,
                beta_hat=float(np.exp(head[0, 0] * features.LB_SD + features.LB_MU)), nis=float(nis[0]), latency_ms=latency,
                trajectory=dict(t=(scn.t_c + tau * dT)[::4].round(2).tolist(), x=pos[::4, 0].round(1).tolist(), z=pos[::4, 2].round(1).tolist()),
                sigma=sig)


def score(result: dict, scn) -> dict:
    miss = float(np.hypot(result["ip"][0] - scn.impact_x, result["ip"][1] - scn.impact_y))
    return dict(miss_m=miss, inside_cep=bool(miss <= result["cep"]))
