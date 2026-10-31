import numpy as np

from trajai import dataset, filters, physics, radar, threat


def _data(n=40, seed=3):
    return dataset.generate(n, seed, frac=0.15)


def test_ekf_is_consistent_and_learns_beta():
    d = _data()
    nz = radar.NOISE_NOMINAL
    x, P, nis, _ = filters.ekf(d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5, nz["sigma_r"], nz["sigma_ang"])
    assert 1.5 < np.median(nis) < 5.0                          # chi-square(3) mean = 3
    err = np.abs(x[:, 6] - np.log(d["beta"]))
    assert np.median(err) < 0.2                                 # beta observable in the dense-air part of the climb
    assert np.all(np.linalg.eigvalsh(P) > 0)                    # Joseph form keeps P positive definite


def test_ukf_runs_and_matches_ekf_scale():
    d = _data(20, 4)
    nz = radar.NOISE_NOMINAL
    xe, *_ = filters.ekf(d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5, nz["sigma_r"], nz["sigma_ang"])
    xu, Pu, nis, _ = filters.ukf(d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5, nz["sigma_r"], nz["sigma_ang"])
    assert np.median(np.linalg.norm(xe[:, :3] - xu[:, :3], axis=1)) < 100.0
    assert 1.5 < np.median(nis) < 5.0


def test_monte_carlo_impact_shapes():
    d = _data(10, 5)
    nz = radar.NOISE_NOMINAL
    x, P, *_ = filters.ekf(d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5, nz["sigma_r"], nz["sigma_ang"])
    mc = filters.monte_carlo_impact(x, P, d["t_c"], n_particles=50)
    assert mc["mpi"].shape == (10, 2) and np.all(mc["cep"] > 0)
    assert np.median(np.abs(mc["tot"] - d["T"])) < 5.0


def test_threat_levels_and_windows():
    probs = threat.impact_probabilities(np.array([120_000.0, -6_000.0]), np.array([500.0, 500.0]))
    assert probs["ASSET-B"] > 0.99 and threat.threat_level(probs) == 5
    far = threat.impact_probabilities(np.array([900_000.0, 0.0]), np.array([500.0, 500.0]))
    assert threat.threat_level(far) == 1
    t = np.linspace(0, 200, 401)
    h = 50_000 * np.sin(np.pi * t / 200)
    w = threat.engagement_windows(t, h)
    assert w["upper"]["t_in"] < w["upper"]["t_out"] < w["lower"]["t_in"]


def test_rts_smoother_beats_filter_and_beta_prior_is_used():
    rng = np.random.default_rng(5)
    N, dt = 30, 0.5
    g = np.radians(rng.uniform(35, 55, N)); v0 = rng.uniform(1000, 1600, N)
    s0 = np.stack([np.zeros(N), np.zeros(N), rng.uniform(10e3, 20e3, N), v0 * np.cos(g), np.zeros(N), v0 * np.sin(g)], 1)
    beta = np.exp(rng.uniform(np.log(800), np.log(8000), N))
    _, pos, vel, _, _ = physics.simulate_ballistic(s0, beta, dt=0.1, out_dt=dt)
    K = 40
    nz = radar.NOISE_NOMINAL
    pol = np.zeros((N, K, 3)); car = np.zeros((N, K, 3))
    for i in range(N):
        pol[i], car[i] = radar.measure(pos[i, :K], rng, sigma_r=nz["sigma_r"], sigma_ang=nz["sigma_ang"])
    n = np.full(N, K)
    st = {}
    filters.ekf(pol, car, n, dt, nz["sigma_r"], nz["sigma_ang"], store=st)
    xs, _ = filters.rts_smooth(pol, car, n, dt, nz["sigma_r"], nz["sigma_ang"])
    k = np.arange(5, K - 5)
    e_f = np.linalg.norm(st["xf"][:, k, 3:6] - vel[:, k], axis=-1)
    e_s = np.linalg.norm(xs[:, k, 3:6] - vel[:, k], axis=-1)
    assert np.median(e_s) < 0.8 * np.median(e_f)                       # smoothing uses future measurements
    lb_true = np.log(beta)
    x0, P0, _, _ = filters.ekf(pol[:, :4], car[:, :4], np.full(N, 4), dt, nz["sigma_r"], nz["sigma_ang"],
                               lb_prior=lb_true, slb_prior=0.05)
    assert np.abs(x0[:, 6] - lb_true).max() < 0.3 and np.all(np.sqrt(P0[:, 6, 6]) < 0.06)


def test_imm_on_ballistic_tracks_stays_in_ballistic_mode():
    d = _data(30, 5)
    nz = radar.NOISE_NOMINAL
    x, P, mu, nis = filters.imm(d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5, nz["sigma_r"], nz["sigma_ang"])
    assert 1.5 < np.nanmedian(nis) < 5.0                        # nominal-mode innovations consistent
    assert np.nanmean(mu) < 0.5                                 # manoeuvre mode not preferred on ballistic flights
    assert np.all(np.linalg.eigvalsh(P) > 0)                    # mixed covariance positive definite
    xe, *_ = filters.ekf(d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5, nz["sigma_r"], nz["sigma_ang"])
    assert np.median(np.linalg.norm(x[:, :3] - xe[:, :3], axis=1)) < 100.0
