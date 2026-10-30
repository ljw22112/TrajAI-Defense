"""Offline tests for the weather atmosphere (synthetic pressure-level profile, no network)."""
import numpy as np

from trajai import dataset, filters, physics, radar, weather


def _synthetic_profile(jet=50.0):
    z = np.array([110, 760, 1460, 3010, 4200, 5570, 7180, 9160, 10360, 11790, 13600, 16200, 18450, 20600, 23850, 26500, 31000], float)
    T = physics.temperature(z) + 2.0
    p = np.array(weather.LEVELS, float) * 100.0
    u = jet * np.exp(-0.5 * ((z - 11_000) / 3_000) ** 2)
    return dict(time="synthetic", z=z, p=p, T=T, u=u, v=np.zeros_like(u))


def test_grid_is_hydrostatic_and_close_to_standard():
    bank = weather.build_grid([_synthetic_profile()])
    ussa = weather.ussa_bank()
    k = np.searchsorted(weather.H_GRID, [0, 10_000, 30_000])
    ratio = bank.rho[0, k] / ussa.rho[0, k]
    assert np.all(np.abs(ratio - 1) < 0.08)              # +2 K offset -> small density change
    assert abs(ussa.rho[0, 0] - 1.225) < 0.01
    assert np.all(np.diff(bank.rho[0]) < 0)              # density decreases with height


def test_wind_lookup_and_decay_above_data():
    bank = weather.build_grid([_synthetic_profile(jet=50.0)])
    atm = bank[np.array([0, 0])]
    w = atm.wind(np.array([11_000.0, 50_000.0]))
    assert abs(w[0, 0] - 50.0) < 2.5 and abs(w[1, 0]) < 1e-9   # peak lies between pressure levels


def test_tailwind_extends_range():
    bank = weather.build_grid([_synthetic_profile(jet=0.0), _synthetic_profile(jet=60.0)])
    p = dataset.sample_params(50, np.random.default_rng(0), dataset.ID_RANGES)
    s0 = dataset.initial_state(p)
    calm = physics.propagate_to_ground(s0, p[:, 4], atm=bank[np.zeros(50, int)])
    jet = physics.propagate_to_ground(s0, p[:, 4], atm=bank[np.ones(50, int)])
    assert np.median(jet["r"][:, 0] - calm["r"][:, 0]) > 0      # westerly jet (+x) pushes impacts downrange


def test_ekf_with_matching_atmosphere_is_consistent():
    bank = weather.build_grid([_synthetic_profile(jet=60.0)])
    d = dataset.generate(30, 5, frac=0.15, bank=bank)
    atm = bank[d["wx_idx"]]
    nz = radar.NOISE_NOMINAL
    x, P, nis, _ = filters.ekf(d["meas_pol"].astype(float), d["meas"].astype(float), d["n_meas"], 0.5, nz["sigma_r"], nz["sigma_ang"], atm=atm)
    assert 1.5 < np.median(nis) < 5.0
    mc = filters.monte_carlo_impact(x, P, d["t_c"], n_particles=40, atm=atm)
    assert np.median(np.linalg.norm(mc["mpi"] - d["impact"][:, :2], axis=1)) < 2_000.0


def test_weather_features_shape():
    bank = weather.build_grid([_synthetic_profile(), _synthetic_profile(jet=20.0)])
    f = bank.features(np.array([0, 1, 1]))
    assert f.shape == (3, weather.N_WX) and np.isfinite(f).all()
