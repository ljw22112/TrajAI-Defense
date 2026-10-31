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


def test_custom_atmosphere_defaults_match_standard():
    lv = weather.custom_levels(dict(rh_pct=0.0, wind_sfc_ms=0.0, jet_ms=0.0))
    bank = weather.build_grid([dict(time="c", **{k: np.array(lv[k]) for k in ("z", "p", "u", "v")}, T=np.array(lv["Tv"]))])
    ussa = weather.ussa_bank()
    k = np.searchsorted(weather.H_GRID, [0, 5_000, 11_000, 25_000])
    assert np.allclose(bank.rho[0, k], ussa.rho[0, k], rtol=0.01)        # dry, calm, 15 C, 1013 hPa == USSA-1976
    assert np.abs(bank.u[0]).max() < 1e-9


def test_custom_atmosphere_humidity_heat_and_wind():
    dry = weather.custom_levels(dict(t_sfc_c=30.0, rh_pct=0.0))
    wet = weather.custom_levels(dict(t_sfc_c=30.0, rh_pct=100.0))
    assert wet["Tv"][0] > dry["Tv"][0] + 4.0                             # ~27 g/kg of vapour at 30 C -> Tv up ~5 K
    # moist air is lighter: lower density at the surface for the same pressure
    rho = lambda lv: lv["p"][0] / (physics.R_AIR * lv["Tv"][0])
    assert rho(wet) < rho(dry)
    w = weather.custom_levels(dict(jet_ms=60.0, jet_km=12.0, jet_dir=270.0, wind_sfc_ms=0.0))
    i = int(np.argmax(np.hypot(w["u"], w["v"])))
    assert abs(w["z"][i] - 12_000) <= 500 and w["u"][i] > 59.0             # westerly jet blows toward +x (east)


def test_custom_tropopause_stays_physical():
    for t in (-50.0, 50.0):
        lv = weather.custom_levels(dict(t_sfc_c=t))
        k = int(np.searchsorted(lv["z"], 11_000))
        assert 190.0 <= lv["T"][k] <= 230.0
