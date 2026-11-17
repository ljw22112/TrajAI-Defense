import numpy as np

from trajai import physics, radar


def test_atmosphere_reference_values():
    assert abs(physics.density(0.0) - 1.225) < 1e-12
    assert abs(physics.speed_of_sound(0.0) - 340.29) < 0.1          # USSA-1976 sea level
    assert abs(physics.speed_of_sound(15_000.0) - 295.07) < 0.1      # tropopause isothermal layer
    assert abs(physics.gravity(0.0) - physics.G0) < 1e-12


def test_cd_ratio_is_continuous():
    for m in (0.8, 1.2):
        lo, hi = physics.cd_ratio(m - 1e-9), physics.cd_ratio(m + 1e-9)
        assert abs(lo - hi) < 1e-6


def test_vacuum_limit_matches_parabola():
    # very large beta -> negligible drag; flat gravity approximately constant over a short arc
    s0 = np.array([[0.0, 0.0, 100_000.0, 100.0, 0.0, 50.0]])
    s = s0.copy()
    for _ in range(100):
        s = physics.rk4_step(s, np.array([1e12]), 0.1)
    g = physics.gravity(100_000.0)
    t = 10.0
    assert abs(s[0, 0] - 1000.0) < 1e-6
    assert abs(s[0, 2] - (100_000.0 + 50.0 * t - 0.5 * g * t * t)) < 1.0


def _run(s0, beta, dt, T=20.0):
    s = s0.copy()
    for _ in range(int(round(T / dt))):
        s = physics.rk4_step(s, beta, dt)
    return s


def test_rk4_fourth_order_convergence_smooth_dynamics():
    # high altitude, drag negligible: smooth right-hand side -> 4th-order convergence (ratio ~16)
    s0 = np.array([[0.0, 0.0, 120_000.0, 1500.0, 0.0, 1500.0]])
    beta = np.array([1e9])
    ref = _run(s0, beta, 0.01, 60.0)
    e1 = np.abs(_run(s0, beta, 2.0, 60.0) - ref)[0, :3].max()
    e2 = np.abs(_run(s0, beta, 1.0, 60.0) - ref)[0, :3].max()
    assert e1 / e2 > 10.0


def test_rk4_error_bounded_across_atmosphere_kinks():
    # crossing the 20 km USSA-1976 temperature kink makes the RHS non-smooth (order drops),
    # but at the dataset step (0.1 s) the position error stays far below sensor noise
    s0 = np.array([[0.0, 0.0, 15_000.0, 900.0, 0.0, 900.0]])
    beta = np.array([1500.0])
    ref = _run(s0, beta, 0.005)
    assert np.abs(_run(s0, beta, 0.1) - ref)[0, :3].max() < 0.01


def test_propagate_to_ground_lands():
    s0 = np.array([[0.0, 0.0, 20_000.0, 1000.0, 0.0, 800.0]] * 3)
    res = physics.propagate_to_ground(s0, np.array([800.0, 2000.0, 8000.0]))
    assert np.all(np.abs(res["r"][:, 2]) < 1e-6)
    assert res["r"][0, 0] < res["r"][1, 0] < res["r"][2, 0]      # less drag -> longer range


def test_radar_roundtrip():
    p = np.array([[10_000.0, -5_000.0, 30_000.0], [200_000.0, 40_000.0, 1_000.0]])
    assert np.allclose(radar.to_cartesian(radar.to_polar(p)), p, atol=1e-6)


def test_polar_jacobian_matches_finite_difference():
    p = np.array([[50_000.0, 10_000.0, 20_000.0]])
    J = radar.polar_jacobian(p)[0]
    eps = 1.0
    for j in range(3):
        dp = p.copy(); dp[0, j] += eps
        fd = (radar.to_polar(dp) - radar.to_polar(p))[0] / eps
        assert np.allclose(J[:, j], fd, rtol=1e-4, atol=1e-10)


def test_glide_extends_range_and_ballistic_reference_unchanged():
    """trajai.hgv: lift only during descent lengthens the flight; zero L/D reproduces the ballistic truth."""
    import numpy as np
    from trajai import dataset, hgv
    a = dataset.generate(50, seed=5, frac=0.12)
    b = dataset.generate(50, seed=5, frac=0.12, truth=lambda m, r: hgv.nominal(m))
    assert np.allclose(a["impact"], b["impact"], atol=1.0)
    c = dataset.generate(50, seed=5, frac=0.12, truth=lambda m, r: hgv.sample(m, r))
    assert np.median(c["T"]) > np.median(a["T"]) + 30


def test_sixdof_reduces_to_3dof_and_trim_moves_impact():
    """trajai.sixdof: zero angle of attack reproduces the 3-DOF simulator; a trim angle moves the impact point."""
    import numpy as np
    from trajai import physics, sixdof
    from trajai.dataset import ID_RANGES, initial_state, sample_params
    rng = np.random.default_rng(2)
    p = sample_params(20, rng, ID_RANGES)
    s0 = initial_state(p)
    prm = sixdof.sample(20, rng)
    zero = {**prm, "alpha0": np.zeros(20), "trim": np.zeros(20)}
    _, _, T0, r0 = sixdof.simulate(s0, p[:, 4], zero, dt=0.02, out_dt=1.0)
    _, _, _, T, r = physics.simulate_ballistic(s0, p[:, 4], dt=0.1, out_dt=0.5)
    assert np.abs(r[:, :2] - r0[:, :2]).max() < 0.5 and np.abs(T - T0).max() < 0.01
    _, _, _, r1 = sixdof.simulate(s0, p[:, 4], prm, dt=0.02, out_dt=1.0)
    assert np.median(np.linalg.norm(r1[:, :2] - r0[:, :2], axis=1)) > 5.0
