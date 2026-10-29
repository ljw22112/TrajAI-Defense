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


def test_rk4_fourth_order_convergence():
    s0 = np.array([[0.0, 0.0, 15_000.0, 900.0, 0.0, 900.0]])
    beta = np.array([1500.0])

    def run(dt, T=20.0):
        s = s0.copy()
        for _ in range(int(round(T / dt))):
            s = physics.rk4_step(s, beta, dt)
        return s

    ref = run(0.01)
    e1 = np.abs(run(0.4) - ref)[0, :3].max()
    e2 = np.abs(run(0.2) - ref)[0, :3].max()
    assert e1 / e2 > 10.0      # ~16 for a 4th-order scheme


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
