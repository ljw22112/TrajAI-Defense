"""Model mismatch for the truth trajectories only (the EKF and the PINN residual keep the nominal model).

A constant scale on C_D is indistinguishable from a different beta, which the filter estimates, so it is not a
mismatch. The two effects here cannot be absorbed by beta:
  * Cd(M) shape error: the transonic peak and the supersonic tail are scaled independently per trajectory
    (k_peak, k_tail ~ U(0.75, 1.25)), i.e. the drag curve has a different shape than the one the estimators assume;
  * weak lift: a lift-to-drag ratio L/D ~ U(0.03, 0.10), acting horizontally across the flight path after apogee
    (a mild cross-range manoeuvre during re-entry), with random sign.
The object duck-types the atmosphere interface (rho, sos, wind, __getitem__) with the standard atmosphere.
"""
from __future__ import annotations

import numpy as np

from . import physics


class Mismatch:
    def __init__(self, k_peak, k_tail, ld, sign):
        self.k_peak, self.k_tail, self.ld, self.sign = (np.asarray(a, float) for a in (k_peak, k_tail, ld, sign))

    def __len__(self):
        return len(self.ld)

    def __getitem__(self, idx):
        return Mismatch(self.k_peak[idx], self.k_tail[idx], self.ld[idx], self.sign[idx])

    def rho(self, h):
        return physics.density(h)

    def sos(self, h):
        return physics.speed_of_sound(h)

    def wind(self, h):
        return np.zeros(np.shape(h) + (3,))

    def cd_ratio(self, mach):
        m = np.asarray(mach, float)
        base = physics.cd_ratio(m)
        peak = np.clip(1.0 - np.abs(m - 1.2) / 0.6, 0.0, 1.0)            # weight of the transonic bump
        tail = np.clip((m - 1.2) / 1.2, 0.0, 1.0)                         # weight of the supersonic tail
        return base * (1.0 + (self.k_peak - 1.0) * peak + (self.k_tail - 1.0) * tail)

    def lift(self, rel, drag_per_v, vel):
        """Lift = L/D * |drag|, horizontal and perpendicular to the air-relative velocity, only while descending."""
        drag = drag_per_v                                                  # |a_drag| = k * |v|
        horiz = np.stack([-rel[..., 1], rel[..., 0], np.zeros_like(rel[..., 0])], -1)
        n = horiz / np.maximum(np.linalg.norm(horiz, axis=-1, keepdims=True), 1e-6)
        on = (vel[..., 2] < 0).astype(float)
        return (self.sign * self.ld * drag * on)[..., None] * n


def sample(n, rng, cd=True, lift=True):
    k_peak = rng.uniform(0.75, 1.25, n) if cd else np.ones(n)
    k_tail = rng.uniform(0.75, 1.25, n) if cd else np.ones(n)
    ld = rng.uniform(0.03, 0.10, n) if lift else np.zeros(n)
    return Mismatch(k_peak, k_tail, ld, rng.choice([-1.0, 1.0], n))
