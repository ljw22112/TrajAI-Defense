"""Glide manoeuvre for the truth trajectories (hypersonic-glide-like, §9.13 / §9.14).

After apogee the body generates aerodynamic lift L = (L/D) * |drag|, perpendicular to the air-relative velocity and
tilted by a constant bank angle sigma out of the vertical plane: cos(sigma) pulls the trajectory up (glide, longer
range), sin(sigma) turns it sideways (cross-range). Because lift scales with drag it is negligible in the thin upper
atmosphere and grows during re-entry, which is when real glide vehicles pull up. Lift is applied only while
descending, so the body glides in a sequence of shallow pull-ups instead of skipping out of the atmosphere.
L/D ~ U(1.0, 2.5) is typical of lifting re-entry / glide bodies (the weak-lift mismatch of §9.9 used 0.03-0.10).
The estimators keep the ballistic model; this is a truth-only effect, like trajai.mismatch.
"""
from __future__ import annotations

import numpy as np

from . import physics


class Glide:
    def __init__(self, ld, bank):
        self.ld, self.bank = np.asarray(ld, float), np.asarray(bank, float)

    def __len__(self):
        return len(self.ld)

    def __getitem__(self, idx):
        return Glide(self.ld[idx], self.bank[idx])

    def rho(self, h):
        return physics.density(h)

    def sos(self, h):
        return physics.speed_of_sound(h)

    def wind(self, h):
        return np.zeros(np.shape(h) + (3,))

    def lift(self, rel, drag, vel):
        vhat = rel / np.maximum(np.linalg.norm(rel, axis=-1, keepdims=True), 1e-6)
        up = np.zeros_like(rel)
        up[..., 2] = 1.0
        n_up = up - np.sum(up * vhat, -1, keepdims=True) * vhat           # vertical-plane normal to the velocity
        n_up /= np.maximum(np.linalg.norm(n_up, axis=-1, keepdims=True), 1e-6)
        n_side = np.cross(vhat, n_up)
        cb, sb = np.cos(self.bank)[..., None], np.sin(self.bank)[..., None]
        on = (vel[..., 2] < 0).astype(float)
        return (self.ld * drag * on)[..., None] * (cb * n_up + sb * n_side)


def sample(n, rng, ld=(1.0, 2.5), bank_deg=45.0):
    return Glide(rng.uniform(*ld, n), np.radians(rng.uniform(-bank_deg, bank_deg, n)))


def nominal(n):
    """Same interface without lift (reference flights for paired comparisons)."""
    return Glide(np.zeros(n), np.zeros(n))
