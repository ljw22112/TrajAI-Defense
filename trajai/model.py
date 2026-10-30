"""PINN trajectory predictor: sequence encoder + continuous-time decoder with an autograd physics residual.

r_hat(tau) = D(tau, c),  tau in [0, 1],  t = t_c + tau * dT
Physics residual (Eq. 7.2):  R = L/dT^2 d2r/dtau2 - a_model(r, L/dT dr/dtau; beta_hat),  normalised by g0.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .features import L_POS, LB_MU, LB_SD, N_FEAT, N_GLOBAL

G0, RE, RHO0, HS = 9.80665, 6_371_000.0, 1.225, 8_500.0


# ----------------------------------------------------------------------------- torch physics (mirrors physics.py)
def t_density(h):
    return RHO0 * torch.exp(-torch.clamp(h, min=0.0) / HS)


def t_sos(h):
    h = torch.clamp(h, min=0.0)
    T = torch.where(h < 11_000, 288.15 - 0.0065 * h,
        torch.where(h < 20_000, torch.full_like(h, 216.65),
        torch.where(h < 32_000, 216.65 + 0.001 * (h - 20_000),
        torch.where(h < 47_000, 228.65 + 0.0028 * (h - 32_000), torch.full_like(h, 270.65)))))
    return torch.sqrt(1.4 * 287.053 * T)


def t_cd_ratio(m):
    return torch.where(m < 0.8, torch.full_like(m, 0.75),
           torch.where(m < 1.2, 0.75 + (m - 0.8) / 0.4 * 0.6, 0.8 + 0.55 * torch.exp(-(m - 1.2) / 1.2)))


def t_gravity(h):
    r = RE / (RE + h)
    return G0 * r * r


class TorchAtmosphere:
    """Torch view of a weather ProfileBank: per-sample profile index, linear interpolation on the 250 m grid."""

    def __init__(self, bank, dh=250.0):
        self.dh = dh
        self.rho = torch.as_tensor(bank.rho, dtype=torch.float32)
        self.sos = torch.as_tensor(bank.sos, dtype=torch.float32)
        self.u = torch.as_tensor(bank.u, dtype=torch.float32)
        self.v = torch.as_tensor(bank.v, dtype=torch.float32)
        self.hmax = (self.rho.shape[1] - 1) * dh

    def lookup(self, arr, z, idx):
        x = z.clamp(0.0, self.hmax - 1e-3) / self.dh
        i = x.detach().floor().long()
        f = x - i
        rows = idx.view(-1, *([1] * (z.dim() - 1))).expand_as(i)
        return arr[rows, i] * (1 - f) + arr[rows, i + 1] * f


def t_accel(z_abs, v, beta, atm=None, idx=None):
    """z_abs (...,), v (..., 3), beta (...,) -> acceleration (..., 3) [m/s^2].
    With atm (TorchAtmosphere) and idx (B,), drag uses date-specific density, speed of sound and wind."""
    if atm is None:
        rel, rho_h, a_h = v, t_density(z_abs), t_sos(z_abs)
    else:
        w = torch.stack([atm.lookup(atm.u, z_abs, idx), atm.lookup(atm.v, z_abs, idx), torch.zeros_like(z_abs)], -1)
        rel, rho_h, a_h = v - w, atm.lookup(atm.rho, z_abs, idx), atm.lookup(atm.sos, z_abs, idx)
    speed = torch.linalg.norm(rel, dim=-1).clamp(min=1e-3)
    k = rho_h * speed * t_cd_ratio(speed / a_h) / (2.0 * beta)
    a = -k.unsqueeze(-1) * rel
    g = torch.zeros_like(a)
    g[..., 2] = t_gravity(z_abs)
    return a - g


# ----------------------------------------------------------------------------- network
class TransformerEncoder(nn.Module):
    def __init__(self, d=128, layers=3, heads=4):
        super().__init__()
        self.inp = nn.Linear(N_FEAT, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(2 * d, d)

    def forward(self, x, pad):
        h = self.enc(self.inp(x), src_key_padding_mask=pad)
        valid = (~pad).float().unsqueeze(-1)
        mean = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
        last_idx = (~pad).sum(1) - 1
        last = h[torch.arange(h.shape[0]), last_idx]
        return self.out(torch.cat([mean, last], -1))


class BiLSTMEncoder(nn.Module):
    def __init__(self, d=128, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(N_FEAT, d, layers, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * d, d)

    def forward(self, x, pad):
        lengths = (~pad).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.out(torch.cat([h[-2], h[-1]], -1))


class PINNPredictor(nn.Module):
    def __init__(self, encoder="transformer", d=128, hybrid=False, dec_width=160, dec_depth=4, n_wx=0):
        super().__init__()
        self.encoder = TransformerEncoder(d) if encoder == "transformer" else BiLSTMEncoder(d)
        self.hybrid = hybrid
        self.n_wx = n_wx
        self.fuse = nn.Sequential(nn.Linear(d + (N_GLOBAL if hybrid else 0) + n_wx, d), nn.SiLU(), nn.Linear(d, d))
        self.head = nn.Linear(d, 4)          # [ln beta (normalised), log dT, log sigma_x, log sigma_y]
        layers = [nn.Linear(d + 1, dec_width), nn.Tanh()]
        for _ in range(dec_depth - 1):
            layers += [nn.Linear(dec_width, dec_width), nn.Tanh()]
        layers += [nn.Linear(dec_width, 3)]
        self.dec = nn.Sequential(*layers)

    def context(self, x, pad, g=None, wx=None):
        c = self.encoder(x, pad)
        if self.hybrid:
            c = torch.cat([c, g], -1)
        if self.n_wx:
            c = torch.cat([c, wx], -1)
        c = self.fuse(c)
        h = self.head(c)
        return c, h

    def decode(self, c, tau):
        """c (B, d), tau (B, M) -> positions (B, M, 3) in units of L_POS relative to track origin."""
        B, M = tau.shape
        inp = torch.cat([c.unsqueeze(1).expand(B, M, c.shape[-1]), tau.unsqueeze(-1)], -1)
        return self.dec(inp)

    @staticmethod
    def unpack_head(h):
        lb = h[:, 0] * LB_SD + LB_MU
        dT = 100.0 * torch.exp(h[:, 1])
        log_sig = h[:, 2:4]               # IP sigma in km (log)
        return lb, dT, log_sig


def physics_residual(model, c, lnbeta, dT, z0, n_col=32, atm=None, atm_idx=None):
    """Mean squared residual of Eq. (7.2) at random collocation points, in units of g0^2."""
    B = c.shape[0]
    tau = torch.rand(B, n_col, device=c.device, requires_grad=True)
    r = model.decode(c, tau)                                   # (B, M, 3), units of L_POS
    d1, d2 = [], []
    for k in range(3):
        g1 = torch.autograd.grad(r[..., k].sum(), tau, create_graph=True)[0]
        g2 = torch.autograd.grad(g1.sum(), tau, create_graph=True)[0]
        d1.append(g1); d2.append(g2)
    d1 = torch.stack(d1, -1); d2 = torch.stack(d2, -1)
    s = dT.view(B, 1, 1)
    v = L_POS / s * d1
    a_nn = L_POS / s ** 2 * d2
    z_abs = r[..., 2] * L_POS + z0.view(B, 1)
    beta = torch.exp(lnbeta).view(B, 1)
    a_phys = t_accel(z_abs, v, beta.expand_as(z_abs), atm, atm_idx)
    res = (a_nn - a_phys) / G0
    return (res ** 2).sum(-1).mean(), res.detach()


def param_count(m):
    return sum(p.numel() for p in m.parameters())
