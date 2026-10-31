"""Train a trajectory predictor.

Examples
  python scripts/train.py --name pinn_tf               --encoder transformer --lambda-p 0.1
  python scripts/train.py --name data_tf               --encoder transformer --lambda-p 0
  python scripts/train.py --name pinn_lstm             --encoder lstm        --lambda-p 0.1
  python scripts/train.py --name pinn_tf_hybrid        --encoder transformer --lambda-p 0.1 --hybrid
  python scripts/train.py --name pinn_tf_2k --n-train 2000 ...
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajai.features import L_POS  # noqa: E402
from trajai.model import PINNPredictor, TorchAtmosphere, param_count, physics_residual  # noqa: E402


def load(split, n=None, global_key="g", feat_key="feats"):
    d = np.load(f"data/prep_{split}.npz")
    sl = slice(None, n)
    out = {k: torch.from_numpy(np.asarray(d[feat_key if k == "feats" else k][sl])) for k in ["feats", "pad", "fut", "dT", "lb", "origin"]}
    out["g"] = torch.from_numpy(np.asarray(d[global_key][sl]))
    out["origin"] = out["origin"].float()
    nn_ = out["feats"].shape[0]
    out["wx"] = torch.from_numpy(np.asarray(d["wx"][sl])) if "wx" in d else torch.zeros(nn_, 0)
    out["wx_idx"] = torch.from_numpy(np.asarray(d["wx_idx"][sl]).astype(np.int64)) if "wx_idx" in d else torch.zeros(nn_, dtype=torch.long)
    return out


def make_atm(kind):
    """Atmosphere used in the physics residual: std (exponential model), clim (mean profile), date (per-sample profile)."""
    if kind == "std":
        return None, False
    from trajai import weather
    banks = weather.load_banks()
    if kind == "clim":
        return TorchAtmosphere(banks["train"].mean()), False
    return TorchAtmosphere(banks["train"]), True


def batches(d, bs, shuffle, gen=None):
    n = d["feats"].shape[0]
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        b = {k: v[j] for k, v in d.items()}
        L = int((~b["pad"]).sum(1).max())      # trim padding
        b["feats"], b["pad"] = b["feats"][:, :L], b["pad"][:, :L]
        yield b


def step_losses(model, b, lam, n_grid=24, n_col=32, atm=None, per_sample=False):
    c, h = model.context(b["feats"], b["pad"], b["g"], b["wx"])
    lb_hat, dT_hat, log_sig = model.unpack_head(h)
    B = c.shape[0]
    K = b["fut"].shape[1]
    idx = torch.cat([torch.zeros(B, 1, dtype=torch.long), torch.full((B, 1), K - 1, dtype=torch.long),
                     torch.randint(1, K - 1, (B, n_grid))], 1)
    tau = idx.float() / (K - 1)
    r = model.decode(c, tau)
    tgt = torch.gather(b["fut"], 1, idx.unsqueeze(-1).expand(B, idx.shape[1], 3))
    l_data = ((r - tgt) ** 2).sum(-1).mean()
    l_T = ((h[:, 1] - torch.log(b["dT"] / 100.0)) ** 2).mean()
    l_b = ((h[:, 0] - b["lb"]) ** 2).mean()
    ip_err_km = (r[:, 1, :2].detach() - b["fut"][:, -1, :2]) * L_POS / 1000.0
    l_nll = (log_sig + 0.5 * ip_err_km ** 2 * torch.exp(-2 * log_sig)).sum(-1).mean()
    l_phys = torch.zeros(())
    if lam > 0:
        z0 = b["origin"][:, 2]
        idx = b["wx_idx"] if per_sample else torch.zeros_like(b["wx_idx"])
        l_phys, _ = physics_residual(model, c, lb_hat, b["dT"], z0, n_col, atm, idx if atm is not None else None)
    total = 100.0 * l_data + l_T + 0.3 * l_b + 0.05 * l_nll + lam * l_phys
    return total, dict(data=l_data.item(), T=l_T.item(), b=l_b.item(), nll=l_nll.item(), phys=float(l_phys.detach()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--encoder", default="transformer", choices=["transformer", "lstm"])
    ap.add_argument("--lambda-p", type=float, default=0.1)
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=None, help="fixed number of optimiser steps (overrides epochs)")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--feat-key", default="feats", help="track features: feats (Savitzky-Golay) or feats_rts (RTS smoother)")
    ap.add_argument("--global-key", default="g", help="EKF posterior features: g (split default) or g_date")
    ap.add_argument("--wx", action="store_true", help="feed the selected date's weather summary (27 features)")
    ap.add_argument("--phys-atm", default="std", choices=["std", "clim", "date"], help="atmosphere in the physics residual")
    ap.add_argument("--save-every", type=int, default=0,
                    help="also save the resume state every N steps inside an epoch (on resume the interrupted epoch restarts its shuffle)")
    ap.add_argument("--init-from", default=None, help="start from the weights of checkpoints/<name>.pt (fine-tuning)")
    ap.add_argument("--no-curriculum", action="store_true", help="full lambda_p from the first step (for fine-tuning)")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/<name>_last.pt if present")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    tr, va = load(a.train_split, a.n_train, a.global_key, a.feat_key), load(a.val_split, 1000, a.global_key, a.feat_key)
    model = PINNPredictor(a.encoder, hybrid=a.hybrid, n_wx=tr["wx"].shape[1] if a.wx else 0)
    if not a.wx:
        tr["wx"], va["wx"] = tr["wx"][:, :0], va["wx"][:, :0]
    atm, per_sample = make_atm(a.phys_atm)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    n = tr["feats"].shape[0]
    steps_per_epoch = math.ceil(n / a.bs)
    total_steps = a.steps or a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total_steps, pct_start=0.1)
    os.makedirs("checkpoints", exist_ok=True)
    log = []
    gen = torch.Generator().manual_seed(a.seed)
    step, t0, best = 0, time.time(), float("inf")
    last_path = f"checkpoints/{a.name}_last.pt"
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, weights_only=False)
        model.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"]); torch.set_rng_state(ck["rng"])
        step, best, log = ck["step"], ck["best"], ck["log"]
        t0 = time.time() - ck["elapsed"]
        print(f"resumed {a.name} at step {step}", flush=True)
    if a.init_from and step == 0:
        model.load_state_dict(torch.load(f"checkpoints/{a.init_from}.pt", weights_only=False)["state"])
        print(f"initialised from {a.init_from}", flush=True)
    print(f"{a.name}: {param_count(model):,} params, {n} train samples, {total_steps} steps", flush=True)
    while step < total_steps:
        model.train()
        agg = {}
        for b in batches(tr, a.bs, True, gen):
            lam = a.lambda_p if a.no_curriculum else a.lambda_p * min(1.0, step / (0.4 * total_steps))   # curriculum 0 -> lambda_p
            loss, parts = step_losses(model, b, lam, atm=atm, per_sample=per_sample)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + v
            if a.save_every and step % a.save_every == 0:
                torch.save(dict(state=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(),
                                rng=torch.get_rng_state(), step=step, best=best, log=log, elapsed=time.time() - t0), last_path)
            if step >= total_steps:
                break
        model.eval()
        with torch.no_grad():
            vl = []
            for b in batches(va, 250, False):
                c, h = model.context(b["feats"], b["pad"], b["g"], b["wx"])
                K = b["fut"].shape[1]
                tau = torch.linspace(0, 1, K).expand(c.shape[0], K)
                r = model.decode(c, tau)
                vl.append(torch.linalg.norm(r - b["fut"], dim=-1).mean(1) * L_POS)
            v_ade = torch.cat(vl).median().item()
        nb = max(1, len(agg) and steps_per_epoch)
        rec = dict(step=step, time=time.time() - t0, val_ade_m=v_ade, **{k: v / nb for k, v in agg.items()})
        log.append(rec)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        if v_ade < best:
            best = v_ade
            torch.save(dict(state=model.state_dict(), args=vars(a)), f"checkpoints/{a.name}.pt")
        torch.save(dict(state=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(),
                        rng=torch.get_rng_state(), step=step, best=best, log=log, elapsed=time.time() - t0), last_path)
    with open(f"results/train_{a.name}.json", "w") as f:
        json.dump(dict(args=vars(a), params=param_count(model), log=log, train_time_s=time.time() - t0), f, indent=1)


if __name__ == "__main__":
    main()
