"""TrajAI-Bench evaluator. Scores a submission CSV against truth/<split>.npz and prints it next to the leaderboard.

Submission: one row per track, columns  split, track_id, ip_x, ip_y, cep, tot
  ip_x, ip_y  predicted impact point [m] in the benchmark's local ENU frame (same frame as the measurements)
  cep         predicted CEP50 radius [m] (radius of the circle that should contain the true impact with 50 % probability)
  tot         predicted impact time [s], same time origin as the measurements (burn-out at t = 0)
Metrics per split: median and 90th-percentile IP miss, IP RMSE, CEP50 coverage (calibrated = 0.50), median |ToT error|.
Usage: python evaluate.py submission.csv [--root .] [--json scores.json]
Only numpy is required.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def score(sub_rows, truth):
    n = len(truth["T"])
    ip, cep, tot = np.full((n, 2), np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for r in sub_rows:
        i = int(r["track_id"])
        ip[i] = float(r["ip_x"]), float(r["ip_y"])
        cep[i], tot[i] = float(r["cep"]), float(r["tot"])
    missing = int(np.isnan(ip[:, 0]).sum())
    ok = ~np.isnan(ip[:, 0])
    miss = np.linalg.norm(ip[ok] - truth["impact"][ok, :2], axis=1)
    return dict(n=int(ok.sum()), missing=missing, miss_median_m=float(np.median(miss)), miss_p90_m=float(np.percentile(miss, 90)),
                miss_rmse_m=float(np.sqrt(np.mean(miss ** 2))), cep_coverage=float(np.mean(miss <= cep[ok])),
                tot_err_median_s=float(np.median(np.abs(tot[ok] - truth["T"][ok]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = defaultdict(list)
    with open(a.submission, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["split"]].append(r)
    board = json.load(open(os.path.join(a.root, "leaderboard.json")))
    out = {}
    for split, desc in board["splits"].items():
        if split not in rows:
            print(f"{split}: no predictions in the submission")
            continue
        t = dict(np.load(os.path.join(a.root, "truth", f"{split}.npz")))
        s = out[split] = score(rows[split], t)
        print(f"\n{split}  ({desc})  n = {s['n']}" + (f", missing {s['missing']}" if s["missing"] else ""))
        print(f"  {'':28s} {'median':>8s} {'p90':>8s} {'RMSE':>8s} {'CEP cov':>8s} {'|ToT|':>7s}")
        print(f"  {'submission':28s} {s['miss_median_m']:8.0f} {s['miss_p90_m']:8.0f} {s['miss_rmse_m']:8.0f} "
              f"{s['cep_coverage']:8.2f} {s['tot_err_median_s']:7.2f}")
        for key, e in board["entries"].items():
            if split in e:
                r = e[split]
                rm = f"{r['miss_rmse_m']:8.0f}" if "miss_rmse_m" in r else f"{'—':>8s}"
                print(f"  {e['label']:28s} {r['miss_median_m']:8.0f} {r['miss_p90_m']:8.0f} {rm} {r['cep_coverage']:8.2f} "
                      f"{r['tot_err_median_s']:7.2f}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
