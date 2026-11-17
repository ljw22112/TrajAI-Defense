"""Package the five test splits as a public benchmark (benchmark/ -> TrajAI-Bench-<version>.zip).

For every split:
  data/<split>.npz         radar input only: polar and Cartesian measurements, track length, noise, radar site
  truth/<split>.npz        targets: impact point, impact time, true future trajectory on the tau grid, beta
  stonesoup/<split>_detections.csv   one row per radar measurement   (Stone Soup CSVDetectionReader)
  stonesoup/<split>_groundtruth.csv  true positions from cut-off to impact (Stone Soup CSVGroundTruthReader)
plus leaderboard.json (the estimators reported in the technical report) and sample_submission_ekf.csv.
Track i of a split starts at epoch TIME0 + i * TRACK_GAP_S in the Stone Soup files, so tracks never overlap in time.

Usage: python scripts/export_benchmark.py [--out benchmark] [--version v1]   (needs data/*.npz and results/baseline_ekf_*.npz)
"""
import argparse
import csv
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajai import radar  # noqa: E402
from trajai.dataset import DT_MEAS, N_TAU  # noqa: E402

SPLITS = {"test_id_f05": "관측 5% (학습 범위 밖의 짧은 관측)", "test_id_f10": "관측 10%", "test_id_f15": "관측 15%",
          "test_id_noise": "관측 12%, 레이더 잡음 3배", "test_ood": "학습 범위 밖 비행 조건(고속·고β), 관측 12%"}
TIME0 = 1_767_225_600          # 2026-01-01T00:00:00Z, arbitrary epoch for the Stone Soup files
TRACK_GAP_S = 1000             # longer than any flight (< 600 s)
BOARD = [("ekf", "EKF + Monte Carlo"), ("ukf", "UKF + Monte Carlo"), ("data_tf_v2", "Transformer, 데이터만"),
         ("pinn_tf_v2", "PINN-Transformer"), ("pinn_tf_v2_rts", "PINN-Transformer, RTS 입력"),
         ("pinn_tf_hybrid_v2", "PINN-Transformer + EKF"), ("data_lstm_v2", "Bi-LSTM, 데이터만"), ("pinn_lstm_v2", "PINN-Bi-LSTM"),
         ("pinn_lstm_v2_rts", "PINN-Bi-LSTM, RTS 입력"), ("pinn_lstm_rts_9k", "PINN-Bi-LSTM, RTS 입력·학습 2배")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark")
    ap.add_argument("--version", default="v1")
    a = ap.parse_args()
    out = a.out
    for sub in ("data", "truth", "stonesoup"):
        os.makedirs(f"{out}/{sub}", exist_ok=True)
    for split in SPLITS:
        d = dict(np.load(f"data/{split}.npz"))      # load once (NpzFile re-reads on every access)
        n = d["n_meas"]
        np.savez_compressed(f"{out}/data/{split}.npz", meas_pol=d["meas_pol"], meas=d["meas"], n_meas=n,
                            sigma=d["sigma"], dt=np.float64(DT_MEAS), radar_site=radar.RADAR_SITE)
        np.savez_compressed(f"{out}/truth/{split}.npz", impact=d["impact"], T=d["T"], t_c=d["t_c"],
                            future=d["future"], beta=d["beta"])
        tau = np.linspace(0, 1, N_TAU)
        with open(f"{out}/stonesoup/{split}_detections.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "time", "x", "y", "z", "range_m", "az_rad", "el_rad"])
            for i in range(len(n)):
                for k in range(int(n[i])):
                    x, y, z = d["meas"][i, k]
                    r, az, el = d["meas_pol"][i, k]
                    w.writerow([i, f"{TIME0 + i * TRACK_GAP_S + k * DT_MEAS:.1f}", f"{x:.2f}", f"{y:.2f}", f"{z:.2f}",
                                f"{r:.2f}", f"{az:.7f}", f"{el:.7f}"])
        with open(f"{out}/stonesoup/{split}_groundtruth.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "time", "x", "y", "z"])
            for i in range(len(n)):
                t = d["t_c"][i] + tau * (d["T"][i] - d["t_c"][i])
                for j in range(N_TAU):
                    x, y, z = d["future"][i, j]
                    w.writerow([i, f"{TIME0 + i * TRACK_GAP_S + t[j]:.3f}", f"{x:.2f}", f"{y:.2f}", f"{z:.2f}"])
        print(f"{split:14s} {len(n)} tracks, {int(n.sum())} detections", flush=True)

    # leaderboard from the report's evaluation (median IP miss, RMSE where per-track errors are available, CEP coverage)
    M = json.load(open("results/metrics.json"))
    rmse_known = {}                       # RMSE recorded by scripts/lstm_compare.py when per-track files are not at hand
    import glob
    for f in glob.glob("results/lstm_*.json"):
        j = json.load(open(f))
        for sp, v in j.get("splits", {}).items():
            for role, k in (("data", "rmse_data_m"), ("pinn", "rmse_pinn_m")):
                if k in v:
                    rmse_known[(j["models"][role], sp)] = v[k]
            if "rmse_ekf_m" in v:
                rmse_known[("ekf", sp)] = v["rmse_ekf_m"]
    if os.path.exists("results/rmse_onnx.json"):
        for mdl, v in json.load(open("results/rmse_onnx.json")).items():
            for sp, r in v.items():
                rmse_known[(mdl, sp)] = r["rmse_m"]
    board = {}
    for key, label in BOARD:
        if key not in M:
            continue
        row = {"label": label}
        for s in SPLITS:
            if s in M[key]:
                r = {"miss_median_m": M[key][s]["miss"]["median"], "miss_p90_m": M[key][s]["miss"]["p90"],
                     "cep_coverage": M[key][s]["coverage"], "tot_err_median_s": M[key][s]["tot"]["median"]}
                f = f"results/{'baseline_' + key if key in ('ekf', 'ukf') else 'pred_' + key}_{s}.npz"
                if os.path.exists(f):
                    r["miss_rmse_m"] = float(np.sqrt(np.mean(np.load(f)["miss"] ** 2)))
                elif (key, s) in rmse_known:
                    r["miss_rmse_m"] = rmse_known[(key, s)]
                row[s] = r
        board[key] = row
    json.dump({"version": a.version, "splits": SPLITS, "entries": board}, open(f"{out}/leaderboard.json", "w"),
              indent=1, ensure_ascii=False)

    # sample submission: the EKF + Monte Carlo baseline (the evaluator must reproduce its leaderboard row)
    with open(f"{out}/sample_submission_ekf.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "track_id", "ip_x", "ip_y", "cep", "tot"])
        for s in SPLITS:
            b = np.load(f"results/baseline_ekf_{s}.npz")
            for i in range(len(b["mpi"])):
                w.writerow([s, i, f"{b['mpi'][i, 0]:.3f}", f"{b['mpi'][i, 1]:.3f}", f"{b['cep'][i]:.3f}", f"{b['tot_pred'][i]:.4f}"])
    for f in ("README.md", "evaluate.py", "stonesoup_example.py"):
        shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark_src", f), f"{out}/{f}")
    print("benchmark written to", out)


if __name__ == "__main__":
    main()
