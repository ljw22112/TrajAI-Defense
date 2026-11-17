"""Add estimates of newly registered models to the existing static demo snapshot (app/demo/) without re-seeding it.

Each scenarios_<id>.json already holds the measurements, weather profile and truth that the server stored, so the
estimates are computed with the same service code the API uses (estimation.run_pinn, the out-of-distribution guard
with EKF fallback, scoring and threat assessment) and appended in the API's output format. models.json, stats.json,
scenarios.json (list counts) and estimates.csv are updated to match.
Usage (from the repository root): python server/scripts/demo_add_models.py pinn_lstm_v2 data_lstm_v2
"""
import csv
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "server")]
from app.main import MODELS  # noqa: E402
from app.models import Method  # noqa: E402
from app.services import estimation, threat  # noqa: E402
from app.services import weather as wsvc  # noqa: E402

DEMO = ROOT / "app" / "demo"
TZ = datetime.now().astimezone().tzinfo


def registry_entry(name, next_id):
    kind, fname, uses_wx, key = next((k, f, u, m) for n, k, f, u, m in MODELS if n == name)
    metrics = {k: v for k, v in json.loads((ROOT / "results" / "metrics.json").read_text())[key].items() if k != "args"}
    cal = json.loads((ROOT / "results" / "calibration.json").read_text()).get(key)
    if cal:
        metrics.update(cep_scale=cal["cep_scale"], cep90_scale=cal["cep90_scale"], cep_scale_fitted_on=cal["fitted_on"])
    return dict(id=next_id, name=name, kind=kind, uses_weather=uses_wx, metrics=metrics, is_active=False,
                created_at=datetime.now(TZ).isoformat()), str(ROOT / "export" / fname)


def estimate(scn_json, mv, assets, next_id):
    meas = [SimpleNamespace(**m) for m in scn_json["measurements"]]
    s = SimpleNamespace(**{k: v for k, v in scn_json.items() if k != "measurements"}, measurements=meas)
    wrow = None
    if scn_json["weather"]:
        w = scn_json["weather"]
        wrow = SimpleNamespace(levels=w["levels"], valid_time=datetime.fromisoformat(w["valid_time"]))
    bank = wsvc.bank_of(wrow)
    res = estimation.run_pinn(s, bank, mv)
    g = estimation.guard_params(mv.name)                     # same guard and fallback as routers/scenarios.py
    if g is not None and res.get(g["input"]) is not None:
        score = estimation.ood_score(res[g["input"]], g)
        out = estimation.outside_envelope(res.get("kin"), g.get("envelope"))
        flag = score > g["threshold"] or bool(out)
        if flag:
            res = estimation.run_filter(s, bank, Method.ekf_wx if bank is not None else Method.ekf_std)
        res["trajectory"] = {**res["trajectory"], "guard": dict(score=round(score, 2), threshold=round(g["threshold"], 2),
                                                                fallback="ekf" if flag else None, input=g["input"], outside=out)}
    sc = estimation.score(res, s)
    th = threat.assess(res["ip"], res["sigma"], res["trajectory"], assets)
    traj = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in res["trajectory"].items()
            if k in ("t", "x", "z", "guard")}
    return dict(id=next_id, scenario_id=s.id, method="pinn", model_version_id=mv.id, model_name=mv.name,
                created_at=datetime.now(TZ).isoformat(), ip_x=float(res["ip"][0]), ip_y=float(res["ip"][1]), cep=float(res["cep"]),
                tot=float(res["tot"]), tot_std=res["tot_std"], beta_hat=float(res["beta_hat"]), nis=float(res["nis"]),
                latency_ms=float(res["latency_ms"]), miss_m=sc["miss_m"], inside_cep=sc["inside_cep"], trajectory=traj,
                threat={k: v for k, v in th.items()})


def main(names):
    models = json.loads((DEMO / "models.json").read_text())
    assets = [SimpleNamespace(**a) for a in json.loads((DEMO / "assets.json").read_text())]
    mvs = []
    for name in names:
        row = next((m for m in models if m["name"] == name), None)
        path = None
        if row is None:
            row, path = registry_entry(name, max(m["id"] for m in models) + 1)
            models.append(row)
        path = path or next(str(ROOT / "export" / f) for n, k, f, u, m in MODELS if n == name)
        mvs.append(SimpleNamespace(id=row["id"], name=name, kind=row["kind"], uses_weather=row["uses_weather"],
                                   file_path=path, metrics=row["metrics"]))
    (DEMO / "models.json").write_text(json.dumps(models, ensure_ascii=False))

    files = sorted(p for p in DEMO.glob("scenarios_*.json") if not p.stem.endswith("_flight"))
    all_est = [e for p in files for e in json.loads(p.read_text())["estimates"]]
    next_id = max(e["id"] for e in all_est) + 1
    added = []
    for p in files:
        scn = json.loads(p.read_text())
        scn["estimates"] = [e for e in scn["estimates"] if e.get("model_name") not in names]   # idempotent
        for mv in mvs:
            e = estimate(scn, mv, assets, next_id)
            next_id += 1
            scn["estimates"].append(e)
            added.append((scn, e))
            print(f"#{scn['id']:<3} {scn['name']:<26} {mv.name:14s} miss {e['miss_m']:8.0f} m  CEP {e['cep']:7.0f} m"
                  + ("  -> EKF fallback" if e["trajectory"].get("guard", {}).get("fallback") else ""))
        ests = scn["estimates"]
        scn["n_estimates"] = len(ests)
        scn["best_miss_m"] = min(x["miss_m"] for x in ests)
        p.write_text(json.dumps(scn, ensure_ascii=False))

    # scenario list counts
    lst_path = DEMO / "scenarios.json"
    lst = json.loads(lst_path.read_text())
    by_id = {json.loads(p.read_text())["id"]: json.loads(p.read_text()) for p in files}
    for item in lst["items"]:
        if item["id"] in by_id:
            item["n_estimates"] = by_id[item["id"]]["n_estimates"]
            item["best_miss_m"] = by_id[item["id"]]["best_miss_m"]
    lst_path.write_text(json.dumps(lst, ensure_ascii=False))

    # dashboard statistics
    stats = json.loads((DEMO / "stats.json").read_text())
    stats["by_method"] = [b for b in stats["by_method"] if b.get("model") not in names]
    for mv in mvs:
        es = [e for _, e in added if e["model_name"] == mv.name]
        stats["by_method"].append(dict(method="pinn", model=mv.name, n=len(es),
                                       median_miss_m=float(np.median([e["miss_m"] for e in es])),
                                       coverage=float(np.mean([e["inside_cep"] for e in es])),
                                       median_latency_ms=float(np.median([e["latency_ms"] for e in es]))))
    stats["estimates"] = sum(len(json.loads(p.read_text())["estimates"]) for p in files)
    lv = {}
    for p in files:
        for e in json.loads(p.read_text())["estimates"]:
            if e.get("threat"):
                lv[str(e["threat"]["level"])] = lv.get(str(e["threat"]["level"]), 0) + 1
    stats["threat_levels"] = lv
    (DEMO / "stats.json").write_text(json.dumps(stats, ensure_ascii=False))

    # CSV export: append rows with the same columns
    csv_path = DEMO / "estimates.csv"
    rows = list(csv.reader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
    head = rows[0]
    keep = [r for r in rows[1:] if len(r) > 2 and r[head.index("model")] not in names] if "model" in head else rows[1:]
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(head)
    new = []
    for scn, e in added:                                     # same columns and rounding as GET /api/export.csv
        probs = sorted(e["threat"]["probabilities"].items(), key=lambda kv: -kv[1])
        new.append([scn["id"], scn["name"], scn["weather"]["valid_time"] if scn["weather"] else "", e["id"], e["method"],
                    e["model_name"], e["created_at"], round(e["ip_x"], 1), round(e["ip_y"], 1), round(e["cep"], 1),
                    round(e["miss_m"], 1), int(e["inside_cep"]), round(e["tot"], 2), round(e["beta_hat"], 1), round(e["nis"], 3),
                    round(e["latency_ms"], 2), e["threat"]["level"], probs[0][0] if probs else "",
                    round(probs[0][1], 4) if probs else ""])
    allrows = keep + [[str(x) for x in r] for r in new]
    allrows.sort(key=lambda r: (int(r[0]), int(r[3])))
    w.writerows(allrows)
    csv_path.write_text(out.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["pinn_lstm_v2", "data_lstm_v2"])
