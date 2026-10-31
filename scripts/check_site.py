"""Consistency checks for the report and README: hand-written numbers against results/*.json, internal links,
section references, figures and stale wording. Exit code 1 on any failure (run before every deploy).
Usage: python scripts/check_site.py
"""
import json
import os
import re
import sys

R = "results"
fails, notes = [], []


def load(name):
    p = f"{R}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


M = load("metrics.json")
L = load("latency.json")
W = load("weather_filters.json")
FB = load("feedback_rts.json")
MM = load("mismatch.json")
OOD = load("ood.json")
html = open("index.html", encoding="utf-8").read()
readme = open("README.md", encoding="utf-8").read()
med = lambda m, s, k="miss": M[m][s][k]["median"]


def near(text, value, tol, where, fmt="{:,.0f}"):
    """`text` (as written) must match `value` within relative tolerance `tol`."""
    num = float(text.replace(",", ""))
    ok = abs(num - value) <= tol * max(abs(value), 1e-9)
    (notes if ok else fails).append(f"{'ok ' if ok else 'BAD'} {where}: written {text}, computed {value:,.3g}")


def claim(pattern, value, tol, where, src=html):
    m = re.search(pattern, src)
    if not m:
        fails.append(f"MISSING {where}: pattern {pattern!r} not found")
        return
    near(m.group(1), value, tol, where)


# ---- hand-written claims in the research-question table, abstract, §6, §12 and README
claim(r"지연은 충족\(([\d.]+) ms", L["onnxruntime_cpu"]["p50_ms"], 0.02, "RQ1 latency")
claim(r"약 (\d+)배 빠름", L["ekf_plus_mc100"]["p50_ms"] / L["onnxruntime_cpu"]["p50_ms"], 0.005, "RQ1 speed-up")
if "pinn_tf_v2_rts" in M:
    claim(r"EKF보다 정확\((\d[\d,]*) m vs", med("pinn_tf_v2_rts", "test_id_noise"), 0.01, "RQ1 RTS high-noise")
    claim(r"EKF보다 정확\(\d[\d,]* m vs (\d[\d,]*) m\)", med("ekf", "test_id_noise"), 0.01, "RQ1 EKF high-noise")
    red = [1 - med("pinn_tf_v2_rts", s) / med("pinn_tf_v2", s) for s in ("test_id_f10", "test_id_f15", "test_id_noise")]
    claim(r"고잡음에서 (\d+)–\d+% 감소", min(red) * 100, 0.05, "RQ3 RTS reduction min")
    claim(r"고잡음에서 \d+–(\d+)% 감소", max(red) * 100, 0.05, "RQ3 RTS reduction max")
    if med("pinn_tf_v2_rts", "test_id_f05") < med("pinn_tf_v2", "test_id_f05") and "관측 5%에서는 이득 없음" in html:
        fails.append("BAD RQ3: says no gain at ID 5% but RTS is better there")
claim(r"잔차 ([\d.]+) → [\d.]+ g\)", M["data_tf_v2"]["test_id_f10"]["phys_rms_g"]["median"], 0.05, "RQ2 residual data-only")
claim(r"잔차 [\d.]+ → ([\d.]+) g\)", M["pinn_tf_v2"]["test_id_f10"]["phys_rms_g"]["median"], 0.1, "RQ2 residual PINN")
if W:
    claim(r"EKF 오차 (\d+)% 감소", (1 - W["ekf_date"]["miss"]["median"] / W["ekf_exp"]["miss"]["median"]) * 100, 0.03, "RQ4 weather reduction")
    claim(r"기상 학습으로 ([\d.]+) → [\d.]+ km", med("pinn_tf_hybrid_v2", "test_wx") / 1000, 0.05, "RQ4 NN std-atm km")
    claim(r"기상 학습으로 [\d.]+ → ([\d.]+) km", med("wx_hybrid_clim", "test_wx") / 1000, 0.08, "RQ4 NN weather km")
if FB:
    rts = FB["rts"]["vel_err"]
    claim(r"속도 오차가 훨씬 작고\(([\d.]+) →", rts["sg"], 0.02, "§6 SG velocity error")
    claim(r"속도 오차가 훨씬 작고\([\d.]+ → ([\d.]+) m/s", rts["rts"], 0.05, "§6 RTS velocity error")
    sp = FB["feedback"]["splits"]
    if f"{len(sp)}개 조건 모두 EKF보다 정확" in html and not all(r["ci_diff"][1] < 0 for r in sp.values()):
        fails.append("BAD RQ3: feedback not better than EKF in every split")
if OOD:
    claim(r"감지\(AUROC ([\d.]+)\)", OOD["detect"]["test_ood"]["latent"]["auroc"], 0.001, "RQ5 OOD AUROC")
    claim(r"OOD 오차 ([\d.]+) → [\d.]+ km", OOD["guard"]["test_ood"]["network"] / 1000, 0.01, "RQ5 OOD before")
    claim(r"OOD 오차 [\d.]+ → ([\d.]+) km", OOD["guard"]["test_ood"]["latent"]["miss"] / 1000, 0.02, "RQ5 OOD after")
if MM:
    cov = [v["coverage"] for k, v in MM["test_mm_lift"].items() if not k.endswith("_mm")]
    claim(r"과신\(포함률 ([\d.]+)–", min(cov), 0.03, "RQ5 lift coverage min")
    claim(r"과신\(포함률 [\d.]+–([\d.]+)\)", max(cov), 0.03, "RQ5 lift coverage max")
if MM and "pinn_rts_mm" in MM["test_mm_lift"]:
    lf = MM["test_mm_lift"]
    claim(r"양력 기동에서 EKF의 ([\d.]+)배", lf["pinn_tf_v2_rts"]["miss"] / lf["ekf"]["miss"], 0.01, "RQ1 lift ratio")
    claim(r"필터를 넘지 못함\(양력 ([\d,]+) m", lf["pinn_rts_mm"]["miss"], 0.01, "RQ1 retrained lift")
    claim(r"필터를 넘지 못함\(양력 [\d,]+ m vs EKF ([\d,]+) m", lf["ekf"]["miss"], 0.01, "RQ1 EKF lift")
    if "넘지 못함" in html and lf["pinn_rts_mm"]["miss"] < lf["ekf"]["miss"]:
        fails.append("BAD RQ1: retrained model does beat the EKF under lift")
if "pinn_rts_ft_l10" in M:
    claim(r"서빙 모델 ([\d.]+) g \(ID 10%\)", M["pinn_tf_v2_rts"]["test_id_f10"]["phys_rms_g"]["median"], 0.01, "REQ-F-03 served residual")
    claim(r"λ = 10 이어 학습으로 ([\d.]+) g까지", M["pinn_rts_ft_l10"]["test_id_f10"]["phys_rms_g"]["median"], 0.01, "REQ-F-03 lambda 10 residual")
    claim(r"IP 오차가 대조 대비 (\d+)% 커짐", (M["pinn_rts_ft_l10"]["test_id_f10"]["miss"]["median"] / M["pinn_rts_ft_l1"]["test_id_f10"]["miss"]["median"] - 1) * 100, 0.03, "REQ-F-03 accuracy cost")
MN = load("maneuver.json")
if MN:
    claim(r"기동 가설을 넣으면 포함률 ([\d.]+)까지", MN["robust"]["test_mm_lift"]["0.10"]["coverage"], 0.02, "RQ5 manoeuvre-hypothesis coverage")
    ratio = MN["robust"]["test_mm_lift"]["0.10"]["cep"] / MN["robust"]["test_mm_lift"]["0.00"]["cep"]
    if "CEP가 약 2배" in html and not 1.8 <= ratio <= 2.2:
        fails.append(f"BAD RQ5: CEP widening is x{ratio:.2f}, not about 2")
    if MN["splits"]["long_lift"]["auroc"]["nis_w"] > 0.65 and "감지되지 않음" in html:
        fails.append("BAD RQ5: innovation test does detect the lift")
DT = load("decision_time.json")
if DT:
    c = DT["radii"]["cal"]["300"]
    claim(r"300 m 기준 중앙값 ([\d.]+) s", c["gain_median"], 0.02, "REQ-P-03 gain median")
    claim(r"300 m 기준 중앙값 [\d.]+ s · 평균 ([\d.]+) s", c["gain_mean"], 0.02, "REQ-P-03 gain mean")
    if c["gain_mean"] >= 25 and "미충족" in html.split("REQ-P-03")[1][:400]:
        fails.append("BAD REQ-P-03: target met but marked unmet")
for pat, val, where in ((r"PINN 단독 오차가 ID 10% (\d[\d,]*) →", med("pinn_tf_v2", "test_id_f10"), "README RTS ID10 before"),
                        (r"PINN 단독 오차가 ID 10% \d[\d,]* → (\d[\d,]*) m", med("pinn_tf_v2_rts", "test_id_f10"), "README RTS ID10 after"),
                        (r"개선되지만\(ID 10% (\d[\d,]*) →", med("pinn_tf_hybrid_v2", "test_id_f10"), "README ensemble before"),
                        (r"개선되지만\(ID 10% \d[\d,]* → (\d[\d,]*) m", med("ens_hybrid_v2", "test_id_f10"), "README ensemble after")):
    if "pinn_tf_v2_rts" in M:
        claim(pat, val, 0.01, where, readme)
# §9.10 validity row must match how many seeds exist
seeds = {"PINN": sum(m in M for m in ("pinn_tf_v2", "pinn_tf_v2_s1", "pinn_tf_v2_s2")),
         "hybrid": sum(m in M for m in ("pinn_tf_hybrid_v2", "pinn_tf_hybrid_v2_s1", "pinn_tf_hybrid_v2_s2"))}
if f"PINN·data-only는 seed {seeds['PINN']}개" not in html:
    fails.append(f"BAD §9.10: PINN/data-only seed count should be {seeds['PINN']}")
if f"하이브리드는 seed {seeds['hybrid']}개" not in html:
    fails.append(f"BAD §9.10: hybrid seed count should be {seeds['hybrid']}")

# ---- structure: anchors, section references, figures, stale words
ids = set(re.findall(r'id="([^"]+)"', html))
for href in set(re.findall(r'href="#([^"]+)"', html)):
    if href not in ids:
        fails.append(f"BAD link #{href} has no target")
subs = set(re.findall(r'<span class="no">(\d+\.\d+)</span>', html))
tops = set(re.findall(r'<span class="no">§(\d+)</span>', html))
for ref in set(re.findall(r"§(\d+(?:\.\d+)?)", re.sub(r"<[^>]+>", " ", html))):
    if "." in ref and ref not in subs or "." not in ref and ref not in tops:
        fails.append(f"BAD reference §{ref} does not exist")
for src in set(re.findall(r'<img[^>]+src="([^"]+)"', html)):
    if not src.startswith("http") and not os.path.exists(src):
        fails.append(f"BAD image {src} missing")
for word in ("TODO", "TBD", "갱신 예정", "아직 구현하지 않았습니다", "RTS 평활은 계획", "되먹이는 것은 계획"):
    if word in html:
        fails.append(f"BAD stale wording: {word!r}")
tables = re.findall(r"<b>Table (R?\d+[ab]?)</b>", html)
dups = {t for t in tables if tables.count(t) > 1}
if dups:
    fails.append(f"BAD duplicate table numbers: {sorted(dups)}")
for tag in ("div", "table", "section"):
    o, c = html.count(f"<{tag}"), html.count(f"</{tag}>")
    if o != c:
        fails.append(f"BAD unbalanced <{tag}>: {o} open, {c} close")

print("\n".join(notes))
print("\n".join(fails) if fails else "all checks passed")
sys.exit(1 if fails else 0)
