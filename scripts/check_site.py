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
claim(r"약 (\d+)배 빠름", L["ekf_plus_mc100"]["p50_ms"] / L["onnxruntime_cpu"]["p50_ms"], 0.05, "RQ1 speed-up")
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
