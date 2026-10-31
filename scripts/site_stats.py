"""Statistical significance and uncertainty calibration for §9.4–9.5 of index.html.

Significance: every comparison is paired (same 1,000 test tracks), so we report
  * the median IP miss of each method with a 95 % percentile-bootstrap CI (B = 2,000),
  * the paired-bootstrap CI of the difference of medians,
  * the win rate P(miss_A < miss_B) and a Wilcoxon signed-rank p-value,
with Holm–Bonferroni correction over all comparisons in the table.

Calibration: the raw CEP50 radius c_i of each method is rescaled by one scalar s,
fitted as the median of miss_i / c_i on a calibration half and scored on the other half
(split conformal, 2-fold cross-fitting, 50 random splits). The 90 % radius uses the
circular-Rayleigh ratio c90 = c50 * sqrt(ln 10 / ln 2). Reported coverages are always
out-of-fold, and the 90 % row uses the same s (so it tests the tail shape, not a second fit); the deployment scale written to results/calibration.json is refitted on the
whole split and is what the console applies.

Usage: python scripts/site_stats.py
"""
import json
import os
import re

import numpy as np
from scipy.stats import wilcoxon

R = "results"
B = 2000
RAY90 = float(np.sqrt(np.log(10) / np.log(2)))
LAB = {
    "ekf": "EKF + MC", "ukf": "UKF + MC", "data_tf_v2": "Transformer (data only)", "pinn_tf_v2": "PINN-Transformer",
    "pinn_tf_hybrid_v2": "PINN + EKF 하이브리드", "pinn_tf_v2_rts": "PINN (RTS 입력)", "wx_hybrid_clim": "하이브리드, 직전 1년 기상 학습",
    "wx_hybrid_date": "하이브리드, 기상 학습 + 일자 입력", "ekf_clim": "EKF, 기후값", "ekf_date": "EKF, 선택 일자 기상",
    "ekf_exp": "EKF, 표준 대기",
}
SPL = {"test_id_f05": "ID 5%", "test_id_f10": "ID 10%", "test_id_f15": "ID 15%", "test_id_noise": "고잡음", "test_ood": "OOD", "test_wx": "기상"}


def load(m, s):
    for f in (f"{R}/pred_{m}_{s}.npz", f"{R}/baseline_{m}_{s}.npz"):
        if os.path.exists(f):
            return dict(np.load(f))
    return None


def boot_median(x, rng):
    idx = rng.integers(0, len(x), (B, len(x)))
    return np.percentile(np.median(x[idx], 1), [2.5, 97.5])


def compare(a, b, s, rng):
    A, Bm = load(a, s), load(b, s)
    if A is None or Bm is None:
        return None
    x, y = A["miss"], Bm["miss"]
    idx = rng.integers(0, len(x), (B, len(x)))
    d = np.median(x[idx], 1) - np.median(y[idx], 1)
    return dict(a=a, b=b, split=s, med_a=float(np.median(x)), ci_a=boot_median(x, rng).tolist(),
                med_b=float(np.median(y)), ci_b=boot_median(y, rng).tolist(),
                diff=float(np.median(x) - np.median(y)), ci_diff=np.percentile(d, [2.5, 97.5]).tolist(),
                win=float((x < y).mean()), p=float(wilcoxon(x, y).pvalue))


def holm(rows):
    order = np.argsort([r["p"] for r in rows])
    m, run = len(rows), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * rows[i]["p"]))
        rows[i]["p_holm"] = run
    return rows


def calibrate(m, s, rng, reps=50):
    d = load(m, s)
    if d is None:
        return None
    miss, c50 = d["miss"], d["cep"]
    r = miss / np.maximum(c50, 1e-6)
    n = len(r)
    cov50, cov90, scales = [], [], []
    for _ in range(reps):
        p = rng.permutation(n)
        for cal, ev in ((p[: n // 2], p[n // 2:]), (p[n // 2:], p[: n // 2])):
            s50 = np.median(r[cal])
            cov50.append((r[ev] <= s50).mean())
            cov90.append((r[ev] <= s50 * RAY90).mean())   # same single scale: tests the shape of the error tail
            scales.append(s50)
    return dict(model=m, split=s, raw50=float((r <= 1).mean()), raw90=float((r <= RAY90).mean()),
                cal50=float(np.mean(cov50)), cal50_sd=float(np.std(cov50)), cal90=float(np.mean(cov90)),
                scale=float(np.median(r)), scale90=float(np.quantile(r, 0.9) / RAY90),
                cep_raw=float(np.median(c50)), cep_cal=float(np.median(c50) * np.median(r)))


def verdict_of(r):
    """Significant only if the Wilcoxon test (Holm) rejects AND the CI of the median difference excludes 0."""
    lo, hi = r["ci_diff"]
    if r["p_holm"] < 0.05 and (hi < 0 or lo > 0):
        return "A 우세" if r["diff"] < 0 else "B 우세"
    if r["p_holm"] < 0.05:
        return "혼재 (분포는 이동, 중앙값 차이는 불확실)"
    return "유의차 없음"


def f0(v):
    return f"{v:,.0f}"


def fp(p):
    return "&lt;0.001" if p < 1e-3 else f"{p:.3f}"


def _feedback_card():
    if not os.path.exists(f"{R}/feedback_rts.json"):
        return []
    fb = json.load(open(f"{R}/feedback_rts.json"))["feedback"]["splits"]
    d, f = fb["test_id_f05"]["default"]["miss"], fb["test_id_f05"]["feedback"]["miss"]
    n_better = sum(v["ci_diff"][1] < 0 for v in fb.values())
    return [("ok", "필터의 사전정보로 쓰면 최선", f"−{(1 - f / d) * 100:.0f}%",
             f"PINN의 β 예측을 EKF 사전으로 넣으면 ID 5% IP 오차 {f0(d)} → {f0(f)} m, 시험 분할 {len(fb)}개 중 {n_better}개에서 유의하게 개선", "#sec-feedback")]


def key_findings(html, rows, cal):
    """Abstract-level summary cards; every number is read from results/*.json so it cannot drift from §9."""
    M = json.load(open(f"{R}/metrics.json"))
    L = json.load(open(f"{R}/latency.json"))
    W = json.load(open(f"{R}/weather_filters.json"))
    med = lambda m, s, k="miss": M[m][s][k]["median"]
    pick = lambda a, b: next((r for r in rows if r["a"] == a and r["b"] == b), None)
    onnx, ekfmc = L["onnxruntime_cpu"]["p50_ms"], L["ekf_plus_mc100"]["p50_ms"]
    best_nn = min([m for m in ("pinn_tf_hybrid_v2", "pinn_tf_v2_rts") if m in M], key=lambda m: med(m, "test_id_f15"))
    h15, u15 = med(best_nn, "test_id_f15"), med("ukf", "test_id_f15")
    we, wd = W["ekf_exp"]["miss"]["median"], W["ekf_date"]["miss"]["median"]
    n0, n1 = med("pinn_tf_hybrid_v2", "test_wx"), med("wx_hybrid_clim", "test_wx")
    dd = pick("wx_hybrid_date", "wx_hybrid_clim")
    ood = med("pinn_tf_hybrid_v2", "test_ood")
    c90 = [c["cal90"] for c in cal]
    cards = [
        ("ok", "지연 목표 충족", f"{onnx:.2f} ms", f"ONNX Runtime p50, CPU 1스레드. EKF + Monte Carlo {ekfmc:.0f} ms보다 약 {ekfmc / onnx:,.0f}배 빠름", "#sec-results"),
        ("no", "신경망 단독은 필터보다 부정확", f"{f0(h15)} m vs {f0(u15)} m", f"최선 신경망({'RTS 입력 PINN' if best_nn.endswith('_rts') else '하이브리드'}) 대 UKF, ID 15% IP 오차 중앙값. 고잡음에서만 신경망이 EKF보다 정확", "#sec-stats"),
        *_feedback_card(),
        ("ok", "실제 기상 반영이 가장 큰 개선", f"−{(1 - wd / we) * 100:.0f}%", f"EKF IP 오차 {f0(we)} → {f0(wd)} m (표준 대기 → 선택 일자 기상). 기본 설정으로 채택", "#sec-weather"),
        ("ok", "신경망도 기상으로 학습하면 개선", f"{f0(n0)} → {f0(n1)} m", "직전 1년 실제 기상 속 발사 2만 회로 학습. 일자 기상 입력의 추가 효과"
         + (f"(Δ {dd['diff']:+.0f} m)는 불확실" if dd else "") , "#sec-stats"),
        ("part", "불확실성은 보정 필요", "CEP50 → 0.50", f"계수 하나로 50% 포함률은 맞지만, 같은 계수의 90% 포함률은 {len(c90)}개 중 {sum(c < 0.85 for c in c90)}개가 0.85 미만({min(c for c in c90 if c < 0.85):.2f}–{max(c for c in c90 if c < 0.85):.2f}). 오차 꼬리가 두꺼움", "#sec-calib"),
        ("no", "분포 밖 일반화 실패", f"{ood / 1000:,.1f} km", "OOD(고속·고β) 시험에서 하이브리드 IP 오차. 물리 손실만으로는 외삽이 보장되지 않음", "#sec-validity"),
    ]
    body = "\n".join(f'    <a class="key {t}" href="{href}"><span class="label">{h}</span><b>{v}</b><span>{d}</span></a>'
                     for t, h, v, d, href in cards)
    block = f"""<!-- KEY:BEGIN -->
  <div class="keys" aria-label="핵심 결과">
{body}
  </div>
<!-- KEY:END -->"""
    if "<!-- KEY:BEGIN -->" in html:
        return re.sub(r"<!-- KEY:BEGIN -->.*?<!-- KEY:END -->", lambda _: block, html, flags=re.S)
    anchor = '  <div class="meta">'
    assert anchor in html
    return html.replace(anchor, block + "\n" + anchor, 1)


def main():
    rng = np.random.default_rng(7)
    pinn = "pinn_tf_v2" if load("pinn_tf_v2", "test_id_f10") else None
    pairs = []
    ids = ["test_id_f10", "test_id_f15", "test_id_noise", "test_ood"]
    if pinn:
        pairs += [(pinn, "data_tf_v2", s, "물리 손실") for s in ids]
        pairs += [("pinn_tf_hybrid_v2", pinn, s, "EKF 정보 결합") for s in ids]
    else:
        pairs += [("pinn_tf_hybrid_v2", "data_tf_v2", s, "물리 손실 + EKF 결합") for s in ids]
    pairs += [("pinn_tf_hybrid_v2", "ukf", s, "신경망 대 필터") for s in ids]
    pairs += [("wx_hybrid_clim", "pinn_tf_hybrid_v2", "test_wx", "기상 학습"),
              ("wx_hybrid_date", "wx_hybrid_clim", "test_wx", "일자 기상 입력"),
              ("ekf_date", "ekf_clim", "test_wx", "일자 기상 (필터)"),
              ("wx_hybrid_date", "ekf_date", "test_wx", "신경망 대 필터 (기상)")]
    rows = []
    for a, b, s, g in pairs:
        r = compare(a, b, s, rng)
        if r:
            r["group"] = g
            rows.append(r)
    holm(rows)

    cal_specs = [(m, s) for m in ["ekf", "ukf", "data_tf_v2"] + ([pinn] if pinn else []) + ["pinn_tf_hybrid_v2"]
                 + (["pinn_tf_v2_rts"] if load("pinn_tf_v2_rts", "test_id_f10") else []) for s in ["test_id_f10", "test_id_noise", "test_ood"]]
    cal_specs += [(m, "test_wx") for m in ["ekf_date", "pinn_tf_hybrid_v2", "wx_hybrid_clim", "wx_hybrid_date"]]
    cal = [c for c in (calibrate(m, s, rng) for m, s in cal_specs) if c]
    # transfer test: scale fitted on ID 10 % applied to shifted splits (no refit)
    for c in cal:
        ref = next((k for k in cal if k["model"] == c["model"] and k["split"] == "test_id_f10"), None)
        if ref and c["split"] != "test_id_f10":
            d = load(c["model"], c["split"])
            c["transfer50"] = float((d["miss"] <= d["cep"] * ref["scale"]).mean())

    deploy = {}
    for c in cal:
        if c["split"] == ("test_wx" if c["model"].startswith("wx_") else "test_id_f10"):
            deploy[c["model"]] = dict(cep_scale=c["scale"], fitted_on=c["split"])
    json.dump(deploy, open(f"{R}/calibration.json", "w"), indent=1)

    trs = []
    for r in rows:
        r["verdict"] = verdict = verdict_of(r)
        trs.append(f'<tr><td>{r["group"]}</td><td>{LAB.get(r["a"], r["a"])}<br><small class="muted">vs {LAB.get(r["b"], r["b"])}</small></td>'
                   f'<td>{SPL[r["split"]]}</td>'
                   f'<td class="n">{f0(r["med_a"])} <small class="muted">[{f0(r["ci_a"][0])}, {f0(r["ci_a"][1])}]</small><br>'
                   f'{f0(r["med_b"])} <small class="muted">[{f0(r["ci_b"][0])}, {f0(r["ci_b"][1])}]</small></td>'
                   f'<td class="n">{r["diff"]:+,.0f} <small class="muted">[{r["ci_diff"][0]:+,.0f}, {r["ci_diff"][1]:+,.0f}]</small></td>'
                   f'<td class="n">{r["win"]:.2f}</td><td class="n">{fp(r["p_holm"])}</td><td>{verdict}</td></tr>')
    crs = []
    for c in cal:
        tr = f'{c["transfer50"]:.2f}' if "transfer50" in c else "—"
        crs.append(f'<tr><td>{LAB.get(c["model"], c["model"])}</td><td>{SPL[c["split"]]}</td>'
                   f'<td class="n">{c["raw50"]:.2f}</td><td class="n"><b>{c["cal50"]:.2f}</b> <small class="muted">±{c["cal50_sd"]:.2f}</small></td>'
                   f'<td class="n">{c["raw90"]:.2f}</td><td class="n"><b>{c["cal90"]:.2f}</b></td>'
                   f'<td class="n">×{c["scale"]:.2f}</td><td class="n">{f0(c["cep_raw"])} → {f0(c["cep_cal"])}</td><td class="n">{tr}</td></tr>')

    json.dump(dict(significance=rows, calibration=cal), open(f"{R}/significance.json", "w"), indent=1)
    n_sig = sum(verdict_of(r).endswith("우세") for r in rows)
    wd = next((r for r in rows if r["a"] == "wx_hybrid_date" and r["b"] == "wx_hybrid_clim"), None)
    wd_txt = ""
    if wd:
        wd_txt = (f' 일자 기상 입력의 효과(Δ {wd["diff"]:+,.0f} m, 95% CI [{wd["ci_diff"][0]:+,.0f}, {wd["ci_diff"][1]:+,.0f}], '
                  f'Holm p = {fp(wd["p_holm"])}, 승률 {wd["win"]:.2f})는 ' + {
                      "A 우세": "통계적으로 유의합니다.",
                      "유의차 없음": "통계적으로 유의하지 않습니다. 개선의 대부분은 직전 1년 기상으로 학습한 데서 나옵니다.",
                  }.get(verdict_of(wd), "부호순위 검정으로는 유의하지만 중앙값 차이의 신뢰구간이 0을 포함하고 승률도 0.5에 가깝습니다. "
                                        "작고 일관된 이동은 있으나 크기는 불확실하며, 개선의 대부분은 직전 1년 기상으로 학습한 데서 나옵니다."))
    thin = [c["cal90"] for c in cal if c["cal90"] < 0.85]
    tail_txt = (f"하지만 같은 s로 잰 90% 포함률은 {len(cal)}개 중 {len(thin)}개에서 0.85 미만({min(thin):.2f}–{max(thin):.2f})입니다. 오차 분포의 꼬리가 가우시안보다 두껍다는 뜻이므로, 높은 신뢰수준이 필요한 위협 판정에는 수준별 보정이 따로 필요합니다."
                if thin else "같은 s로 잰 90% 포함률도 모두 0.85 이상으로, 오차 꼬리가 가우시안 가정과 크게 어긋나지 않습니다.")
    ood = [c for c in cal if c["split"] == "test_ood" and "transfer50" in c]
    ood_txt = ", ".join(f'{LAB.get(c["model"], c["model"])} {c["transfer50"]:.2f}' for c in ood)

    block = f"""<!-- STATS:BEGIN -->
  <h3 class="sub" id="sec-stats"><span class="no">9.4</span>통계적 유의성</h3>
  <div class="prose">
    <p>모든 비교는 같은 1,000개 시험 궤적에 대한 쌍대 비교입니다. 중앙값의 95% 신뢰구간은 percentile bootstrap(B = {B:,}), 중앙값 차이의 구간은 쌍대 bootstrap, p값은 Wilcoxon 부호순위 검정이며 표 전체 {len(rows)}개 비교에 Holm–Bonferroni 보정을 했습니다. 승률은 같은 궤적에서 A의 IP 오차가 B보다 작은 비율입니다.</p>
  </div>
  <div class="tw">
    <table>
      <caption><b>Table R5</b>쌍대 비교, IP 오차 중앙값 [m]과 95% CI. Δ = A − B (음수면 A가 더 정확)</caption>
      <thead><tr><th>효과</th><th>A vs B</th><th>분할</th><th class="n">A / B 중앙값</th><th class="n">Δ 중앙값</th><th class="n">승률</th><th class="n">p (Holm)</th><th>판정</th></tr></thead>
      <tbody>
        {chr(10).join("        " + t for t in trs).strip()}
      </tbody>
    </table>
  </div>
  <div class="prose">
    <p>{len(rows)}개 비교 중 {n_sig}개가 보정 후 유의수준 0.05에서 유의하고 중앙값 차이의 95% 구간도 0을 포함하지 않습니다(두 조건을 모두 만족할 때만 "우세"로 판정).{wd_txt}</p>
  </div>
  <h3 class="sub" id="sec-calib"><span class="no">9.5</span>불확실성 보정 (split conformal)</h3>
  <div class="prose">
    <p>CEP 포함률이 0.5에서 벗어나면 위협 평가의 확률이 틀립니다. 각 추정기의 CEP<sub>50</sub> 반경을 스칼라 s 하나로 늘리거나 줄이되, s는 시험 분할의 절반(보정용)에서 IP 오차 / CEP 비의 중앙값으로 정하고 나머지 절반에서 포함률을 쟀습니다. 두 절반을 바꿔 가며 50번 무작위 분할한 평균입니다. 90% 열은 같은 s에 원형 Rayleigh 비 √(ln 10 / ln 2) ≈ {RAY90:.3f}를 곱한 반경으로, 따로 맞추지 않으므로 오차 분포 꼬리의 모양을 검사합니다. 마지막 열은 ID 10%에서 정한 s를 다시 맞추지 않고 분포가 바뀐 분할에 그대로 쓴 결과입니다.</p>
  </div>
  <div class="tw">
    <table>
      <caption><b>Table R6</b>CEP 포함률, 보정 전 → 보정 후 (목표 0.50 / 0.90). 보정 후 값은 모두 보정에 쓰지 않은 절반에서 잰 것</caption>
      <thead><tr><th>추정기</th><th>분할</th><th class="n">50% 전</th><th class="n">50% 후</th><th class="n">90% 전</th><th class="n">90% 후</th><th class="n">s</th><th class="n">CEP 중앙값 [m]</th><th class="n">ID 10% s 이전</th></tr></thead>
      <tbody>
        {chr(10).join("        " + t for t in crs).strip()}
      </tbody>
    </table>
  </div>
  <div class="prose">
    <p>스칼라 하나로 50% 포함률은 모든 추정기에서 목표에 맞춰집니다. {tail_txt} 분포가 바뀌면 ID에서 정한 s가 통하지 않습니다(OOD 50% 포함률: {ood_txt}). 그래서 콘솔은 모델마다 학습 분포에 맞는 분할에서 정한 s(<code>results/calibration.json</code>)를 PINN의 CEP에 곱해 위협 확률을 계산하고, 분포 밖 입력의 보정은 보장하지 않습니다.</p>
  </div>
<!-- STATS:END -->"""
    html = open("index.html").read()
    html = key_findings(html, rows, cal)
    if "<!-- STATS:BEGIN -->" in html:
        html = re.sub(r"<!-- STATS:BEGIN -->.*?<!-- STATS:END -->", lambda _: block, html, flags=re.S)
    else:
        html = html.replace("<!-- WX:END -->", "<!-- WX:END -->\n" + block, 1)
    open("index.html", "w").write(html)
    print(f"stats block written: {len(rows)} comparisons ({n_sig} significant), {len(cal)} calibration rows")


if __name__ == "__main__":
    main()
