"""Render §9.6 (PINN -> EKF feedback, RTS smoothing) and §9.7 (deep ensemble, seed variance) of index.html
from results/feedback_rts.json and results/ensemble.json. Blocks missing their inputs are skipped.
Usage: python scripts/site_extra.py
"""
import json
import os
import re

SPL = {"test_id_f05": "ID 5%", "test_id_f10": "ID 10%", "test_id_f15": "ID 15%", "test_id_noise": "고잡음", "test_ood": "OOD", "test_wx": "기상", "test_mm_cd": "Cd 곡선 오차", "test_mm_lift": "약한 양력"}


def f0(v):
    return f"{v:,.0f}"


def rts_retrain_block(base="pinn_tf_v2", rts="pinn_tf_v2_rts"):
    """Same PINN, same seed and schedule; only the track features differ (SG vs RTS). Paired over test tracks."""
    import numpy as np
    from scipy.stats import wilcoxon
    splits = [s for s in ("test_id_f05", "test_id_f10", "test_id_f15", "test_id_noise", "test_ood")
              if os.path.exists(f"results/pred_{base}_{s}.npz") and os.path.exists(f"results/pred_{rts}_{s}.npz")]
    if not splits:
        return ""
    rng = np.random.default_rng(3)
    rows, wins = [], 0
    for s in splits:
        a, b = np.load(f"results/pred_{base}_{s}.npz"), np.load(f"results/pred_{rts}_{s}.npz")
        x, y = b["miss"], a["miss"]
        idx = rng.integers(0, len(x), (2000, len(x)))
        lo, hi = np.percentile(np.median(x[idx], 1) - np.median(y[idx], 1), [2.5, 97.5])
        p = wilcoxon(x, y).pvalue * len(splits)                      # Bonferroni over the splits shown
        verdict = "RTS 우세" if hi < 0 and p < 0.05 else ("SG 우세" if lo > 0 and p < 0.05 else "유의차 없음")
        wins += verdict == "RTS 우세"
        rows.append(f'<tr><td>{SPL[s]}</td><td class="n">{f0(np.median(y))} → <b>{f0(np.median(x))}</b></td>'
                    f'<td class="n">{np.median(x) - np.median(y):+,.0f} <small class="muted">[{lo:+,.0f}, {hi:+,.0f}]</small></td>'
                    f'<td class="n">{np.median(a["phys_rms_g"]):.3f} → {np.median(b["phys_rms_g"]):.3f}</td>'
                    f'<td class="n">{np.median(a["ade"]):,.0f} → {np.median(b["ade"]):,.0f}</td><td>{verdict}</td></tr>')
    msg = ("속도·가속도 입력의 잡음을 줄이는 것이 신경망 정확도로 이어진다는 뜻입니다. RTS 평활기는 EKF 위에서 동작하므로, 이것도 칼만 필터의 정보를 신경망 입력으로 쓰는 결합 방식(Table 6의 전처리 역할)입니다. "
           "고잡음 조건에서는 이 PINN 단독이 EKF + MC(629 m)보다 정확하고, ID 15%에서는 EKF 사후 상태를 받는 하이브리드(428 m)보다도 정확합니다. 관측이 5%로 아주 짧으면 평활할 구간 자체가 짧아 이득이 없습니다." if wins >= len(splits) / 2 else
           "입력 특징이 훨씬 정확해졌는데도 예측 정확도 이득이 제한적이므로, 이 모델의 오차는 입력 잡음보다 짧은 관측에서의 외삽(β 불확실성)에서 주로 온다고 볼 수 있습니다.")
    body = "\n".join("        " + r for r in rows).strip()
    return ('</div>\n  <div class="tw">\n    <table>\n'
            '      <caption><b>Table R8b</b>PINN-Transformer, 입력 특징만 Savitzky–Golay → RTS 평활로 바꿔 같은 seed·같은 설정으로 다시 학습: '
            'IP 오차 중앙값 [m], 차이의 95% CI, 물리 잔차 RMS [g], ADE [m]</caption>\n'
            '      <thead><tr><th>분할</th><th class="n">IP 오차</th><th class="n">Δ 중앙값</th><th class="n">물리 잔차</th><th class="n">ADE</th><th>판정 (Wilcoxon, Bonferroni)</th></tr></thead>\n'
            f'      <tbody>\n        {body}\n      </tbody>\n    </table>\n  </div>\n  <div class="prose">\n'
            f'    <p>RTS 특징은 {len(splits)}개 분할 중 {wins}개에서 유의하게 더 정확합니다. {msg} '
            '단일 seed 비교이므로 차이가 작은 분할은 §9.7의 seed 변동 범위와 함께 읽어야 합니다.</p>')


MMLAB = {"ekf": "EKF + MC", "ukf": "UKF + MC", "ekf_pinn": "EKF · PINN β 사전", "data_tf_v2": "Transformer (data only)",
         "pinn_tf_v2": "PINN-Transformer", "pinn_tf_v2_rts": "PINN (RTS 입력)", "pinn_tf_hybrid_v2": "PINN + EKF 하이브리드",
         "pinn_rts_mm": "PINN (RTS 입력) · 불일치 데이터 학습", "data_rts_mm": "data only (RTS 입력) · 불일치 데이터 학습"}
OODLAB = {"state": "EKF 상태 Mahalanobis", "latent": "신경망 내부 표현 Mahalanobis", "spread": "앙상블 불일치 (seed 3개)"}


def ood_block():
    if not os.path.exists("results/ood.json"):
        return ""
    O = json.load(open("results/ood.json"))
    det = O["detect"]
    rows = []
    for k, lab in OODLAB.items():
        cells = "".join(f'<td class="n">{det[s][k]["auroc"]:.3f} <small class="muted">({det[s][k]["flagged"] * 100:.0f}%)</small></td>'
                        if s in det and k in det[s] else '<td class="n">—</td>' for s in ("test_ood", "test_mm_cd", "test_mm_lift"))
        rows.append(f"<tr><td>{lab}</td>{cells}</tr>")
    G = O["guard"]
    grows = []
    for s in ("test_id_f10", "test_id_f15", "test_ood", "test_mm_cd", "test_mm_lift"):
        if s not in G:
            continue
        g = G[s]; lat = g.get("latent", {})
        grows.append(f'<tr><td>{SPL.get(s, s)}</td><td class="n">{f0(g["network"])} <small class="muted">/ {f0(g["network_p90"])}</small></td>'
                     f'<td class="n">{f0(g["ekf"])} <small class="muted">/ {f0(g["ekf_p90"])}</small></td>'
                     f'<td class="n"><b>{f0(lat.get("miss", float("nan")))}</b> <small class="muted">/ {f0(lat.get("miss_p90", float("nan")))}</small></td>'
                     f'<td class="n">{lat.get("flagged", 0) * 100:.0f}%</td></tr>')
    o = G.get("test_ood", {}); i10 = G.get("test_id_f10", {})
    return f"""
  <h3 class="sub" id="sec-ood"><span class="no">9.8</span>분포 밖 입력 감지와 EKF 대체</h3>
  <div class="prose">
    <p>§9.2·§9.7에서 신경망은 학습 범위 밖(OOD)에서 10 km 이상 틀리면서도 그 사실을 알리지 못했습니다. 그래서 예측마다 "학습 데이터와 얼마나 다른가"를 재고, 기준을 넘으면 신경망 대신 EKF 결과를 쓰는 장치를 붙였습니다. 기준은 검증 분할 점수의 {O["threshold_quantile"] * 100:.0f}번째 백분위수(정상 트랙의 약 {100 - O["threshold_quantile"] * 100:.0f}%가 넘는 값)이며 시험 자료는 쓰지 않았습니다.</p>
  </div>
  <div class="tw">
    <table>
      <caption><b>Table R11</b>감지 점수별 AUROC(정상 ID 10·15% 트랙 대비)와 괄호 안의 감지율</caption>
      <thead><tr><th>점수</th><th class="n">OOD (고속·고β)</th><th class="n">Cd(M) 곡선 오차</th><th class="n">약한 양력</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
  </div>
  <div class="tw">
    <table>
      <caption><b>Table R12</b>RTS 입력 PINN 단독, EKF, 내부 표현 감지로 전환하는 전략: IP 오차 중앙값 / p90 [m], EKF로 넘긴 비율</caption>
      <thead><tr><th>분할</th><th class="n">신경망</th><th class="n">EKF</th><th class="n">감지 + 전환</th><th class="n">전환 비율</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in grows).strip()}
      </tbody>
    </table>
  </div>
  <div class="prose">
    <p>신경망 내부 표현의 거리는 OOD를 거의 완벽하게 가려내고(AUROC {det["test_ood"]["latent"]["auroc"]:.3f}), 전환 전략의 OOD 오차는 {f0(o.get("network", 0))} m에서 {f0(o.get("latent", {}).get("miss", 0))} m로, p90은 {f0(o.get("network_p90", 0))} m에서 {f0(o.get("latent", {}).get("miss_p90", 0))} m로 EKF 수준까지 내려갑니다. 정상 입력은 {i10.get("latent", {}).get("flagged", 0) * 100:.0f}%만 넘겨져 비용이 거의 없고, 넘겨진 트랙이 원래 어려운 경우라 ID 10% 오차는 오히려 {f0(i10.get("network", 0))} → {f0(i10.get("latent", {}).get("miss", 0))} m로 줄어듭니다. 반면 같은 비행 범위 안에서 동역학만 다른 모델 불일치는 입력 분포가 그대로라 감지하지 못합니다(AUROC 약 0.5). 이 장치는 "입력이 낯선 경우"를 막을 뿐 "물리 모델이 틀린 경우"를 막지는 못하며, 후자는 §9.9에서 다룹니다. 운용 콘솔은 같은 기준으로 RTS 입력 PINN의 예측마다 점수를 계산해, 기준을 넘으면 EKF 결과로 바꾸고 화면에 "분포 밖 → EKF"로 표시합니다.</p>
  </div>"""


def maneuver_block():
    """Longer tracking, IMM, innovation test and a manoeuvre-hypothesis CEP (scripts/maneuver.py)."""
    if not os.path.exists("results/maneuver.json"):
        return ""
    R = json.load(open("results/maneuver.json"))
    S, rb, la = R["splits"], R["robust"], R.get("lift_accel")
    lab = {"long_nom": "공칭 비행", "long_lift": "약한 양력", "long_cd": "Cd(M) 곡선 오차"}
    rows = []
    for k, e in S.items():
        au = e.get("auroc", {}).get("nis_w")
        rows.append(f'<tr><td>{lab[k]}</td><td class="n">{f0(e["ekf"]["miss"])} <small class="muted">({e["ekf"]["coverage"]:.2f})</small></td>'
                    f'<td class="n">{f0(e["imm"]["miss"])} <small class="muted">({e["imm"]["coverage"]:.2f})</small></td>'
                    f'<td class="n">{e["flagged"] * 100:.0f}%</td><td class="n">{"—" if au is None else f"{au:.2f}"}</td></tr>')
    rrows = []
    for ld in ("0.00", "0.05", "0.10"):
        a, b, c = rb["test_id_f10"][ld], rb["test_mm_lift"][ld], rb["test_mm_cd"][ld]
        rrows.append(f'<tr><td>{"없음 (기본 EKF + MC)" if ld == "0.00" else f"L/D ≤ {ld}"}</td>'
                     + "".join(f'<td class="n">{f0(z["cep"])} / {z["coverage"]:.2f}</td>' for z in (a, b, c)) + "</tr>")
    nom, lift = S["long_nom"], S["long_lift"]
    acc = (f" 양력 가속도는 공기가 짙은 하강 후반에 커져, 레이더 창 안의 최대값은 중앙값 {la['window_median']:.2f} m/s²로 비행 전체 최대값({la['flight_median']:.1f} m/s²)의 "
           f"{la['window_median'] / la['flight_median'] * 100:.0f}%에 불과합니다. 관측 동안에는 측정 잡음(15 m)과 구분하기 어려운 크기라, 기동 오차는 대부분 관측이 끝난 뒤에 생깁니다.") if la else ""
    b0, b1 = rb["test_mm_lift"]["0.00"], rb["test_mm_lift"]["0.10"]
    _mm = json.load(open("results/mismatch.json"))["test_mm_lift"]
    mlc = f'{_mm["pinn_rts_mm"]["coverage"]:.2f}' if "pinn_rts_mm" in _mm else "—"
    n0, n1 = rb["test_id_f10"]["0.00"], rb["test_id_f10"]["0.10"]
    return f"""
  <div class="tw">
    <table>
      <caption><b>Table R14</b>정점 이후까지 추적(관측 {R["frac"] * 100:.0f}%, 최대 100 s, 정점 통과 트랙 {nom["past_apogee"]:.0%}): IP 오차 중앙값 [m]과 CEP 포함률, 혁신(NIS) 검정 감지율과 AUROC</caption>
      <thead><tr><th>실제 비행</th><th class="n">EKF</th><th class="n">IMM (탄도 + 기동 모드)</th><th class="n">감지율 (오경보 기준 5%)</th><th class="n">AUROC</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
  </div>
  <div class="tw">
    <table>
      <caption><b>Table R15</b>기동 가설 CEP: Monte Carlo 입자마다 L/D ~ U(0, 상한)의 양력을 무작위 부호로 넣어 전파. CEP 중앙값 [m] / 포함률, 관측 12%</caption>
      <thead><tr><th>가정한 기동</th><th class="n">공칭 (ID 10%)</th><th class="n">약한 양력</th><th class="n">Cd(M) 곡선 오차</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rrows).strip()}
      </tbody>
    </table>
  </div>
  <div class="prose">
    <p><b>기동을 감지하거나 대비할 수 있는가.</b> 위 시험은 관측 12%에서 끊어 모든 트랙이 아직 상승 중이었으므로, 정점 이후에 시작하는 양력은 측정에 들어 있지 않았습니다. 그래서 추적을 정점 이후까지 이어 두 가지 표준 대응을 시험했습니다. 하나는 최근 {R["window"]}개 측정의 정규화 혁신 제곱(NIS)으로 모델 오차를 감지하는 검정이고(기준은 별도 공칭 트랙의 95번째 백분위수), 다른 하나는 탄도 모드와 큰 가속도 잡음의 기동 모드를 섞는 IMM 필터입니다. 결과는 부정적입니다. 양력 트랙의 감지율은 {lift["flagged"] * 100:.0f}%로 공칭의 오경보율({nom["flagged"] * 100:.0f}%)과 같고(AUROC {lift["auroc"]["nis_w"]:.2f}), IMM도 오차를 줄이지 못하면서 공칭 비행에서는 {f0(nom["ekf"]["miss"])} → {f0(nom["imm"]["miss"])} m로 손해를 봅니다.{acc}</p>
    <p>관측으로 알 수 없는 기동에는 불확실성을 미리 넓히는 수밖에 없습니다. 예측 입자마다 양력을 무작위로 넣으면(L/D ≤ 0.10) 양력 트랙의 포함률이 {b0["coverage"]:.2f}에서 {b1["coverage"]:.2f}까지 오르지만, CEP가 {f0(b0["cep"])} → {f0(b1["cep"])} m로 넓어지고 공칭 트랙에서도 {f0(n0["cep"])} → {f0(n1["cep"])} m(포함률 {n1["coverage"]:.2f})로 지나치게 보수적이 됩니다. 이 상한은 시험 자료를 만든 범위와 같아 낙관적인 가정입니다. 불일치 데이터로 학습한 신경망(포함률 {mlc})이 같은 문제를 데이터로 푸는 다른 길이며, 어느 쪽이든 위협 목록(예상 기동 범위)을 먼저 정해야 한다는 것이 운용상 결론입니다.</p>
  </div>"""


def mismatch_block():
    if not os.path.exists("results/mismatch.json"):
        return ""
    Mm = json.load(open("results/mismatch.json"))
    import json as _j
    met = _j.load(open("results/metrics.json"))
    ref = {k: met[k]["test_id_f10"]["miss"]["median"] for k in ("ekf", "ukf", "data_tf_v2", "pinn_tf_v2", "pinn_tf_v2_rts", "pinn_tf_hybrid_v2") if k in met}
    fb = _j.load(open("results/feedback_rts.json"))["feedback"]["splits"]["test_id_f10"]["feedback"]["miss"] if os.path.exists("results/feedback_rts.json") else None
    if fb:
        ref["ekf_pinn"] = fb
    keys = [k for k in MMLAB if k in Mm["test_mm_cd"]]
    rows = []
    for k in keys:
        e = Mm["test_mm_cd"]["ekf"]["miss"], Mm["test_mm_lift"]["ekf"]["miss"]
        cd, lf = Mm["test_mm_cd"][k]["miss"], Mm["test_mm_lift"][k]["miss"]
        rows.append(f'<tr><td>{MMLAB[k]}</td><td class="n">{f0(ref[k]) if k in ref else "—"}</td>'
                    f'<td class="n">{f0(cd)} <small class="muted">(×{cd / e[0]:.2f})</small></td>'
                    f'<td class="n">{f0(lf)} <small class="muted">(×{lf / e[1]:.2f})</small></td>'
                    f'<td class="n">{Mm["test_mm_cd"][k]["coverage"]:.2f} / {Mm["test_mm_lift"][k]["coverage"]:.2f}</td></tr>')
    rts_e = [Mm[s]["pinn_tf_v2_rts"]["miss"] / Mm[s]["ekf"]["miss"] for s in ("test_mm_cd", "test_mm_lift")]
    rts0 = ref["pinn_tf_v2_rts"] / ref["ekf"]
    sh = json.load(open("results/mismatch_shift.json"))   # median truth IP shift vs the same flight without mismatch
    nom = [k for k in keys if not k.endswith("_mm")]
    cov = [Mm["test_mm_lift"][k]["coverage"] for k in nom]
    retrain = ""
    if "pinn_rts_mm" in Mm["test_mm_cd"]:
        a, b = Mm["test_mm_cd"], Mm["test_mm_lift"]
        best = min(("pinn_rts_mm", "data_rts_mm"), key=lambda k: a[k]["miss"] + b[k]["miss"])
        dcd, dlf = a["pinn_rts_mm"]["miss"] - a["data_rts_mm"]["miss"], b["pinn_rts_mm"]["miss"] - b["data_rts_mm"]["miss"]
        beats = [s for s, M_ in (("Cd(M) 곡선 오차", a), ("약한 양력", b)) if M_[best]["miss"] < min(M_["ekf"]["miss"], M_["ukf"]["miss"])]
        retrain = (f' 불일치가 섞인 데이터(2만 개)로 다시 학습하면 {MMLAB[best]}이 Cd 오차 {f0(a[best]["miss"])} m, 양력 {f0(b[best]["miss"])} m로, '
                   + (f'{", ".join(beats)}에서 EKF·UKF보다 정확해집니다. 필터는 모르는 동역학을 신경망은 데이터에서 배울 수 있다는, 이 연구에서 신경망이 필터를 앞서는 첫 조건입니다.' if beats else 'EKF·UKF보다는 여전히 부정확합니다.')
                   + (" 또한" if beats else " 대신") + f' 불확실성은 훨씬 정직해져 양력에서 CEP 포함률이 {b[best]["coverage"]:.2f}까지 오릅니다(공칭 학습 {b["pinn_tf_v2_rts"]["coverage"]:.2f}).'
                   + f' 물리 손실(공칭 모델 기준)이 있는 모델과 없는 모델의 차이는 Cd 오차 {dcd:+,.0f} m, 양력 {dlf:+,.0f} m(PINN − data only)입니다.'
                   + (' 실제 동역학과 다른 공칭 물리를 강제하면 손해가 된다는 점은 PINN 설계의 주의점입니다.' if dcd > 0 and dlf > 0 else
                      ' 공칭 물리가 실제와 조금 달라도 물리 손실은 해가 되지 않고 오히려 도움이 되었습니다(궤적 모양의 큰 틀은 여전히 맞기 때문으로 보입니다).' if dcd <= 0 and dlf <= 0 else
                      ' 효과의 방향이 오차 종류에 따라 갈려, 틀린 공칭 물리의 득실은 단정할 수 없습니다.')
                   + (f' 대가로 불일치가 없는 공칭 시험(ID 10%)에서는 {f0(met["pinn_tf_v2_rts"]["test_id_f10"]["miss"]["median"])} → {f0(met["pinn_rts_mm"]["test_id_f10"]["miss"]["median"])} m로 '
                      + ("나빠집니다." if met["pinn_rts_mm"]["test_id_f10"]["miss"]["median"] > met["pinn_tf_v2_rts"]["test_id_f10"]["miss"]["median"] else "오히려 좋아집니다.")
                      if "pinn_rts_mm" in met else ""))

    return f"""
  <h3 class="sub" id="sec-mismatch"><span class="no">9.9</span>모델 불일치 시험</h3>
  <div class="prose">
    <p>지금까지의 시험은 EKF와 데이터 생성기가 같은 동역학을 써서 필터에 유리했습니다(§9.10의 "inverse crime"). 실제 표적은 공칭 모델과 다르게 날기 때문에, 실제 비행에만 두 가지 오차를 넣은 시험 분할(각 1,000회, 관측 12%)을 만들었습니다. 추정기는 모두 공칭 모델을 그대로 씁니다. (1) Cd(M) 곡선 모양 오차: 천음속 정점과 초음속 꼬리를 궤적마다 ±25% 독립적으로 바꿉니다. Cd 전체에 같은 배율을 곱하는 오차는 β로 흡수되므로 불일치가 아닙니다. (2) 약한 양력 기동: 정점 이후 L/D 0.03–0.10의 횡방향 양력. 탄착점은 공칭 비행 대비 중앙값 약 {sh['test_mm_cd'] / 1000:.1f} km(Cd), {sh['test_mm_lift'] / 1000:.1f} km(양력) 움직입니다.</p>
  </div>
  <div class="tw">
    <table>
      <caption><b>Table R13</b>모델 불일치에서의 IP 오차 중앙값 [m], 괄호는 EKF 대비 배율, CEP 포함률 (Cd / 양력)</caption>
      <thead><tr><th>추정기</th><th class="n">불일치 없음 (ID 10%)</th><th class="n">Cd(M) 곡선 오차</th><th class="n">약한 양력</th><th class="n">포함률</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
  </div>
  <div class="prose">
    <p>불일치가 생기면 모든 추정기의 오차가 커지지만, 필터와 신경망의 격차는 줄어듭니다. RTS 입력 PINN은 불일치가 없을 때 EKF의 {rts0:.2f}배였으나 Cd 오차에서 {rts_e[0]:.2f}배, 양력에서 {rts_e[1]:.2f}배입니다. 여기의 신경망도 공칭 동역학으로만 학습했으므로 같은 불일치를 겪고 있다는 점에 유의해야 합니다. 더 심각한 문제는 불확실성입니다. 양력 기동에서는 공칭 모델만 아는 모든 추정기의 CEP 포함률이 {min(cov):.2f}–{max(cov):.2f}로 떨어져, 예측 반경이 실제 오차를 크게 과소평가합니다. 모르는 기동 앞에서 필터와 신경망 모두 과신한다는 뜻이며, §9.8의 감지 장치도 이를 잡지 못합니다.{retrain}</p>
  </div>{maneuver_block()}"""


def decision_block():
    """REQ-P-03: decision-time gain of the beta prior (scripts/decision_time.py)."""
    if not os.path.exists("results/decision_time.json"):
        return ""
    D = json.load(open("results/decision_time.json"))
    rows = []
    for mode, lab in (("raw", "보정 전"), ("cal", "50% 보정")):
        for R, g in D["radii"][mode].items():
            rows.append(f'<tr><td>{lab}</td><td class="n">{int(R):,}</td><td class="n">{g["ekf_decided"]:.2f} → {g["fb_decided"]:.2f}</td>'
                        f'<td class="n">{g["ekf_left_median"]:.0f} → {g["fb_left_median"]:.0f}</td>'
                        f'<td class="n">{g["gain_median"]:.1f} / {g["gain_mean"]:.1f}</td><td class="n">{g["earlier"]:.2f} / {g["later"]:.2f}</td></tr>')
    c, raw = D["radii"]["cal"]["300"], D["radii"]["raw"]
    cuts = D["per_cut"]
    return f"""
  <div class="tw">
    <table>
      <caption><b>Table R8c</b>교전 판단 시점: 같은 {D["n"]}개 비행을 관측 {cuts[0]["frac"] * 100:.0f}–{cuts[-1]["frac"] * 100:.0f}%에서 잘라, CEP<sub>50</sub> ≤ R이 처음 되는 시점에 판단. EKF → EKF + PINN β 사전</caption>
      <thead><tr><th>CEP</th><th class="n">R [m]</th><th class="n">판단 가능 비율</th><th class="n">판단 시 남은 시간 중앙값 [s]</th><th class="n">앞당긴 시간 중앙값 / 평균 [s]</th><th class="n">빨라진 / 늦어진 비율</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
  </div>
  <div class="prose">
    <p><b>교전 판단 시점(REQ-P-03).</b> 계산이 빨라지는 것(0.23 s)과 별개로, 같은 측정으로 더 일찍 믿을 만한 탄착점을 얻으면 판단 시간이 늘어납니다. 보정 전 EKF CEP는 관측이 늘수록 보수적이어서(포함률 {cuts[2]["cov_ekf"]:.2f}–{cuts[-1]["cov_ekf"]:.2f}, 관측 10–30%) β 사전이 줄인 오차가 CEP에 드러나지 않고, 앞당겨지는 시간도 1 s 미만입니다. 짝수 번째 트랙으로 포함률을 50%에 맞춘 CEP(배율 EKF {D["scale"]["ekf"]:.2f}, β 사전 {D["scale"]["fb"]:.2f})로 판단하면 R = 300 m에서 트랙의 {c["earlier"]:.0%}가 더 일찍 판단되고 늦어지는 트랙은 {c["later"]:.0%}이며, 앞당긴 시간은 중앙값 {c["gain_median"]:.1f} s, 평균 {c["gain_mean"]:.1f} s입니다(평균에는 EKF가 끝까지 판단하지 못한 트랙이 포함). 요구한 +25 s에는 못 미칩니다. β 사전의 이득은 β가 아직 관측되지 않은 짧은 관측 구간에 몰려 있고, 관측이 길어지면 EKF가 β를 스스로 추정해 차이가 사라지기 때문입니다. 시점의 해상도는 자르는 간격(관측 2.5–5%, 약 5–9 s)으로 제한됩니다.</p>
  </div>"""


def feedback_block():
    if not os.path.exists("results/feedback_rts.json"):
        return ""
    R = json.load(open("results/feedback_rts.json"))
    fb, rts = R["feedback"], R["rts"]
    rows = []
    for s, r in fb["splits"].items():
        d, f = r["default"], r["feedback"]
        lo, hi = r["ci_diff"]
        sig = "개선" if hi < 0 else ("악화" if lo > 0 else "차이 불확실")
        rows.append(f'<tr><td>{SPL[s]}</td><td class="n">{f0(d["miss"])} → <b>{f0(f["miss"])}</b></td>'
                    f'<td class="n">{f["miss"] - d["miss"]:+,.0f} <small class="muted">[{lo:+,.0f}, {hi:+,.0f}]</small></td>'
                    f'<td class="n">{r["win"]:.2f}</td><td class="n">{d["coverage"]:.2f} → {f["coverage"]:.2f}</td>'
                    f'<td class="n">{d["lb_err"]:.3f} → {f["lb_err"]:.3f}</td><td>{sig}</td></tr>')
    best = min(fb["splits"].items(), key=lambda kv: kv[1]["feedback"]["miss"] / kv[1]["default"]["miss"])
    ood = fb["splits"].get("test_ood")
    v, a = rts["vel_err"], rts["acc_err"]
    return f"""
  <h3 class="sub" id="sec-feedback"><span class="no">9.6</span>PINN → EKF 되먹임과 RTS 평활</h3>
  <div class="prose">
    <p>Table 6에서 계획으로 남겼던 결합의 나머지 방향을 구현했습니다. 트랙만 보는 PINN(<code>{fb["model"]}</code>, EKF 입력 없음)이 예측한 ln β를 EKF의 사전분포로 넣습니다. 기본 사전은 ln 2450 ± {fb["default_sd"]:.1f}(학습 범위 전체)이고, 되먹임 사전의 폭 {fb["prior_sd"]:.3f}은 검증 분할에서 PINN 예측 오차의 강건 표준편차로 정했습니다. 나머지 EKF + Monte Carlo는 그대로입니다.</p>
  </div>
  <div class="tw">
    <table>
      <caption><b>Table R8</b>EKF 기본 사전 → PINN β 사전: IP 오차 중앙값 [m], 차이의 95% bootstrap CI, 승률, CEP 포함률, |Δ ln β| 중앙값</caption>
      <thead><tr><th>분할</th><th class="n">IP 오차</th><th class="n">Δ 중앙값</th><th class="n">승률</th><th class="n">CEP 포함률</th><th class="n">|Δ ln β|</th><th>판정</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
  </div>
  <div class="prose">
    <p>β를 관측하기 어려운 짧은 관측에서 효과가 가장 큽니다({SPL[best[0]]}: {f0(best[1]["default"]["miss"])} → {f0(best[1]["feedback"]["miss"])} m). 학습된 사전은 EKF가 관측만으로 좁히지 못하는 β 불확실성을 메우는 역할을 하며, 신경망을 대체가 아니라 필터의 사전정보로 쓰는 것이 이 시스템에서 가장 효과적인 결합 방식입니다.{(f' OOD(β 10,000–16,000, 학습 범위 밖)에서도 {f0(ood["default"]["miss"])} → {f0(ood["feedback"]["miss"])} m로 개선되는데, PINN의 ln β 오차({ood["feedback"]["lb_err"]:.3f})가 EKF 기본 사전으로 수렴한 값의 오차({ood["default"]["lb_err"]:.3f})보다 작기 때문입니다. 신경망 단독으로는 OOD에서 실패하지만(§9.2), 필터 안의 사전정보로 쓰면 필터의 물리 모델이 잘못된 사전을 측정으로 바로잡을 수 있습니다.' if ood and ood["feedback"]["miss"] < ood["default"]["miss"] else (f' OOD에서는 PINN의 β 예측이 학습 범위 밖이라 사전이 틀릴 수 있습니다(OOD: {f0(ood["default"]["miss"])} → {f0(ood["feedback"]["miss"])} m).' if ood else ''))}</p>
    <p><b>RTS 평활.</b> 같은 동역학으로 새로 만든 트랙 {rts["n"]}개(측정 {rts["steps"]}개)에서 측정 시각의 실제 속도 대비 오차 중앙값은 Savitzky–Golay 미분 {v["sg"]:.1f} m/s, EKF(필터) {v["ekf"]:.1f} m/s, RTS 평활 {v["rts"]:.1f} m/s입니다. 가속도는 SG {a["sg"]:.1f} → RTS {a["rts"]:.1f} m/s²입니다. 평활은 마지막 상태를 바꾸지 않으므로 필터 기반 추정기에는 영향이 없고, 신경망 입력 특징을 정제하는 용도입니다.</p>{rts_retrain_block()}
  </div>{decision_block()}"""


def _ens_text(ens):
    ids = [k for k in ("test_id_f05", "test_id_f10", "test_id_f15", "test_id_noise") if k in ens]
    better = [k for k in ids if ens[k]["ensemble"]["miss"]["median"] < ens[k]["single"]["miss"]["median"]]
    t = ""
    if "test_id_f10" in ens:
        r = ens["test_id_f10"]
        t += (f"학습 분포 안(ID)에서는 {len(better)}/{len(ids)}개 분할에서 IP 오차가 줄고(ID 10% {f0(r['single']['miss']['median'])} → "
              f"{f0(r['ensemble']['miss']['median'])} m), 50% 포함률이 0.5 쪽으로 올라갑니다. ")
    if "test_ood" in ens:
        o = ens["test_ood"]
        t += (f"반면 OOD에서는 오차가 {f0(o['single']['miss']['median'])} → {f0(o['ensemble']['miss']['median'])} m로 "
              + ("줄지만" if o["ensemble"]["miss"]["median"] < o["single"]["miss"]["median"] else "오히려 늘고")
              + f" 90% 포함률도 {o['ensemble']['coverage90']:.2f}에 그칩니다. 모델 간 분산이 전체 예측 분산에서 차지하는 비율이 OOD에서도 "
              f"{o['epistemic_share']:.2f}로 ID와 비슷해, 같은 데이터로 seed만 바꾼 3개 모델은 분포 밖 입력을 알아채지 못합니다. "
              "분포 이동 탐지에는 입력 특징의 분포 거리나 데이터 다양화 같은 별도 장치가 필요합니다.")
    return t


def ensemble_block():
    if not os.path.exists("results/ensemble.json"):
        return ""
    E = json.load(open("results/ensemble.json"))
    ens, sv = E["ensemble"], E["seeds"]
    rows = [f'<tr><td>{SPL[s]}</td><td class="n">{f0(r["single"]["miss"]["median"])} → <b>{f0(r["ensemble"]["miss"]["median"])}</b></td>'
            f'<td class="n">{r["single"]["coverage"]:.2f} → {r["ensemble"]["coverage"]:.2f}</td>'
            f'<td class="n">{r["single"]["coverage90"]:.2f} → <b>{r["ensemble"]["coverage90"]:.2f}</b></td>'
            f'<td class="n">{r["epistemic_share"]:.2f}</td></tr>' for s, r in ens.items()]
    srows = []
    for label, g in sv.get("groups", {}).items():
        cells = "".join(f'<td class="n">{f0(g["splits"][s]["mean"])} <small class="muted">± {f0(g["splits"][s]["sd"])}</small></td>' if s in g["splits"] else '<td class="n">—</td>'
                        for s in ("test_id_f10", "test_id_f15", "test_id_noise", "test_ood"))
        srows.append(f'<tr><td>{label} <small class="muted">(seed {len(g["models"])}개)</small></td>{cells}</tr>')
    pl = sv.get("physics_loss_by_seed", {})
    pl_txt = ""
    if pl:
        cons = [SPL[s] for s, r in pl.items() if r["consistent"]]
        incons = [SPL[s] for s, r in pl.items() if not r["consistent"]]
        pl_txt = (f" 물리 손실의 IP 오차 효과(PINN − data only)는 seed를 바꿔도 부호가 같은 분할이 {', '.join(cons) or '없음'}"
                  f"{', 바뀌는 분할이 ' + ', '.join(incons) if incons else ''}입니다. 부호가 바뀌는 분할의 효과는 학습 무작위성 수준이므로 결론에서 뺍니다.")
    if not ens:
        head = ""
    else:
        id10 = ens.get("test_id_f10")
        head = f"""
  <div class="tw">
    <table>
      <caption><b>Table R9</b>하이브리드 단일 모델(seed 0) → 앙상블(seed 0–2): IP 오차 중앙값 [m], CEP<sub>50</sub>·CEP<sub>90</sub> 포함률(보정 전, 목표 0.50 / 0.90), 예측 분산 중 모델 간 분산 비율</caption>
      <thead><tr><th>분할</th><th class="n">IP 오차</th><th class="n">50% 포함률</th><th class="n">90% 포함률</th><th class="n">모델 간 분산 비율</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in rows).strip()}
      </tbody>
    </table>
  </div>
  <div class="prose">
    <p>앙상블 예측 분산은 구성원 분산의 평균(관측 잡음에서 오는 불확실성)과 구성원 예측의 분산(모델 불확실성)의 합입니다. {_ens_text(ens)} 콘솔 모델 레지스트리에 앙상블(<code>ens_hybrid_v2</code>, ONNX 3개)을 등록했습니다.</p>
  </div>"""
    _m = json.load(open("results/metrics.json"))
    gap = min(_m[k]["test_id_f10"]["miss"]["median"] for k in ("pinn_tf_v2_rts", "pinn_tf_hybrid_v2", "ens_hybrid_v2", "pinn_tf_v2") if k in _m) - _m["ekf"]["test_id_f10"]["miss"]["median"]

    def sd_rng(k, div=1):
        v = [g["splits"][k]["sd"] / div for g in sv["groups"].values() if k in g["splits"]]
        return f"{min(v):,.0f}–{max(v):,.0f}" if div == 1 else f"{min(v):.1f}–{max(v):.1f}"
    seed_tbl = "" if not srows else f"""
  <div class="tw">
    <table>
      <caption><b>Table R10</b>학습 seed에 따른 IP 오차 중앙값 [m], 평균 ± 표준편차</caption>
      <thead><tr><th>모델</th><th class="n">ID 10%</th><th class="n">ID 15%</th><th class="n">고잡음</th><th class="n">OOD</th></tr></thead>
      <tbody>
        {chr(10).join("        " + r for r in srows).strip()}
      </tbody>
    </table>
  </div>
  <div class="prose"><p>Table R5의 신뢰구간은 시험 표본 변동만 반영합니다. 학습 seed를 바꾼 반복으로 학습 변동을 따로 쟀습니다. seed 간 표준편차는 ID 10% {sd_rng('test_id_f10')} m, 고잡음 {sd_rng('test_id_noise')} m로, ID 10%에서 EKF와 가장 가까운 신경망의 차이({gap:,.0f} m)보다 작지만 물리 손실 유무 같은 신경망끼리의 차이(수십 m)와는 비슷한 크기입니다. OOD는 seed에 따라 {sd_rng('test_ood', 1000)} km 흔들리므로 OOD 수치는 단일 모델 값보다 이 범위로 읽어야 합니다.{pl_txt or " 물리 손실 비교용 PINN·data-only 모델의 추가 seed는 CPU 학습 시간 제약으로 돌리지 못해, §9.4의 물리 손실 효과(특히 ID 10%의 −61 m)는 단일 seed 결과로 남아 있습니다."}</p></div>"""
    return f"""
  <h3 class="sub" id="sec-ensemble"><span class="no">9.7</span>Deep ensemble과 학습 seed 변동</h3>{head}{seed_tbl}"""


if __name__ == "__main__":
    block = "<!-- EXTRA:BEGIN -->" + feedback_block() + ensemble_block() + ood_block() + mismatch_block() + "\n<!-- EXTRA:END -->"
    s = open("index.html").read()
    if "<!-- EXTRA:BEGIN -->" in s:
        s = re.sub(r"<!-- EXTRA:BEGIN -->.*?<!-- EXTRA:END -->", lambda _: block, s, flags=re.S)
    else:
        anchor = '  <h3 class="sub" id="sec-validity">'
        assert anchor in s
        s = s.replace(anchor, block + "\n" + anchor, 1)
    s = re.sub(r'<h3 class="sub" id="sec-validity"><span class="no">9\.\d+</span>', '<h3 class="sub" id="sec-validity"><span class="no">9.10</span>', s)
    open("index.html", "w").write(s)
    print("extra block written", "feedback" if "sec-feedback" in block else "", "ensemble" if "sec-ensemble" in block else "")
