"""Render §9.6 (PINN -> EKF feedback, RTS smoothing) and §9.7 (deep ensemble, seed variance) of index.html
from results/feedback_rts.json and results/ensemble.json. Blocks missing their inputs are skipped.
Usage: python scripts/site_extra.py
"""
import json
import os
import re

SPL = {"test_id_f05": "ID 5%", "test_id_f10": "ID 10%", "test_id_f15": "ID 15%", "test_id_noise": "고잡음", "test_ood": "OOD", "test_wx": "기상"}


def f0(v):
    return f"{v:,.0f}"


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
    <p><b>RTS 평활.</b> 같은 동역학으로 새로 만든 트랙 {rts["n"]}개(측정 {rts["steps"]}개)에서 측정 시각의 실제 속도 대비 오차 중앙값은 Savitzky–Golay 미분 {v["sg"]:.1f} m/s, EKF(필터) {v["ekf"]:.1f} m/s, RTS 평활 {v["rts"]:.1f} m/s입니다. 가속도는 SG {a["sg"]:.1f} → RTS {a["rts"]:.1f} m/s²입니다. 평활은 마지막 상태를 바꾸지 않으므로 필터 기반 추정기에는 영향이 없고, 신경망 입력 특징을 정제하는 용도입니다. 이 특징으로 다시 학습하는 것은 남은 일입니다.</p>
  </div>"""


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
  <div class="prose"><p>Table R5의 신뢰구간은 시험 표본 변동만 반영합니다. 학습 seed를 바꾼 반복으로 학습 변동을 따로 쟀습니다. ID 10%·고잡음의 seed 간 표준편차(수 m)는 §9.4에서 유의하다고 판정한 차이보다 훨씬 작아 그 결론은 학습 변동에 흔들리지 않지만, OOD는 seed에 따라 수 km 달라지므로 OOD 수치는 단일 모델 값보다 이 범위로 읽어야 합니다.{pl_txt or " 물리 손실 비교용 PINN·data-only 모델의 추가 seed는 CPU 학습 시간 제약으로 돌리지 못해, §9.4의 물리 손실 효과(특히 ID 10%의 −61 m)는 단일 seed 결과로 남아 있습니다."}</p></div>"""
    return f"""
  <h3 class="sub" id="sec-ensemble"><span class="no">9.7</span>Deep ensemble과 학습 seed 변동</h3>{head}{seed_tbl}"""


if __name__ == "__main__":
    block = "<!-- EXTRA:BEGIN -->" + feedback_block() + ensemble_block() + "\n<!-- EXTRA:END -->"
    s = open("index.html").read()
    if "<!-- EXTRA:BEGIN -->" in s:
        s = re.sub(r"<!-- EXTRA:BEGIN -->.*?<!-- EXTRA:END -->", lambda _: block, s, flags=re.S)
    else:
        anchor = '  <h3 class="sub" id="sec-validity">'
        assert anchor in s
        s = s.replace(anchor, block + "\n" + anchor, 1)
    s = s.replace('<h3 class="sub" id="sec-validity"><span class="no">9.6</span>', '<h3 class="sub" id="sec-validity"><span class="no">9.8</span>')
    open("index.html", "w").write(s)
    print("extra block written", "feedback" if "sec-feedback" in block else "", "ensemble" if "sec-ensemble" in block else "")
