"""Render the §9.1 results block of index.html from results/metrics.json and results/latency.json.

The block between <!-- RESULTS:BEGIN --> and <!-- RESULTS:END --> is replaced in place.
Usage: python scripts/site_results.py <primary-hybrid> <pinn> <data-only> [--label v1] [--pair2k data_2k pinn_2k] [--lstm name]
"""
import argparse
import json
import re

M = json.load(open("results/metrics.json"))
LAT = json.load(open("results/latency.json"))
TESTS = ["test_id_f05", "test_id_f10", "test_id_f15", "test_id_noise", "test_ood"]
TLAB = ["ID 5%", "ID 10%", "ID 15%", "ID 고잡음", "OOD 12%"]


def med(m, t, k):
    return M[m][t][k]["median"]


def f0(v):
    return f"{v:,.0f}"


def _mixed(D, P):
    """Spell out where the physics loss helps and where it does not (IP miss, all test splits)."""
    better = [TLAB[i] for i, t in enumerate(TESTS) if med(P, t, "miss") < med(D, t, "miss")]
    worse = [TLAB[i] for i, t in enumerate(TESTS) if med(P, t, "miss") >= med(D, t, "miss")]
    if not worse:
        return " IP 오차는 모든 시험 분할에서 data-only보다 작습니다."
    return (f" IP 오차 이득은 분할에 따라 갈립니다. {', '.join(better) or '없음'}에서는 PINN이 더 작고, {', '.join(worse)}에서는 data-only와 같거나 더 큽니다. "
            "물리 손실의 확실한 효과는 궤적의 물리 일관성(잔차)이며, 정확도 이득은 조건부입니다(유의성은 §9.4).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hybrid"); ap.add_argument("pinn"); ap.add_argument("data")
    ap.add_argument("--label", default="v1")
    ap.add_argument("--steps", default="1,500")
    ap.add_argument("--train-note", default="공칭 잡음만으로 학습")
    ap.add_argument("--pair2k", nargs=2, default=None)
    ap.add_argument("--lstm", default=None)
    ap.add_argument("--pinn-label", default="PINN-Transformer")
    ap.add_argument("--prev", nargs=3, default=None, help="previous hybrid/pinn/data names for the comparison table")
    a = ap.parse_args()
    H, P, D = a.hybrid, a.pinn, a.data
    methods = [("ekf", "EKF + MC"), ("ukf", "UKF + MC"), (D, "Transformer (data only)"), (P, a.pinn_label)]
    if a.lstm:
        methods.append((a.lstm, "PINN-BiLSTM"))
    methods.append((H, "PINN-Transformer + EKF"))

    rows = []
    for m, lab in methods:
        prop = ' class="prop"' if m in (P, H, a.lstm) else ""
        cells = "".join(f'<td class="n">{f0(med(m, t, "miss"))} <span class="sub-c">({M[m][t]["coverage"]:.2f})</span></td>' for t in TESTS)
        rows.append(f"<tr{prop}><td>{lab}</td>{cells}</tr>")
    t_main = "\n".join(rows)

    def detail(t):
        r = []
        for m, lab in methods:
            res = f'{M[m][t]["phys_rms_g"]["median"]:.3f}' if "phys_rms_g" in M[m][t] else "—"
            prop = ' class="prop"' if m in (P, H, a.lstm) else ""
            r.append(f'<tr{prop}><td>{lab}</td><td class="n">{f0(med(m, t, "ade"))}</td><td class="n">{med(m, t, "tot"):.2f}</td>'
                     f'<td class="n">{f0(med(m, t, "cep"))}</td><td class="n">{res}</td></tr>')
        return "\n".join(r)

    pair = ""
    if a.pair2k:
        d2, p2 = a.pair2k
        pr = []
        for m, lab in [(d2, "Transformer (data only), 2k"), (p2, "PINN-Transformer, 2k"), (D, "Transformer (data only), 20k"), (P, "PINN-Transformer, 20k")]:
            pr.append(f'<tr{" class=\"prop\"" if "PINN" in lab else ""}><td>{lab}</td>' + "".join(
                f'<td class="n">{f0(med(m, t, "miss"))}</td>' for t in ["test_id_f10", "test_id_noise", "test_ood"]) +
                f'<td class="n">{M[m]["test_id_f10"]["phys_rms_g"]["median"]:.3f}</td></tr>')
        pair = f"""
  <div class="tw">
    <table>
      <caption><b>Table R3</b>학습 데이터 규모 ablation: IP 오차 중앙값 [m]과 물리 잔차 [g]</caption>
      <thead><tr><th>모델</th><th>ID 10%</th><th>ID 고잡음</th><th>OOD</th><th>잔차 (ID 10%)</th></tr></thead>
      <tbody>
{chr(10).join(pr)}
      </tbody>
    </table>
  </div>"""

    same_run = a.prev and P == a.prev[1]          # no current-generation PINN: fall back to the previous pair
    pd_, pp_ = (a.prev[2], a.prev[1]) if same_run else (D, P)
    rd, rp = med(pd_, "test_id_f10", "phys_rms_g"), med(pp_, "test_id_f10", "phys_rms_g")
    md, mp = med(pd_, "test_id_f10", "miss"), med(pp_, "test_id_f10", "miss")
    cond = "같은 v1 학습 조건에서 " if same_run else f"같은 {a.label} 학습 조건에서 "
    phys_txt = (f"{cond}ID 10%의 물리 잔차 RMS가 {rd:.2f} g(data only)에서 {rp:.2f} g(PINN)로 줄었고, IP 오차도 {f0(md)} m에서 {f0(mp)} m로 "
                + ("줄었습니다." if mp < md else "늘었습니다.") + f" 잡음 증강 하이브리드({a.label})의 잔차는 {med(H, 'test_id_f10', 'phys_rms_g'):.3f} g입니다."
                + ("" if same_run else _mixed(D, P)))
    prev = ""
    if a.prev:
        ph, pp, pd = a.prev
        pv = []
        for (m0, m1, lab) in [(pd, D, "Transformer (data only)"), (pp, P, "PINN-Transformer"), (ph, H, "PINN-Transformer + EKF")]:
            if m0 == m1:
                continue
            pv.append(f"<tr><td>{lab}</td>" + "".join(f'<td class="n">{f0(med(m0, t, "miss"))} → <b>{f0(med(m1, t, "miss"))}</b></td>' for t in TESTS) + "</tr>")
        prev = f"""
  <div class="tw">
    <table>
      <caption><b>Table R4</b>v1 → {a.label}: IP 오차 중앙값 [m]. v1은 공칭 잡음 학습·1,500 step, {a.label}는 잡음 증강 학습·{a.steps} step</caption>
      <thead><tr><th>모델</th>{"".join(f"<th>{l}</th>" for l in TLAB)}</tr></thead>
      <tbody>
{chr(10).join(pv)}
      </tbody>
    </table>
  </div>"""

    L = LAT
    speed = L["ekf_plus_mc100"]["p50_ms"] / L["onnxruntime_cpu"]["p50_ms"]
    res_d, res_p = med(D, "test_id_f10", "phys_rms_g"), med(P, "test_id_f10", "phys_rms_g")
    best_nn_15 = min(med(m, "test_id_f15", "miss") for m in (D, P, H))
    html = f"""<!-- RESULTS:BEGIN -->
  <h3 class="sub" id="sec-results"><span class="no">9.1</span>실험 결과 ({a.label})</h3>
  <div class="prose">
    <p>모든 수치는 각 시험 분할 1,000개 궤적의 중앙값입니다. 신경망은 CPU에서 {a.steps} step(batch 128) 학습했고 {a.train_note}했습니다. EKF/UKF는 데이터 생성기와 같은 동역학 모델을 쓰므로 모델 불일치가 없는, baseline에 유리한 조건입니다. 재현 방법은 저장소의 README에 있습니다.</p>
  </div>
  <div class="tw">
    <table>
      <caption><b>Table R1</b>IP 오차 중앙값 [m]과 괄호 안의 CEP<sub>50</sub> 포함률 (보정된 추정기는 0.50)</caption>
      <thead><tr><th>모델</th>{"".join(f"<th>{l}</th>" for l in TLAB)}</tr></thead>
      <tbody>
{t_main}
      </tbody>
    </table>
  </div>
  <figure><div class="sheet fig-img"><img src="figures/ip_miss.png" alt="시험 분할별 모델의 IP 오차 중앙값 막대그래프 (로그 축)"></div>
    <figcaption><b>Fig. 5</b>IP 오차 중앙값 (로그 축). 같은 색 순서는 Table R1의 행 순서와 같습니다.</figcaption></figure>
  <div class="tw">
    <table>
      <caption><b>Table R2</b>세부 지표, ID 10% / OOD: ADE [m], ToT 오차 [s], CEP<sub>50</sub> [m], 물리 잔차 RMS [g]</caption>
      <thead><tr><th>모델</th><th>ADE</th><th>ToT</th><th>CEP</th><th>잔차</th></tr></thead>
      <tbody>
<tr><td colspan="5" class="grp">ID 10%</td></tr>
{detail("test_id_f10")}
<tr><td colspan="5" class="grp">OOD 12%</td></tr>
{detail("test_ood")}
      </tbody>
    </table>
  </div>{pair}{prev}
  <div class="fig-row">
    <figure><div class="sheet fig-img"><img src="figures/physics_residual.png" alt="모델별 물리 잔차 RMS 막대그래프"></div>
      <figcaption><b>Fig. 6</b>식 (7.2)의 물리 잔차 RMS. 물리 손실이 있는 모델은 ID에서 잔차가 크게 줄어듭니다.</figcaption></figure>
    <figure><div class="sheet fig-img"><img src="figures/cep_coverage.png" alt="모델별 CEP 포함률 막대그래프"></div>
      <figcaption><b>Fig. 7</b>CEP<sub>50</sub> 포함률. 점선 0.5가 보정된 값이며, 아래는 과신, 위는 과소신입니다.</figcaption></figure>
  </div>
  <figure><div class="sheet fig-img"><img src="figures/latency.png" alt="단일 표적 추론 지연 비교 막대그래프 (로그 축)"></div>
    <figcaption><b>Fig. 8</b>단일 표적 지연 (CPU 1스레드, 다른 학습 작업과 동시 실행). PINN은 ONNX Runtime 기준 p50 {L["onnxruntime_cpu"]["p50_ms"]:.2f} ms, EKF + Monte Carlo(100 표본, NumPy)는 {L["ekf_plus_mc100"]["p50_ms"]:.0f} ms로 약 {speed:,.0f}배 차이입니다. ONNX 출력은 PyTorch와 최대 {L["onnx_vs_torch_max_diff_m"]*100:.1f} cm 이내로 일치합니다.</figcaption></figure>
  <h3 class="sub"><span class="no">9.2</span>해석</h3>
  <div class="prose">
    <p><b>물리 손실은 궤적의 물리 일관성을 확실히 높입니다.</b> {phys_txt} REQ-F-03의 목표(0.05 g)에는 아직 못 미칩니다.</p>
    <p><b>EKF 정보 결합이 가장 효과가 큽니다.</b> EKF 사후 상태를 입력으로 받는 하이브리드 모델이 신경망 중 가장 정확하며(ID 15%에서 {f0(med(H, "test_id_f15", "miss"))} m), 필터가 단기 상태를, 신경망이 장기 궤적을 맡는 Table 6의 구조를 뒷받침합니다.</p>
    <p><b>정확도는 아직 고전 필터에 못 미칩니다.</b> 신경망의 최선 IP 오차(ID 15% {f0(best_nn_15)} m)는 EKF({f0(med("ekf", "test_id_f15", "miss"))} m)와 UKF({f0(med("ukf", "test_id_f15", "miss"))} m)보다 큽니다. 모델 불일치가 없는 합성 환경에서는 정확한 물리 모델을 가진 필터가 상한에 가깝기 때문입니다. 신경망의 이점은 현재로서는 정확도보다 지연(약 {speed:,.0f}배)과 일정한 추론 시간에 있습니다.</p>
    <p><b>분포 이동에 취약합니다.</b> 학습 분포 밖의 OOD 조건에서는 모든 신경망의 IP 오차가 10 km 이상으로 커지고 CEP 포함률이 0.1 수준으로 떨어집니다. 반면 EKF/UKF는 물리 모델이 그대로 성립하므로 영향이 작습니다. 물리 잔차를 손실에 넣는 것만으로는 외삽 일반화가 보장되지 않는다는 결과입니다.</p>
    <p><b>필터 쪽의 결과.</b> 관측 5%에서 EKF는 CEP 포함률 {M["ekf"]["test_id_f05"]["coverage"]:.2f}로 과신하고 UKF는 {M["ukf"]["test_id_f05"]["coverage"]:.2f}로 보정이 유지되어, 공분산이 큰 초기 구간에서 선형화 오차가 문제가 된다는 §6.1의 예상과 일치합니다.</p>
  </div>
<!-- RESULTS:END -->"""
    # ---------------- requirements table (Table 2): measured value + verdict
    nn = [x for x in (D, P, H, a.lstm) if x]
    best = min(nn, key=lambda m: med(m, "test_id_f15", "miss"))
    best10 = min(nn, key=lambda m: med(m, "test_id_f10", "miss"))
    res_best = min(med(m, "test_id_f10", "phys_rms_g") for m in (P, H))
    rel = (med(best10, "test_id_f10", "miss") / med("ekf", "test_id_f10", "miss") - 1) * 100
    saved_s = (L["ekf_plus_mc100"]["p50_ms"] - L["onnxruntime_cpu"]["p50_ms"]) / 1000
    st = lambda k: {"ok": '<span class="st ok">충족</span>', "part": '<span class="st part">부분 충족</span>',
                    "no": '<span class="st no">미충족</span>', "ni": '<span class="st no">미구현</span>'}[k]
    req = [
        ("REQ-F-01", "초기 관측 구간만으로 탄착까지의 전 궤적 예측", "관측 ≤ 15 %",
         f"전 구간 출력, IP 오차 {f0(med(best, 'test_id_f15', 'miss'))} m (ID 15%)", "part", "T"),
        ("REQ-F-02", "IP, ToT, CEP<sub>50</sub> 동시 출력", "3종 + 1σ",
         f"출력됨, CEP 포함률 {M[H]['test_id_f10']['coverage']:.2f} (ID) / {M[H]['test_ood']['coverage']:.2f} (OOD). split-conformal 보정 후 ID 0.50, OOD는 보장 안 됨 (§9.5)", "part", "D"),
        ("REQ-F-03", "예측 궤적의 물리 일관성 (운동 방정식 잔차)", "잔차 RMS ≤ 0.05 g", f"{res_best:.2f} g (ID 10%)", "no", "A"),
        ("REQ-F-04", "보호 자산 DB 대비 위협 등급과 요격 고도 창 산출", "5단계", "구현, 가상 자산으로 시연 (운용 콘솔, 3D 뷰어)", "ok", "D"),
        ("REQ-P-01", "단일 표적 추론 지연 (전처리 제외, p99)", "≤ 50 ms", f"{L['onnxruntime_cpu']['p99_ms']:.1f} ms (ONNX Runtime, CPU)", "ok", "T"),
        ("REQ-P-02", "IP / CEP 추정 오차, EKF baseline 대비", "−40 %", f"{rel:+.0f} % (최선 신경망, ID 10%)", "no", "T"),
        ("REQ-P-03", "교전 판단 가용 시간 증가", "+25 s", f"계산 시간 단축 {saved_s:.2f} s, 25 s 근거 없음", "no", "A"),
        ("REQ-I-01", "센서 트랙 스트리밍 입력 (gRPC / Socket)", "≥ 20 Hz", "WebSocket 측정 스트림 + 측정당 증분 EKF 갱신 (처리량 약 660 Hz, CPU). 실제 센서 연동이 아닌 시뮬레이션 재생", "part", "D"),
    ]
    req_html = "<!-- REQ:BEGIN -->\n" + "\n".join(
        f'        <tr><td class="id">{i}</td><td>{d}</td><td class="c">{t}</td><td>{m}</td><td>{st(v)}</td><td><span class="vm">{vm}</span></td></tr>'
        for i, d, t, m, v, vm in req) + "\n<!-- REQ:END -->"

    s = open("index.html", encoding="utf-8").read()
    s = re.sub(r"<!-- REQ:BEGIN -->.*?<!-- REQ:END -->", lambda _: req_html, s, flags=re.S)
    if "<!-- RESULTS:BEGIN -->" in s:
        s = re.sub(r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->", lambda _: html, s, flags=re.S)
    else:
        anchor = "결과는 §9.1에 있습니다.</p></div>"
        assert anchor in s
        s = s.replace(anchor, anchor + "\n" + html)
    open("index.html", "w", encoding="utf-8").write(s)
    print("results block written")


if __name__ == "__main__":
    main()
