"""§9.11 of index.html: encoder swap (Bi-LSTM pair) from results/lstm_vs_pinn.json and results/metrics.json.
The block between <!-- LSTM:BEGIN --> and <!-- LSTM:END --> is replaced in place (inserted at the end of §9 the first time).
Usage: python scripts/site_lstm.py
"""
import json
import re

M = json.load(open("results/metrics.json"))
J = json.load(open("results/lstm_vs_pinn.json"))
SIG = json.load(open("results/significance.json"))["significance"]
SPL = {"test_id_f05": "ID 5%", "test_id_f10": "ID 10%", "test_id_f15": "ID 15%", "test_id_noise": "고잡음", "test_ood": "OOD"}


def f0(v):
    return f"{v:,.0f}"


def sg(v):
    return f"{v:+,.0f}"


def fp(p):
    return "&lt; 0.001" if p < 0.001 else f"{p:.3f}"


def main():
    sp, tests = J["splits"], {r["split"]: r for r in J["tests"]}
    rows = []
    for s, name in SPL.items():
        v, t = sp[s], tests[s]
        sig = t["p_holm"] < 0.05
        verdict = ("PINN 우세" if t["ci_diff"][1] < 0 else "data only 우세" if t["ci_diff"][0] > 0 else "혼재 (분포는 이동, 중앙값 차이는 불확실)") if sig else "불확실"
        rows.append(
            f'        <tr><td>{name}</td><td class="n">{f0(v["miss_data_m"])} → {f0(v["miss_pinn_m"])}</td>'
            f'<td class="n">{sg(t["diff"])} <small class="muted">[{sg(t["ci_diff"][0])}, {sg(t["ci_diff"][1])}]</small></td>'
            f'<td class="n">{fp(t["p_holm"])}</td><td>{verdict}</td>'
            f'<td class="n">{f0(v["rmse_data_m"])} → {f0(v["rmse_pinn_m"])}</td>'
            f'<td class="n">{v["phys_data_g"]:.2f} → {v["phys_pinn_g"]:.2f}</td>'
            f'<td class="n">{v["cov_data"]:.2f} → {v["cov_pinn"]:.2f}</td>'
            f'<td class="n">{f0(v["miss_ekf_m"])} <small class="muted">/ {f0(v["rmse_ekf_m"])}</small></td></tr>')

    t10 = "test_id_f10"
    tf = {r["split"]: r for r in SIG if r["a"] == "pinn_tf_v2" and r["b"] == "data_tf_v2"}
    enc = []
    for label, d, p in (("Transformer", "data_tf_v2", "pinn_tf_v2"), ("Bi-LSTM", "data_lstm_v2", "pinn_lstm_v2")):
        md, mp = M[d][t10], M[p][t10]
        sig_splits = [SPL[s] for s in ("test_id_f10", "test_id_f15", "test_id_noise")
                      if (tf.get(s) if label == "Transformer" else tests.get(s))
                      and (tf.get(s) if label == "Transformer" else tests[s])["p_holm"] < 0.05
                      and (tf.get(s) if label == "Transformer" else tests[s])["ci_diff"][1] < 0]
        enc.append(
            f'        <tr><td>{label}</td><td class="n">{f0(md["miss"]["median"])} → {f0(mp["miss"]["median"])}</td>'
            f'<td class="n">{md["phys_rms_g"]["median"]:.2f} → {mp["phys_rms_g"]["median"]:.2f} '
            f'<small class="muted">({(1 - mp["phys_rms_g"]["median"] / md["phys_rms_g"]["median"]) * 100:.0f}% 감소)</small></td>'
            f'<td>{", ".join(sig_splits) or "없음"}</td>'
            f'<td class="n">{M[p]["latency"]["p50_ms"]:.1f} / {M[p]["latency"]["p99_ms"]:.1f}</td></tr>')

    v10, v05 = sp[t10], sp["test_id_f05"]
    red10 = v10["phys_reduction"] * 100
    ekf_rmse10 = v10["rmse_ekf_m"]
    html = f"""<!-- LSTM:BEGIN -->
  <h3 class="sub" id="sec-lstm"><span class="no">9.11</span>인코더를 바꿔도 물리 손실의 효과가 유지되는가: Bi-LSTM 쌍</h3>
  <div class="prose">
    <p>§9.4의 물리 손실 비교는 Transformer 인코더 한 쌍뿐이었습니다. 효과가 인코더 구조에 묶인 것인지 확인하려고 Bi-LSTM 인코더를 같은 조건(잡음 증강 학습 분할, 4,500 step, λ<sub>p</sub> = 1.0, seed 0)으로 물리 손실 없이(data only)와 있이(PINN) 학습했습니다. 검정은 Table R5와 같습니다(쌍대 bootstrap 95% CI, Wilcoxon, 5개 분할에 Holm 보정). 학습·평가는 <code>.github/workflows/train-lstm.yml</code>, 집계는 <code>scripts/lstm_compare.py</code>로 재현합니다.</p>
  </div>
  <div class="tw" tabindex="0" role="region" aria-label="표 (가로로 스크롤)">
    <table>
      <caption><b>Table R16</b>Bi-LSTM data only → PINN. IP 오차 중앙값·RMSE [m], 물리 잔차 RMS 중앙값 [g], CEP<sub>50</sub> 포함률(보정 전). Δ = PINN − data only</caption>
      <thead><tr><th>분할</th><th class="n">IP 중앙값</th><th class="n">Δ 중앙값 [95% CI]</th><th class="n">p (Holm)</th><th>판정</th><th class="n">IP RMSE</th><th class="n">잔차</th><th class="n">포함률</th><th class="n">EKF 중앙값 / RMSE</th></tr></thead>
      <tbody>
{chr(10).join(rows)}
      </tbody>
    </table>
  </div>
  <div class="tw" tabindex="0" role="region" aria-label="표 (가로로 스크롤)">
    <table>
      <caption><b>Table R16b</b>인코더별 물리 손실 효과 (ID 10%). 지연은 PINN의 PyTorch CPU 1스레드 p50 / p99 [ms]</caption>
      <thead><tr><th>인코더</th><th class="n">IP 중앙값 data → PINN</th><th class="n">잔차 data → PINN</th><th>IP 중앙값이 유의하게 준 분할 (ID 10·15%·고잡음 중)</th><th class="n">지연</th></tr></thead>
      <tbody>
{chr(10).join(enc)}
      </tbody>
    </table>
  </div>
  <div class="prose">
    <p><b>물리 일관성은 인코더와 무관하게 개선됩니다.</b> Bi-LSTM에서도 잔차가 {v10["phys_data_g"]:.2f} → {v10["phys_pinn_g"]:.2f} g({red10:.0f}% 감소)로, Transformer 쌍과 거의 같은 수준까지 줄었습니다. RQ2의 "물리 손실은 물리 일관성을 확실히 개선한다"는 답이 한 구조에만 해당하는 결과가 아님을 뜻합니다.</p>
    <p><b>IP 정확도 이득은 중앙값에서만 뚜렷합니다.</b> Bi-LSTM에서는 ID 10%·15%·고잡음 세 분할 모두 PINN의 IP 오차 중앙값이 유의하게 작아(Δ {sg(tests[t10]["diff"])} m, ID 10%) Transformer 쌍보다 일관됩니다. 그러나 RMSE는 ID 15%에서만 줄고({f0(sp["test_id_f15"]["rmse_data_m"])} → {f0(sp["test_id_f15"]["rmse_pinn_m"])} m), ID 10%({f0(v10["rmse_data_m"])} → {f0(v10["rmse_pinn_m"])} m)와 고잡음({f0(sp["test_id_noise"]["rmse_data_m"])} → {f0(sp["test_id_noise"]["rmse_pinn_m"])} m)에서는 거의 같습니다. ID 10%의 RMSE는 개정 기획서의 평가 기준(1,000 m 이하, EKF {f0(ekf_rmse10)} m의 1.2배 이내)을 충족하지 못합니다. 물리 손실이 전형적인 트랙의 오차는 줄이지만 드문 대형 오차(꼬리)는 대체로 줄이지 못합니다.</p>
    <p><b>학습 범위 밖에서는 두 모델 모두 실패합니다.</b> 관측 5%는 학습 분할의 관측 비율(8–20%) 밖이라 두 모델 모두 IP 오차가 수 km이고, PINN이 오히려 큽니다({f0(v05["miss_data_m"])} → {f0(v05["miss_pinn_m"])} m, EKF {f0(v05["miss_ekf_m"])} m). OOD도 같습니다. 인코더를 바꿔도 외삽은 해결되지 않으므로 §9.8의 분포 밖 감지·EKF 전환이 필요합니다.</p>
  </div>
<!-- LSTM:END -->"""

    s = open("index.html", encoding="utf-8").read()
    if "<!-- LSTM:BEGIN -->" in s:
        s = re.sub(r"<!-- LSTM:BEGIN -->.*?<!-- LSTM:END -->", lambda _: html, s, flags=re.S)
    else:
        anchor = "</section>\n\n<!-- 10 IMPL -->"
        assert s.count(anchor) == 1
        s = s.replace(anchor, html + "\n" + anchor, 1)
    open("index.html", "w", encoding="utf-8").write(s)
    print("lstm block written")


if __name__ == "__main__":
    main()
