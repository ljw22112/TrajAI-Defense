"""§9.12 multi-radar fusion and §9.13 extension roadmap (after the LSTM block), and §10.2 public benchmark.
Numbers come from results/multiradar.json, results/rmse_onnx.json and results/metrics.json.
Blocks between <!-- EXT:BEGIN/END --> and <!-- BENCH:BEGIN/END --> are replaced in place.
Usage: python scripts/site_ext.py
"""
import json
import re

R = json.load(open("results/multiradar.json"))
SPL = {"test_id_f05": "ID 5%", "test_id_f10": "ID 10%", "test_id_noise": "고잡음"}


def f0(v):
    return f"{v:,.0f}"


def sg(v):
    return f"{v:+,.0f}"


def ext_block():
    rows = []
    for s, name in SPL.items():
        v = R["splits"][s]
        a, b = v["A"], v["A+B"]
        rows.append(f'        <tr><td>{name}</td><td class="n">{f0(a["miss_median_m"])} → {f0(b["miss_median_m"])}</td>'
                    f'<td class="n">{sg(v["diff_median_m"])} <small class="muted">[{sg(v["ci_diff"][0])}, {sg(v["ci_diff"][1])}]</small></td>'
                    f'<td class="n">{v["win"]:.2f}</td><td class="n">{f0(a["miss_rmse_m"])} → {f0(b["miss_rmse_m"])}</td>'
                    f'<td class="n">{f0(a["cep_median_m"])} → {f0(b["cep_median_m"])}</td><td class="n">{a["coverage"]:.2f} → {b["coverage"]:.2f}</td></tr>')
    red = {s: 1 - R["splits"][s]["A+B"]["miss_median_m"] / R["splits"][s]["A"]["miss_median_m"] for s in SPL}
    same = all(R["splits"][s]["identical_to_dataset"] for s in SPL)
    rb = R["radar_b"]
    return f"""<!-- EXT:BEGIN -->
  <h3 class="sub" id="sec-multiradar"><span class="no">9.12</span>다중 레이더 융합</h3>
  <div class="prose">
    <p>지금까지는 레이더 1대만 썼습니다. 같은 시험 트랙을 두 번째 가상 레이더(지역 좌표 ({rb[0] / 1000:.0f} km, {rb[1] / 1000:.0f} km), 같은 잡음)로도 관측하게 하고, EKF가 매 시각 1번 레이더 갱신 뒤 2번 레이더 측정으로 한 번 더 갱신하도록 했습니다(<code>filters.ekf(..., extra=)</code>). 1번 레이더 측정은 데이터셋과 같은 난수열로 다시 만들어 기존 시험 분할과 {"완전히 같고" if same else "다를 수 있고"}, 단일 레이더 결과는 §9.1의 EKF 값과 일치합니다. 재현: <code>python scripts/multiradar.py</code>.</p>
  </div>
  <div class="tw" tabindex="0" role="region" aria-label="표 (가로로 스크롤)">
    <table>
      <caption><b>Table R17</b>EKF + Monte Carlo, 레이더 1대 → 2대 융합. IP 오차 중앙값과 쌍대 bootstrap 95% CI [m], 승률(융합이 더 정확한 트랙 비율), RMSE, CEP<sub>50</sub> 중앙값, 포함률</caption>
      <thead><tr><th>분할</th><th class="n">IP 중앙값</th><th class="n">Δ 중앙값</th><th class="n">승률</th><th class="n">RMSE</th><th class="n">CEP</th><th class="n">포함률</th></tr></thead>
      <tbody>
{chr(10).join(rows)}
      </tbody>
    </table>
  </div>
  <div class="prose">
    <p>융합하면 세 분할 모두 IP 오차가 줄고 신뢰구간이 0을 포함하지 않습니다(ID 5% −{red["test_id_f05"] * 100:.0f}%, ID 10% −{red["test_id_f10"] * 100:.0f}%, 고잡음 −{red["test_id_noise"] * 100:.0f}%). 이득은 관측이 짧거나 잡음이 클 때 크고, 관측이 충분하면 작습니다. 두 레이더가 같은 시각에 같은 구간을 보므로 측정 수가 늘어날 뿐 관측 구간이 길어지지는 않아, 탄도계수 β를 알아내기 어려운 근본 한계(§2, Phase II)는 그대로이기 때문입니다. 실제 운용에서는 갱신 주기가 다르고 레이더 사이 정렬 오차(bias)가 있으므로, 이를 함께 추정하는 것이 다음 단계입니다(§9.13).</p>
  </div>
  <h3 class="sub" id="sec-roadmap"><span class="no">9.13</span>확장 로드맵: 실측 검증, 6-DOF, 기동 비행체</h3>
  <div class="tw" tabindex="0" role="region" aria-label="표 (가로로 스크롤)">
    <table>
      <caption><b>Table R18</b>이 연구의 범위 밖 항목과 확장에 필요한 것</caption>
      <thead><tr><th>항목</th><th>현재</th><th>확장에 필요한 것</th></tr></thead>
      <tbody>
        <tr><td>실측 검증</td><td>합성 자료만 사용. 공개된 탄도 비행체 레이더 궤적 자료는 찾지 못함(국내 선행 연구도 같은 이유로 합성 궤적 사용, §2.2)</td><td>협력 기관의 관측 자료(비공개 환경에서만), 또는 공개된 관측 로켓 비행 기록이나 우주 물체 재진입 관측으로 정성 비교. 궤도 요소(TLE) 기반 재진입 예측은 수일 단위 궤도 감쇠 문제라 이 과제와 물리가 달라 직접 비교에는 쓰지 않음</td></tr>
        <tr><td>6-DOF 동역학</td><td>3-DOF 점질량(§5)</td><td>받음각·옆미끄럼각에 따른 공력 계수(C<sub>L</sub>, C<sub>D</sub>, C<sub>m</sub>)와 관성 자료, 자세 상태를 포함한 EKF, PINN 잔차를 6-DOF 운동 방정식으로 확장</td></tr>
        <tr><td>기동 비행체(HGV)</td><td>약한 양력 기동만 시험(§9.9). 활공 기동은 국내 연구에서도 탄착점 오차가 수십 km(§2.2)</td><td>양력·뱅크각 제어 모델로 기동 궤적 생성, 기동 가설을 가진 IMM과 신경망의 결합, 탄착점보다 위협 영역(확률 지도) 출력</td></tr>
        <tr><td>다중 레이더</td><td>2대, 같은 시각, 정렬 오차 없음(§9.12)</td><td>서로 다른 갱신 주기의 비동기 융합, 레이더별 정렬 오차 추정, 다표적 데이터 연관(Stone Soup와 연동, §10.2)</td></tr>
        <tr><td>추진 구간</td><td>연소 종료 이후만</td><td>추력·질량 변화 모델과 연소 종료 시점 추정</td></tr>
      </tbody>
    </table>
  </div>
<!-- EXT:END -->"""


def bench_block():
    O = json.load(open("results/rmse_onnx.json"))
    M = json.load(open("results/metrics.json"))
    J = json.load(open("results/lstm_vs_pinn_rts.json")) if __import__("os").path.exists("results/lstm_vs_pinn_rts.json") else None
    rows = [("EKF + Monte Carlo", M["ekf"]["test_id_f10"]["miss"]["median"], json.load(open("results/lstm_vs_pinn.json"))["splits"]["test_id_f10"]["rmse_ekf_m"], M["ekf"]["test_id_f10"]["coverage"])]
    for key, label in (("pinn_tf_v2", "PINN-Transformer"), ("pinn_tf_hybrid_v2", "PINN-Transformer + EKF"), ("pinn_tf_v2_rts", "PINN-Transformer, RTS 입력")):
        rows.append((label, M[key]["test_id_f10"]["miss"]["median"], O[key]["test_id_f10"]["rmse_m"], M[key]["test_id_f10"]["coverage"]))
    L = json.load(open("results/lstm_vs_pinn.json"))["splits"]["test_id_f10"]
    rows.append(("PINN-Bi-LSTM", L["miss_pinn_m"], L["rmse_pinn_m"], L["cov_pinn"]))
    if J:
        rows.append(("PINN-Bi-LSTM, RTS 입력", J["splits"]["test_id_f10"]["miss_pinn_m"], J["splits"]["test_id_f10"]["rmse_pinn_m"], J["splits"]["test_id_f10"]["cov_pinn"]))
    tr = "\n".join(f'        <tr><td>{a}</td><td class="n">{f0(b)}</td><td class="n">{f0(c)}</td><td class="n">{d:.2f}</td></tr>' for a, b, c, d in rows)
    return f"""<!-- BENCH:BEGIN -->
  <h3 class="sub" id="sec-bench"><span class="no">10.2</span>공개 벤치마크 TrajAI-Bench</h3>
  <div class="prose">
    <p>이 플랫폼의 비교를 다른 사람도 같은 조건으로 재현할 수 있도록 시험 분할 5종(각 1,000 트랙)을 벤치마크로 묶었습니다. 입력(레이더 측정)과 정답(탄착점·탄착 시각·실제 잔여 궤적)을 나눈 numpy 파일, Stone Soup의 CSV 판독기로 바로 읽히는 검출·실제 궤적 파일, numpy만 쓰는 채점기(<code>evaluate.py</code>), 이 문서의 결과로 만든 리더보드가 들어 있습니다. 채점기는 예시 제출(EKF 기준선)로 리더보드의 EKF 값을 정확히 재현하는지 CI에서 확인하고, 태그 <code>benchmark-v*</code>를 올리면 GitHub 릴리스로 배포됩니다(<code>.github/workflows/benchmark.yml</code>). 모든 자료는 합성이며 실제 체계 정보가 없어 공개해도 안전합니다(§12.3).</p>
  </div>
  <div class="tw" tabindex="0" role="region" aria-label="표 (가로로 스크롤)">
    <table>
      <caption><b>Table 12b</b>리더보드 일부 (ID 10%). IP 오차 중앙값·RMSE [m], 보정 전 CEP<sub>50</sub> 포함률. RMSE는 배포 모델을 다시 실행해 계산(<code>scripts/rmse_onnx.py</code>)</caption>
      <thead><tr><th>추정기</th><th class="n">중앙값</th><th class="n">RMSE</th><th class="n">포함률</th></tr></thead>
      <tbody>
{tr}
      </tbody>
    </table>
  </div>
<!-- BENCH:END -->"""


def main():
    s = open("index.html", encoding="utf-8").read()
    for tag, html, anchor in (("EXT", ext_block(), "<!-- LSTM:END -->"),
                              ("BENCH", bench_block(), '</section>\n\n<!-- 11 PLAN -->')):
        if f"<!-- {tag}:BEGIN -->" in s:
            s = re.sub(rf"<!-- {tag}:BEGIN -->.*?<!-- {tag}:END -->", lambda _: html, s, flags=re.S)
        else:
            assert s.count(anchor) == 1, anchor
            s = s.replace(anchor, (anchor + "\n" + html) if tag == "EXT" else (html + "\n" + anchor), 1)
    open("index.html", "w", encoding="utf-8").write(s)
    print("ext + bench blocks written")


if __name__ == "__main__":
    main()
