"""§9.12 multi-radar fusion and §9.13 extension roadmap (after the LSTM block), and §10.2 public benchmark.
Numbers come from results/multiradar.json, results/rmse_onnx.json and results/metrics.json.
Blocks between <!-- EXT:BEGIN/END --> and <!-- BENCH:BEGIN/END --> are replaced in place.
Usage: python scripts/site_ext.py
"""
import json
import re

import numpy as np

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
  <h3 class="sub" id="sec-roadmap"><span class="no">9.13</span>확장 실험과 남은 과제</h3>
  <div class="tw" tabindex="0" role="region" aria-label="표 (가로로 스크롤)">
    <table>
      <caption><b>Table R18</b>처음 범위 밖으로 둔 항목: 이번에 시험한 것과 남은 것</caption>
      <thead><tr><th>항목</th><th>이번에 한 것</th><th>남은 것</th></tr></thead>
      <tbody>
        <tr><td>다중 레이더</td><td>레이더 2대 같은 시각 EKF 융합 (§9.12)</td><td>서로 다른 갱신 주기의 비동기 융합, 레이더별 정렬 오차 추정, 다표적 연관</td></tr>
        <tr><td>기동 비행체(HGV)</td><td>활공 기동 진값 모델과 시험 분할, 탄도 추정기 평가, 기동 가능 영역 예측 (§9.14)</td><td>관측 길이별 영역 보정, 휘어진 영역을 따르는 밀도 기반 형상, 기동 중 관측이 있을 때의 추적, 기동 표적으로 학습한 신경망</td></tr>
        <tr><td>6-DOF 동역학</td><td>받음각(원뿔 운동·트림)과 롤을 넣은 축소 6-DOF 진값으로 3-DOF 가정의 비용 측정 (§9.15)</td><td>실제 공력 계수표와 관성 자료를 쓰는 완전 6-DOF, 자세 상태를 추정하는 필터</td></tr>
        <tr><td>실측 검증</td><td>공개 관측된 재진입(Hayabusa 캡슐) 궤적으로 항력·중력·대기 모델의 외삽 정확도 확인 (§9.16)</td><td>탄도 비행체의 레이더 실측 자료로 추정기 전체를 검증(비공개 환경에서만 가능)</td></tr>
        <tr><td>추진 구간</td><td>—</td><td>추력·질량 변화 모델과 연소 종료 시점 추정</td></tr>
      </tbody>
    </table>
  </div>
{hgv_block()}{sixdof_block()}{real_block()}
<!-- EXT:END -->"""


def hgv_block():
    H = json.load(open("results/hgv.json"))
    rows = []
    for k, label in (("nom_f12", "탄도, 관측 12%"), ("hgv_f12", "활공, 관측 12%"), ("nom_long", "탄도, 관측 최대 100 s"), ("hgv_long", "활공, 관측 최대 100 s")):
        v = H[k]
        pin = f'{f0(v["pinn_rts"]["miss_median_m"])}' if "pinn_rts" in v else "—"
        rows.append(f'        <tr><td>{label}</td><td class="n">{v["flight_median_s"]:.0f} s</td><td class="n">{f0(v["ekf"]["miss_median_m"])} <small class="muted">/ {v["ekf"]["coverage"]:.2f}</small></td>'
                    f'<td class="n">{f0(v["imm"]["miss_median_m"])} <small class="muted">/ {v["imm"]["coverage"]:.2f}</small></td><td class="n">{pin}</td>'
                    f'<td class="n">{v["footprint"]["coverage90"]:.2f} → {v["footprint_cal"]["coverage90"]:.2f}</td><td class="n">{f0(v["footprint_cal"]["area_median_km2"])}</td></tr>')
    g, n = H["hgv_f12"], H["nom_f12"]
    return f"""  <h3 class="sub" id="sec-hgv"><span class="no">9.14</span>기동 비행체: 활공 기동 표적</h3>
  <div class="prose">
    <p>진값에 활공 기동을 넣었습니다(<code>trajai/hgv.py</code>). 정점 이후 하강하는 동안 항력에 비례하는 양력(L/D 1.0–2.5)이 속도에 수직으로 작용하고, 뱅크각(±45°)만큼 기울어 진행 방향을 옆으로도 틉니다. 양력이 대기 밀도와 함께 커지므로 재진입 때 비행체가 끌어올려지며 활공하고, 비행 시간이 중앙값 {n["flight_median_s"]:.0f} → {g["flight_median_s"]:.0f} s로 늘어납니다. 추정기는 모두 탄도 모델을 그대로 씁니다. 재현: <code>python scripts/hgv.py</code>.</p>
  </div>
  <div class="tw" tabindex="0" role="region" aria-label="표 (가로로 스크롤)">
    <table>
      <caption><b>Table R19</b>활공 기동 표적(각 600 트랙). IP 오차 중앙값 [m] / CEP<sub>50</sub> 포함률, 그리고 기동 가설을 섞은 Monte Carlo 영역의 90% 포함률(χ² 타원 → 보정 후)과 보정 후 넓이 중앙값 [km²]</caption>
      <thead><tr><th>분할</th><th class="n">비행 시간</th><th class="n">EKF</th><th class="n">IMM</th><th class="n">PINN (RTS)</th><th class="n">영역 포함률</th><th class="n">영역 넓이</th></tr></thead>
      <tbody>
{chr(10).join(rows)}
      </tbody>
    </table>
  </div>
  <div class="prose">
    <p><b>점 예측은 기동 표적에서 실패합니다.</b> 기동은 레이더 관측 창(최대 100 s)이 끝난 뒤 재진입에서 시작되므로, EKF·IMM·PINN 모두 탄착 오차가 약 {g["ekf"]["miss_median_m"] / 1000:.0f} km이고 CEP 포함률은 0입니다. 기동 가설을 가진 IMM도 기동을 관측하지 못하면 도움이 되지 않습니다. 국내 연구가 활공 구간의 탄착점 예측 오차를 수십 km로 보고한 것과 같은 결과입니다(§2.2).</p>
    <p><b>대안은 탄착점 대신 기동 가능 영역입니다.</b> EKF 사후 분포에서 뽑은 입자마다 활공 가설(L/D 0–2.5, 뱅크 ±45°, L/D 0은 탄도)을 붙여 지면까지 전파하고 90% 타원을 내면, 기동 표적의 실제 탄착점을 {g["footprint"]["coverage90"] * 100:.0f}%(장기 관측 {H["hgv_long"]["footprint"]["coverage90"] * 100:.0f}%) 포함합니다. 영역이 휘어 있어 χ² 타원이 작게 나오므로, CEP와 같은 split-conformal 방식(§9.5)으로 별도 활공 분할 600개에서 타원 크기를 정하면(반지름 ×{H["calibration"]["scale_radius"]:.2f}) 포함률이 {g["footprint_cal"]["coverage90"] * 100:.0f}%(장기 관측 {H["hgv_long"]["footprint_cal"]["coverage90"] * 100:.0f}%)로 오릅니다. 보정 분할과 관측 길이가 다른 장기 관측에서는 덜 맞습니다. 대가는 넓이로, 중앙값 약 {f0(g["footprint_cal"]["area_median_km2"])} km²(한 변 약 {np.sqrt(g["footprint_cal"]["area_median_km2"]):.0f} km)입니다. 탄도 표적이라면 CEP 원 수 km² 수준이면 충분한 것과 대비됩니다.</p>
  </div>
"""


def sixdof_block():
    D = json.load(open("results/sixdof.json"))
    rows = []
    for f, name in (("f10", "관측 10%"), ("f15", "관측 15%")):
        e, p = D[f]["ekf"], D[f].get("pinn_rts")
        rows.append(f'        <tr><td>{name}</td><td class="n">{f0(e["median_3dof_m"])} → {f0(e["median_6dof_m"])}</td>'
                    f'<td class="n">{sg(e["diff_median_m"])} <small class="muted">[{sg(e["ci_diff"][0])}, {sg(e["ci_diff"][1])}]</small></td>'
                    f'<td class="n">{e["coverage_3dof"]:.2f} → {e["coverage_6dof"]:.2f}</td>'
                    f'<td class="n">{sg(e["slow_roll"]["diff_median_m"])} <small class="muted">[{sg(e["slow_roll"]["ci_diff"][0])}, {sg(e["slow_roll"]["ci_diff"][1])}]</small></td>'
                    + (f'<td class="n">{f0(p["median_3dof_m"])} → {f0(p["median_6dof_m"])}</td>' if p else "<td>—</td>") + "</tr>")
    ts, tsl = D["truth_shift"], D["truth_shift_slow_roll"]
    return f"""  <h3 class="sub" id="sec-6dof"><span class="no">9.15</span>3-DOF 가정은 얼마나 비싼가: 축소 6-DOF 진값</h3>
  <div class="prose">
    <p>이 연구의 동역학은 기수가 항상 속도 방향을 향한다고 보는 3-DOF 점질량입니다. 실제 탄두는 분리 때 받음각을 가지고 원뿔 운동을 하며, 비대칭 때문에 작은 트림 받음각이 롤과 함께 돕니다. 이를 축소 6-DOF 진값으로 넣었습니다(<code>trajai/sixdof.py</code>): 받음각 진폭은 동압이 커질수록 줄고(단열 불변량) 공력으로 감쇠하며, 받음각이 만드는 유도 항력과 양력이 궤적을 바꿉니다. 받음각을 0으로 두면 3-DOF와 1 cm 이내로 같습니다. 같은 비행을 두 진값으로 만들고 같은 측정 잡음을 써서 쌍대로 비교했습니다(각 {D["n"]} 트랙). 재현: <code>python scripts/sixdof.py</code>.</p>
  </div>
  <div class="tw" tabindex="0" role="region" aria-label="표 (가로로 스크롤)">
    <table>
      <caption><b>Table R20</b>진값만 3-DOF → 축소 6-DOF로 바꿨을 때 EKF + Monte Carlo의 IP 오차 중앙값 [m], 차이와 95% CI, 포함률, 롤이 느린 트랙(0.3 rev/s 미만)의 차이, RTS 입력 PINN</caption>
      <thead><tr><th>분할</th><th class="n">EKF</th><th class="n">Δ 중앙값</th><th class="n">포함률</th><th class="n">느린 롤 Δ</th><th class="n">PINN (RTS)</th></tr></thead>
      <tbody>
{chr(10).join(rows)}
      </tbody>
    </table>
  </div>
  <div class="prose">
    <p>받음각 운동은 탄착점을 실제로 옮깁니다(중앙값 {f0(ts["median_m"])} m, 상위 10% {f0(ts["p90_m"])} m 이상, 롤이 느리면 중앙값 {f0(tsl["median_m"])} m). 그런데 EKF와 PINN의 탄착 오차는 바뀌지 않습니다(Δ의 95% CI가 0을 포함하고 폭이 수 m). 받음각이 가장 클 때는 연소 종료 직후 비교적 짙은 대기 속이라 그 영향이 관측 구간 안에 들어오고, 필터가 이를 상태와 탄도계수 추정으로 흡수하기 때문입니다. 재진입에서는 진폭이 줄고 회전하는 양력이 상쇄됩니다. 이 범위의 비행에서는 3-DOF 가정이 결과를 바꾸지 않는다는 근거이며, 완전 6-DOF가 필요한 경우는 큰 트림이나 기동처럼 관측 이후에 생기는 비대칭 힘입니다(§9.14).</p>
  </div>
"""


def real_block():
    Rr = json.load(open("results/realdata_hayabusa.json"))
    f4 = Rr["fits"]["4"]
    rows = []
    for i, t in enumerate(f4["predicted_times_s"]):
        rows.append(f'        <tr><td class="n">{t:.1f} s</td><td class="n">{t - Rr["times_s"][3]:.0f} s</td><td class="n">{f0(f4["drag_model"]["pred_err_m"][i])}</td>'
                    f'<td class="n">{f0(f4["vacuum"]["pred_err_m"][i])}</td><td class="n">{f0(f4["constant_velocity_err_m"][i])}</td></tr>')
    return f"""  <h3 class="sub" id="sec-realdata"><span class="no">9.16</span>공개 실측 궤적으로 본 물리 모델</h3>
  <div class="prose">
    <p>탄도 비행체의 레이더 실측 자료는 공개되어 있지 않지만, 2010년 Hayabusa 귀환 캡슐의 재진입은 여러 관측소에서 사진으로 측정되어 시각별 위치(정밀도 약 100 m)가 공개되어 있습니다[17]. 12 km/s로 들어오는 둥근 캡슐이라 이 연구의 추정기를 그대로 시험할 수는 없으므로, 추정기 대신 그 바탕인 물리 모델(항력, 중력, 표준 대기, 탄도계수)을 시험했습니다. 앞쪽 4개 관측점(고도 64.7–50.4 km)에 초기 상태와 ln β를 최소제곱으로 맞추고(잔차 RMS {f0(f4["drag_model"]["fit_rms_m"])} m) 나머지 점을 예측했습니다. 이 속도와 거리에서는 평면 지구 근사가 맞지 않아 같은 항력 식을 지구 중심 회전 좌표계(WGS84, 코리올리·원심력)에서 적분했습니다. 재현: <code>python scripts/realdata_hayabusa.py</code>.</p>
  </div>
  <div class="tw" tabindex="0" role="region" aria-label="표 (가로로 스크롤)">
    <table>
      <caption><b>Table R21</b>Hayabusa 캡슐 관측 위치 예측 오차 [m]. 마지막 점은 원 논문에서 시각이 불확실하다고 표시</caption>
      <thead><tr><th class="n">시각</th><th class="n">외삽 길이</th><th class="n">이 연구의 항력 모델</th><th class="n">항력 없음</th><th class="n">등속 외삽</th></tr></thead>
      <tbody>
{chr(10).join(rows)}
      </tbody>
    </table>
  </div>
  <div class="prose">
    <p>3 s 앞은 관측 정밀도(100 m)보다 작은 {f0(f4["drag_model"]["pred_err_m"][0])} m로 맞고, 7 s 앞도 {f0(f4["drag_model"]["pred_err_m"][1])} m입니다. 항력을 빼거나 등속으로 외삽하면 같은 시점에 수 km–수십 km 틀립니다. 실제 대기 속 강한 감속을 이 연구의 항력·대기 모델이 수 초 단위에서 정확히 따라간다는 정성적 근거입니다. 적합한 β는 마하수 보정을 포함한 유효값이라 캡슐의 실제 탄도계수와 직접 비교하지는 않았습니다.</p>
  </div>
"""


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
    <p>이 플랫폼의 비교를 다른 사람도 같은 조건으로 재현할 수 있도록 시험 분할 5종(각 1,000 트랙)을 벤치마크로 묶었습니다. 입력(레이더 측정)과 정답(탄착점·탄착 시각·실제 잔여 궤적)을 나눈 numpy 파일, Stone Soup의 CSV 판독기로 바로 읽히는 검출·실제 궤적 파일, numpy만 쓰는 채점기(<code>evaluate.py</code>), 이 문서의 결과로 만든 리더보드가 들어 있습니다. 채점기는 예시 제출(EKF 기준선)로 리더보드의 EKF 값을 정확히 재현하는지 CI에서 확인하고, GitHub 릴리스로 배포합니다(<code>.github/workflows/benchmark.yml</code>). 내려받기: <a href="https://github.com/ljw22112/TrajAI-Defense/releases/tag/benchmark-v2">TrajAI-Bench v2</a>. 모든 자료는 합성이며 실제 체계 정보가 없어 공개해도 안전합니다(§12.3).</p>
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
