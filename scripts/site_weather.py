"""Render §9.3 (date-selected weather experiment) into index.html between <!-- WX:BEGIN --> and <!-- WX:END -->.

Usage: python scripts/site_weather.py [nn models evaluated on test_wx ...]
       e.g. python scripts/site_weather.py pinn_tf_hybrid pinn_tf_hybrid_v2 wx_hybrid_clim wx_hybrid_date
"""
import json
import re
import sys

M = json.load(open("results/metrics.json"))
F = json.load(open("results/weather_filters.json"))
LAB = {"exp": "표준 모델 (지수 대기, 무풍)", "ussa": "USSA-1976 (무풍)", "clim": "기후값 (직전 1년 평균)",
       "date": "선택 일자 API 프로파일", "oracle": "실제 대기 (상한, 운용 불가)"}
NNLAB = {"pinn_tf_hybrid": "하이브리드 v1 (표준 대기 학습)", "pinn_tf_hybrid_v2": "하이브리드 v2 (표준 대기, 잡음 증강)",
         "wx_hybrid_clim": "하이브리드, 직전 1년 기상으로 학습 (일자 입력 없음)",
         "wx_hybrid_date": "하이브리드, 직전 1년 기상 학습 + 선택 일자 기상 입력"}


def f0(v):
    return f"{v:,.0f}"


def main(nn):
    rows = []
    for k in ["exp", "ussa", "clim", "date", "oracle"]:
        for fl in ("ekf", "ukf"):
            key = f"{fl}_{k}"
            if key not in F:
                continue
            r = F[key]
            prop = ' class="prop"' if k == "date" else ""
            rows.append(f'<tr{prop}><td>{fl.upper()}</td><td>{LAB[k]}</td><td class="n">{f0(r["ade"]["median"])}</td><td class="n">{f0(r["miss"]["median"])}</td>'
                        f'<td class="n">{f0(r["cep"]["median"])}</td><td class="n">{r["coverage"]:.2f}</td><td class="n">{r["lb_err"]["median"]:.3f}</td></tr>')
    e = {k: F[f"ekf_{k}"]["miss"]["median"] for k in ["exp", "ussa", "clim", "date", "oracle"]}
    nn_rows, nn_text = "", ""
    have = [m for m in nn if m in M and "test_wx" in M[m]]
    if have:
        nr = []
        for m in have:
            r = M[m]["test_wx"]
            prop = ' class="prop"' if m == "wx_hybrid_date" else ""
            nr.append(f'<tr{prop}><td>{NNLAB.get(m, m)}</td><td class="n">{f0(r["ade"]["median"])}</td><td class="n">{f0(r["miss"]["median"])}</td>'
                      f'<td class="n">{r["tot"]["median"]:.2f}</td><td class="n">{f0(r["cep"]["median"])}</td><td class="n">{r["coverage"]:.2f}</td></tr>')
        nn_rows = f"""
  <div class="tw">
    <table>
      <caption><b>Table W2</b>신경망, 같은 기상 시험 분할: ADE [m], IP 오차 [m], ToT 오차 [s], CEP<sub>50</sub> [m], CEP 포함률</caption>
      <thead><tr><th>모델</th><th>ADE</th><th>IP 오차</th><th>ToT</th><th>CEP</th><th>포함률</th></tr></thead>
      <tbody>
{chr(10).join(nr)}
      </tbody>
    </table>
  </div>"""
        if "wx_hybrid_date" in have and "wx_hybrid_clim" in have:
            a, b = M["wx_hybrid_clim"]["test_wx"]["miss"]["median"], M["wx_hybrid_date"]["test_wx"]["miss"]["median"]
            base = [m for m in ("pinn_tf_hybrid_v2", "pinn_tf_hybrid") if m in have]
            bl = f"표준 대기로 학습한 하이브리드({f0(M[base[0]]['test_wx']['miss']['median'])} m)보다 " if base else ""
            nn_text = (f"<p><b>신경망도 같은 경향입니다.</b> 직전 1년의 실제 기상으로 시뮬레이션한 발사 자료로 학습하면 {bl}"
                       f"IP 오차가 {f0(a)} m로 줄고, 선택 일자의 기상을 입력과 물리 잔차에 함께 넣으면 {f0(b)} m"
                       f"({(b / a - 1) * 100:+.0f}%)가 됩니다. 다만 같은 조건의 EKF({f0(e['date'])} m)보다는 여전히 큽니다.</p>")
    html = f"""<!-- WX:BEGIN -->
  <h3 class="sub" id="sec-weather"><span class="no">9.3</span>기상 반영 실험: 선택 일자의 기온·바람</h3>
  <div class="prose">
    <p>Open-Meteo 기압면 자료(1000–10 hPa, 약 0–31 km의 기온, 바람, 지오퍼텐셜 고도)를 날짜별로 받아 250 m 격자의 기온·밀도·음속·바람 분포로 만들고, 비행 동역학의 항력을 공기 상대속도 기준으로 계산했습니다. 시험 발사 1,000회는 2026-10-01–28의 프로파일(하루 4회, 112개)을 따르되, 분석장 오차를 흉내 내도록 상관 길이 4 km의 무작위 섭동(바람 1σ 4 m/s, 기온 1σ 1.5 K)을 더한 대기를 실제로 사용했습니다. "그 전 발사 데이터"로는 직전 1년(2025-10–2026-09, 1,460개 프로파일)의 실제 기상 속에서 시뮬레이션한 발사 20,000회를 썼습니다. 실제 발사 기록은 공개 자료가 없어 쓰지 않았습니다.</p>
  </div>
  <div class="tw">
    <table>
      <caption><b>Table W1</b>추정기의 대기 모델별 성능 (관측 12%, 1,000회 중앙값): ADE [m], IP 오차 [m], CEP<sub>50</sub> [m], CEP 포함률, ln β 오차</caption>
      <thead><tr><th>필터</th><th>대기 모델</th><th>ADE</th><th>IP 오차</th><th>CEP</th><th>포함률</th><th>|Δ ln β|</th></tr></thead>
      <tbody>
{chr(10).join(rows)}
      </tbody>
    </table>
  </div>{nn_rows}
  <div class="prose">
    <p><b>대기 모델이 IP 예측 오차의 가장 큰 원인이었습니다.</b> 지수 밀도 모델은 30 km 부근에서 실제보다 밀도를 약 2배 크게 잡아 IP 오차가 {f0(e['exp'])} m입니다. USSA-1976으로 바꾸면 {f0(e['ussa'])} m, 기후값이면 {f0(e['clim'])} m, 선택 일자 프로파일이면 {f0(e['date'])} m로 줄어듭니다. 일자 선택은 기후값 대비 {(1 - e['date'] / e['clim']) * 100:.0f}%, 표준 모델 대비 {(1 - e['date'] / e['exp']) * 100:.0f}%의 오차 감소이며, CEP 포함률도 {F['ekf_exp']['coverage']:.2f}에서 {F['ekf_date']['coverage']:.2f}로 회복됩니다. 분석장 오차를 모르는 상한(실제 대기)은 {f0(e['oracle'])} m입니다.</p>
    {nn_text}
    <p><b>채택.</b> 기상 반영의 효과가 크므로 Fig. 1 시연에 날짜 선택을 넣었고, 기본 추정기 설정을 "선택 일자 기상 반영"으로 바꿨습니다. 바람 자료가 없는 31 km 이상은 바람을 0으로 줄여 나가므로, 고고도 구간의 바람 영향은 아직 반영되지 않습니다.</p>
  </div>
<!-- WX:END -->"""
    s = open("index.html", encoding="utf-8").read()
    if "<!-- WX:BEGIN -->" in s:
        s = re.sub(r"<!-- WX:BEGIN -->.*?<!-- WX:END -->", lambda _: html, s, flags=re.S)
    else:
        s = s.replace("<!-- RESULTS:END -->", "<!-- RESULTS:END -->\n" + html, 1)
    open("index.html", "w", encoding="utf-8").write(s)
    print("weather block written", have)


if __name__ == "__main__":
    main(sys.argv[1:])
