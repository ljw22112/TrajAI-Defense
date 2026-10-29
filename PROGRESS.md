# 작업 진행 현황 (2026-10-29 22:50 중단 시점, 브랜치 `wip`)

## 완료
- **GitHub Pages 사이트** (`index.html`): 기술 보고서 형식, 7-상태 EKF + Monte Carlo 궤도 추정 시연 (브라우저 실시간 계산)
- **물리 시뮬레이터** (`trajai/physics.py`): RK4, 고도 의존 g(h), 지수 대기 밀도, USSA-1976 음속, 마하수 의존 C_D(M), 배치 벡터화. 추진 구간 포함 단일 궤적(`simulate_powered`, 질량 유량 dm/dt = -T/(g0 Isp))
- **레이더 관측 모델** (`trajai/radar.py`): 거리·방위각·고각 잡음, ENU 변환, 측정 야코비안
- **합성 데이터셋** (`trajai/dataset.py`, `scripts/make_dataset.py`): 연소 종료 후 탄도 구간, train 20,000 / val 2,000 / test 5종(관측 5·10·15 %, 고잡음, OOD) 각 1,000 생성 완료 (`data/`, git 제외 — 스크립트로 재생성)
- **EKF / UKF baseline** (`trajai/filters.py`): 배치 처리, 극좌표 측정, 수치 야코비안, Joseph 갱신, NIS, Monte Carlo 탄착 분산 전파
  - 스모크 테스트(관측 12 %, 200 궤적): EKF 중앙 NIS 2.89, ln β 오차 0.015, ADE 166 m, IP 오차 292 m, CEP 540 m, CEP 포함률 0.75
  - `test_id_f05` (관측 5 %): EKF ADE 715 m, IP 오차 1,429 m, CEP 1,168 m, CEP 포함률 0.37 (과신 → PINN 도입 근거)
- **특징 추출** (`trajai/features.py`): Savitzky–Golay 미분 기반 12개 시계열 특징 + EKF 사후 10개 전역 특징(하이브리드용)
- **평가 지표** (`trajai/metrics.py`): ADE, IP miss, ToT 오차, CEP 포함률, 가우시안 CEP
- **위협 평가** (`trajai/threat.py`): 가상 보호 자산 대비 탄착 확률, 위협 등급 1–5, 상층/하층 교전 고도 창과 PIP
- **PINN 모델 코드** (`trajai/model.py`): Transformer / Bi-LSTM 인코더, 연속시간 디코더 r(τ), autograd 물리 잔차 (식 7.2), 하이브리드(EKF 사후 입력) 옵션
- **학습·평가 스크립트** (`scripts/train.py`, `scripts/evaluate.py`): λ 커리큘럼, IP 불확실성 NLL, 지연 측정
- **테스트** (`tests/`): 물리(대기 기준값, RK4 4차 수렴, 진공 극한), 레이더 야코비안, EKF/UKF 일관성, 위협 평가

## 2차 세션 (22:05–22:50)에 추가로 한 것
- 테스트 12개 전부 통과 (RK4 수렴 테스트를 매끄러운 동역학 4차 수렴 + 대기 꺾임 구간 오차 상한으로 분리)
- 전처리와 EKF/UKF baseline 5개 시험 분할 모두 완료 (`results/baseline_*.npz`, `results/prepare.log`)
  - 관측 10%: EKF IP 오차 340 m / CEP 포함률 0.70, UKF 303 m / 0.78
  - 관측 15%: EKF 225 m / 0.84, UKF 212 m / 0.89
  - 관측 5%: EKF 1,429 m / 0.37 (과신), UKF 1,207 m / 0.54 (보정됨) → 선형화 오차 효과
- 모델 학습 (1,500 step, CPU 1스레드씩 2개 병렬, `scripts/run_experiments.sh`)
  - `data_tf` 완료: 검증 ADE 1,328 m (아직 하강 중, 학습 부족)
  - `pinn_tf_hybrid` 완료: 검증 ADE 1,209 m
  - `pinn_tf` 1,256/1,500 step에서 중단 (검증 ADE 1,431 m, 체크포인트 저장됨)
  - `pinn_lstm` 314 step에서 중단, `data_tf_2k` / `pinn_tf_2k` 미시작
  - 체크포인트는 `checkpoints/`에 커밋 (wip 브랜치만)
- 새로 작성: `scripts/export_onnx.py`(ONNX 변환 + 지연 비교), `scripts/make_figures.py`, `scripts/export_samples.py`, `prototype/viewer.html`(결과 연동 3D 뷰어), `.github/workflows/tests.yml`(CI)
- `index.html`: Table 7(모델 구성), 8(데이터셋 v2), 9, 11 갱신. 결과 섹션 §9.1은 아직 없음 → main에 배포하지 않음

## 미완료 (내일 이어서)
1. 남은 학습: `pinn_tf`(재학습 또는 체크포인트 사용), `pinn_lstm`, `data_tf_2k`, `pinn_tf_2k`
   - NN이 EKF(ADE ~180 m)보다 아직 크게 뒤처짐. step 수를 늘릴지 결정 필요 (CPU 기준 1,500 step ≈ 20–30분/모델)
2. `python scripts/evaluate.py data_tf pinn_tf pinn_lstm pinn_tf_hybrid data_tf_2k pinn_tf_2k`
3. `python scripts/export_onnx.py pinn_tf_hybrid` → `results/latency.json`
4. `python scripts/make_figures.py`, `python scripts/export_samples.py pinn_tf_hybrid`
5. `index.html`에 §9.1 실험 결과 추가, REQ 표에 측정값, §10 구현 현황 갱신 → wip를 main에 병합해 배포

## 이전 미완료 목록
1. `scripts/prepare.py` 나머지 test split 처리 (test_id_f05까지 완료, 중단됨)
2. PyTorch 설치 후 모델 학습: data_tf, pinn_tf, pinn_lstm, pinn_tf_hybrid, 2k 저데이터 ablation
3. `scripts/evaluate.py` 실행 → `results/metrics.json`, 결과 그림
4. ONNX 변환과 ONNX Runtime 지연 측정 (TensorRT는 GPU 필요)
5. `pytest` 실행 확인, GitHub Actions CI
6. 사이트에 실험 결과 섹션 추가, `prototype/viewer.html`에 예측 결과 연동

## 재현
```bash
pip install -r requirements.txt
python scripts/make_dataset.py      # data/ 생성
python scripts/prepare.py           # 특징 + EKF/UKF baseline
python scripts/train.py --name pinn_tf --encoder transformer --lambda-p 0.1
python scripts/evaluate.py pinn_tf
```
