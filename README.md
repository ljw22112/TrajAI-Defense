# TrajAI-Defense

물리 제약 신경망(PINN) 기반 실시간 탄도 궤도 예측 및 위협 평가 시스템.
레이더 초기 관측 구간(전체 비행의 5~15%)만으로 잔여 궤적, 탄착점(IP), 탄착 시각(ToT), CEP를 추론하고,
같은 물리 모델을 쓰는 EKF/UKF + Monte Carlo baseline과 비교합니다.

- **기술 문서 / 프로젝트 페이지:** https://ljw22112.github.io/TrajAI-Defense/
- **3D 결과 뷰어:** https://ljw22112.github.io/TrajAI-Defense/prototype/viewer.html

## 구성

| 경로 | 내용 |
|---|---|
| `index.html` | 기술 문서. 브라우저에서 실시간 계산하는 7-상태 EKF + Monte Carlo 시연, 실험 결과(§9) |
| `trajai/physics.py` | 3-DOF 점질량, RK4, g(h), USSA-1976 음속, 마하수 의존 C_D(M), 배치 벡터화 |
| `trajai/radar.py` | 거리·방위각·고각 측정 잡음, ENU 변환, 측정 야코비안 |
| `trajai/dataset.py` | Monte Carlo 합성 데이터셋 (ID / OOD / 고잡음 / 잡음 증강 분할) |
| `trajai/filters.py` | 배치 EKF / UKF (극좌표 측정, Joseph 갱신, NIS), Monte Carlo 탄착 분산 전파 |
| `trajai/features.py` | Savitzky–Golay 기반 12개 시계열 특징, EKF 사후 전역 특징 |
| `trajai/model.py` | Transformer / Bi-LSTM 인코더, 연속시간 디코더, autograd 물리 잔차, 하이브리드 |
| `trajai/threat.py` | 가상 보호 자산 대비 탄착 확률, 위협 등급 1–5, 교전 고도 창과 PIP |
| `scripts/` | 데이터 생성, 전처리, 학습, 평가, ONNX 변환·지연 측정, 그림, 사이트 결과 생성 |
| `tests/` | 물리·레이더·필터·위협 평가 단위 테스트 (GitHub Actions) |
| `results/`, `figures/` | 평가 지표(`metrics.json`), 지연(`latency.json`), 그림 |
| `prototype/viewer.html` | 시험 케이스의 궤적·예측·CEP·위협 평가 Three.js 뷰어 |
| `legacy/` | 초기 프로토타입 코드 (Euler 생성기, 1-step PINN) |
| `docs/` | AI 서비스 컨셉 정의서, 프로젝트 기획서 |

## 재현

```bash
pip install -r requirements.txt
python scripts/make_dataset.py                 # data/ (train 20k, val 2k, test 5종 × 1k), 약 2분
python scripts/prepare.py                      # 특징 추출 + EKF/UKF baseline
python scripts/train.py --name pinn_tf_hybrid --encoder transformer --lambda-p 1.0 --hybrid --steps 1500
python scripts/evaluate.py pinn_tf_hybrid      # results/metrics.json
python scripts/export_onnx.py pinn_tf_hybrid   # export/*.onnx, results/latency.json
python scripts/make_figures.py
python -m pytest -q tests
```

## 결과 요약 (v1: 공칭 잡음 학습, 1,500 step, CPU)

| 모델 | IP 오차 ID 10% | IP 오차 ID 15% | IP 오차 OOD | 물리 잔차 (ID 10%) |
|---|---|---|---|---|
| EKF + MC | 340 m | 225 m | 438 m | — |
| UKF + MC | 303 m | 212 m | 357 m | — |
| Transformer (data only) | 1,674 m | 1,394 m | 23,406 m | 0.57 g |
| PINN-Transformer | 1,516 m | 1,321 m | 23,612 m | 0.22 g |
| PINN-Transformer + EKF | 1,193 m | 875 m | 21,119 m | 0.23 g |

- 물리 손실은 운동 방정식 잔차를 약 2.6배 줄이고 IP 오차도 줄입니다.
- 정확도는 아직 같은 물리 모델을 쓰는 EKF/UKF에 못 미치며, 분포 밖(OOD, 고잡음)에서 크게 나빠집니다.
- 단일 표적 지연은 PINN 0.87 ms(ONNX Runtime) 대 EKF + MC 228 ms(NumPy)입니다.
- 자세한 표와 해석은 프로젝트 페이지 §9에 있습니다. 잡음 증강·장기 학습(v2) 결과로 갱신 예정입니다.

## 작성

인공지능 프론티어 · 이정원 (23012146)

학술 목적의 개념 연구이며 모든 데이터는 공개 물리 모델로 생성한 합성 데이터입니다.
