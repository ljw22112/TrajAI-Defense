# TrajAI-Defense

물리 제약 신경망(PINN) 기반 실시간 탄도 궤도 예측 및 위협 평가 시스템.
레이더 초기 관측 구간(전체 비행의 10~15%)만으로 잔여 궤적, 탄착점(IP), 탄착 시각(ToT), CEP를 추론하는 것을 목표로 합니다.

**프로젝트 페이지:** `https://ljw22112.github.io/TrajAI-Defense/`

## 구성

| 경로 | 내용 |
|---|---|
| `index.html` | 기술 문서 겸 프로젝트 페이지. 7-상태 EKF + Monte Carlo 분산 전파 baseline을 브라우저에서 실시간 계산하는 시연 포함 |
| `src/data_generator.py` | 비행 시뮬레이션 기반 합성 궤적 생성기 (현재 3-DOF 점질량, Euler 적분) |
| `src/pinn_model.py` | Bi-LSTM 인코더 + PINN 손실 프로토타입 (PyTorch) |
| `prototype/viewer.html` | Three.js 3D 궤적 뷰어 초기 프로토타입 |
| `docs/` | AI 서비스 컨셉 정의서, 프로젝트 기획서 |
| `requirements.txt` | Python 의존성 |

## 실행

```bash
pip install -r requirements.txt
python src/data_generator.py   # missile_trajectories.npy 생성
python src/pinn_model.py       # PINN 1-step 학습 동작 확인
```

웹 페이지는 빌드 없이 `index.html`을 브라우저로 열면 됩니다.

## GitHub Pages 배포

1. 이 폴더 내용을 저장소 루트에 그대로 push 합니다 (`index.html`이 루트에 있어야 함).
2. 저장소 **Settings → Pages → Build and deployment**에서 Source를 *Deploy from a branch*, Branch를 `main` / `/ (root)`로 지정합니다.
3. 1~2분 뒤 `https://ljw22112.github.io/TrajAI-Defense/`에서 확인합니다.

`.nojekyll` 파일은 Jekyll 처리를 끄기 위한 것이므로 지우지 마세요.

## 시연 모델 요약

- 동역학: 3-DOF 점질량, RK4, `g(h)`, 지수 대기 밀도, USSA-1976 온도 기반 음속, 마하수 의존 `C_D(M)`
- 추정: 상태 `[r, v, ln β]`의 EKF, 수치 야코비안, Joseph 형식 공분산 갱신, NIS 일관성 검정
- 분산 전파: EKF 사후분포에서 160개 표본 → IP 분포, CEP50, ToT ± 1σ

## 작성

인공지능 프론티어 · 이정원 (23012146)

학술 목적의 개념 연구이며 모든 데이터는 공개 물리 모델로 생성한 합성 데이터입니다.
