# TrajAI-Defense

물리 제약 신경망(PINN) 기반 실시간 탄도 궤도 예측 및 위협 평가 시스템.
레이더 초기 관측 구간(전체 비행의 5~15%)만으로 잔여 궤적, 탄착점(IP), 탄착 시각(ToT), CEP를 추론하고,
같은 물리 모델을 쓰는 EKF/UKF + Monte Carlo baseline과 비교합니다.

- **기술 문서 / 프로젝트 페이지:** https://ljw22112.github.io/TrajAI-Defense/
- **3D 결과 뷰어:** https://ljw22112.github.io/TrajAI-Defense/prototype/viewer.html

## 풀스택 운용 콘솔

연구 코드를 감싼 웹 애플리케이션입니다 (풀스택 프로그래밍 과제).

- **프론트엔드** `web/`: React 18 + TypeScript (strict) + Vite + TanStack Query. 대시보드(교전 우선순위, 위협 알림), 시나리오 생성(표준·일자 기상·**기온·습도·기압·바람·날씨 직접 설정**), 상세·실시간 재생(WebSocket, PINN 예측 갱신), 환경·센서 분석(기상 영향·자료 지연·센서 조건·환경 민감도), 다표적 교전 타임라인, 시나리오 비교, PDF 저장, 자산 CRUD, 모델 레지스트리, 사용자·감사 로그
- **백엔드** `server/`: FastAPI + Pydantic v2. REST 37개 + WebSocket 2개, JWT 인증과 로그인 실패 제한, 추정 결과 CSV 내보내기, viewer / analyst / admin 역할, 감사 로그, OpenAPI 문서 `/api/docs`
- **데이터베이스**: PostgreSQL 16 (개발·테스트는 SQLite), SQLAlchemy 2.0, Alembic 마이그레이션, 11개 테이블
- **정적 데모**: https://ljw22112.github.io/TrajAI-Defense/app/ (실제 서버 스냅샷, 읽기 전용)
- **설계 문서**: [docs/architecture.md](docs/architecture.md) (구성도, ERD, 시퀀스, 보안, 테스트 전략)

```bash
# 로컬 (SQLite)
pip install -r server/requirements.txt
cd server && uvicorn app.main:app --reload       # API + 문서: http://127.0.0.1:8000/api/docs
cd web && npm ci && npm run dev                   # 콘솔: http://127.0.0.1:5173  (admin@trajai.dev / admin1234)

# PostgreSQL + 단일 컨테이너
export TRAJAI_JWT_SECRET=$(openssl rand -hex 32)
docker compose up --build                         # http://localhost:8000

# 테스트
python -m pytest -q tests server/tests            # 코어 21 + API 24
cd web && npm test                                # 프론트엔드 9
E2E_BASE=http://127.0.0.1:8000 python -m pytest e2e   # Playwright 12 (서버 실행 + seed_demo.py 필요, 만든 데이터는 끝나고 삭제)
```

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
| `server/` | FastAPI 앱, Alembic 마이그레이션, API 테스트, 시드·데모 내보내기 스크립트 |
| `web/` | React + TypeScript 운용 콘솔 |
| `app/` | 정적 데모 빌드 (GitHub Pages) |
| `e2e/` | Playwright 종단 간 테스트 |
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

## 결과 요약 (CPU, 시험 분할별 1,000개 궤적의 중앙값)

**v2: 잡음 증강 학습, 4,500 step** (IP 오차)

| 모델 | ID 10% | ID 15% | 고잡음 | OOD | 물리 잔차 (ID 10%) |
|---|---|---|---|---|---|
| EKF + MC | 340 m | 225 m | 629 m | 438 m | — |
| UKF + MC | 303 m | 212 m | 608 m | 357 m | — |
| Transformer (data only, v2) | 794 m | 615 m | 1,596 m | 18,081 m | 0.40 g |
| PINN-Transformer (v2) | 733 m | 645 m | 1,704 m | 14,224 m | 0.10 g |
| PINN-Transformer + EKF (v2) | 578 m | 428 m | 1,026 m | 10,404 m | 0.086 g |

**기상 반영** (2026-10 실제 기상 + 분석장 오차 섭동, IP 오차 / CEP 포함률)

| 추정기 | IP 오차 | 포함률 |
|---|---|---|
| EKF, 표준 대기 | 2,633 m | 0.08 |
| EKF, 직전 1년 기후값 | 524 m | 0.47 |
| EKF, 선택 일자 Open-Meteo 프로파일 | 339 m | 0.65 |
| 하이브리드 v2 (표준 대기 학습) | 2,779 m | 0.11 |
| 하이브리드, 직전 1년 기상으로 학습 | 720 m | 0.63 |
| 하이브리드, 직전 1년 기상 학습 + 선택 일자 기상 입력 | 683 m | 0.60 |

- 잡음 증강으로 고잡음 붕괴(v1 12.2 km)가 사라졌고, 물리 잔차는 0.23 g → 0.086 g으로 줄었습니다.
- 물리 손실(PINN 대 data only, 같은 v2 조건)은 잔차를 0.40 → 0.10 g으로 확실히 줄이지만, IP 오차 이득은 ID 10%(−61 m)와 OOD(−3.9 km)에서만 유의하고 ID 15%·고잡음에서는 불확실합니다.
- 대기 모델이 IP 오차의 가장 큰 원인입니다. 선택 일자 기상을 쓰면 EKF 오차가 표준 대기 대비 87% 줄어 기본 설정으로 채택했습니다.
- Savitzky–Golay 특징을 쓰는 신경망은 같은 물리 모델을 쓰는 EKF/UKF보다 부정확하고 OOD에 취약합니다. 확실한 이점은 지연입니다(ONNX Runtime 0.87 ms 대 EKF + MC 228 ms).
- **PINN → EKF 되먹임**: PINN의 탄도계수 예측을 EKF 사전정보로 쓰면 5개 시험 분할 모두에서 EKF보다 정확합니다(ID 5% 1,429 → 956 m, OOD 438 → 363 m). 신경망은 필터를 대체할 때보다 필터 안에서 쓸 때 가장 효과적이었습니다.
- **RTS 평활 특징**: 입력 속도·가속도를 Savitzky–Golay 대신 EKF-RTS 평활로 만들면 PINN 단독 오차가 ID 10% 733 → 437 m, 고잡음 1,704 → 607 m(EKF 629 m보다 정확)로 줄어듭니다.
- **seed 반복**: 물리 손실의 IP 정확도 이득은 seed 2쌍 모두에서 ID 10%·OOD에만 일관됩니다.
- **앙상블(seed 3개)**: ID에서 IP 오차와 CEP 포함률이 개선되지만(ID 10% 578 → 514 m), OOD는 개선되지 않습니다. seed만 바꾼 앙상블은 분포 밖 입력을 알아채지 못합니다.
- **분포 밖 감지(§9.8)**: PINN 내부 표현의 Mahalanobis 거리로 OOD를 가려내고(AUROC 0.999), 기준을 넘는 트랙을 EKF로 넘기면 OOD 오차가 11.7 → 0.44 km로 줄어듭니다. 콘솔도 같은 기준으로 "분포 밖 → EKF" 전환을 합니다.
- **모델 불일치(§9.9)**: 실제 비행에만 Cd(M) 곡선 오차나 약한 양력을 넣으면 필터와 신경망의 격차가 줄지만, 불일치 데이터로 다시 학습한 PINN도 EKF를 넘지 못합니다(양력 1,142 m vs 1,125 m). 다만 CEP 포함률은 0.11 → 0.60으로 훨씬 정직해집니다.
- 콘솔은 표준 대기 시나리오의 PINN 추정에 RTS 입력 모델(`pinn_tf_v2_rts`, 서버에서 EKF-RTS 평활 후 ONNX 추론)을 씁니다.
- 기상 시나리오에는 `wx_hybrid_date`를 씁니다. 자세한 표와 해석은 프로젝트 페이지 §9에 있습니다.

## 작성

인공지능 프론티어 · 이정원 (23012146)

학술 목적의 개념 연구이며 모든 데이터는 공개 물리 모델로 생성한 합성 데이터입니다.
