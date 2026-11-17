# TrajAI-Bench: 짧은 레이더 관측으로 탄도 비행체 탄착점 예측하기

칼만 필터, 신경망, 물리 기반 신경망 같은 탄착점 예측 방법을 **같은 트랙·같은 지표**로 비교하기 위한 공개 벤치마크입니다. TrajAI-Defense 기술 문서(https://ljw22112.github.io/TrajAI-Defense/)의 §9 실험과 같은 시험 분할이며, 생성 코드는 저장소(https://github.com/ljw22112/TrajAI-Defense)의 `scripts/make_dataset.py`로 그대로 재현됩니다.

## 과제

연소 종료 후 비행의 앞부분(관측 구간)에서 레이더 1대가 0.5초마다 잰 거리·방위각·고각을 받아, 같은 비행체의 **탄착점(IP), 탄착 시각(ToT), CEP50**을 예측합니다.

- 동역학: 3-DOF 점질량, 마하수에 따른 항력계수 C_D(M), 탄도계수 β, 고도에 따른 중력, USSA-1976 대기(무풍)
- 레이더: 지역 ENU 좌표의 가상 위치 (−30 km, 20 km, 0), 측정 잡음은 분할별 `sigma` (거리 [m], 각도 [rad])
- 모든 자료는 합성이며 실제 체계의 제원이나 관측 자료가 아닙니다.

## 분할 (각 1,000개 트랙)

| 분할 | 내용 |
|---|---|
| `test_id_f05` | 관측 5% (학습 분할 8–20%보다 짧음) |
| `test_id_f10` | 관측 10% (기준 분할) |
| `test_id_f15` | 관측 15% |
| `test_id_noise` | 관측 12%, 레이더 잡음 약 3배 |
| `test_ood` | 학습 범위 밖 비행 조건(고속·고β), 관측 12% |

학습용 분할은 같은 스크립트로 만들 수 있습니다(`python scripts/make_dataset.py --only train_aug val_aug`).

## 파일

| 경로 | 내용 |
|---|---|
| `data/<split>.npz` | 입력. `meas_pol` (N, 200, 3) [거리 m, 방위각 rad, 고각 rad], `meas` (N, 200, 3) 같은 측정의 ENU 좌표 [m], `n_meas` 트랙별 유효 측정 수(이후는 0으로 채움), `sigma` (N, 2), `dt` = 0.5 s, `radar_site` |
| `truth/<split>.npz` | 정답. `impact` (N, 3) [m], `T` 탄착 시각 [s], `t_c` 관측 종료 시각 [s], `future` (N, 129, 3) 관측 종료부터 탄착까지 균등 시간 격자의 실제 위치, `beta` |
| `stonesoup/<split>_detections.csv` | 같은 측정을 Stone Soup `CSVDetectionReader` 형식으로 (`track_id`, `time`, `x`, `y`, `z`, `range_m`, `az_rad`, `el_rad`) |
| `stonesoup/<split>_groundtruth.csv` | 관측 종료 이후 실제 궤적을 `CSVGroundTruthReader` 형식으로 (`track_id`, `time`, `x`, `y`, `z`) |
| `leaderboard.json` | 기술 문서에 보고한 추정기들의 점수 |
| `sample_submission_ekf.csv` | 제출 형식 예시 (EKF + Monte Carlo 기준선) |
| `evaluate.py` | 채점기 (numpy만 필요) |
| `stonesoup_example.py` | Stone Soup로 읽는 예 |

시각: 각 트랙의 측정은 연소 종료(t = 0)부터 0.5초 간격입니다. Stone Soup CSV에서는 트랙 i를 `1767225600 + 1000·i` 초(유닉스 시각)에서 시작하게 옮겨 트랙끼리 시간이 겹치지 않게 했습니다.

## 제출과 채점

```bash
python evaluate.py my_submission.csv              # 분할별 점수와 리더보드를 나란히 출력
python evaluate.py sample_submission_ekf.csv      # EKF 기준선: 리더보드의 EKF 줄과 같은 값이 나와야 함
```

제출 CSV 열: `split, track_id, ip_x, ip_y, cep, tot` (좌표 m, 시각 s, 측정과 같은 좌표계·시간 기준)

지표 (분할별):
- **IP 오차** 중앙값, 90번째 백분위, RMSE [m]: 예측 탄착점과 실제 탄착점의 수평 거리
- **CEP50 포함률**: 실제 탄착점이 예측 CEP 원 안에 든 비율. 잘 보정되었다면 0.50 (과신하면 더 낮음)
- **ToT 오차** 중앙값 [s]

기술 문서의 규칙대로 비교할 때는 같은 1,000개 트랙에 대한 쌍대 비교(중앙값 차이의 bootstrap 신뢰구간과 Wilcoxon 부호순위 검정, 여러 비교에는 Holm 보정)를 권장합니다.

## 인용

이정원, “TrajAI-Defense: 탄도 궤적·탄착점 예측 알고리즘 비교·평가 플랫폼,” 세종대학교 인공지능 프론티어 프로젝트, 2026.
