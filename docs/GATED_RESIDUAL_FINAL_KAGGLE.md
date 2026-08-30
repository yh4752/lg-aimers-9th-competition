# 최종 Gated Residual Kaggle 실행 안내

## 무엇을 확인하는 실험인가

현재 제출 기준선인 Tree Expert E2를 버리고 새 모델로 갈아타는 실험이 아니다. 이미 OOF에서
일부 신호가 확인된 정규시즌 전문가 D5를 작은 보정값으로만 사용하고, 투수·타자 표본이
적거나 정규시즌이 아니면 E2로 돌아가는 최종 제출 후보 실험이다.

후보 구조는 네 개뿐이다.

1. `G0`: 정규시즌에 고정 강도로만 보정
2. `G1`: 선수별 과거 표본 수에 따라 보정 강도를 축소
3. `G2`: G1에 global·game type calibration 추가
4. `G3`: G1에 hand·pitcher·batter 계층 calibration 추가

2022·2023 OOF에서 구조를 고정하고, 2024는 한 번만 확인한다. 2024 결과를 본 뒤 가중치를
다시 맞추지 않는다.

## 필요한 Kaggle Input

두 Dataset만 연결한다.

- 공식 데이터: `lg-aimers-9th-data`
- 최종 실험 입력: 로컬의 `artifacts/gated_residual_final_input.zip`을 Dataset으로 등록한 것

Stage A, Stage B, E2 파일을 따로 추가하지 않는다. 세 파일은 최종 실험 입력 안에 해시와
함께 들어 있다. ZIP이 Kaggle에서 폴더로 풀려도 같은 입력으로 인식한다.

## 실행 설정

- Accelerator: `GPU T4 x2`
- Internet: `On` (`catboost==1.2.10` 설치용이며 외부 데이터는 받지 않음)
- 셀: `experiments/gated_residual_final/KAGGLE_CELL.py` 전체를 한 셀에 복사
- 실행: `Save Version` 한 번

코드는 먼저 입력 구조와 해시, T4 두 장, CatBoost smoke test를 확인한다. OOF gate가
실패하면 GPU 전체 학습을 시작하지 않고 30~60분 안에 종료될 수 있다. 통과하면 D0·D5를
각 3개 seed로 학습하고 감사까지 진행한다. 통과 경로의 예상 시간은 약 5~8시간이며 코드
상한도 8시간이다.

## 정상 로그

초기에는 다음 순서가 보여야 한다.

```text
FINAL_CANDIDATE_CODE_READY ...
FINAL_CANDIDATE_INPUTS_FOUND ...
FINAL_CANDIDATE_DEPENDENCIES_READY catboost=1.2.10
FINAL_CANDIDATE_INPUTS_VERIFIED ...
FINAL_CANDIDATE_GPU_READY count=2 names=Tesla T4 | Tesla T4
FINAL_CANDIDATE_SMOKE_SUCCESS
FINAL_CANDIDATE_STAGE phase=P1
FINAL_CANDIDATE_DECISION status=accepted|rejected candidate=...
```

`rejected`는 실행 실패가 아니다. 사전에 정한 제출 기준을 넘지 못해 전체 학습을 생략한
정상 종료다.

승인되면 다음 로그가 이어진다.

```text
FINAL_CANDIDATE_STAGE phase=P3
FINAL_CANDIDATE_JOB_START job=D0_s42 ...
FINAL_CANDIDATE_JOB_END job=D0_s42 status=completed
...
FINAL_CANDIDATE_STAGE phase=P4
FINAL_CANDIDATE_STAGE phase=P5
FINAL_CANDIDATE_CAMPAIGN_STATUS status=completed
```

## 받아야 할 결과

항상 `/kaggle/working`에 두 파일이 생긴다.

- `gated_residual_final_review.zip`
- `gated_residual_final_handoff.zip`

모든 gate와 행 독립 감사까지 통과했을 때만 다음 파일이 추가된다.

- `gated_residual_final_delivery.zip`

세 파일이 있으면 모두 전달한다. Delivery가 없으면 review와 handoff만 전달한다. Kaggle
셀은 DACON 제출 ZIP을 만들지 않는다. Delivery를 로컬에서 다시 검증한 다음에만 별도
패키징 단계로 넘어간다.

## 중단됐을 때

`gated_residual_final_handoff.zip`을 Kaggle Dataset으로 등록한 뒤, 공식 데이터와 최종
실험 입력에 더해 연결하고 같은 셀을 다시 Save Version한다. 코드·계약·입력 해시가 같은
경우에만 완료된 모델을 재사용한다. 다른 버전의 셀에서 만든 handoff는 섞지 않는다.

## 대회 규칙 안전선

- 평가 데이터 전체의 평균·빈도·순위·calibration을 계산하지 않는다.
- 평가 행 사이 rolling, lag, 누적 통계를 만들지 않는다.
- 현재 평가 행의 ID·손잡이·game type으로 학습 데이터에서 고정한 표만 조회한다.
- 행을 하나만 넣거나 순서를 바꾸거나 배치 크기를 바꿔도 같은 확률인지 감사한다.
- Public 점수로 강도를 다시 맞추지 않는다.
- 승인 근거와 현재 artifact 해시가 없으면 제출 패키징을 차단한다.
