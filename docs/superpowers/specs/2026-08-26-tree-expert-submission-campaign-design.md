# CatBoost 트리 전문가 제출 후보 캠페인 설계

**작성일:** 2026-08-26

**대회:** DACON 236743, LG Aimers 9기 Phase 2

**상태:** 구현 전 승인 설계

**실행 환경:** Kaggle T4 x2, 사용자가 전체 데이터 학습 실행

## 1. 목적

Public `872.3920184667`을 기록한 기존 `TabM + hand_matchup` 제출물을 기준으로,
고차원 범주형 상호작용과 시즌 스냅샷을 사용하는 CatBoost 전문가가 시간 전이 OOF를
안정적으로 개선하는지 확인한다. 후보가 사전 등록한 성능·안정성·규칙·실행 gate를
모두 통과한 경우에만 전체 데이터 모델 delivery를 만들고, 별도 로컬 감사 이후 한 번의
명령으로 DACON 제출 ZIP을 생성한다.

이 캠페인은 1130점 제출물의 바이너리 모델이나 저장 통계를 복사하지 않는다. 해당
제출물에서 관찰한 구조적 가설만 공식 학습 데이터와 공식 TrackMan 데이터로 독립
재현한다.

## 2. 현재 근거

### 2.1 기존 기준선

- 후보 ID: `tabm_hand_matchup_version_d_seed3407_v1`
- Public: `872.3920184667`
- `2022 -> 2023` Brier: `0.2508657359`
- `2023 -> 2024` Brier: `0.2481108023`
- 제출 ZIP SHA-256:
  `bcea66999322721e60016761ce041ca151c71b034d0e4122f0b2852f5b06697a`
- 단일 TabM 구조, seed `3407`, 전체 데이터 3 epoch 학습

### 2.2 이미 확인한 결과

- S1 현재 시즌 스냅샷은 2022, 2023, 2024 세 시간 fold에서 모두 기준선보다
  개선됐다.
- S1의 세 fold 결합 개선량은 약 `0.00026836`이었다.
- S1에 대한 추가 seed 확인에서는 2022 fold 불안정성이 커 최종 후보가 기각됐다.
- `game_type=F` fallback 다중 seed 후보는 결합 개선량이
  `-0.0000816644`로 기각됐다.
- 기존 CatBoost·TabM 단순 확률 혼합은 fold별 적정 tree 수가 달라 전체 학습 정렬
  gate를 통과하지 못했다.
- 기존 계층형 TabM과 행 피처 TabM은 현재 단일 TabM을 안정적으로 넘지 못했다.

따라서 이번 캠페인은 S1을 재탐색하지 않고 고정 피처 근거로 사용한다. 실패한 F
fallback도 반복하지 않는다. 신규성은 CatBoost ordered boosting, 고차원 범주형 CTR,
S1 anchor 잔차 학습, CatBoost 안에서의 TrackMan 재평가와 감사 가능한 실패 유형
학습이다.

### 2.3 1130점 제출물에서 가져오는 가설

정적 검토 대상:

- 파일: `submit_gylee_1130_v2.0.0.zip`
- SHA-256:
  `48d367af315681040cceeb3013d2c87bbdf9ea7539a8dfb55ec781d0e71a9a69`
- CatBoost `.cbm` 59개, XGBoost 5개, LightGBM 5개, CatBoost pipeline 5개,
  작은 PyTorch 보정 모델 1개
- 시즌별 스냅샷, 공식 TrackMan 요약, 범주형 CTR, 실패 유형 전문가, 다중 seed,
  잔차 결합을 사용

가져오지 않는 요소:

- 제공된 모델, pickle, 통계, 코드의 직접 복사
- Public 리더보드 점수로 정한 F 구간 강도
- 전체 75개 예측기 복제
- 계보를 공식 데이터로 증명할 수 없는 저장 자산
- 테스트 분포 또는 예측 평균을 이용한 후처리

## 3. 공식 규칙과 제출 제한

평가 행 하나의 예측은 다음 정보만 사용한다.

- 해당 평가 행의 입력 변수
- 해당 평가 행 변수로 만든 파생변수
- 공식 학습 데이터
- 공식 학습 데이터와 공식 TrackMan 데이터만으로 미리 고정한 통계·모델

금지한다.

- 평가 행 간 평균, 빈도, 순위, 집계, 이웃, rolling, lag, 누적 상태
- 평가 행에서 범주 사전, 결측 대치값, 정규화값, calibration 또는 ID 매핑 학습
- 평가 예측 평균에 맞춘 scale·shift
- 공식 제공 범위를 벗어난 외부 데이터
- 추론 시 네트워크·외부 서버·추가 다운로드
- 샘플 제출 ID 불일치 시 기존 확률을 남기는 fallback

제출 사양:

- `submit.zip` 최상위는 `model/`, `script.py`, `requirements.txt`
- 설치 시간 10분 이하
- 245,789행 추론 시간 10분 이하
- 제출 ZIP 10GB 이하, 압축 해제 후 32GB 이하
- Ubuntu 22.04.5, Python 3.11.15, 6 vCPU, RAM 28GB, L4 22.4GiB
- 오프라인 추론

참조:

- https://dacon.io/competitions/official/236743/overview/evaluation
- https://dacon.io/competitions/official/236743/overview/rules
- https://dacon.io/competitions/official/236743/talkboard/417123?dtype=recent&page=1

## 4. 전체 실행 구조

캠페인을 한 번의 10시간 실행으로 만들지 않는다.

```text
Kaggle E1: 최신 fold 구조 선별, 최대 4시간
    -> tree_expert_e1_review.zip
    -> tree_expert_e1_resume.zip

Kaggle E2: 세 fold·다중 seed 확인, 조건부 전체 학습, 최대 6시간
    -> tree_expert_e2_review.zip
    -> tree_expert_e2_resume.zip
    -> tree_expert_delivery.zip  # 모든 acceptance gate 통과 시에만

로컬 감사 및 패키징
    -> submit_tree_expert_v1.zip # delivery 재검증 후에만
```

E1과 E2는 완료 job을 원자적으로 기록한다. 같은 contract, 코드, 데이터와 resume을
다시 실행하면 완료 job은 재사용하고 진행 중이거나 미완료인 job만 시작한다. 기존
review와 resume을 덮어쓰지 않는다.

## 5. 공통 데이터와 시간 fold

입력 파일:

- `train.csv`
- `trackman_history.csv`
- 기존 TabM OOF 및 승인 모델 계보
- E2에서는 E1 review와 resume

검증 fold:

| fold | 학습 가능 시즌 | 검증 시즌 | TrackMan cutoff |
|---|---|---:|---:|
| F1 | 2021까지 | 2022 | 2021 |
| F2 | 2022까지 | 2023 | 2022 |
| F3 | 2023까지 | 2024 | 2023 |
| Final | 2024까지 | 평가 데이터 | 2024 |

모든 학습 통계와 TrackMan 매핑은 해당 fold cutoff 이전 자료만 사용한다. 검증 행의
타깃은 해당 fold의 metric에만 사용하며, 그 fold를 예측하는 피처 상태, 모델,
calibration 또는 결합 계수 생성에는 사용하지 않는다. 완료된 과거 fold의 OOF와
타깃으로 다음 연도 fold에 적용할 결합 계수를 고르는 순차 검증만 허용한다.

### 5.1 기준 TabM OOF binding

비교 기준은 Public 점수만 같은 임의의 TabM이 아니라 872점 제출물과 동일한 모델
설정, 전처리, seed를 fold cutoff에 맞춰 재학습한 OOF다.

- 기존 F2·F3 OOF는 후보 ID, 설정, 전처리와 코드 SHA-256이 제출 영수증과 연결될
  때만 재사용한다.
- 동일 설정의 F1 OOF가 없으면 E1에서 F1 기준 job 하나를 추가한다.
- temporal T1 anchor가 동일 identity를 증명하지 못하면 기준 TabM OOF로 대체하지
  않는다.
- 세 fold 모두에서 CatBoost와 TabM의 `row_id`, target, fold 정의가 정확히 같아야
  결합 평가를 허용한다.
- binding이 불완전하면 CatBoost 단독 연구 결과는 만들 수 있지만 TabM 결합과
  acceptance는 차단한다.

## 6. 공통 행 피처

### 6.1 원본 범주형

- `pitcher_id`, `batter_id`
- `pitcher_team_id`, `batter_team_id`
- `pitcher_hand`, `batter_hand`
- `top_bottom`, `game_type`, `base_state`
- `balls_before`, `strikes_before`, `outs_before`

### 6.2 명시적 범주형 상호작용

- `count_state = balls_before | strikes_before`
- `hand_matchup = pitcher_hand | batter_hand`
- `pitcher_id | batter_hand`
- `pitcher_id | count_state`
- `pitcher_id | base_state`
- `pitcher_id | game_type`
- `batter_id | pitcher_hand`
- `pitcher_team_id | count_state`
- `count_state | base_state`
- `hand_matchup | count_state`
- `inning_bin | score_bin`
- `leverage_bin | score_bin`
- `leverage_bin | num_runners_on | count_state`

문자열은 현재 행에서만 조합한다. 평가 데이터 전체에서 범주 목록이나 빈도를 만들지
않는다. CatBoost가 학습 데이터에서만 ordered target statistics를 학습한다.

### 6.3 수치형 파생변수

- 최근 1·3·5경기 성공률 평균, 표준편차, 기울기
- 최근 middle 비율 평균과 기울기
- 투수·타자 노출 수 기반 신뢰도
- 투수·타자 성공률 차이
- 성공률 logit
- 구종 비율 entropy
- leverage, 점수 차, 주자 수를 이용한 행 단위 압박 지표

### 6.4 S1 시즌 스냅샷

검증 시즌 이전 학습 데이터의 시즌 시작 전 누적 상태를 고정한다. 현재 행의 공식
`asof` 누적값에서 이 상태를 빼 현재 시즌 표본 수와 성공 건수를 복원한다.

```text
season_n = current_asof_n - train_only_before_season_n
season_success = current_asof_n * current_asof_rate - train_only_before_season_success
season_rate = shrink(season_success, season_n, train_only_prior)
```

계산 불가, 신규 선수, 음수 보정과 낮은 표본 신뢰도를 별도 flag로 제공한다. 평가
데이터의 다른 행은 사용하지 않는다.

### 6.5 TrackMan

TrackMan은 후보 C2부터 사용한다.

- 공식 TrackMan cutoff 이전 데이터만 집계
- 투수 ID 매핑은 공식 train과 TrackMan으로만 고정
- 매핑 coverage, confidence, 모호성, 미매핑 flag 기록
- 구속, 회전, 무브먼트, 익스텐션, 릴리스, zone speed
- 구종군 비율과 구종군별 요약
- 최근 시즌과 이전 시즌 차이
- 작은 표본은 타깃을 사용하지 않는 train-only prior로 축소

매핑 coverage 또는 신뢰도가 사전 기준에 못 미치면 C2만 생략한다.

## 7. 실패 유형 복원 감사

공식 `train.csv`에는 `control_success`만 있고 `middle`, `wild`, `reverse` 타깃 열은
없다. 실패 유형을 임의로 만들지 않는다.

학습 데이터 안에서 같은 투수의 다음 누적 상태가 정확히 한 투구 증가한 경우에만
현재 투구의 누적률 변화량을 계산한다. `middle`과 `reverse`는 해당 누적률 변화로
복원하고, 성공은 공식 `control_success`를 사용한다. 남은 실패는 의미를 과장하지
않고 `other_failure`로 이름 붙인다.

감사 gate:

- 연결 가능한 행 coverage `>= 0.98`
- 누적 변화량의 `{0, 1}` 근접 비율 `>= 0.999`
- 복원 성공 여부와 `control_success` 일치율 `>= 0.999`
- `middle`과 `reverse` 동시 양성률 `<= 0.001`
- 각 실패 범주의 학습 표본 `>= 10,000`
- 연결은 fold 학습 시즌 안에서만 수행
- 검증 시즌 행은 해당 fold 복원 모델 학습에 사용하지 않음

하나라도 실패하면 C3를 `skipped_unreliable_labels`로 기록한다. C0~C2와 전체
캠페인은 계속 진행한다.

## 8. E1: 최신 fold 구조 선별

공통 조건:

- fold: F3, `2023 -> 2024`
- seed: `3407`
- early stopping과 best iteration 기록
- Brier를 최종 선택 metric으로 사용
- T4 두 장에 독립 job을 하나씩 배치

후보:

### C0 `native_ctr`

- 이진 `control_success` CatBoost
- 공통 행 피처 + S1
- CatBoost native categorical와 ordered boosting

### C1 `anchor_residual`

- S1 train-only 계층 확률을 anchor로 생성
- CatBoostRegressor가 `control_success - anchor` 잔차를 학습
- 최종 확률은 `clip(anchor + residual)`

### C2 `trackman_residual`

- C1 + TrackMan 피처
- TrackMan mapping gate를 통과한 경우에만 실행

### C3 `failure_aware`

- 실패 유형 복원 gate를 통과한 행으로 다중 클래스 CatBoost 학습
- 클래스: `success`, `middle`, `reverse`, `other_failure`
- 최종 이진 확률은 `P(success)`
- 복원되지 않은 학습 행은 C3에서만 제외

E1 선택:

- 기존 TabM 기준 OOF와 독립 CatBoost Brier를 모두 기록
- 필요한 경우 동일 설정의 F1 기준 TabM OOF job 하나를 추가
- 기준 대비 개선량, bootstrap 하한, 세그먼트 회귀 기록
- 상위 최대 두 구조만 E2로 승급
- 후보가 모두 최신 fold에서 `0.00005` 이상 회귀하면 E2를 중단

## 9. E2: 세 fold·다중 seed·결합 확인

### 9.1 구조 확인

E1 상위 최대 두 구조를 F1과 F2에서 seed `3407`로 학습한다. F3 결과와 합쳐 세
fold의 방향을 판단한다.

최종 구조는 다음 순서로 선택한다.

1. 최악 fold 개선량 최대화
2. 세 fold 행 수 가중 개선량 최대화
3. 최신 fold 개선량 최대화
4. 동일하면 TrackMan과 실패 유형 의존성이 적고 추론이 빠른 구조

### 9.2 seed 확인

선택 구조만 seed `42`, `2026`으로 F1~F3에서 추가 학습한다. seed `3407`과 함께
확률 평균한 세 seed ensemble을 평가한다. 단일 seed가 더 좋아도 사전 안정성 gate를
통과하지 못하면 임의로 그 seed만 선택하지 않는다.

### 9.3 기존 TabM과 결합

결합 후보를 사전 고정한다.

- CatBoost 단독
- 확률 평균: TabM/CatBoost `90/10`, `80/20`, `70/30`
- logit 평균: TabM/CatBoost `90/10`, `80/20`, `70/30`

같은 OOF 전체에서 가중치를 선택하고 같은 OOF로 성능을 과대평가하지 않는다.

- 2022 OOF로 2023 적용 가중치 선택
- 2022+2023 OOF로 2024 적용 가중치 선택
- 순차 적용 결과와 고정 가중치의 세 fold 결과를 모두 기록
- 최종 가중치는 사전 후보 중 순차 검증이 가장 안정적인 하나만 사용
- Public 리더보드는 가중치 선택에 사용하지 않음

## 10. Acceptance gate

전체 데이터 학습과 delivery 생성을 허용하려면 다음 조건을 모두 만족해야 한다.

### 성능

- 세 fold 행 수 가중 Brier 개선 `>= 0.00015`
- 최신 2024 fold 개선 `>= 0.00010`
- 최악 fold 회귀 `<= 0.00005`
- 선수 block bootstrap 개선량 95% 하한 `> 0`
- eligible 세그먼트 최대 회귀 `<= 0.00050`

성능 등급은 별도로 기록한다.

- `incremental`: `0.00015` 이상
- `competitive`: `0.00025` 이상
- `breakthrough`: `0.00045` 이상

### seed 안정성

- 세 seed ensemble의 가중 개선량이 seed `3407` 방향과 일치
- 새 seed 중 하나의 대규모 붕괴가 ensemble로 가려지지 않음
- 각 seed·fold의 best iteration과 확률 분포가 유한하고 정상

### 실행·규칙

- 출력 확률 유한, `[1e-5, 1-1e-5]`
- 입력과 출력 `row_id` 정확히 일치, 중복·누락·fallback 없음
- 단독 행, 역순, 셔플, 배치 크기 변경에서 허용 오차 내 동일 예측
- 동일 행을 다른 평가 행과 함께 넣어도 동일 예측
- 평가 행 간 groupby, rolling, lag, 누적 상태 없음
- 245,789행 모의 추론 `<= 480초`
- peak RSS `<= 22GB`, peak GPU memory `<= 20GiB`
- 제출 ZIP `<= 10GB`, 압축 해제 `<= 32GB`
- 패키지 설치 모의 실행 `<= 480초`

하나라도 실패하면 상태는 `rejected` 또는 `blocked`이고 delivery와 제출 ZIP을 만들지
않는다.

## 11. 전체 데이터 학습

Acceptance gate를 통과한 구조만 공식 2019~2024 학습 데이터로 최종 학습한다.

- CatBoost seed: `3407`, `42`, `2026`
- fold별 best iteration의 사전 고정 집계값 사용
- 전체 데이터 성능을 보고 iteration을 다시 탐색하지 않음
- 전처리 상태, S1 스냅샷, TrackMan 매핑과 요약을 공식 학습 데이터로 재생성
- 기존 TabM 결합이 선택된 경우 승인된 모델 해시와 정확히 연결
- optimizer와 중간 checkpoint는 delivery에 포함하지 않음

전체 학습 중단 시 완료 seed는 재사용한다. 미완료 seed만 resume한다.

## 12. Kaggle 산출물과 로그

E1 성공 로그:

```text
TREE_E1_INPUTS_VERIFIED ...
TREE_E1_GPU_READY device_count=2 ...
TREE_E1_LABEL_AUDIT status=passed|skipped ...
TREE_E1_JOB_START candidate=... fold=2023->2024 seed=3407 gpu=...
TREE_E1_JOB_PROGRESS candidate=... iteration=... brier=... eta_seconds=...
TREE_E1_JOB_END candidate=... status=completed
TREE_E1_SUCCESS review=... resume=...
```

E2 성공 로그:

```text
TREE_E2_INPUTS_VERIFIED ...
TREE_E2_RESUME_READY ...
TREE_E2_JOB_START candidate=... fold=... seed=... gpu=...
TREE_E2_DECISION status=accepted|rejected reason=...
TREE_E2_FULL_FIT_START seed=...
TREE_E2_FULL_FIT_END seed=... model_sha256=...
TREE_E2_SUCCESS review=... resume=... delivery=...|none
```

오류는 다음 형식으로 끝난다.

```text
TREE_EXPERT_ERROR stage=<stage> type=<type> message=<message>
```

브라우저 자동 다운로드는 각 version 최종 산출물에만 요청한다. 반복 snapshot을 여러
개 다운로드하지 않는다. 중간 checkpoint는 Kaggle working output과 resume bundle에
기록한다.

## 13. Delivery 계약

모든 gate를 통과했을 때만 `tree_expert_delivery.zip`을 쓴다.

포함 항목:

- 최종 CatBoost 모델 최대 3개
- 선택된 기존 TabM 모델 또는 그 정확한 해시 참조
- frozen 전처리와 범주 schema
- S1 스냅샷
- 선택 시 공식 TrackMan 매핑·요약
- 선택 시 실패 유형 복원 감사 결과와 클래스 정의
- 세 fold OOF와 지표
- 순차 결합 판정
- 행 독립성 사전 검사 결과
- 학습 로그
- 파일별 SHA-256, 데이터 계보, 코드 identity
- `accepted` decision

delivery는 제출물이 아니며 `submission_package: false`를 명시한다.

## 14. 로컬 재검토와 제출 패키징

사용자가 delivery를 전달하면 다음을 순서대로 수행한다.

1. ZIP 경로 안전성, 중복, 암호화, CRC 검사
2. manifest member set과 파일별 SHA-256 검사
3. 공식 데이터·기준 TabM·코드 identity binding 검사
4. acceptance decision과 OOF 수치 재계산
5. 실패 유형·TrackMan 계보 정적 감사
6. 실제 제출 runtime으로 단독·역순·셔플·배치 독립성 검사
7. 245,789행 모의 추론 시간과 메모리 검사
8. Python 3.11.15 및 제출 패키지 의존성 검사
9. 최상위 구조와 `output/submission.csv` 계약 검사

모두 통과한 경우에만 `submission/package.py`가
`submit_tree_expert_v1.zip`을 원자적으로 생성한다. 기존 제출 파일을 덮어쓰지 않고
제출 ZIP SHA-256과 영수증을 남긴다.

최종 ZIP:

```text
submit_tree_expert_v1.zip
├── model/
├── script.py
└── requirements.txt
```

`script.py`는 `data/test.csv`와 `data/sample_submission.csv`만 읽고
`output/submission.csv`만 쓴다. ID가 다르면 즉시 실패하며 sample submission의 기존
확률을 남기지 않는다. 제출 업로드는 자동화하지 않고 사용자가 직접 수행한다.

CatBoost가 선택되면 `catboost==1.2.10`만 추가 설치 대상으로 고정한다. 평가 서버
기본 버전과 충돌할 수 있는 `pandas`, `numpy`, `scikit-learn`, `torch`는 임의로
재설치하거나 다른 버전으로 고정하지 않는다. TabM이 결합되면 기존 872점 제출에서
검증한 정확한 런타임과 모델 해시를 재사용한다.

## 15. 자원 예산

| 단계 | 최대 벽시계 | 주요 작업 |
|---|---:|---|
| E1 준비·감사 | 0.5시간 | 입력·label·TrackMan mapping 감사 |
| E1 학습 | 3.0시간 | F3 최대 4후보, 필요 시 F1 기준 TabM 1개 |
| E1 정리 | 0.5시간 | metric·review·resume |
| E2 fold 확인 | 2.0시간 | 상위 2구조 F1/F2 |
| E2 seed 확인 | 2.5시간 | 최종 구조 2개 seed x 3fold |
| E2 전체 학습 | 1.0시간 | 3 seed, gate 통과 시만 |
| E2 감사·산출물 | 0.5시간 | review·resume·delivery |
| 합계 | 최대 10시간 | 보통 8시간 전후 |

각 version은 자체 deadline 10분 전에 새 job을 시작하지 않고 산출물 publication을
우선한다. E1 최대 4시간, E2 최대 6시간으로 나눠 한 번의 장시간 실패로 전체 계산을
잃지 않는다.

## 16. 비목표

- 1130 제출물의 모델·코드·통계 복사
- 6 seed와 75개 모델의 즉시 복제
- F 전용 리더보드 튜닝
- 새 DL 구조 탐색
- 제출 이후 점수로 내부 가중치 재조정
- 테스트 행 관계를 사용하는 어떤 피처도 허용하지 않음
- acceptance 이전 제출 ZIP 생성
- 자동 DACON 업로드

## 17. 완료 조건

구현 단계의 완료는 다음을 뜻한다.

- E1/E2 계약과 재시작 가능한 runner가 합성 fixture에서 검증됨
- 한 셀 Kaggle 실행 코드가 1MB kernel source 제한을 만족함
- 사용자 실행 목적·입력·출력·예상 시간·재실행 안전성·성공/오류 로그가 문서화됨
- rejected 후보가 full fit, delivery 또는 submission package에 도달할 수 없음
- 독립 후보 하나의 실패가 다른 후보 실행을 막지 않음
- 기존 사용자 변경과 기존 제출 후보를 수정하지 않음
