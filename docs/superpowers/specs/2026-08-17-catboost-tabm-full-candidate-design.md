# CatBoost·TabM 전체 학습 후보 설계

**작성일:** 2026-08-17  
**대회:** DACON 236743, LG Aimers 9기 Phase 2  
**상태:** 승인된 설계

## 목적

시간 전이 OOF에서 승급한 `0.70 × TabM + 0.30 × CatBoost`를 실제 제출 후보로
옮긴다. 기존 제출 서버에서 정상 실행된 TabM 전체 학습 모델은 다시 학습하지 않는다.
CatBoost만 공식 학습 데이터 전체로 학습하고, 두 모델이 평가 행마다 독립적으로
예측하도록 만든다.

이 설계는 제출 ZIP을 바로 만들지 않는다. 먼저 배포 가능한 단일 CatBoost tree 수를
시간 전이 OOF로 검증하고, 전체 학습 모델과 추론 감사를 완료한다. acceptance와 현재
산출물 해시가 모두 맞기 전에는 패키징 경로를 열지 않는다.

## 현재 근거

검증된 CatBoost·TabM OOF delivery는 다음과 같다.

```text
artifact: catboost_tabm_blend_delivery.zip
sha256: ab7ca41e98e2b94b997369d7c777f8293c64acf61114f314110d17bf743edcfa
decision: blend_promoted
selected blend: 0.70 TabM + 0.30 CatBoost
weighted Brier: 0.24928747548299018
TabM baseline Brier: 0.24946623656342970
weighted gain: 0.00017876108043951566
```

두 fold 모두 개선됐다.

| fold | TabM | 70:30 blend | 개선량 |
|---|---:|---:|---:|
| 2022→2023 | 0.25086573594721084 | 0.25064939807593730 | 0.00021633787127354 |
| 2023→2024 | 0.24811080225115084 | 0.24796843480705882 | 0.00014236744409202 |

다만 CatBoost early stopping 결과는 첫 fold 4 trees, 둘째 fold 296 trees였다. 이
상태로는 하나의 전체 학습 tree 수를 정할 수 없다. 296을 바로 쓰거나 두 값을
평균하는 방식은 OOF에서 직접 검증되지 않았으므로 배포 정렬 실험을 먼저 수행한다.

## 공식 규칙 경계

2026-08-17에 아래 공식 페이지를 다시 확인했다.

- [대회 규칙](https://dacon.io/competitions/official/236743/overview/rules)
- [평가 및 코드 제출 사양](https://dacon.io/competitions/official/236743/overview/evaluation)
- [평가 데이터 독립 예측 재안내](https://dacon.io/competitions/official/236743/talkboard/417123?page=1&dtype=recent)

한 평가 행의 예측에는 그 행의 입력 변수, 그 행에서 만든 파생변수, 공식 학습
데이터로 고정한 모델과 상태만 사용할 수 있다. 평가 데이터의 다른 행이나 전체
분포로 만든 평균, 빈도, 순위, rolling, lag, 선수·팀·월·경기 집계는 사용하지 않는다.
행 하나만 넣었을 때와 전체 배치에 넣었을 때 같은 행의 예측이 같아야 한다.

평가 서버의 공식 사양은 Python 3.11.15, Ubuntu 22.04.5, NVIDIA L4 22.4 GiB,
CPU RAM 28 GB다. 패키지 설치와 추론은 각각 10분 이내, 제출 ZIP은 10 GB 이내,
압축 해제 후 32 GB 이내여야 한다. 인터넷은 패키지 설치 이후 사용할 수 없다.

## 1단계: 배포 정렬 OOF

기존 blend weight `0.70`은 고정하고 다시 탐색하지 않는다. 두 temporal fold에서
CatBoost를 400 trees까지 한 번씩 학습한다.

```text
2022까지 학습 → 2023 검증
2023까지 학습 → 2024 검증
```

기존 설정 `tree_native + hand_matchup`, seed 42, RMSE, depth 7, learning rate
0.05, border count 128, Bayesian bootstrap, bagging temperature 1.0,
l2 leaf reg 3.0, model size reg 0.5, max CTR complexity 1을 그대로 쓴다.

이 단계에서는 early stopping과 `use_best_model`을 끈다. fold별 400-tree 모델
하나에서 다음 prefix의 prediction을 계산한다.

```text
4, 32, 64, 128, 192, 296, 400
```

각 prefix마다 아래 하나의 배포 후보만 평가한다.

```text
p = 0.70 * p_tabm + 0.30 * clip(p_catboost, 0, 1)
```

승급 조건은 기존 계약과 같다.

- 검증 행 수 가중 Brier가 TabM보다 `0.00003` 이상 낮음
- 어느 fold도 TabM보다 `0.00003`를 초과해 악화하지 않음

여러 prefix가 통과하면 가중 Brier가 가장 낮은 하나를 선택한다. 차이가
`1e-12` 이내면 tree 수가 작은 후보를 고른다. segment 결과는 진단용이며 실행 후
새 차단 기준을 만들지 않는다. 어느 prefix도 통과하지 못하면 상태는
`deployment_blocked`다. 이 경우 전체 학습과 제출 후보 제작을 시작하지 않는다.

후보 수를 일곱 개로 제한하고 blend weight를 고정해 같은 OOF에 대한 사후 탐색 폭을
막는다. fold마다 일곱 모델을 따로 학습하지 않고 400-tree 모델 하나의 prefix를
평가한다.

## 2단계: CatBoost 전체 학습

배포 정렬 gate를 통과한 경우에만 같은 Colab 실행에서 공식 학습 데이터 전체
1,475,092행으로 CatBoost를 한 번 학습한다. tree 수는 1단계에서 선택한 값으로
고정하고 early stopping과 검증 데이터는 사용하지 않는다.

CatBoost 전체 학습 delivery에는 다음 파일만 넣는다.

```text
alignment/alignment_decision.json
alignment/fold_metrics.json
frozen_catboost/model.cbm
frozen_catboost/preprocessing_state.json
frozen_catboost/inference_manifest.json
logs/campaign.log
policy/policy.json
manifest.json
```

`preprocessing_state.json`은 공식 학습 데이터로 fit한 source/output schema,
categorical column, `tree_native + hand_matchup` 계약을 담는다. 평가 데이터에서
새 범주 사전이나 통계를 만들지 않는다. inference artifact에는 optimizer,
snapshot, validation prediction, 평가 데이터, cached prediction을 넣지 않는다.

`inference_manifest.json`은 다음 identity를 봉인한다.

- 공식 train과 Trackman SHA-256
- 배포 정렬 입력 delivery와 decision SHA-256
- CatBoost 1.2.10과 전체 parameter
- 선택 tree 수, seed 42와 전체 학습 행 수
- 전처리 상태와 model SHA-256
- 코드와 계약 SHA-256

## Colab 실행과 복구

배포 정렬과 전체 학습은 직접 업로드 방식의 한 셀로 실행한다. 첫 실행 입력은 다음
세 개다.

```text
catboost_tabm_blend_input.zip
tabm_colab_stage_C_delivery.zip
catboost_tabm_blend_delivery.zip
```

재개할 때만 새 캠페인의 resume ZIP 하나를 추가한다. 파일명 대신 member set,
manifest와 SHA-256으로 입력을 구분한다. Drive, GitHub, 외부 URL에서 코드나
데이터를 읽지 않는다.

CatBoost native snapshot은 300초 간격, 브라우저 emergency resume은 변경된
snapshot에 한해 20분 간격으로 만든다. 각 fold와 전체 학습 완료 직후에는 최신
resume을 즉시 내려받는다. 전체 세션 deadline은 업로드와 설치를 포함해 3시간이며,
deadline이 가까우면 새 job을 시작하지 않는다.

정상 완료 산출물은 review-only
`catboost_full_training_delivery.zip` 하나다. `deployment_blocked`일 때는 alignment
review와 resume만 만들며 frozen full model을 만들지 않는다.

## 3단계: 고정 TabM과 결합

기존 모델은 아래 candidate의 bytes와 해시를 그대로 재사용한다.

```text
candidate_id: tabm_hand_matchup_version_d_seed3407_v1
source: artifacts/tabm_submission_validation_handoff_v2/candidate/model/
```

새 candidate importer는 기존 TabM 네 파일과 검증된 CatBoost 세 파일만 복사한다.
두 source delivery, OOF decision, 전체 학습 decision과 모든 member SHA-256을 하나의
candidate identity로 묶는다. 새 candidate ID는 선택된 tree 수를 포함하되 사람이
읽을 수 있는 고정 문자열로 생성한다.

## 독립 추론 runtime

제출용 runtime은 다음 순서로 동작한다.

1. 모델 member와 SHA-256을 검증
2. `data/test.csv`와 `data/sample_submission.csv`의 schema와 row ID를 검증
3. 기존 고정 상태로 TabM 행 단위 전처리와 예측
4. 공식 학습 상태로 CatBoost `tree_native + hand_matchup` 변환과 예측
5. CatBoost 예측을 `[0,1]`로 clip
6. `0.70 * TabM + 0.30 * CatBoost`
7. 유한성, 범위, 행 순서를 확인한 뒤 `output/submission.csv`를 원자적으로 게시

test mean, 표준편차, 빈도, OOV 비율, 순위, groupby, rolling과 calibration fit은
금지한다. 배치 처리는 계산 효율만 위한 것이며 행 사이에서 상태를 공유하지 않는다.
네트워크, 자동 다운로드, CPU/GPU별 다른 예측식, cached row-ID prediction도 없다.

## 검증과 acceptance

전체 학습 delivery가 돌아오면 먼저 재귀 해시를 확인한다. 그다음 기존 제출 감사
구조를 확장한 review-only validation handoff를 만든다.

- 공식 5행을 원본·역순·shuffle·singleton·여러 batch partition으로 예측
- 각 모델 prediction과 최종 blend를 row ID 기준으로 비교
- 전처리 state digest가 실행 전후 같은지 확인
- 245,789행 합성 scale fixture로 시간, RAM과 VRAM 측정
- Python 3.11.15, PyTorch 2.7.1, pandas 2.0.3, NumPy 1.26.4 호환성 확인
- CatBoost 1.2.10 설치 시간과 model load/predict 확인
- 설치 480초, 추론 480초, RAM 22 GiB, VRAM 20 GiB의 내부 gate 적용

실제 hidden 평가 행은 제출 전에 볼 수 없으므로, independence evidence는
`official_sample_plus_synthetic_scale`로 정확히 표시한다.

acceptance는 temporal validation, deployment alignment, provenance, row
independence, evaluator runtime, dependency/license와 당일 공식 규칙 검토가 모두
통과할 때만 생성한다. acceptance가 참조하는 candidate, runtime, model, policy와
audit SHA-256이 현재 파일과 하나라도 다르면 무효다.

## 패키징 차단

이 설계 구현 단계에서는 blend 제출 ZIP 생성 코드를 제공하거나 실행하지 않는다.
전체 학습 delivery와 validation evidence가 돌아오고 acceptance가 생성된 뒤 별도
승인을 받아 기존 `submission/package.py`의 단일 writer를 확장한다.

최종 package가 허용될 경우에도 구조는 공식 형식을 지킨다.

```text
submit.zip
├── model/
├── script.py
└── requirements.txt
```

requirements 후보는 `tabm==0.0.3`, `rtdl-num-embeddings==0.0.12`,
`catboost==1.2.10` 세 개뿐이다. 실제 package 허용 전 Python 3.11 평가 환경에서
설치 시간과 dependency closure를 다시 확인한다.

## 실패 처리

- 잘못된 입력, 계약·코드·데이터 hash drift: GPU 작업 전에 실패
- 배포 정렬 gate 실패: `deployment_blocked`, 전체 학습 금지
- snapshot 손상: 이전 검증 resume 보존
- deadline 도달: 최신 검증 resume 게시 후 종료
- 전체 모델 또는 전처리 state 검증 실패: candidate import 금지
- independence·호환성·용량 gate 실패: acceptance와 package 금지

실패한 blend 후보가 기존 단일 TabM 후보의 기록이나 재사용 가능성을 막지는 않는다.

## 비목표

- TabM 재학습 또는 seed 변경
- blend weight 재탐색
- CatBoost parameter, feature bundle 또는 calibration 추가 탐색
- 평가 데이터 분포를 이용한 보정
- 전체 학습 전에 submit.zip 생성
- Phase 3 재현 학습 코드와 PPT 제작
- 현재 main에 남아 있는 Stage C/Stage P 관련 사용자 변경 수정

## 성공 기준

1. 고정 prefix 배포 정렬이 두 fold gate를 통과한다.
2. 선택된 tree 수로 CatBoost 전체 학습 delivery가 완성되고 재귀 검증된다.
3. 기존 TabM bytes와 새 CatBoost bytes가 하나의 candidate identity로 봉인된다.
4. 행 독립성, Python 3.11 호환성, 설치·추론 시간과 자원 gate를 통과한다.
5. 같은 identity로 acceptance가 생성된다.
6. 그 이후 별도 승인 전까지 제출 ZIP은 존재하지 않는다.
