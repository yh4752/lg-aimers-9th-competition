# T4×2 분할형 전처리 캠페인 설계

## 목적

기존 `preprocessing_campaign_v1`의 760개 Wave A 전체 실행은 연구 질문은 넓지만,
첫 실측 기준 약 1,407 GPU-hours가 필요해 Kaggle T4×2의 10시간 예산과 맞지 않는다.
이번 캠페인은 모델을 최종 확정하거나 가능한 전처리를 모두 소진하는 실험이 아니다.
다음 후속 모델 실험에서 사용할 모델별 전처리 기본 후보를, 시간 누출 없이 제한된
계산량으로 선별하는 것이 목적이다.

Codex는 코드와 합성 fixture만 실행한다. 공식 데이터 학습, GPU 실행과 Kaggle
Save Version은 사용자가 수행한다. 제출 파일이나 제출 패키지는 만들지 않는다.

## 고정 근거

- 학습 데이터는 2019~2024년 1,475,092행이며 시즌별 행 수는 약 23.7만~25.4만이다.
- 인접 시즌 adversarial AUC가 모두 거의 1이므로 무작위 fold 결과를 채택 근거로
  사용하지 않는다.
- 완료된 TabM P3 `2019→2020`, seed 42, `dl_standard`의 Brier는
  `0.2476358521`이고 약 6,663초가 걸렸다.
- 이 결과는 train-prior Brier보다 약 `0.00231543` 좋지만 한 fold·한 seed이므로
  TabM을 주력 DL로 확정하지 않는다.
- EDA에서 투수 성공률 평활화는 집계상 `K=100`, 타자 성공률 평활화는 `K=250`이
  우선 후보였다.
- 2024년 투수 ID 미관측률은 약 19.9%이며 ID/OOV 처리는 별도 검증 가치가 있다.
- 손잡이 조합은 5개 fold 중 4개에서 방향이 개선됐지만 집계 효과는 작았다.
- 결측 지표, count interaction과 투수 팀 승리 기대값은 EDA 효과가 더 작으므로
  10시간 캠페인의 우선순위를 낮춘다.

## 검토한 접근

### 모든 최신 모델을 동일하게 탐색

CatBoost, TabM, FT-Transformer, TabNet과 TabR에 같은 예산을 배정하면 누락 위험은
줄지만, 제한 시간 성능이 수렴 성능처럼 해석되고 전처리 비교 시간이 사라진다.
채택하지 않는다.

### CatBoost와 TabM만 깊게 탐색

실행 안정성과 효율은 좋지만 전처리 효과가 TabM의 귀납적 편향에만 의존하는지 알 수
없다. 전처리 연구의 대표 패널로는 너무 좁아 채택하지 않는다.

### 핵심 모델, 누락 방지 슬롯과 단계적 승급

이 접근을 채택한다. CatBoost를 정형 모델 기준선으로 유지하고, TabM과
FT-Transformer를 DL 핵심 후보로 비교한다. TabNet에는 짧지만 명시적인 승급 기회를
준다. 모델 선별 뒤 한 DL 대표에만 단일 전처리 ablation을 적용한다.

TabR는 성능 때문에 제외하지 않는다. 현재 저장소 구현은 대규모 fold에서 retrieval
key 재계산 비용 문제가 있으므로, 캐시 확장성 검증 전까지 별도 연구 트랙으로 보류한다.
TabICLv2 등 foundation model도 대회 사용 계약이 확정될 때까지 별도 트랙으로 둔다.
일반 MLP/ResNet은 TabM이 같은 MLP 계열의 더 강한 대표 역할을 하므로 이번 예산에서는
실행하지 않는다.

## 실행 예산과 Save Version 경계

Kaggle 가속기는 T4×2이며 10시간은 벽시계 예산, 최대 약 20 GPU-hours로 해석한다.
한 Save Version에서 전체 캠페인을 실행하지 않는다.

- 동일한 단일 Kaggle 셀을 최대 5번 Save Version으로 실행한다.
- 각 실행은 셀 시작 시각부터 6,300초를 절대 상한으로 둔다.
- 새 학습 시작은 종료 900초 전에 막고, 마지막 600초는 체크포인트 검증과 산출물
  압축에 예약한다.
- 다섯 실행의 절대 상한은 8시간 45분이다. 남은 약 1시간 15분은 설치, 시작 지연과
  실패 재실행 여유다.
- 각 실행은 이전 resume bundle을 검사해 다음 미완료 stage를 자동 선택한다.
- 사용자는 정상 경로에서는 stage 사이 결과를 Codex에 전달하지 않는다.
- `STAGE_NEEDS_REVIEW` 또는 `STAGE_ERROR`일 때만 중간 결과를 전달한다.

Kaggle 셀은 GitHub에 접속하지 않는다. 공식 데이터는 Kaggle Input에서 읽고, 실행
코드는 셀 안에 포함한다. 이전 실행 결과도 Kaggle Input으로 연결한 resume bundle에서
읽는다.

입력에 resume bundle이 없으면 Stage 1로 시작한다. 하나 이상이면 campaign ID와
artifact hash가 유효한 bundle 중 완료 stage가 가장 높은 하나를 선택한다. 같은 stage의
서로 다른 유효 bundle이 함께 있어 어느 쪽이 최신인지 결정할 수 없으면 임의 선택하지
않고 `STAGE_NEEDS_REVIEW`로 종료한다.

## 병렬 실행과 상태 소유권

GPU 0과 GPU 1은 하나의 모델을 분산 학습하지 않고 서로 다른 작업을 실행한다.
각 worker는 `CUDA_VISIBLE_DEVICES`로 물리 GPU 하나에 고정하고 자신의 격리된 job
디렉터리에만 기록한다. worker는 공통 manifest를 수정할 수 없다.

부모 실행기만 다음 작업을 수행한다.

1. 실행할 job과 GPU를 배정한다.
2. worker 완료 코드를 확인한다.
3. metrics와 predictions의 해시를 검증한다.
4. 검증된 결과만 공통 manifest에 원자적으로 병합한다.
5. 남은 시간을 계산해 새 작업 시작 여부를 결정한다.

완료 job은 설정·데이터·코드 해시가 모두 일치할 때만 재실행을 건너뛴다. 체크포인트는
epoch 경계에서 저장한다. 시간 종료, KeyboardInterrupt와 worker 종료는 완료가 아니라
`pending`으로 기록한다.

## 데이터 프로토콜

### 탐색용 proxy

- 주 fold: `2019~2023 → 2024`
- 안정성 fold: `2019~2022 → 2023`
- 학습 행: fold별 최대 400,000행
- 표본: 학습 시즌 비율을 유지하는 결정적 `row_id` 해시 표본
- 검증 행: 해당 검증 시즌 전체
- seed: 42
- test: 스키마 확인만 허용

전처리 상태, 범주 사전, 결측 대치값, 표준화 통계와 target을 사용하는 smoothing
통계는 해당 fold의 실제 학습 표본에서만 fit한다. 검증 또는 test 행은 fit에 사용할 수
없다. 표본 row ID와 해시는 산출물에 기록한다.

400,000행 proxy는 최종 채택 증거가 아니다. 방향 선별용이며 ID/OOV처럼 표본 크기에
민감한 후보는 마지막 stage의 전체 학습 행 비교를 통과해야 한다.

### 전체 행 확인

최종 확인은 `2019~2023 → 2024`에서 2019~2023의 전체 학습 행과 2024년 전체 검증
행을 사용한다. proxy에서 정한 모델, 설정과 seed를 바꾸지 않는다.

## 모델 패널과 승급

### 핵심 패널

- CatBoost: `tree_native` 기준선
- TabM: 현대적 MLP·parameter-efficient ensemble 대표
- FT-Transformer: attention 기반 feature interaction 대표
- TabNet: 순차 feature selection 가설을 확인하는 누락 방지 슬롯

모든 DL은 `raw_typed + dl_standard`, 같은 proxy 행, 검증 행과 seed를 사용한다.
후보별 최대 학습 시간은 45분이며 TabNet의 최초 승급 시험은 30분이다. 각 후보는
early stopping과 Brier 최적 checkpoint를 사용하고 학습 곡선, 처리속도와 GPU 사용량을
남긴다.

TabM과 FT-Transformer는 기존 `campaign_v1.json`의 p2 모델·`level_2` 학습 설정을
사용한다. TabNet은 `n_d=64`, `n_a=64`, `n_steps=5`, `gamma=1.5`,
`lambda_sparse=0.0001`, `momentum=0.02`, `mask_type=sparsemax`로 고정하고 나머지
optimizer·early-stopping 설정은 `level_2`와 맞춘다. CatBoost 기준선은 기존 검증
구조인 depth 7, 400 iterations를 사용한다. 실행 전 고정 dependency를 불러오지 못하면
TabNet을 조용히 건너뛰지 않고 학습 시작 전에 `STAGE_ERROR`로 종료한다.

최소 유효 학습량은 DL의 경우 10개 전체 epoch와 3개 이상의 검증 지점이다. 시간 제한
전에 이를 채우지 못하면 성능 순위에 넣지 않는다. CatBoost는 early stopping으로
종료되거나 400 iterations를 완료해야 유효하다. 모델 혼합은 CatBoost 확률에 후보
확률을 각각 `0.10`, `0.25`, `0.50` 가중한 세 비율만 평가한다. 결과를 본 뒤 새 혼합
비율을 추가하지 않는다.

DL 대표는 다음 사전순 기준으로 한 개를 고른다.

1. NaN, OOM 또는 산출물 오류 없이 최소 유효 학습량을 완료했다.
2. 검증 Brier가 가장 낮다.
3. 최고 후보와 `0.0002` 이내면 더 빠르고 안정적인 후보를 고른다.
4. 단독 Brier가 조금 낮아도 CatBoost와 사전 고정 혼합에서 Brier를 `0.0001` 이상
   개선하면 후속 후보로 보존한다.

TabNet은 다음 중 하나를 만족하면 정식 후속 후보로 승급한다.

- 같은 시간 지점에서 DL 최고 모델보다 Brier가 `0.0005` 이내다.
- CatBoost와 사전 고정 혼합에서 Brier를 `0.0001` 이상 개선한다.
- OOV 투수 또는 타자 구간을 `0.0002` 이상 개선하고 전체 Brier 악화가
  `0.00005` 이하다.

최소 유효 학습량을 채우지 못한 모델은 `rejected`가 아니라 `inconclusive`다.

## 전처리 후보

### DL 단일 ablation

기준선은 `dl_standard`다.

1. `selective_yeo_johnson`
2. `pitcher_smoothing_k100`
3. `batter_smoothing_k250`
4. `id_frequency_and_oov`
5. `asof_count_log1p`
6. `grouped_recent_missing`
7. `hand_matchup`

각 job은 기준선에서 정확히 한 요소만 바꾼다. 처음부터 조합을 실행하지 않는다.
`selective_yeo_johnson`의 대상 열, smoothing 수식, ID 빈도·OOV 정책과 결측 그룹은
기존 `eda-informed-preprocessing-design` 계약을 그대로 사용한다. count log는
`asof_pitcher_n`과 `asof_batter_n`의 `log1p`만 추가하고 원본을 유지한다.

### CatBoost 단일 ablation

기준선은 `tree_native`다.

1. `pitcher_smoothing_k100`
2. `batter_smoothing_k250`
3. `id_frequency_and_oov`
4. `hand_matchup`

CatBoost에는 표준화, Yeo-Johnson과 `log1p`를 적용하지 않는다. 그룹 결측 지표도
효과가 작고 CatBoost가 결측을 직접 처리하므로 이번 캠페인에서는 생략한다.

## 다섯 stage

### Stage 1: 실행 보정과 모델 선별

2024 proxy fold에서 CatBoost, TabM과 FT-Transformer를 비교하고 TabNet 승급 시험을
수행한다. 실제 GPU 이름, 수, VRAM, AMP 상태와 처리속도를 기록한다. 두 T4가 독립
worker로 실제 사용되지 않으면 `STAGE_NEEDS_REVIEW`로 중단한다.

### Stage 2: DL 근거 우선 단일 ablation

선정된 DL 대표에서 기준선, selective Yeo-Johnson, 투수 `K=100`, 타자 `K=250`을
2024 proxy fold에 실행한다.

### Stage 3: DL 구조 단일 ablation과 CatBoost 후보

DL 대표에서 ID/OOV, count log1p, grouped recent missing과 hand matchup을 실행한다.
남은 GPU 시간은 CatBoost 단일 ablation에 배정한다. CatBoost job은 다음 stage에서도
해시 기반으로 이어서 실행할 수 있다.

### Stage 4: 조합과 시간 안정성

DL과 CatBoost 각각 승급한 후보를 최대 두 개로 제한한다. 기준선, 최고 단일 후보와
상위 두 후보 조합만 비교한다. DL은 2023 proxy fold에서 방향을 확인하고, CatBoost는
2023·2024 시간 fold와 2024 전체 학습 행 확인을 이 stage에서 마친다. 모든 조합이나
beam search는 실행하지 않는다. CatBoost가 시간 안에 최소 판정 근거를 완료하지 못하면
DL의 마지막 stage를 빼앗지 않고 `inconclusive`로 남긴다.

### Stage 5: 전체 행 최종 확인

2024 fold 전체 학습 행에서 선정된 DL 기준선과 최종 후보만 두 GPU에 하나씩 배정해
짝지어 비교한다. 두 작업은 같은 최대 epoch와 고정 학습 설정을 사용하며, 4,800초 뒤
새 epoch 진입을 막는다. 둘 다 최소 10개 전체 epoch를 완료했으면 공통으로 완료한 epoch
범위 안의 best Brier로 capped-budget 비교를 완료한다. 하나라도 최소량을 채우지 못하면
성능을 억지로 판정하지 않고 DL 전처리를 `inconclusive`로 남긴 뒤 최종 review bundle을
생성한다. 정상적인 시간 상한 도달만으로 여섯 번째 Save Version을 요구하지 않는다.

## 전처리 승급과 최종 판정

단일 후보는 2024 proxy에서 다음 중 하나를 만족할 때만 Stage 4로 승급한다.

- 전체 Brier가 기준선보다 `0.0001` 이상 개선된다.
- 전체 Brier가 개선되고 OOV 투수 또는 타자 구간이 `0.0002` 이상 개선된다.
- 전체 Brier 악화가 `0.00005` 이하이며 CatBoost와 사전 고정 혼합 Brier를
  `0.0001` 이상 개선한다.

DL과 CatBoost에서 각각 최대 두 개만 승급한다. 결과 상태는 다음 네 종류다.

- `recommended`: 이후 모델 실험의 검증된 기본 후보
- `model_specific`: 해당 모델 계열에서만 사용
- `inconclusive`: 추가 seed 또는 fold 필요
- `not_recommended`: 현재 근거에서는 사용하지 않음

`recommended`는 다음을 모두 만족해야 한다.

1. 2023과 2024 proxy에서 Brier 개선 방향이 같다.
2. 두 proxy fold 가중 평균 개선이 최소 `0.0001`이다.
3. 2024 전체 학습 행 비교에서도 Brier가 개선된다.
4. 투수 OOV, 타자 OOV와 game type 어느 구간도 `0.0002` 넘게 악화되지 않는다.
5. 데이터 누출, 행 정렬과 산출물 해시 검사를 모두 통과한다.

행 간 상관과 seed 변동이 있으므로 행 단위 bootstrap p-value만으로 채택하지 않는다.
이 캠페인의 `recommended`도 모든 모델에 영구 확정된 전처리가 아니라, 다음 full OOF와
multi-seed 실험에 기본으로 적용할 후보를 의미한다.

## 로그 계약

표준 출력은 unbuffered로 동작한다. 최소 60초마다 heartbeat를 출력하고 epoch 종료마다
검증 Brier와 best epoch를 출력한다. epoch가 5분보다 길면 batch 진행률을 별도로
출력한다.

필수 이벤트는 다음과 같다.

- `STAGE_START`
- `DATA_READY`
- `GPU_READY`
- `JOB_START`
- `TRAINING_PROGRESS`
- `GPU_STATUS`
- `CHECKPOINT_SAVED`
- `JOB_COMPLETE`
- `HEARTBEAT`
- `STAGE_SUMMARY`

정상 stage 종료는 `STAGE_COMPLETE_READY_FOR_NEXT`, 재개가 필요하면
`STAGE_INCOMPLETE_RESUME_SAME_STAGE`, 자동 판정이 불가능하면
`STAGE_NEEDS_REVIEW`, 실패는 `STAGE_ERROR`로 끝난다. traceback은 원문을 run log에
남긴다.

## 산출물과 사용자 인계

각 Save Version은 두 bundle을 만든다.

### Resume bundle

`preprocessing_stage_XX_resume_bundle.zip`에는 다음 실행에 필요한 checkpoint,
manifest, job 산출물과 stage 상태를 넣는다. 사용자가 다음 Save Version의 Kaggle
Input으로 연결하며 Codex에 전달할 필요는 없다.

### Review bundle

`preprocessing_stage_XX_review_bundle.zip`에는 다음 작은 검토 산출물을 넣는다.

- `stage_summary.json`
- `decision_table.csv`
- `model_metrics.csv`
- `segment_metrics.csv`
- `learning_curves.csv`
- `resource_usage.csv`
- `validation_predictions.csv.gz`
- `artifact_manifest.json`
- `errors.json`
- `run.log`

Stage 5가 완료되면 모든 stage의 검증된 작은 산출물을 합친
`preprocessing_campaign_final_review_bundle.zip`을 만든다. 정상 경로에서는 사용자가
이 파일 하나만 Codex에 전달한다. `STAGE_NEEDS_REVIEW` 또는 `STAGE_ERROR`일 때는 해당
stage review bundle과 첫 오류 traceback을 전달한다.

## 코드 변경 범위

기존 전처리 변환기와 모델 adapter를 재사용하고 다음 범위만 변경한다.

- 시간 제한형 다섯 stage 설정 계약
- GPU별 격리 worker와 부모 전용 manifest 병합
- 결정적 400,000행 temporal proxy 생성
- TabNet 승급 시험 adapter와 필요한 고정 dependency
- stage별 자동 승급·중단 판정
- resume/review/final bundle 생성 및 해시 검증
- 위 동작에 직접 대응하는 합성·fixture 테스트
- 사용자가 복사할 수 있는 하나의 self-contained `KAGGLE_CELL.py`

기존 760개 캠페인을 삭제하거나 결과를 덮어쓰지 않는다. 노트북은 수정하지 않는다.
범용 HPO 프레임워크, 대시보드, 자동 Kaggle dataset 갱신, GitHub 접속, 제출 파일과
제출 패키징은 추가하지 않는다.

## 완료 기준

1. 하나의 Kaggle 셀을 반복 실행하면 이전 bundle에서 다음 미완료 stage를 찾는다.
2. 한 Save Version은 셀 시작부터 6,300초를 넘겨 새 작업을 계속하지 않는다.
3. T4 두 개가 서로 다른 worker에서 사용되고 공통 manifest는 부모만 수정한다.
4. stage 중단 뒤 완료 job을 해시로 확인해 건너뛰고 checkpoint부터 재개한다.
5. temporal proxy와 전체 행 확인 모두 fold 학습 구간 밖의 정보로 fit하지 않는다.
6. 모델 및 전처리 승급은 문서의 고정 기준만 사용한다.
7. 정상 경로에서는 Stage 5 최종 review bundle 하나로 검토할 수 있다.
8. Codex의 작은 합성 테스트가 통과하고 공식 데이터 GPU 학습은 사용자가 수행한다.
9. 제출 파일 또는 제출 패키지는 생성하지 않는다.
