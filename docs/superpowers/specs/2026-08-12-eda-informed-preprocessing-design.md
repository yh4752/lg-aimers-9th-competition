# EDA 근거 전처리 설계

## 목적

`preprocessing_eda_v1`의 전체 시간 전이 결과를 실제 모델 전처리 계약으로 옮긴다.
트리 모델과 DL 모델에 같은 변환을 강제하지 않고, 공통 원본 특징에서 다음 세 전처리
프로필을 명시적으로 구분한다.

- `tree_native`: CatBoost의 범주형 및 결측값 처리를 유지하는 트리 기준선
- `dl_standard`: DL용 중앙값 대치, 표준화와 OOV 범주를 적용하는 비교 기준선
- `dl_selective_transform`: `dl_standard`에 제한된 Yeo-Johnson 변환만 추가한 비교군

Codex는 코드와 작은 합성 검증만 수행한다. 공식 전체 데이터의 전처리와 모델 학습은
사용자가 Colab에서 실행하고, 결과를 받은 뒤 다음 전처리 채택 여부를 결정한다.

## 근거와 해석 경계

이번 EDA는 1,475,092개 학습 행과 `2019→2020`부터 `2023→2024`까지 다섯 시간
전이 fold를 사용했다. 다음 결과만 구현 근거로 사용한다.

- `asof_pitcher_n`과 `asof_pitcher_pitchmix_n`은 모든 행에서 같다.
- 투수 성공률 평활화의 집계 최적 후보는 `K=100`, 타자 성공률은 `K=250`이다.
- 최근 1, 3, 5경기 투수 비율 여섯 열은 같은 행에서 약 1.98%가 결측이다.
- 투수 누적 비율군과 타자 누적 비율군도 각 그룹 안에서 결측 패턴이 같다.
- 다음 시즌 미관측 비율이 투수 ID는 약 13.8~22.4%, 타자 ID는 약 8.8~11.6%다.
- Yeo-Johnson은 일부 수치 분포의 왜도를 줄였지만 모델 성능 개선은 증명하지 않았다.
- 손잡이 조합, 볼·스트라이크 상태와 투수 팀 승리 기대값은 개별 ablation 후보지만,
  여러 교차 특징을 한 번에 넣을 근거는 없다.
- 인접 시즌 adversarial AUC가 거의 1이므로 무작위 fold 성능은 채택 근거가 될 수 없다.

분포 진단, 단변량 Brier와 실제 모델 Brier는 서로 다른 증거다. 따라서 위 항목은
자동 채택 목록이 아니라 누출 없는 모델 ablation의 우선순위를 정하는 자료로만 쓴다.

## 검토한 접근

### 1. 기존 독립 DL 캠페인의 네 feature view를 전부 교체

코드 양은 적지만 원본 특징 구성과 수치 전처리가 동시에 바뀐다. 성능 차이가 어느
변경에서 왔는지 알 수 없고, 아직 실행하지 않은 64개 캠페인의 계약도 한꺼번에
바뀐다. 채택하지 않는다.

### 2. 전처리 조합을 모든 모델과 feature view에 완전 교차

탐색 폭은 넓지만 첫 비교부터 모델, 특징과 전처리 차원이 섞인다. 결과 해석이 어렵고
EDA에서 근거가 약한 조합까지 대량 실행한다. 초기 단계로는 채택하지 않는다.

### 3. 전처리를 독립 축으로 추가하고 동일 모델에서 짝지어 비교

이 접근을 채택한다. 기존 `raw_typed`, `engineered`, `entity_context`,
`trackman_augmented`는 특징 출처 축으로 유지하고, 전처리 프로필을 별도 축으로 둔다.
첫 본 비교는 같은 fold, 모델 설정, seed와 특징 출처에서 프로필만 바꾼다. 유망한
프로필만 넓은 모델 및 feature view 탐색에 포함한다.

## 전처리 프로필

### 공통 정합성 규칙

- `row_id`와 `control_success`는 입력 특징에서 제외한다.
- fit 상태는 해당 fold의 학습 시즌 행만 사용한다.
- 검증 및 테스트에는 저장된 상태로 `transform`만 수행한다.
- `asof_pitcher_pitchmix_n`은 학습 및 변환 대상의 모든 행에서 `asof_pitcher_n`과
  결측 위치가 같고, 결측이 아닌 값이 정확히 같을 때만 제거한다. 다르면 오류를 내고
  조용히 제거하지 않는다.
- 원본 성공률, 원본 카운트와 원본 ID를 대체하지 않는다. 파생 특징은 새 열로 추가한다.
- 범주 사전에 없는 값은 예약된 OOV 인덱스 `0`을 사용한다.
- 행 순서, 행 개수와 `row_id`는 변환 전후에 같아야 한다.

### `tree_native`

- 수치형 결측값을 대치하거나 표준화하지 않는다.
- 수치형의 `NaN`은 그대로 두고 범주형 결측만 고정 문자열로 바꾼다.
- ID는 정수 크기를 갖는 수치로 사용하지 않고 범주형으로 전달한다.
- 중복 카운트 제거 외의 파생 특징은 ablation 스위치로만 추가한다.
- CatBoost 기준선과의 실제 성능 비교 전에는 어떤 파생 특징도 기본 채택하지 않는다.

### `dl_standard`

- 학습 구간 중앙값으로 수치형 결측을 대치하고, 학습 구간 평균과 모집단 표준편차로
  표준화한다. 중앙값, 평균과 표준편차는 상태에 함께 저장한다.
- 원본 수치형 열은 유지하되 `log1p`와 빈도 특징은 기본으로 추가하지 않는다.
- 범주형과 ID는 embedding 입력으로 유지하며 미관측 값은 OOV 인덱스 0을 사용한다.

다음 보조 특징은 EDA만으로 성능이 증명되지 않았으므로 독립 스위치로 비교한다.

- `asof_count_log1p`: `asof_pitcher_n`과 `asof_batter_n`의 `log1p`
- `entity_frequency_log1p`: 학습 구간에서 계산한 투수 및 타자 ID 빈도와 그 `log1p`.
  미관측 ID의 빈도는 0이다.
- `grouped_missing_indicators`: 다음 세 결측 지표
  - `pitcher_recent_missing`: 최근 1, 3, 5경기 성공률 및 middle rate 여섯 열 중 하나라도
    결측이면 1
  - `pitcher_career_missing`: `asof_pitcher_ball_rate`,
    `asof_pitcher_breaking_rate`, `asof_pitcher_fastball_rate`,
    `asof_pitcher_middle_rate`, `asof_pitcher_offspeed_rate`,
    `asof_pitcher_reverse_rate`, `asof_pitcher_strike_rate`,
    `asof_pitcher_success_rate` 중 하나라도 결측이면 1
  - `batter_career_missing`: `asof_batter_success_rate` 또는
    `asof_batter_middle_rate`가 결측이면 1

### `dl_selective_transform`

`dl_standard`와 동일한 열을 사용하되 다음 열이 존재하고 학습 구간에서 상수가 아닐
때만 Yeo-Johnson을 fit한다.

- `li`
- `run_top_before`, `run_bot_before`, `run_total_before`
- `score_diff_home`, `score_diff_pitcher_team`
- `asof_pitcher_middle_rate`, `asof_batter_middle_rate`

변환 전 중앙값 대치와 변환 후 표준화를 학습 구간 상태로 고정한다. 성공률 전체,
카운트 전체와 이산 상황 열에는 Yeo-Johnson이나 quantile 변환을 일괄 적용하지 않는다.
원본 열을 함께 복제하지 않으므로 두 DL 프로필의 수치 입력 차원은 같다.

## 선택적 파생 특징

다음 파생 특징도 전처리 프로필과 분리된 단일 스위치로 구현한다.

1. `pitcher_success_smooth_K`: `K∈{25, 50, 100, 250}`
2. `batter_success_smooth_K`: `K∈{10, 25, 100, 250, 500, 1000, 2500}`
3. `hand_matchup`
4. `count_state`
5. `pitcher_team_win_expectancy`

평활화 prior는 fold 학습 행의 `control_success` 평균이다. 성공률 또는 카운트가
유효하지 않거나 카운트가 0 이하면 prior를 사용한다. 원본 성공률과 카운트는 유지한다.
`hand_matchup`은 투수와 타자 손잡이의 범주형 조합이다. `count_state`는 볼과
스트라이크의 범주형 조합이다. `pitcher_team_win_expectancy`는 초·말과 홈·원정 관계로
투수 팀 관점의 승리 기대값을 선택한다.

기존 `engineered`가 여러 파생 특징을 한꺼번에 추가하더라도 이번 연구에서는 위 스위치를
개별적으로 켠 결과만 전처리 근거로 인정한다. `game_type×count_state`, 투수×타자 ID와
고차 교차는 이번 범위에서 제외한다.

## 성능 실험 규모

작은 최신 시즌 선별을 먼저 하고 나중에 안정성을 확인하는 방식은 사용하지 않는다.
시즌 이동이 매우 크므로 첫 본 실험부터 다섯 시간 전이 fold를 모두 사용한다.

### 파동 A: DL 단독 효과 전수 비교

기존 독립 DL 캠페인의 4개 모델 계열과 중대형 p3, p4를 사용해 8개 기준점을 만든다.
p3와 p4는 TabM/TabMmini, MLP/ResNet을 모두 포함하고 FT-Transformer와 TabR의
서로 다른 용량도 포함한다. 각 기준점은 동일한 모델 설정, seed와 fold에서 전처리만
바꾼다. 작은 p1, p2는 전처리 채택 근거를 반복하는 데 쓰지 않고 최선 결과가 모델
용량의 하한에 있을 때 경계 확장 후보로 사용한다.

- 모델 기준점: `4개 계열 × 2개 중대형 용량 = 8개`
- fold: `2019→2020`, `2020→2021`, `2021→2022`, `2022→2023`,
  `2023→2024`
- seed: `42`
- 특징 출처: 전처리 효과를 분리하기 위한 `raw_typed`
- 학습 행: 각 fold의 전체 과거 시즌 행
- epoch: p3와 p4의 기존 `240, 400`을 그대로 사용

각 기준점에서 다음 19개 설정을 실행한다.

1. `dl_standard`
2. `dl_selective_transform`
3. `asof_count_log1p`
4. `entity_frequency_log1p`
5. `grouped_missing_indicators`
6. `hand_matchup`
7. `count_state`
8. `pitcher_team_win_expectancy`
9. 투수 평활화 `K=25`
10. 투수 평활화 `K=50`
11. 투수 평활화 `K=100`
12. 투수 평활화 `K=250`
13. 타자 평활화 `K=10`
14. 타자 평활화 `K=25`
15. 타자 평활화 `K=100`
16. 타자 평활화 `K=250`
17. 타자 평활화 `K=500`
18. 타자 평활화 `K=1000`
19. 타자 평활화 `K=2500`

따라서 파동 A는 `8 × 19 × 5 = 760회`의 전체 fold 학습이다. 이는 smoke가
아니며 모델 계열, 구조, 용량, 시간 이동과 평활화 범위를 동시에 충분히 넓게 보는
본 실험이다. 실행기는 후보와 fold 단위로 재개하며 완료된 작업을 반복하지 않는다.

### 파동 B: seed 확인

파동 A에서 다음 중 하나를 만족하는 설정은 탈락시키지 않고 seed `2026`, `3407`을
추가한다.

- 한 기준점에서 5개 fold 가중 평균 Brier가 기준선보다 낮고 최소 3개 fold가 개선됨
- 각 기준점의 상위 2개 설정
- 각 모델 계열에서 가장 좋은 설정
- 투수 OOV, 타자 OOV 또는 `game_type` 구간 중 하나에서 최소 3개 fold의 방향이
  개선되고, 해당 구간의 다섯 fold 가중 평균 Brier가 기준선보다 낮음

파동 A에서 조건을 만족하지 못한 설정은 `rejected`가 아니라 `inconclusive`로 남긴다.
단일 설정의 결과로 전처리 종류나 모델 계열을 닫지 않는다.

### 파동 C: 조합 효과

단일 효과만으로는 상쇄 또는 상승작용을 놓칠 수 있다. 다음 9개 구성요소를 대상으로
모든 2개 조합을 생성한다.

- 해당 기준점에서 가장 좋은 투수 평활화 `K`
- 해당 기준점에서 가장 좋은 타자 평활화 `K`
- `dl_selective_transform`
- `asof_count_log1p`
- `entity_frequency_log1p`
- `grouped_missing_indicators`
- `hand_matchup`
- `count_state`
- `pitcher_team_win_expectancy`

2개 조합은 기준점마다 최대 36개다. p3와 p4의 8개 중대형 기준점, 다섯 fold와 seed
`42`에서 모두 실행하므로 최대 `36 × 8 × 5 = 1,440회`다. 단일 효과가 약하다는
이유만으로 조합에서 제외하지 않는다.

2개 조합 뒤에는 기준점별 beam search를 수행한다. 다섯 fold 가중 평균 Brier,
개선 fold 수, 최악 fold와 OOV 악화를 사전순으로 정렬해 상위 8개 조합을 유지한다.
각 조합에 아직 없는 구성요소를 하나씩 추가하고 중복을 제거하는 과정을 최대 5개
구성요소까지 반복한다. 각 깊이의 상위 8개가 모두 이전 깊이보다 나쁘면 그 기준점의
확장을 멈춘다. 이렇게 3개 이상 조합도 탐색하되 `2^9` 완전 탐색을 무조건 실행하지
않는다. 이후 각 모델 계열의 상위 조합과 단일 후보를 3개 seed로 확인한다.

### 파동 D: 특징 출처 상호작용

원본 특징에서 살아남은 단일 및 조합 프로필을 다음 두 조건에서 다시 평가한다.

- `raw_plus_trackman`: 원본에 fold cutoff를 지킨 Trackman 특징만 추가
- 독립 DL 본 캠페인에서 해당 모델 계열의 최상위 비원본 feature view

기존 `trackman_augmented`는 여러 engineered 및 context 특징도 함께 넣으므로
전처리 단독 효과 확인에는 사용하지 않는다. `raw_plus_trackman`을 별도로 만들어
Trackman 추가 효과와 전처리 효과를 분리한다. 파동 D는 다섯 fold와 3개 seed를
사용한다.

### 파동 E: 강한 CatBoost 전수 비교

트리 모델은 DL과 별도로 검증한다. 로컬 원본 저장소의 검증된 commit
`9454d68b93971627e3d3f613ce30be690cb5dce2`에서 CatBoost 특징 계약과 다음 네
구조를 출처와 함께 이식한다.

- `champion`: depth 7, 400 iterations
- `depth5`: depth 5, 600 iterations
- `depth8`: depth 8, 300 iterations
- `lr003`: learning rate 0.03, 700 iterations

각 구조에서 `tree_native`, ID 빈도, 그룹 결측 지표, 11개 평활화 설정과 세 의미
특징을 비교한다. 총 17개 설정, 다섯 fold와 seed `42`, `2026`, `3407`을 사용하므로
`4 × 17 × 5 × 3 = 1,020회`다. CatBoost에서는 순서만 바꾸는 `log1p` 카운트가
단일 수치 분할의 표현력을 늘리지 않으므로 별도 후보로 중복 실행하지 않는다.

## 판정 계약

모든 비교는 같은 모델 기준점, fold와 seed의 `tree_native` 또는 `dl_standard`를
짝으로 사용한다. 서로 다른 모델 계열이나 검증 프로토콜의 절대 Brier를 전처리 효과로
비교하지 않는다.

전역 기본값으로 채택하려면 다음을 모두 만족해야 한다.

- 다섯 fold 가중 평균 Brier 개선
- 다섯 fold 중 최소 네 fold의 seed 평균 개선
- 세 seed 중 최소 두 seed의 전체 평균 개선
- 최악 fold Brier 악화가 `0.0001` 이하
- 투수 및 타자 OOV 구간에서 각각 Brier 악화가 `0.0002` 이하

전역 기준을 통과하지 않아도 특정 모델 계열 또는 용량에서 같은 조건을 만족하면 해당
범위의 전처리로 채택할 수 있다. 평균은 개선하지만 방향 조건을 통과하지 못하면
`inconclusive`로 남긴다. 최고 `K`가 현재 범위의 25 또는 2500이면 해당 방향으로
다음 범위를 확장한다.

평균 Brier, fold별 delta, seed 평균과 표준편차, 최악 fold, `game_type`, 투수 OOV,
타자 OOV와 두 ID가 모두 OOV인 구간을 기록한다. 행 단위 표본 수가 많다는 이유로
훈련 seed와 시즌 이동의 불확실성을 무시하지 않으며, 행 독립 가정의 유의확률을 채택
근거로 사용하지 않는다.

단일 fold, seed, 설정, OOM 또는 런타임 종료는 해당 작업만 막는다. 비용과 실행 시간은
실행 순서와 자원 안내에만 사용하고 후보 제거 기준으로 사용하지 않는다.

## 실행 순서와 자원 추정

파동 A 안에서는 p3의 네 모델 계열을 먼저 순환하고 p4를 실행한다. 각 기준점은
`dl_standard`의 다섯 fold를 먼저 완료한 뒤 18개 설정을 같은 fold 순서로 실행한다.
따라서 장시간 캠페인이 끝나기 전에도 완결된 짝 비교가 순차적으로 쌓인다. 파동 E는
DL과 독립적으로 실행할 수 있지만 한 Colab 런타임에서는 GPU 메모리 경합을 막기 위해
동시에 학습하지 않는다.

사전에 근거 없는 고정 시간을 약속하지 않는다. 첫 기준점의 기준선 다섯 fold가 끝나면
실측 wall time, 최대 GPU 메모리와 peak RAM을 기록하고, 완료된 동일 계열 작업의
중앙값으로 남은 실행 시간을 갱신한다. 실행 요약에는 전체 예상 GPU-hours와 현재
처리율을 표시한다. 예상 시간이 길어도 설정을 삭제하지 않으며 Drive checkpoint와
원자적 manifest로 Colab 런타임 초기화 뒤 이어서 실행한다.

## 산출물

기존 캠페인과 구분된 Drive 경로를 사용한다.

```text
outputs/preprocessing_campaign_v1/
├── campaign_manifest.json
├── resolved_candidates.json
├── fold_metrics.csv
├── segment_metrics.csv
├── paired_deltas.csv
├── resource_usage.csv
├── promotion_decisions.json
├── feature_cache/
├── checkpoints/
├── predictions/
└── campaign_summary.json
```

`resolved_candidates.json`에는 모델 구조, 용량, fold, seed, 전처리 프로필,
구성요소와 smoothing `K`를 모두 기록한다. `promotion_decisions.json`에는 파동 B~D의
각 승격 규칙 boolean과 입력 산출물 SHA-256을 남긴다. 예측은 `row_id`, fold, season,
target, probability, model anchor, seed와 preprocessing ID를 포함한다. 대용량 예측,
checkpoint와 cache는 Git에 커밋하지 않는다.

## 코드 경계

기존 독립 DL 캠페인을 다시 만들지 않고 다음 최소 변경만 수행한다.

- `experiments/independent_dl/preprocessing.py`: 프로필, fold-fit 상태, 선택적 파생 특징
- `experiments/independent_dl/features.py`: 기존 feature view 결과에 전처리 상태를 적용
- `experiments/independent_dl/contracts.py`: 후보에 전처리 프로필과 파생 특징 ID를 기록
- `experiments/independent_dl/configs/preprocessing_ablation_v1.json`: 파동 A의 고정
  후보와 실행 순서
- `experiments/independent_dl/preprocessing_campaign.py`: 파동 B~D의 사전 고정
  승격 규칙, 조합 생성과 지표 집계
- `experiments/catboost_preprocessing/`: 검증된 CatBoost 계약과 파동 E 실행 코드
- 관련 작은 합성 테스트

기존 `campaign_v1.json`과 `INDEPENDENT_DL_CAMPAIGN.ipynb`는 변경하지 않는다. 기존
후보의 의미와 재개 상태를 보존하기 위해 새 ablation 설정은 별도 campaign ID와 출력
경로를 사용한다. CatBoost는 임의의 새 설정을 만들지 않고 원본 저장소의 검증된 특징
계약과 네 구조만 고정 출처에서 이식한다. 기존 CatBoost 점수와 이번 다섯 fold 점수는
검증 프로토콜이 다르므로 하나의 순위로 비교하지 않는다.

## 오류 처리와 산출물 식별

- 필수 열이 없으면 어떤 스위치가 어떤 열을 요구하는지 포함해 오류를 낸다.
- 중복 카운트의 동일성 검사가 실패하면 두 열을 모두 남기는 대신 실행을 중단한다.
- Yeo-Johnson fit 실패, 비유한 상태 또는 변환 후 비유한 값은 해당 후보의 실패로
  기록하고 다른 후보를 막지 않는다.
- cache identity에는 feature view, 전처리 프로필, 파생 특징 ID, cutoff, 행 해시와
  전처리 코드 해시를 포함한다.
- 후보 ID와 metrics에는 전처리 프로필과 파생 특징 ID를 기록한다.

## 작은 검증 범위

합성 테스트로 다음만 확인한다.

- 중앙값 대치와 변환 상태가 검증 행에서 다시 fit되지 않는다.
- OOV ID가 0으로 인코딩된다.
- 세 결측 지표가 정의된 그룹과 일치한다.
- 두 평활화 특징이 학습 prior와 지정된 `K`를 사용한다.
- `tree_native`는 수치형 `NaN`을 보존한다.
- 선택된 열에만 Yeo-Johnson이 적용되고 두 DL 프로필의 열 순서와 차원이 같다.
- 동일한 전처리 계약만 cache를 재사용한다.

공식 데이터 전처리, CPU 및 GPU 학습, 실제 Brier 비교와 장시간 Colab 실행은 하지
않는다.

## 완료 기준

1. 세 전처리 프로필이 fold-fit 상태와 함께 재현된다.
2. DL 기준선과 18개 선택적 설정, 17개 CatBoost 설정을 각각 독립적으로 실행할 수 있다.
3. 기존 독립 DL 캠페인과 노트북의 계약은 바뀌지 않는다.
4. 새 ablation 후보의 ID, cache와 metrics가 전처리 차이를 숨기지 않는다.
5. 작은 합성 테스트가 누출 방지, 행 정렬과 변환 수치 계약을 검증한다.
6. 파동 A~E가 후보 및 fold 단위로 재개되고 한 후보의 실패가 다른 후보를 막지 않는다.
7. 사용자 요청 없이 노트북, 제출 파일 또는 제출 ZIP을 만들지 않는다.
