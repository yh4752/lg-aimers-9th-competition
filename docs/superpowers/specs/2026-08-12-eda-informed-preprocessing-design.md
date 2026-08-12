# EDA 근거 전처리 설계

## 목적

`preprocessing_eda_v1`의 전체 시간 전이 결과를 실제 모델 전처리 계약으로 옮긴다.
트리 모델과 DL 모델에 같은 변환을 강제하지 않고, 공통 원본 특징에서 다음 세 전처리
프로필을 명시적으로 구분한다.

- `tree_native`: CatBoost의 범주형 및 결측값 처리를 유지하는 트리 기준선
- `dl_standard`: DL용 중앙값 대치, 표준화, OOV 범주와 선택적 보조 특징
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
- `asof_pitcher_n`, `asof_batter_n`, `pitcher_id_frequency`와
  `batter_id_frequency`가 존재하면 원본과 함께 `log1p` 열을 추가한다.
- 다음 세 결측 지표만 추가한다.
  - `pitcher_recent_missing`: 최근 1, 3, 5경기 성공률 및 middle rate 여섯 열 중 하나라도
    결측이면 1
  - `pitcher_career_missing`: `asof_pitcher_ball_rate`,
    `asof_pitcher_breaking_rate`, `asof_pitcher_fastball_rate`,
    `asof_pitcher_middle_rate`, `asof_pitcher_offspeed_rate`,
    `asof_pitcher_reverse_rate`, `asof_pitcher_strike_rate`,
    `asof_pitcher_success_rate` 중 하나라도 결측이면 1
  - `batter_career_missing`: `asof_batter_success_rate` 또는
    `asof_batter_middle_rate`가 결측이면 1
- 선수 ID 빈도는 학습 구간에서만 계산한다. 미관측 ID의 빈도와 `log1p` 빈도는 0이다.
- 범주형과 ID는 embedding 입력으로 유지하며 미관측 값은 OOV 인덱스 0을 사용한다.

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

다음 파생 특징은 전처리 프로필과 분리된 단일 스위치로 구현한다.

1. `pitcher_success_smooth_100`
2. `batter_success_smooth_250`
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

## Ablation 순서

각 비교는 작은 smoke가 아니라 전체 학습 구간과 전체 다음 시즌 holdout을 사용한다.
동일 모델 설정, seed, fold와 특징 출처를 고정하고 한 번에 한 축만 바꾼다.

1. CatBoost에서는 `tree_native`, 각 DL 계열에서는 `dl_standard`를 해당 계열의
   기준선으로 기록한다. 서로 다른 모델 계열의 절대 점수로 전처리 우열을 판정하지
   않는다.
2. `dl_standard`에 `pitcher_success_smooth_100`만 추가한다.
3. 2번의 채택 여부와 독립적으로 `batter_success_smooth_250`만 추가한다.
4. `hand_matchup`, `count_state`, `pitcher_team_win_expectancy`를 각각 단독 추가한다.
5. `dl_standard`와 `dl_selective_transform`을 같은 열 구성으로 짝지어 비교한다.
6. 단독 개선이 확인된 스위치만 누적 조합한다.
7. 유망 조합을 다섯 시간 전이 fold에서 확인하고, 최고 결과가 `K` 탐색 경계면 해당
   평활화 범위를 확장한다.

첫 DL 전처리 본 실험은 기존 네 계열의 `p3` 용량, `raw_typed`, seed `42`와
`2023→2024` 전체 holdout을 고정한다. 각 계열에서 `dl_standard` 기준선 1개,
다섯 단독 파생 특징 5개와 `dl_selective_transform` 1개를 실행하므로 최초 설정은
총 28개다. `p3`는 각 계열의 240 epoch 중대형 설정이며 smoke로 취급하지 않는다.
단독 결과를 받기 전에는 누적 조합을 `preprocessing_ablation_v1`에 넣지 않는다.
누적 조합과 다섯 fold 확인은 반환된 결과로 유망한 설정을 확인한 뒤 별도 설정으로
고정한다.

단일 fold 또는 단일 seed 실패는 해당 설정만 중단한다. 평균 Brier뿐 아니라 fold별
방향, 최악 fold 변화와 ID OOV 구간을 함께 기록한다. 비용과 실행 시간은 순서를
정하는 정보일 뿐 후보 제거 기준이 아니다.

## 코드 경계

기존 독립 DL 캠페인을 다시 만들지 않고 다음 최소 변경만 수행한다.

- `experiments/independent_dl/preprocessing.py`: 프로필, fold-fit 상태, 선택적 파생 특징
- `experiments/independent_dl/features.py`: 기존 feature view 결과에 전처리 상태를 적용
- `experiments/independent_dl/contracts.py`: 후보에 전처리 프로필과 파생 특징 ID를 기록
- `experiments/independent_dl/configs/preprocessing_ablation_v1.json`: 고정된 비교 순서
- 관련 작은 합성 테스트

기존 `campaign_v1.json`과 `INDEPENDENT_DL_CAMPAIGN.ipynb`는 변경하지 않는다. 기존
후보의 의미와 재개 상태를 보존하기 위해 새 ablation 설정은 별도 campaign ID와 출력
경로를 사용한다. 트리 모델 학습기는 이번 변경에 추가하지 않는다. 현재 저장소에는
기존 점수와 같은 검증 계약으로 재실행할 강한 CatBoost 학습기가 없으므로 임의의 약한
설정을 새로 만들지 않는다. `tree_native`는 향후 강한 CatBoost 기준선이 이식될 때
같은 변환 계약을 재사용할 수 있는 프레임 변환 인터페이스까지만 제공한다.

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
2. 다섯 선택적 파생 특징을 각각 독립적으로 켜고 끌 수 있다.
3. 기존 독립 DL 캠페인과 노트북의 계약은 바뀌지 않는다.
4. 새 ablation 후보의 ID, cache와 metrics가 전처리 차이를 숨기지 않는다.
5. 작은 합성 테스트가 누출 방지, 행 정렬과 변환 수치 계약을 검증한다.
6. 사용자 요청 없이 노트북, 제출 파일 또는 제출 ZIP을 만들지 않는다.
