# 전처리 연구용 EDA 설계

## 목적

`train.csv`를 사용해 ML과 DL의 전처리 후보를 넓게 발굴한다. 단순 분포 요약에
그치지 않고, 각 후보가 다음 시즌에서도 유지되는지 누출 없는 시간 전이 진단으로
확인한다.

EDA는 피처를 자동 채택하거나 제거하지 않는다. 결과는 후속 CatBoost, XGBoost,
TabM과 DeepFM 계열의 개별 ablation 후보를 정하는 근거로만 사용한다. 최종 채택은
동일한 시간 전이 검증에서 모델 계열별로 판단한다.

## 실행 소유권과 전달 형식

- Codex는 순서대로 복사해 실행할 수 있는 Colab 셀과 작은 합성 테스트를 작성한다.
- 사용자는 공식 전체 데이터로 CPU 장시간 실행을 수행하고 결과 파일 또는 오류
  메시지를 Codex에 전달한다.
- `.ipynb`와 실행 완료 산출물은 저장소에 추가하지 않는다.
- 구현물은 `docs/PREPROCESSING_EDA_COLAB.md` 한 파일에 실행 순서, 완결된 코드 셀,
  예상 시간, 산출물, 재실행 안전성과 반환할 메시지를 함께 둔다.
- 공식 데이터, 추출된 CSV, 그래프와 EDA 결과는 Git에 커밋하지 않는다.

예상 전체 실행 시간은 일반적인 Colab CPU 런타임에서 약 30~120분이다. 데이터
읽기 속도, 사용 가능한 코어 수와 adversarial validation 표본 크기에 따라 달라질 수
있다. CPU 코어는 가능한 범위에서 모두 사용하되, seed와 표본 인덱스를 고정한다.

## 입력 경계

### 사용 데이터

- 공식 배포 ZIP 안의 `train.csv`
- 공식 배포 ZIP 안의 `test.csv`

Trackman, sample submission, 기존 OOF와 모델 예측은 이번 EDA에 사용하지 않는다.
ZIP은 읽기 전용으로 열고 필요한 두 CSV만 Colab 런타임의 임시 작업 공간에서
읽는다. 구현 코드에는 개인 Google Drive 경로를 고정하지 않는다. Drive를 마운트한
뒤 `official_open.zip` 후보를 검색하고, 후보가 0개 또는 2개 이상이면 임의 선택하지
않고 사용자에게 정확한 경로를 입력하도록 중단한다.

### `test.csv` 사용 제한

현재 배포 ZIP의 `test.csv`는 5행 스키마 샘플이다. 다음 작업에만 사용한다.

- 입력 컬럼 이름과 순서 확인
- 타깃 부재 확인
- 기본 dtype 호환성 확인
- 전처리 함수가 같은 스키마를 받아들이는지 확인

다음 작업에는 절대 사용하지 않는다.

- train-test drift 계산
- 결측률, 범주 빈도 또는 수치 분포 추정
- 인코더, 변환기와 imputer 적합
- 피처 선택, threshold 또는 smoothing 상수 선택

실제 test 행 수가 5가 아니더라도 기본 정책은 바뀌지 않는다. 이번 EDA의 모든
통계 적합과 후보 선정은 `train.csv` 내부 시간 전이만 사용한다.

## 고정 시간 전이 규칙

시즌을 오름차순으로 정렬한다. 첫 시즌을 제외한 각 시즌을 검증 시즌으로 삼고,
그보다 과거인 모든 시즌을 학습 구간으로 사용하는 expanding-window fold를 만든다.
예를 들어 데이터가 2021~2024 시즌이면 다음 세 fold를 사용한다.

1. 2021 → 2022
2. 2021~2022 → 2023
3. 2021~2023 → 2024

모든 bin 경계, 범주 통계, 결측 대체값, scaling과 모델 상태는 해당 fold의 과거
학습 구간에서만 적합한다. 검증 시즌에는 고정된 상태만 적용한다. 같은 검증 시즌의
다른 행, 행 순서와 배치 크기를 피처 생성에 사용하지 않는다.

각 probe는 검증 시즌별 결과와 검증 행 수 가중 전체 결과를 모두 기록한다. 기준
예측은 해당 fold 학습 구간의 `control_success` 평균이며, 비교 지표는 Brier Score
차이 `probe_brier - prior_brier`다. 음수이면 단변량 probe가 기준 예측보다 좋다.
이 차이는 피처 제거 기준이 아니라 후속 ablation 우선순위다.

### 고정 기본 설정

첫 구현의 기본값은 다음과 같다. 사용자는 실행 전 설정 셀에서 값을 볼 수 있지만,
한 실행 도중에는 변경하지 않는다. 변경한 실행은 다른 `RUN_ID`로 기록한다.

| 항목 | 기본값 |
|---|---|
| seed | `20260812` |
| 수치 단변량 bin 수 | 최대 `20`개 학습 분위수 bin |
| 단변량 범주 통계 smoothing K | `50` |
| 희귀 범주 기준 | 학습 구간 빈도 `10` 이하 |
| adversarial 표본 상한 | 시즌별 `200,000`행 |
| adversarial holdout | stratified `20%` |
| 판별 모델 | `HistGradientBoostingClassifier`, 300 iteration, 31 leaves |
| importance 표본 상한 | holdout `50,000`행 |
| permutation 반복 | `5`회 |
| 상관 군집 기준 | Spearman 절댓값 `0.95` 이상인 연결 성분 |
| smoothing K grid | `0, 5, 10, 25, 50, 100, 125, 250, 500, 1000, 2500, 5000, 10000` |
| quantile 수 | 학습 고유값 수와 `1000` 중 작은 값 |
| QuantileTransformer 표본 상한 | 학습 `200,000`행 |

피처의 의미 타입은 CSV dtype만으로 추론하지 않는다. ID, 손잡이, 팀, 경기 유형,
초·말, 주자 상태, 월·요일, 볼·스트라이크·아웃과 주자 indicator는 범주형으로
취급한다. 이들을 제외한 수치 컬럼만 DL 수치 변환 probe의 대상이 된다.

## 분석 블록

### 1. 무결성 및 의미 검사

전체 분석 전에 다음 항목을 확인한다.

- ZIP과 내부 CSV의 SHA-256, 행 수, 컬럼 순서와 dtype
- `row_id` 결측과 중복
- `control_success`의 결측 여부와 `{0, 1}` 이외 값
- 시즌별 행 수와 타깃률
- `balls_before` 0~3, `strikes_before` 0~2, `outs_before` 0~2
- `game_month` 1~12, `game_dayofweek` 0~6
- 주자 indicator가 0 또는 1인지 확인
- `num_runners_on`과 세 주자 indicator 합의 일치
- `base_state`와 세 주자 indicator의 일치
- `run_total_before = run_top_before + run_bot_before`
- `score_diff_home = run_bot_before - run_top_before`
- 초에는 투수 팀이 홈 팀, 말에는 원정 팀이라는 규칙에 따른
  `score_diff_pitcher_team`의 부호 관계
- `home_win_expectancy`와 `away_win_expectancy`의 0~100 범위 및 허용 오차 안의
  합계 100
- 누적 표본 수가 음수가 아닌지 확인
- `asof_*_rate` 계열의 0~1 범위
- 표본 수가 0일 때 관련 rate의 결측 및 비결측 패턴
- `asof_pitcher_n`과 `asof_pitcher_pitchmix_n`의 완전 중복 여부
- 상수 컬럼, 완전 중복 컬럼과 결정적으로 파생 가능한 컬럼

검사 결과는 `pass`, `warn`, `fail`로 구분한다. 스키마, 타깃, ID 또는 시간 전이
실행을 무효화하는 항목은 `fail`로 처리하고 이후 분석을 중단한다. 데이터 설명과
다르지만 안전하게 분석할 수 있는 의미 관계 위반은 `warn`으로 기록하고 계속한다.
값을 자동 수정하지 않는다.

### 2. 전체 피처 시간 진단

#### 기본 프로파일

모든 원본 입력 피처에 대해 전체 및 시즌별로 dtype, 결측률, 고유값 수, 상수 여부,
최빈값 비율을 기록한다. 수치형에는 평균, 표준편차, 최소·최대, 주요 분위수, 왜도,
0의 비율과 극단값 비율을 추가한다. 범주형에는 singleton 비율과 희귀 범주 비율을
추가한다.

#### 인접 시즌 분포 이동

모든 인접 시즌 쌍을 비교한다.

- 연속형: KS statistic, Wasserstein distance와 학습 시즌 IQR로 정규화한 거리
- 범주형·ID: total variation, Jensen-Shannon distance와 다음 시즌 미등록률
- 공통: 결측률 차이, 고유값 범위 밖 비율과 가장 크게 이동한 시즌 쌍

약 147만 행에서는 작은 차이도 유의할 수 있으므로 p-value를 순위 근거로 사용하지
않는다. effect size와 실제 이동량을 저장한다.

#### 누출 없는 단변량 Brier probe

`row_id`, `season`, `control_success`를 제외한 모든 입력 피처를 평가한다.

- 수치형: 학습 구간 분위수로 최대 20개 bin을 만들고, bin별 타깃률을 학습 구간
  전체 prior로 smoothing한 뒤 다음 시즌에 적용한다.
- 범주형·ID: 학습 구간 범주별 타깃률을 같은 prior로 smoothing하고, 미등록 범주는
  prior로 대체한다.
- 모든 예측 확률은 수치 안정성을 위해 `[1e-6, 1 - 1e-6]`로 제한한다.
- 결과에는 fold, 피처, 타입, 검증 행 수, prior Brier, probe Brier, 차이, coverage,
  미등록률과 방향 일치 여부를 기록한다.

공통 smoothing 강도와 bin 수는 결과 manifest에 저장한다. EDA 실행 후 그 값을
성능에 맞춰 사후 변경하지 않는다. 다른 값을 비교하려면 별도 실험 ID로 실행한다.

#### 선택적 결측 indicator probe

결측이 한 건이라도 있는 각 피처의 `is_missing`을 독립적으로 평가한다. 전체와
시즌별 결측률, 결측·비결측 타깃률, 시간 전이 Brier 차이를 기록한다. 결과는 후속
모델에서 `native missing`, `selective indicator`, `all indicators`를 비교할 후보를
만들 뿐, 모든 indicator를 자동 추가하지 않는다.

### 3. 다변량 고비용 진단

#### 인접 시즌 adversarial validation

각 인접 시즌 쌍에서 시즌별 같은 수의 행을 고정 seed로 표본 추출하고, 어느 시즌의
행인지 판별하는 CPU 모델을 학습한다. 전체 데이터 분포를 훼손하지 않도록 표본
상한, 실제 선택 인덱스의 SHA-256과 클래스별 행 수를 manifest에 기록한다.
표본은 정렬된 `row_id`와 seed에서 결정해 원본 파일의 현재 행 순서에 의존하지
않는다.

- 범주형과 ID는 판별 모델의 학습 부분에서 적합한 빈도 인코딩을 사용한다.
- stratified holdout에서 ROC-AUC와 log loss를 계산한다.
- holdout을 재사용한 모델 선택은 하지 않는다.
- AUC가 높다는 사실만으로 피처를 제거하지 않는다. 이동의 존재와 주도 피처를
  후속 실험 후보로 기록한다.

#### 상관 군집과 그룹 중요도

수치형 간 Spearman 상관으로 군집을 만든다. 강하게 상관된 피처에서 개별
permutation importance가 분산되는 문제를 줄이기 위해 다음 두 결과를 함께 낸다.

- 개별 피처 permutation importance
- 상관 군집의 모든 피처를 함께 섞는 그룹 permutation importance

importance는 고정된 holdout 일부와 반복 횟수를 사용한다. 결과는 시즌 판별에
기여한 피처를 설명하는 용도이며, 타깃 예측 피처 중요도로 해석하지 않는다.

### 4. 전처리 특화 진단

#### `asof_*` 신뢰도와 smoothing

`control_success`와 같은 사건의 확률을 나타내는 success rate 계열만 시즌 및 관련
표본 수 구간별 행 수, 평균 rate, 실제 타깃률, Brier와 calibration gap을 계산한다.
구종, reverse, middle, ball과 strike rate는 정답 확률로 해석하지 않고 결측·범위·
시즌 이동 진단만 수행한다. 관련 표본 수가 0인 행과 cold-start 행을 별도 구간으로
둔다.

정확한 누적 분모가 있는 `asof_pitcher_success_rate`와
`asof_batter_success_rate`에만 고정 기본 설정의 K grid를 적용한다. 각각
`asof_pitcher_n`과 `asof_batter_n`을 사용한다. 최근 1·3·5경기 success rate는
정확한 경기별 분모가 없으므로 K-grid smoothing에서 제외하고 표본 수 구간별
신뢰도만 본다. 각 K는 과거 시즌에서 정한 prior로 smoothing한 뒤 다음 시즌
Brier로 평가한다. fold별 결과와 전체 결과를 모두 남기며, 최적 K가 탐색 경계에
있으면 후속 실험에서 범위를 확장할 후보로 표시한다. EDA 자체가 최종 K를 모델
공통값으로 확정하지 않는다.

#### DL 수치 변환 진단

각 수치 피처에 대해 고유값 수, 분위수 중복, 0의 비율, 꼬리 비율, 다음 시즌의
학습 범위 초과율을 계산한다. 다음 변환을 fold 학습 구간에서만 적합하고, 검증
시즌에 적용한 뒤 변환별 분포 안정성을 비교한다.

- 원본값과 학습 median 대체
- StandardScaler
- RobustScaler
- QuantileTransformer의 normal 출력
- Yeo-Johnson PowerTransformer

변환기와 median은 fold 학습 구간에서만 적합한다. QuantileTransformer의
subsample과 quantile 수는 manifest에 고정한다. 결과에는 변환 전후의 finite 비율,
왜도, 중앙 절대 편차, 1%·99% 분위수, 검증 시즌의 변환 범위 초과율을 기록한다.
StandardScaler와 RobustScaler는 선형 변환이므로 별도의 단변량 성능 비교를 하지
않는다. 이 결과는 TabM 등 DL의 수치 표현 ablation 후보를 정하는 자료다.
piecewise-linear 및 periodic embedding은 이번 EDA에서 학습하지 않고,
고유값·분위수 안정성을 바탕으로 후속 후보 여부만 표시한다.

#### 범주형·ID 시간 coverage

`pitcher_id`, `batter_id`, `pitcher_team_id`, `batter_team_id`와 저 cardinality
범주형을 구분한다. fold별로 다음 값을 계산한다.

- 학습 vocabulary 크기
- singleton 및 희귀 범주 비율
- 다음 시즌 미등록률
- 빈도 decile별 행 수, 타깃률과 Brier
- 등록 행과 cold-start 행의 prior Brier 및 probe Brier

이 결과는 CatBoost native category, TabM category encoding과 DeepFM field
embedding 후보를 나누는 근거로 사용한다. ID를 연속 수치로 간주하지 않는다.

#### 제한된 의미 기반 상호작용

전수 피처 쌍을 만들지 않고 다음처럼 경기 의미가 명확한 소수 후보만 단변량 시간
전이 probe로 평가한다.

- `balls_before` × `strikes_before`의 count state
- `pitcher_hand` × `batter_hand`의 hand matchup
- `base_state` × `outs_before`
- `game_type` × count state
- `score_diff_pitcher_team`의 절댓값, 리드·동점·열세 상태
- 투수 팀 관점의 기대 승률

각 파생 규칙, 필요한 원본 컬럼과 결측 처리 방식을 결과에 기록한다. 상호작용이
누락된 컬럼 때문에 계산 불가능하면 조용히 생략하지 않고 `warn`으로 남긴다.

## Colab 셀 순서

구현 문서는 다음 순서를 고정한다. 각 셀은 앞선 셀의 성공 상태를 확인하고, 필요한
변수가 없으면 사람이 이해할 수 있는 오류로 중단한다.

1. 목적, 자원, 예상 시간과 반환할 메시지 안내
2. 라이브러리 import, seed와 분석 설정 고정
3. Google Drive 마운트, ZIP 탐색 또는 명시 경로 검증
4. ZIP 해시·멤버 확인과 `train.csv`, `test.csv` 로드
5. 스키마 및 무결성 검사
6. 기본·시즌 프로파일과 의미 관계 검사
7. 분포 이동 및 전체 피처 단변량 probe
8. 결측 indicator와 ID coverage probe
9. adversarial validation과 그룹 중요도
10. `asof_*` reliability 및 smoothing grid
11. DL 수치 변환 진단과 의미 기반 상호작용 probe
12. 핵심 그래프 생성, 산출물 검증 및 원자적 게시

한 셀이 실패하면 임시 실행 디렉터리를 최종 결과 디렉터리로 승격하지 않는다.
재실행은 동일한 `RUN_ID` 임시 디렉터리를 새로 계산하고, 완성된 동일 이름 결과를
덮어쓰지 않는다. 기존 완성 결과가 있으면 새 `RUN_ID`를 요구한다.

## 산출물 계약

최종 결과 디렉터리에는 다음 구조화 파일을 생성한다.

| 파일 | 내용 |
|---|---|
| `eda_summary.json` | 실행 상태, 핵심 경고, 상위 진단과 후속 실험 후보 |
| `run_manifest.json` | 해시, seed, 설정, fold, 표본 인덱스 해시, 라이브러리 버전 |
| `integrity_checks.csv` | 스키마·범위·의미 관계 검사 결과 |
| `feature_profile.csv` | 전체 및 시즌별 피처 프로파일 |
| `season_drift.csv` | 인접 시즌별 단변량 분포 이동 |
| `temporal_univariate_probes.csv` | 모든 원본 피처의 시간 전이 Brier probe |
| `missingness_probes.csv` | 결측 패턴과 indicator probe |
| `id_coverage.csv` | 범주형·ID의 미등록 및 cold-start 진단 |
| `adversarial_validation.csv` | 인접 시즌 판별 성능 |
| `adversarial_importance.csv` | 개별·상관 군집 permutation importance |
| `correlation_clusters.csv` | 수치형 상관 군집 정의 |
| `asof_reliability.csv` | rate별 표본 수 구간 reliability |
| `asof_smoothing_grid.csv` | fold·K별 smoothing 결과 |
| `numeric_transform_diagnostics.csv` | 수치 변환별 분포 안정성 결과 |
| `interaction_probes.csv` | 의미 기반 상호작용 결과 |

`plots/`에는 8~12개의 핵심 PNG만 저장한다. 최소 그래프는 시즌별 행 수·타깃률,
결측률, drift 상위 피처, 단변량 probe 상위 피처, adversarial AUC·중요도, ID
미등록률, `asof_*` reliability, smoothing K와 수치 변환 비교다. 컬럼마다 그래프를
하나씩 만들거나 HTML 보고서를 생성하지 않는다.

모든 CSV는 컬럼 순서와 정렬 키를 고정한다. JSON은 중복 키, `NaN`과 `Infinity`를
허용하지 않는다. 요약 JSON의 후속 후보에는 반드시 `근거 파일`, `적용 모델 계열`,
`후속 ablation`과 `자동 채택 아님`을 기록한다.

## 성공 및 오류 계약

성공하려면 다음 조건을 모두 만족해야 한다.

- 예정된 JSON과 CSV가 모두 존재하고 비어 있지 않다.
- 모든 Brier, AUC, 거리와 중요도 값이 finite이거나 명시적 `not_applicable`이다.
- fold의 학습 시즌 최댓값이 검증 시즌보다 작다.
- fit 관련 감사 로그에 test 행이 0개다.
- `test.csv`가 drift, 통계 적합 또는 후보 선정에 사용되지 않았다.
- 동일 입력·설정·seed의 재실행에서 파일 정렬과 구조화 수치가 허용 오차 안에서
  같다.
- 임시 결과 검증 후에만 최종 디렉터리를 게시한다.

최종 셀은 성공 시 아래 형식으로 출력한다.

```text
EDA_SUCCESS run_id=<RUN_ID> output_dir=<OUTPUT_DIR> summary=<SUMMARY_PATH>
```

오류는 아래 형식으로 출력하고 원래 예외를 다시 발생시킨다.

```text
EDA_ERROR stage=<STAGE> type=<ERROR_TYPE> message=<MESSAGE>
```

사용자는 성공 시 `eda_summary.json`, `run_manifest.json`과 상세 CSV 묶음 또는 해당
경로를 전달한다. 실패 시 `EDA_ERROR` 한 줄과 전체 traceback을 전달한다.

## 작은 테스트 범위

Codex는 공식 전체 데이터를 실행하지 않는다. 구현 단계에서 작은 합성 데이터로
다음 계약만 검사한다.

- expanding-window fold가 미래 시즌을 학습에 넣지 않는다.
- 변환기, bin과 범주 통계가 학습 행에만 적합된다.
- 미등록 범주와 결측값이 고정 prior 또는 학습 median으로 처리된다.
- test 스키마 샘플이 어떤 fit 함수에도 전달되지 않는다.
- 무결성 위반이 `warn`과 `fail`로 정확히 분리된다.
- JSON에 비유한값이 없고 CSV 정렬이 결정적이다.
- 성공 전에는 임시 결과가 최종 결과로 게시되지 않는다.

## 명시적 제외

- 약 1,100개에 이르는 모든 피처 쌍의 전수 탐색
- 5행 `test.csv`를 이용한 train-test drift
- 모든 피처별 장식용 히스토그램과 대형 HTML 보고서
- EDA 점수만으로 피처를 제거하거나 최종 전처리를 자동 확정하는 기능
- Trackman 결합, 모델 본 학습, OOF 생성과 submission 패키징
- Colab 런타임 재시작, 패키지 설치와 장시간 실행의 Codex 수행
- 별도 대시보드, 데이터베이스 또는 실험 오케스트레이터

## 후속 결정 방식

EDA 결과를 받은 뒤 한 번에 모든 후보를 합치지 않는다. 먼저 무결성 경고를
해석하고, 다음으로 모델 계열별 전처리 ablation 표를 고정한다.

- CatBoost·XGBoost: 중복 제거, smoothing, 선택적 결측 indicator와 의미 기반
  파생 피처를 각각 비교한다.
- TabM: raw·standard·robust·quantile·power와 numerical embedding 후보를
  비교한다.
- DeepFM: field 구분, 미등록 ID 처리, frequency bucket과 의미 기반 interaction을
  비교한다.

단변량 probe가 약해도 다변량 상호작용 가능성이 있으므로 자동 제거하지 않는다.
Adversarial AUC가 높아도 shift가 타깃 성능을 해친다는 뜻은 아니므로 제거 근거로
단독 사용하지 않는다. 서로 다른 검증 프로토콜의 점수를 하나의 순위처럼 비교하지
않는다.

## 설계 근거

- 누출 없는 범주 통계와 cross-fitting:
  [CatBoost 논문](https://papers.neurips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features),
  [scikit-learn TargetEncoder 예제](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html)
- 다변량 분포 이동:
  [Classifier Two-Sample Tests](https://openreview.net/forum?id=SJkXfE5xx),
  [TableShift](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a76a757ed479a1e6a5f8134bea492f83-Abstract-Datasets_and_Benchmarks.html)
- DL 수치 표현:
  [Numerical Feature Embeddings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html),
  [TabM 공식 저장소](https://github.com/yandex-research/tabm)
- 결측 indicator 선택:
  [Missing Indicator Method](https://arxiv.org/abs/2211.09259)
