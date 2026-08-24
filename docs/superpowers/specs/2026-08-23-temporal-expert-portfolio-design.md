# 시간 전문가·TrackMan 포트폴리오 설계

**작성일:** 2026-08-23  
**대회:** DACON 236743, LG Aimers 9기 Phase 2  
**상태:** 구현 전 승인 설계

## 목적

규칙을 지킨 첫 TabM 제출의 Public `872.3920184667`에서 벗어나기 위해, 기존
단일 다년 학습 모델을 최근 시즌 전문가와 감쇠 다중 시즌 전문가로 나눈다. 여기에
현재 시즌 누적 통계 복원, TrackMan 과거 이력, 학습 전용 TrackMan teacher와
제한된 이질 모델을 단계적으로 검증한다.

이번 작업은 기존 제출물을 조금 보정하는 후속 실험이 아니다. 검증된 TabM 전처리와
실행 계약은 기준선으로 재사용하지만, 학습 행 구성과 시간 가중, 전문가 결합 구조를
새로 정의하는 독립 캠페인이다.

이 설계는 제출 ZIP을 만들지 않는다. 시간 전이 OOF와 규칙·추론 gate를 모두 통과한
후보만 전체 학습 delivery로 만들며, 제출 패키징은 그 delivery를 다시 검증하는 별도
작업으로 남긴다.

## 현재 근거와 방향 전환

기존 실험에서 확인한 내용은 다음과 같다.

- 규칙 준수 단일 TabM은 `dl_standard + hand_matchup`, piecewise-linear 수치
  임베딩, BCE, plateau scheduler, seed `3407`을 사용했다.
- 같은 구조의 단순 다중 seed 평균은 seed 3407 단독보다 시간 전이 Brier가
  `0.000318` 이상 나빠졌다.
- FT-Transformer와 TabNet sentinel은 기존 비교에서 TabM보다 낮았다.
- CatBoost는 TabM과의 OOF 혼합에서 개선을 보인 적이 있지만, fold별 적정 tree
  수가 크게 달라 전체 학습 배포 정렬에 실패했다.
- 행 단위 파생변수와 계층형 문맥 후보는 기존 단일 TabM을 안정적으로 넘지 못했다.
- 기존 실험은 최신 한 시즌 전문가와 감쇠 다중 시즌 전문가를 같은 OOF에서
  직접 비교·결합하지 않았다.

따라서 이번 우선순위는 모델 이름을 더 늘리는 것이 아니라 시간 변화에 맞는 학습
구조를 먼저 세우는 것이다.

## 공식 지표와 규칙 경계

공식 평가는 Brier Skill Score다.

```text
Score = max(0, 100000 * (1 - Brier / (r * (1 - r))))
```

`r`은 비공개 평가 데이터의 평균 성공률이다. 한 평가셋 안에서는 점수가 Brier의
단조 함수이므로 OOF Brier 최소화가 올바른 목표다.

`r * (1-r) <= 0.25`를 이용하면 현재 점수에서 필요한 Brier 개선량의 상한을 대략
계산할 수 있다.

| 목표 Public | 점수 상승 | 필요한 Brier 개선량 상한 |
|---:|---:|---:|
| 1,000 | 127.608 | 0.000319 |
| 1,070 | 197.608 | 0.000494 |
| 1,100 | 227.608 | 0.000569 |

후보의 성능 규모는 다음처럼 별도 기록한다.

- `incremental`: 시간 가중 Brier 개선 `0.00005` 이상
- `competitive`: 시간 가중 Brier 개선 `0.00025` 이상
- `breakthrough`: 시간 가중 Brier 개선 `0.00045` 이상

작은 개선도 통계적으로 유효하면 보존하지만, 캠페인이 끝나기 전에 incremental
후보만으로 제한된 리더보드 제출 기회를 소비하지 않는다.

공식 규칙상 평가 행은 다른 평가 행이나 평가셋 전체 분포를 참조하면 안 된다. 반면
공식 학습 데이터에서 만든 통계를 각 평가 행에 독립적으로 결합하는 것, 공식
TrackMan 기간의 ID 대응과 요약, 학습 중 TrackMan privileged information을
teacher·soft label·auxiliary task로 사용하는 것은 운영진 답변에서 허용됐다.

참조:

- [공식 평가·코드 제출 사양](https://dacon.io/competitions/official/236743/overview/evaluation)
- [공식 대회 규칙](https://dacon.io/competitions/official/236743/overview/rules)
- [평가 행 독립성과 TrackMan 관련 운영진 답변](https://dacon.io/competitions/official/236743/talkboard/417082?page=1&dtype=recent)

추론 코드는 다음을 금지한다.

- 평가 행 간 집계, 순위, 이웃, 평균 또는 빈도
- 평가 예측 평균에 맞춘 scale·shift
- 평가 데이터에서 ID 사전, 결측 대치값, 표준화값 또는 calibration 재학습
- 평가 기간 TrackMan 또는 외부 데이터
- 제출 후 외부 API·서버·다운로드 의존

## 자원 배분

주 실행 환경은 Kaggle T4 두 장이며 총 GPU 예산은 30시간이다. Colab T4 한 장은
복구와 짧은 확인에만 사용한다.

| 단계 | 목적 | 최대 벽시계 시간 |
|---|---|---:|
| T1 | 최근·감쇠 다중 시즌 전문가 | 6시간 |
| T2-A | 현재 시즌·TrackMan 단일 피처 선별 | 4시간 |
| T2-B | 상위 피처 다중 fold 확인 | 4시간 |
| T3 | 구조·손실·CatBoost·LUPI | 6시간 |
| T4 | OOF 앙상블·anchor·calibration | 3시간 |
| T5 | seed 확인과 전체 학습 | 5시간 |
| 예비 | 복구 또는 유망 exploratory 확인 | 2시간 |

합계는 30시간이다. 한 Kaggle Save Version은 6시간 미만으로 끝내고 종료 전
10분을 산출물 생성에 남긴다.

## 공통 시간 fold

최종 평가 시즌 직전의 변화에 맞추면서 세 번의 독립적인 시간 이동을 확인한다.

| 검증 시즌 | 최근 시즌 전문가 학습 | 다중 시즌 전문가 학습 |
|---:|---|---|
| 2022 | 2021 | 2019~2021 |
| 2023 | 2022 | 2019~2022 |
| 2024 | 2023 | 2020~2023 |
| 최종 평가 | 2024 | 2021~2024 |

다중 시즌 전문가는 항상 최근 네 시즌 이내만 사용한다. 시즌 `s`의 sample weight는
다음과 같다.

```text
weight(s) = decay ** (latest_training_season - s)
```

가중치는 loss에 직접 적용한다. 행 복제나 확률적 재표집은 사용하지 않는다.

## T1: 시간 전문가 구조

### 학습 후보

첫 실험에서는 기존 P2 TabM, 기존 전처리와 seed 3407을 고정한다.

- 최근 시즌 전문가: 세 fold에서 각 1개, 총 3개
- 다중 시즌 전문가: 감쇠율 네 개와 세 fold, 총 12개
- 감쇠율: `0.40`, `0.55`, `0.70`, `1.00`

GPU 학습은 총 15개다. 최근 전문가 비중과 결합 방식은 저장한 OOF 예측으로
계산한다.

- 최근 전문가 비중: `0.50`, `0.65`, `0.75`, `0.85`, `1.00`
- 확률 평균
- logit 평균

감쇠율까지 포함한 기본 조합은 40개다.

### Train-only anchor

각 fold의 학습 행에서만 다음 값을 만든다.

```text
recent_rate = 최근 학습 시즌 target mean
multi_rate = 감쇠 sample weight를 적용한 다중 시즌 target mean
anchor_rate = alpha * recent_rate + (1 - alpha) * multi_rate
```

`alpha`는 같은 후보의 최근 전문가 비중을 그대로 사용해 별도 탐색 축을 만들지 않는다.
예측에 대한 anchor shrinkage는 다음과 같다.

```text
logit(p_anchor) = (1 - beta) * logit(p) + beta * logit(anchor_rate)
beta in {0, 0.025, 0.05, 0.10}
```

검증·평가 데이터의 target mean이나 예측 평균은 사용하지 않는다.

### T1 승급

Champion 조건:

- 세 fold 모두 개선하거나 각 fold 회귀가 `0.00003` 이하
- 행 수 가중 Brier 개선 `0.00005` 이상
- 최신 fold 개선 `0.00003` 이상
- 선수 block bootstrap 개선량의 95% 하한이 0보다 큼
- eligible segment 최대 회귀 `0.00050` 이하

Exploratory 조건:

- 세 fold 중 두 fold 이상 개선
- 행 수 가중 개선 `0.00003` 이상
- 최악 fold 회귀 `0.00015` 이하
- 최신 fold 회귀 `0.00005` 이하
- segment 최대 회귀 `0.00100` 이하

최대 두 구조만 남긴다. 가장 좋은 champion과 구조적으로 다른 exploratory 하나다.
승급 구조는 seed 42에서도 방향을 확인한다.

## T2: 현재 시즌·TrackMan 피처

피처를 한꺼번에 넣지 않고 정보군별로 기여를 분리한다. 모든 집계와 ID 대응은
검증 시즌 이전까지만 사용한다.

| 검증 시즌 | 사용할 수 있는 TrackMan |
|---:|---|
| 2022 | 2021까지 |
| 2023 | 2022까지 |
| 2024 | 2023까지 |
| 최종 평가 | 2024까지 |

### S1: 현재 시즌 누적 통계 복원

공식 행에 제공된 통산 `asof` 누적값에서 학습 데이터로 고정한 시즌 시작 전 누적
상수를 뺀다.

이 계산의 cutoff-bound 구현은 이미
`experiments/independent_dl/feature_sources/seasonal.py`에 존재한다. 다만 현재
주력 TabM 제출 경로에는 연결되어 있지 않고, 새 시간 전문가 구조에서 독립 S1
ablation으로 판정된 적도 없다. 신규 캠페인은 이 함수를 다시 작성하지 않고
fixture·cutoff 감사를 보강한 뒤 피처 bundle로 연결한다.

```text
season_n = asof_n - before_season_n
season_success = asof_n * asof_rate - before_season_success
season_rate = smoothed(season_success, season_n)
```

투수와 타자 각각에 대해 다음을 만든다.

- 현재 시즌 표본 수와 성공률
- 통산 대비 현재 시즌 차이
- 표본 수 기반 train-only smoothing
- 신규 선수, 계산 불가, 반올림 오차와 음수 보정 flag

각 validation fold의 시즌 시작 상수는 fold 학습 시즌만으로 만든다. 최종 평가에서는
2024년까지의 공식 학습 데이터 상수만 사용한다. 평가 행 자신의 `asof` 값 외 다른
평가 행은 사용하지 않는다.

### TrackMan 피처군

| ID | 내용 |
|---|---|
| P0 | 이력 수, 최근성, 매핑 신뢰도, 미매핑 여부 |
| P1 | 구속, 회전, 수직·수평 무브먼트, 익스텐션, 릴리스, zone speed |
| P2 | 구종군 사용률, 구종군별 구위, 구속 차이 |
| P3 | 최근 시즌과 커리어 차이, 구속·회전·구종 비율 추세 |
| B1 | 타자가 과거에 상대했던 구종군·구속·무브먼트 노출 분포 |
| M1 | 투수 레퍼토리와 타자 노출 차이, 좌우 손 상성 |

작은 표본의 연속 통계는 타깃을 사용하지 않는 시즌·손·구종군 prior 쪽으로 축소한다.

### ID 대응 gate

투수·타자 TrackMan ID는 cutoff마다 공식 train과 history만으로 대응한다.

- 손잡이 일치
- 팀·시즌 이력 일치
- 기록 수 규모 일치
- 후보 간 score margin
- 승인된 one-to-one 대응
- 불확실한 후보는 미매핑으로 유지

승인 coverage가 30% 미만이면 B1·M1 학습을 생략하고 `insufficient_mapping`으로
기록한다. 30~60%는 exploratory로만 취급하고 coverage·confidence flag를 항상
모델에 제공한다.

### T2-A: 최신 fold 선별

2023→2024와 T1 선택 구조를 고정한다.

1. 최근 전문가에 S1, P0, P1, P2, P3, B1, M1을 각각 적용한다.
2. 성능 상위 두 개와 구조적으로 다른 wildcard 한 개를 다중 시즌 전문가에 적용한다.
3. 최대 10개 신규 학습, seed 3407, 최대 12 epochs, patience 3을 사용한다.

최신 fold Brier `0.00003` 이상 개선 또는 일관된 segment·bootstrap 증거가 있는
후보만 남긴다. 최대 세 개가 T2-B로 간다.

### T2-B: 다중 fold와 조합

- T2-A 생존 후보를 2021→2022와 2022→2023에서 확인한다.
- 상호 보완적인 피처군 조합만 최신 fold에서 최대 두 개 확인한다.
- 가장 좋은 조합 하나만 과거 두 fold에서 다시 확인한다.
- 전체 최대 10개 학습이다.

허용할 수 있는 조합 예시는 `S1+P3`, `P1+P3`, `P2+B1`, `P2+M1`이다. T1의
감쇠율·전문가 비중·anchor grid는 바꾸지 않아 개선 원인을 분리한다.

## T3: 모델과 학습 신호 다양성

T1의 시간 구조와 T2 피처를 고정한 뒤 다음 축만 확인한다.

### TabM 구조와 loss

| 후보 | 변경 |
|---|---|
| M0 | P2 + piecewise-linear + BCE + plateau 기준선 |
| M1 | P3-lite + BCE |
| M2 | P2 + Brier loss |

M1과 M2를 최근·다중 시즌 전문가에 각각 적용해 최대 네 번 학습한다. P3-full,
periodic embedding, one-cycle, focal loss와 무조건적인 seed 평균은 기존 증거와
배포 비용 때문에 다시 열지 않는다.

### CatBoost 시간 전문가

최근 시즌과 감쇠 다중 시즌 CatBoost를 각각 학습한다. 한 번 384 trees까지
학습하면서 `16`, `64`, `192`, `384` prefix의 OOF 예측을 저장한다.

세 fold에서 동일하거나 인접한 prefix가 선택될 때만 배포 가능하다. 다시 매우 짧은
tree 수와 긴 tree 수로 갈리면 `unstable_for_deployment`로 종료한다. 단독 성능이
낮아도 TabM과의 고정 혼합에서 개선하면 다양성 후보로 남을 수 있다.

### TrackMan LUPI teacher–student

현재 투구 TrackMan 값은 평가 시 없지만 공식 학습 기간 안에서는 학습 전용 정보로
사용할 수 있다.

1. 신뢰도 높게 train 행과 current-pitch TrackMan 행이 연결된 부분집합을 만든다.
2. teacher는 사전 투구 피처와 현재 투구 구종·구속·회전·무브먼트를 사용한다.
3. teacher soft label은 시즌 또는 선수 그룹 cross-fitting으로 생성한다.
4. teacher가 직접 학습한 행에 대한 in-sample 확률은 student label로 쓰지 않는다.
5. student TabM은 평가 시 사용할 수 있는 사전 피처만 입력받는다.

Student loss:

```text
loss = (1 - lambda) * BCE(y, p) + lambda * BCE(p_teacher, p)
lambda in {0.10, 0.25}
```

매칭되지 않은 행은 원래 BCE만 사용한다. 최신 fold에서 단독 또는 전문가 혼합
Brier를 `0.00010` 이상 개선한 경우에만 과거 fold로 확장한다. 최종 student 추론은
current-pitch TrackMan 없이 완전히 동작해야 한다.

### T3 승급과 seed

최신 fold에서 최대 세 개의 서로 다른 후보만 과거 두 fold로 보낸다.

- 독립 Brier가 가장 좋은 TabM
- 혼합 잔차 다양성이 확인된 CatBoost
- 기준을 통과한 LUPI student

seed 3407로 선별하고 seed 42는 구조 안정성 확인에만 사용한다. 두 seed가 같은
방향일 때 구조를 승인하며, seed 평균은 OOF gate를 별도로 통과할 때만 허용한다.

## T4: OOF 조합과 확률 보정

T4 입력 스트림은 최대 네 개다.

- C0: T1 시간 전문가
- C1: T2 피처 개선 시간 전문가
- C2: T3 구조·loss 또는 LUPI 개선 TabM
- C3: 이질 CatBoost

주 모델과 보조 모델의 고정 혼합만 계산한다.

- `90:10`, `80:20`, `70:30`
- 확률 평균과 logit 평균
- 두 보조 모델이 각각 pair gate를 통과한 경우에만 `80:10:10` 하나

연속 가중치 최적화와 탈락 후보 전체의 조합은 금지한다.

### 시간 가중 순위

```text
selection_delta = 0.20 * delta_2022
                + 0.30 * delta_2023
                + 0.50 * delta_2024
```

행 수 가중 Brier와 각 fold 값도 함께 보고한다. 시간 가중 점수만 좋고 한 fold나
세그먼트가 크게 무너지면 승급하지 않는다.

### Anchor와 calibration

Anchor는 T1에서 승인된 설정만 비교한다. 상위 원본 조합 최대 다섯 개에 대해서만
다음 calibration을 확인한다.

- 없음
- intercept-only logit 보정
- 정규화된 Platt `a + b * logit(p)`

Anchor와 calibration은 중첩하지 않는다. isotonic, 고차 보정, 평가 평균 맞추기는
사용하지 않는다.

Forward calibration:

- 2023 예측 보정기는 2022 OOF로만 fit
- 2024 예측 보정기는 2022~2023 OOF로만 fit
- 최종 평가 보정기는 2022~2024 OOF로 fit

2023과 2024에서 같은 개선 방향, 두 시즌 가중 개선 `0.00003` 이상, slope
`0.8~1.2`, intercept `-0.15~0.15`일 때만 승인한다.

### 불확실성과 segment

validation 시즌 안에서 `pitcher_id` 단위 1,000회 block bootstrap을 수행한다.
최소 5,000행인 다음 segment에 정식 gate를 적용한다.

- game type
- 투수·타자 손 조합
- ID OOV
- TrackMan 매핑 여부
- 이력 표본 수 구간
- 득점권·주자 상태
- leverage 구간

최종 champion과 exploratory 조건은 T1 gate를 유지한다. 성능 규모
`incremental/competitive/breakthrough`는 acceptance와 별도로 표시한다.

리더보드 제출은 최대 세 번으로 관리한다.

1. T5를 통과한 competitive 또는 breakthrough champion
2. 구조적으로 다른 exploratory
3. 오류 수정 또는 최종 확인용 예비

캠페인 종료 전에 competitive 후보가 없으면 전체 근거를 검토한 뒤에만 가장 좋은
incremental 후보의 제출 여부를 결정한다.

## T5: 최종 확인과 전체 학습

T5는 한 번의 긴 실행이 아니라 두 Save Version으로 나눈다.

| 실행 | 목적 | 최대 시간 |
|---|---|---:|
| T5-A | champion과 exploratory의 누락 seed·fold 확인 | 2시간 30분 |
| T5-B | 공식 학습 데이터 전체 학습 | 2시간 30분 |

### 최종 epoch와 tree 수

검증 정답이 없는 전체 학습에서 early stopping하지 않는다. 세 fold best epoch의
시간 가중 median을 사용한다.

```text
E_final = weighted_median(E_2022, E_2023, E_2024; 0.2, 0.3, 0.5)
```

최소 2 epochs이며 OOF에서 확인한 안정 구간을 넘지 않는다. fold별 best epoch가
지나치게 다르면 전체 학습을 차단한다. CatBoost는 세 fold에서 안정된 공통 prefix가
있을 때만 전체 학습한다.

### 전체 학습 구성

- 최근 전문가: 2024년만 학습
- 다중 전문가: 2021~2024, 선택된 감쇠율
- TrackMan state: 공식 2024년까지의 history로 fit
- S1 시즌 시작 상수: 공식 2024년까지의 train으로 고정
- anchor 또는 calibration: 승인된 하나만 동결
- 모델·전문가 혼합 비율: T4에서 승인된 상수

평가 데이터는 모든 상태와 모델이 동결된 뒤 `transform`과 `predict`에만 전달한다.

### 행 독립성 검사

같은 평가 행의 확률이 다음 조건에서 같아야 한다.

- 단일 행 입력
- 원래 배치와 임의 순서 배치
- batch size `1`, `32`, `512`
- 다른 평가 행 일부를 제거한 부분집합

허용 오차를 넘으면 `rule_blocked`다.

### T5 산출물

T5-A:

```text
temporal_portfolio_confirmation_review.zip
```

T5-B:

```text
temporal_portfolio_training_delivery.zip
```

Delivery에는 모델, 전처리 상태, ID/OOV 정책, S1 상수, TrackMan state,
anchor/calibration, 혼합 비율, epoch/tree 수, manifest와 추론 smoke-test를 넣는다.
이 파일 자체는 `submit.zip`이 아니다.

## 중복 실행 방지

기존 제출과 산출물은 삭제하거나 성공으로 재해석하지 않는다. 신규 작업의 identity는
다음 값으로 만든다.

```text
data row hash
+ training seasons and validation season
+ decay and sample weights
+ feature specification
+ model architecture and loss
+ seed and fold
```

완료 artifact와 identity가 정확히 같으면 checkpoint와 OOF를 검증 후 재사용한다.
모델 이름이 같아도 학습 시즌, 감쇠, 피처 중 하나가 다르면 새 작업이다.

## 공통 엔진과 실행 환경

Kaggle과 Colab은 하나의 실험 엔진과 artifact 형식을 공유하고 launcher만 다르게
둔다.

```text
common engine
├── temporal folds and features
├── training and OOF
├── decisions and gates
├── checkpoint and resume
└── artifacts and verification
    ├── Kaggle T4x2 launcher
    └── Colab T4x1 recovery launcher
```

Kaggle 단계:

1. `temporal_portfolio_t1`
2. `temporal_portfolio_t2a`
3. `temporal_portfolio_t2b`
4. `temporal_portfolio_t3a`
5. `temporal_portfolio_t3b_t4`
6. `temporal_portfolio_t5a`
7. `temporal_portfolio_t5b`

Kaggle input의 resume이 ZIP이거나 dataset으로 풀려 있어도 내부 manifest로 찾는다.
파일명은 identity가 아니다.

Colab은 Drive 없이 input ZIP과 최신 handoff를 업로드한다. 같은 런타임에서는 업로드
cache를 재사용한다. 런타임이 초기화되면 Drive나 외부 저장소 없이 재업로드를 피할
수 없다는 한계는 명시한다.

## Handoff와 로그

매 stage의 사용자 전달 파일은 하나다.

```text
temporal_portfolio_stage_<STAGE>_handoff.zip
├── review.zip
├── resume.zip
├── run.log
├── stage_summary.json
└── handoff_manifest.json
```

epoch마다 브라우저 다운로드하지 않는다. checkpoint와 최신 handoff는 런타임
디스크에서 원자적으로 갱신하고, 정상 종료 또는 오류 시 handoff 하나만 다운로드한다.

주요 로그:

```text
PORTFOLIO_CODE_READY sha256=...
PORTFOLIO_DATA_READY train_rows=... test_rows=... trackman_rows=...
PORTFOLIO_GPU_READY count=2 names=...
PORTFOLIO_RESUME_READY stage=... sequence=...
STAGE_START stage=... jobs=... budget_seconds=...
JOB_START candidate=... gpu=...
TRAINING_PROGRESS candidate=... epoch=... batch=... brier=... eta_seconds=...
CHECKPOINT_SAVED candidate=... epoch=... sha256=...
JOB_COMPLETE candidate=... best_epoch=... best_brier=...
CANDIDATE_DECISION candidate=... status=...
STAGE_COMPLETE stage=... completed=... rejected=... inconclusive=...
HANDOFF_READY path=... sha256=...
```

오류 시 마지막 두 줄은 다음 형태다.

```text
PORTFOLIO_ERROR stage=... candidate=... type=... message=...
EMERGENCY_HANDOFF_READY path=... sha256=...
```

화면 진행 로그는 30~60초 간격으로 제한하고 전체 JSONL은 `run.log`에 보존한다.

## Resume와 코드 호환성

Manifest 필수 필드:

- artifact kind, campaign ID, stage, sequence
- parent artifact SHA-256
- data·contract SHA-256
- training identity
- state schema version
- runtime build SHA-256

같은 lineage에서 가장 높은 sequence를 선택한다. 같은 sequence의 서로 다른 artifact,
다른 data hash나 분기된 lineage는 자동 선택하지 않는다.

Checkpoint binding은 학습 의미와 런타임 구현을 구분한다.

반드시 같아야 하는 항목:

- 행 집합, fold, feature spec
- 모델·optimizer·scheduler·loss
- seed, epoch와 RNG state
- 핵심 라이브러리 버전

기록하되 호환성 검사를 거칠 수 있는 항목:

- launcher, 로그, 다운로드, artifact 작성 코드

런타임 버그 수정 후 기존 checkpoint를 이어가려면 state schema, model key·shape,
optimizer, RNG와 고정 fixture prediction이 모두 일치해야 한다. 단순히 code hash를
무시하는 우회는 허용하지 않는다.

## 제출 패키징 gate

이 캠페인의 어떤 경로도 다음 조건 전에는 제출 ZIP을 만들 수 없다.

- 후보가 champion 또는 명시적으로 승인한 exploratory
- 필수 fold와 seed 완료
- OOF review acceptance 통과
- 현재 data·code·config·model hash 일치
- Python 3.11.15, Ubuntu 22.04 호환
- 설치 10분, L4 추론 10분 제한 충족
- 인터넷 없는 추론
- 행 독립성 검사 통과
- sample submission과 행 수·순서·컬럼·dtype 일치
- 확률이 유한하고 `[0,1]` 안에 존재

패키징 진입점도 같은 gate를 fail-closed로 다시 검사해야 한다.

## 오류·예산 상태

- `completed`: 필요한 작업과 검증 완료
- `promoted`: 다음 stage 진급
- `rejected`: 유효하게 완료됐으나 gate 미달
- `insufficient_mapping`: ID 대응 근거 부족
- `unstable_for_deployment`: OOF 구조를 하나의 전체 모델로 정렬할 수 없음
- `budget_inconclusive`: 최소 판정 전 시간 종료
- `failed`: 코드·환경·데이터 오류
- `rule_blocked`: 시간 cutoff 또는 평가 행 독립성 위반

한 후보의 실패나 기각은 해당 후보만 막고 독립 후보와 기존 champion을 막지 않는다.

## 구현 전후 검증

Codex는 전체 GPU 학습을 실행하지 않고 작은 fixture와 정적 검사를 수행한다.

- fold와 TrackMan cutoff
- 감쇠 sample weight
- S1 누적 통계 복원과 반올림 예외
- 투수·타자 ID 매핑, 중복·미매핑
- teacher cross-fit soft label의 in-sample 차단
- OOF row_id 정렬과 중복 방지
- Brier, score 규모, 확률·logit 혼합
- forward calibration
- pitcher block bootstrap
- ZIP·풀린 directory resume 탐지
- 손상 artifact, 다른 data와 분기 lineage 거부
- checkpoint save·restore와 stale module 제거
- 평가 행 순서·batch·부분집합 불변성
- 미승인 후보의 패키징 차단

실제 전체 데이터 전처리, OOF, GPU 학습, 전체 학습과 평가 서버 제출은 사용자가
실행한다.

## 완료 기준

1. 기존 artifact와 신규 job의 중복 감사가 완료된다.
2. T1~T4에서 세 시간 fold의 정렬된 OOF와 모든 판정 근거가 생성된다.
3. 규칙·segment·bootstrap·seed gate를 통과한 후보만 T5에 들어간다.
4. T5-A와 T5-B가 서로 독립적으로 재개 가능하다.
5. 전체 학습 delivery가 평가 행 독립성·해시·Python 3.11·추론시간 검사를 통과한다.
6. 제출 ZIP은 이 설계 범위에서 생성되지 않는다.
