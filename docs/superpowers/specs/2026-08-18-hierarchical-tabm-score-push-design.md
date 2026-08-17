# 계층적 문맥 TabM 점수 개선 설계

## 배경

규칙을 지킨 첫 TabM 제출은 Public `872.3920184667`을 기록했다. 현재 모델은
`dl_standard + hand_matchup`, TabM 한 구조, seed 3407 하나와 고정 3 epoch로
구성된다. 이 제출은 안전한 기준선이지만, 현재 1위의 1,100점 초과와 비교하면
최소 약 228점이 부족하다. Brier Skill Score로 환산하면 평균 성공률 기준 Brier를
대략 `0.00057` 더 낮춰야 한다.

후속 실험에서 같은 구조의 seed 평균은 기준선보다 `0.000318` 이상 나빴다. 여섯
row-local 피처 묶음과 고정 `TabM 0.70 + CatBoost 0.30`도 충분한 배포 안정성을
보이지 못했다. CatBoost 고정 트리 실험의 최선 개선은 `0.000032`로, Public
환산 약 13점 수준이었다. 학습 시간을 늘리거나 같은 모델 주변을 미세 조정하는
방식만으로는 현재 차이를 메우기 어렵다.

이번 라운드는 규칙·누출·패키지 검사를 완화하지 않는다. 대신 기존의 지나치게
경직된 성능 gate를 개선 폭, 최신 fold와 이전 fold 허용 오차로 분리하고, 기존
`asof_*` 통계를 계층적 야구 문맥과 확률 보정에 제대로 연결한다.

## 목표와 비목표

목표는 다음 세 후보를 하나의 고정된 연구 라운드에서 검증하는 것이다.

- `H1`: 계층적 문맥 피처를 넣은 TabM 원확률
- `H2`: H1에 전역 affine-logit calibration을 적용한 확률
- `H3`: H1에 부분 수축 계층적 calibration을 적용한 확률

라운드당 Public 제출은 최대 세 개다. 후보와 계수는 제출 전에 모두 고정하며,
Public 결과를 보고 같은 라운드의 피처, 가중치나 calibration 계수를 다시 맞추지
않는다.

이번 범위에서는 다음을 하지 않는다.

- TabM 구조·optimizer·embedding 전면 재탐색
- 새로운 선수 ID target encoding
- 평가 데이터의 평균·빈도·순서·이전 행을 이용한 피처나 보정
- 기존 CatBoost 블렌드의 사후 가중치 조정
- AutoML 또는 모델 계열 전체 재실험
- acceptance가 끝나기 전 제출 ZIP 생성

## 검토한 접근

### 채택: 계층적 피처와 교차 적합 calibration

현재 데이터에는 이미 투수·타자의 `asof_*` 비율과 표본 수가 있다. 이를 다시
계산하지 않고, 학습 데이터에서만 적합한 저차원 문맥 사전확률과 결합한다. 새로운
TabM은 한 번의 계열만 학습하고 H2·H3는 같은 모델의 OOF 예측을 CPU에서 보정한다.
규칙을 지키면서 새 정보를 추가하고 GPU 비용을 제한할 수 있다.

### 보류: TabM 구조 앙상블

서로 다른 구조는 같은 구조의 seed 평균보다 오류 다양성이 클 수 있다. 하지만
Stage A~C에서 이미 구조와 embedding을 비교했고 현재 추가 GPU의 첫 사용처로 삼을
새 근거가 없다. H1~H3가 모두 실패한 경우에만 별도 설계로 연다.

### 기각: 대규모 AutoML 포트폴리오

후보 수는 늘지만 이전에 낮았던 모델 계열을 다시 넓게 학습하게 된다. 규칙 검증과
재현 가능한 패키징 비용도 후보 수에 따라 늘어난다. 이번 목표에 비해 범위가 크고
성공 기준이 불명확하므로 포함하지 않는다.

## 전체 흐름

```text
공식 train + trackman_history
          │
          ▼
fold별 학습 전용 HierarchicalContextEncoder
          │
          ├─ 상황별 사전확률
          ├─ 투수·타자 표본 신뢰도
          └─ 상황 대비 투수·타자 residual
          │
          ▼
기존 dl_standard + hand_matchup TabM
          │
          ├─ H1: 원확률
          ├─ H2: 전역 affine-logit calibration
          └─ H3: 부분 수축 계층적 calibration
```

기존 `experiments.independent_dl`의 TabM 학습기와 전처리 인터페이스를 재사용한다.
계층적 상태, calibration, 판정과 산출물은 새 `experiments/hierarchical_tabm`
패키지에 격리한다. 기존 Stage C와 row-feature 코드는 수정하지 않는다.

## 계층적 문맥 피처

### 문맥 계층

문맥 성공확률은 아래 순서로 부모와 자식 관계를 가진다.

1. 전체 성공률
2. `balls_before × strikes_before`
3. count × `pitcher_hand × batter_hand`
4. count × 손 조합 × `base_state × outs_before`
5. 위 문맥 × `game_type`

각 단계의 성공확률은 다음 식으로 부모 확률에 수축한다.

```text
rate = (success_sum + K × parent_rate) / (row_count + K)
```

학습 행에는 해당 행의 `control_success`를 합계와 개수에서 뺀 leave-one-out 값을
쓴다. fold 검증 행과 최종 평가 행에는 fold 학습 구간 또는 전체 공식 train에서
만든 고정 lookup만 적용한다. 자식 key가 없으면 부모 단계로 올라가며, 최종 fallback은
학습 구간 전체 성공률이다.

`K` 후보는 `(32, 128, 512)`로 고정한다. `2021→2022` 개발 fold에서 문맥
사전확률 자체의 Brier가 가장 낮은 K 하나만 선택한다. 차이가 `1e-6` 이내면 더 큰
K를 택한다. 이후 fold와 Public 결과로 K를 다시 선택하지 않는다.

### TabM에 추가할 출력

기존 원본 열과 `hand_matchup`에 다음 수치 피처만 추가한다.

- `hier_context_rate`
- `hier_context_logit`
- `hier_pitcher_reliability`
- `hier_batter_reliability`
- `hier_pitcher_context_gap`
- `hier_batter_context_gap`
- `hier_pitcher_weighted_gap`
- `hier_batter_weighted_gap`

투수와 타자의 신뢰도는 기존 `asof_pitcher_n`, `asof_batter_n`으로 계산한다.
투수는 기존 실험의 K=100, 타자는 K=250을 사용하며 이 값을 다시 탐색하지 않는다.
gap은 기존 `asof_*_success_rate`와 `hier_context_rate`의 차이다. weighted gap은
해당 gap에 신뢰도를 곱한 값이다. 입력 비율과 출력은 유한성 및 범위 검사를 거친다.

## 시간 검증과 모델 학습

### 개발 fold

`2021→2022`는 K 선택에만 사용한다. 이 단계는 문맥 사전확률의 CPU 계산이며
TabM 후보 선택이나 calibration 확인에는 사용하지 않는다.

### H1 OOF

H1은 다음 두 fold에서 현재 anchor와 같은 행으로 비교한다.

- `2022→2023`
- `2023→2024`

TabM 구조, seed 3407, optimizer, batch와 기본 전처리는 첫 제출과 동일하게 둔다.
epoch checkpoint는 검증 Brier로 고르고, 두 fold가 끝난 뒤 최적 epoch의 행 수 가중
중앙값을 최종 전체 학습 epoch로 고정한다. 값은 최소 2, 최대 8 epoch로 제한한다.
전체 train 학습에서는 평가 자료나 내부 평가 지표로 epoch를 변경하지 않는다.

### H2 전역 calibration

확률을 `p`, 절단한 logit을 `z`라 할 때 H2는 아래 형태다.

```text
calibrated = sigmoid(bias + slope × z)
```

패널티는 `bias=0`, `slope=1`로 수축한다. `2022→2023` OOF 안에서 월 7 이하를
적합 구간, 월 8 이상을 regularization 선택 구간으로 사용한다. regularization
후보는 `(0.0001, 0.001, 0.01, 0.1)`이며 평균 Brier에 L2 패널티를 더한 같은
목적함수를 쓴다. 선택한 값을 잠근 뒤 2022→2023 전체 OOF로 다시 적합하고,
`2023→2024`에서 한 번만 확인한다.

### H3 계층적 calibration

H3는 H2의 bias와 slope에 다음 저차원 main-effect offset만 더하고 모든 계수를
하나의 Brier 목적함수에서 함께 적합한다.

- `game_type`
- count state
- hand matchup
- base-out state

고차원 교차항과 선수 ID offset은 넣지 않는다. 각 offset은 0으로 수축하며 H2와
같은 regularization 후보와 시간 분리를 사용한다. 출력 확률은 계산 전후
`[1e-6, 1-1e-6]` 범위에서 다루고 유한성을 검사한다.

H2·H3의 구조와 regularization은 2024 확인 결과로 다시 바꾸지 않는다. 최종 후보가
결정된 뒤에는 선택된 구조를 2023·2024 OOF 전체에 적합해 전체 학습 모델과 결합할
고정 calibration state를 만든다.

## 성능 gate

규칙 gate와 성능 gate는 별개다. 규칙·누출·패키지 gate는 항상 하드 실패다.

### H1 strong

- 두 확인 fold의 행 수 가중 Brier 개선이 anchor 대비 `0.00010` 이상
- `2023→2024` 개선이 `0.00005` 이상
- `2022→2023` 악화가 있더라도 `0.00015` 이하
- 5,000행 이상 사전 정의 segment의 최대 악화가 `0.00075` 이하

### H1 frontier

- 가중 Brier와 `2023→2024`가 모두 anchor보다 개선
- `2022→2023` 악화가 `0.00025` 이하
- strong 조건에는 미달

frontier는 라운드당 한 개만 규칙 준수 Public 진단 후보가 될 수 있다. 최종 모델로
자동 승인하지 않는다.

### H2와 H3

H2는 다음을 모두 만족해야 한다.

- `2023→2024`에서 H1보다 `0.00003` 이상 개선
- `2023→2024`에서 anchor보다 `0.00008` 이상 개선
- 5,000행 이상 segment에서 H1 대비 최대 악화 `0.00020` 이하

H3도 위 조건을 만족하고 `2023→2024`에서 H2보다 `0.00002` 이상 좋아야 한다.
차이가 작으면 더 단순한 H2를 선택한다. H1 raw가 strong이 아니더라도 H2 또는 H3가
자기 전체 gate를 통과하면 해당 calibration 후보는 독립적으로 승급할 수 있다.

사전 정의 segment는 `game_type`, count state, hand matchup, base-out state,
투수 ID known/OOV와 타자 ID known/OOV다. 행 수 미달 segment는 수치를 기록하되
하드 gate에는 쓰지 않는다.

## 제출 정책

한 라운드의 후보는 H1, H2, H3로 고정한다. strong과 허용된 frontier만 별도
acceptance를 받을 수 있고 최대 세 후보를 각각 한 번 제출한다. 통과 후보가 적으면
제출 수를 채우지 않는다.

Public은 OOF와 실제 점수의 연결을 진단하고, 규칙을 통과한 제출 중 최종 제출을
선택하는 데 사용할 수 있다. 그러나 같은 라운드의 K, 피처, regularization, 계수나
가중치를 Public에 맞춰 변경하지 않는다. 다음 라운드는 새 독립 가설과 새 계약을
먼저 작성한 뒤 시작한다.

## 규칙과 누출 방지

각 fold는 다음 순서를 강제한다.

1. fold 학습 행만으로 hierarchy state를 적합한다.
2. 학습 행에는 leave-one-out 값을 만든다.
3. 검증 행에는 동결한 lookup만 적용한다.
4. OOF 정답과 예측은 `row_id`가 정확히 일치할 때만 calibration에 사용한다.
5. 모델과 calibration state가 동결된 뒤에만 평가 입력을 열 수 있다.

추론은 현재 행의 원본 값, 공식 train에서 고정한 lookup과 calibration 계수만 쓴다.
평가 행의 `groupby`, 평균, 빈도, 순위, 누적값, rolling, lag, 행 수와 순서를 쓰지
않는다. 외부 데이터나 외부 API도 사용하지 않는다.

행 단독, 역순, shuffle과 여러 batch 크기에서 같은 `row_id`의 예측이 같아야 한다.
state 동결 전 `test.csv` 접근, 현재 행 target 접근, lookup 변경 또는 비유한 확률이
발생하면 후보를 실패 처리한다.

## 구현 경계

새 패키지는 다음 책임으로 나눈다.

```text
experiments/hierarchical_tabm/
├── contract.json          # 입력 해시, fold, 피처, 후보와 gate
├── contracts.py           # 계약 파싱과 strict validation
├── context_features.py    # LOO hierarchy fit/transform/state
├── calibration.py         # H2/H3 fit/transform/state
├── metrics.py             # paired Brier와 segment 판정
├── campaign.py            # 개발→OOF→판정→전체 학습 순서
├── artifacts.py           # review/resume/delivery write/verify
├── colab.py               # subprocess, checkpoint와 재개
└── COLAB_HIERARCHICAL_TABM_CELL.py
```

기존 TabM 학습기에는 hierarchy feature frame을 전달하는 최소 adapter만 둔다. 기존
Stage C, row-feature proxy, CatBoost 실험과 제출 패키징 권한은 바꾸지 않는다.

## 입력, 실행과 산출물

Colab 입력은 기존 파일을 재사용한다.

- `catboost_tabm_blend_input.zip`
- `tabm_colab_stage_C_delivery.zip`
- 재개 시 가장 최근 `hierarchical_tabm_resume.zip` 하나

한 번의 T4 실행은 입력 검증, K 선택, 두 H1 OOF, H1~H3 판정과 필요한 경우 전체
H1 학습을 순서대로 수행한다. 예상 시간은 1~3시간이다. epoch checkpoint와 검증된
resume를 남기며, 브라우저 다운로드는 20분 간격과 fold 완료 시점에 요청한다.
완료된 fold와 CPU 판정은 재사용한다.

주요 로그 marker는 다음과 같다.

```text
HIER_INPUTS_VERIFIED
HIER_CONTEXT_SELECTED k=<K>
HIER_JOB_START fold=<fold>
HIER_TRAINING_PROGRESS
HIER_JOB_END fold=<fold>
HIER_DECISION candidate=<H1|H2|H3> status=<status>
HIER_FULL_TRAIN_START
HIER_DELIVERY_READY
HIER_ERROR stage=<stage> type=<type> message=<message>
```

산출물은 다음 세 종류다.

- `hierarchical_tabm_review.zip`: OOF, fold·segment 지표와 판정
- `hierarchical_tabm_resume.zip`: 완료 fold와 활성 checkpoint
- `hierarchical_tabm_candidate_delivery.zip`: 통과 후보의 전체 모델·lookup·calibration

candidate delivery는 제출물이 아니다. 각 후보의 모델·상태·규칙 evidence를 다시
검증한 뒤 별도 승인 단계에서만 제출 패키지를 만들 수 있다.

## 오류 처리

- 없는 문맥 key는 정해진 부모와 global prior로 fallback한다.
- 필수 열 누락, 불가능한 count, 음수 표본 수와 비유한 값은 즉시 실패한다.
- OOF row 수, ID 또는 target이 다르면 calibration과 판정을 수행하지 않는다.
- resume의 데이터·코드·계약·모델 identity가 다르면 재사용하지 않는다.
- 한 후보의 수치·성능 실패는 다른 후보의 검증을 막지 않지만, 실패 후보 자신의
  candidate delivery와 제출 패키지는 만들지 않는다.
- deadline 전에 새 작업을 시작할 시간이 없으면 현재 검증 상태를 resume로 남기고
  `incomplete`로 끝낸다. 이를 성능 기각으로 바꾸지 않는다.

## 테스트

작은 합성 fixture로 다음을 검증한다.

- hierarchy와 leave-one-out 손계산 일치
- 현재 행 target이 hierarchy 값에 포함되지 않음
- validation·evaluation transform의 target 비의존
- unseen key의 단계별 fallback
- K 선택과 `1e-6` tie break
- 신뢰도, gap과 weighted gap의 정확한 값과 범위
- calibration 시간 분리와 H2/H3 계수 수축
- H1/H2/H3 gate 경계값과 후보별 독립 실패
- OOF row ID·target 완전 정렬
- 행 단독·역순·shuffle·batch 크기 불변성
- ZIP 누락·추가·경로·변조·일관되게 재해시한 위조 거부
- resume identity와 완료 fold 재사용
- 평가 자료 접근 전 state 동결
- 기존 전체 테스트 회귀

Codex는 코드, 정적 검사와 합성 fixture만 실행한다. 공식 전체 자료 처리, GPU OOF,
전체 학습, 평가 추론과 실제 제출은 사용자가 수행한다.

## 완료 조건

구현 완료와 모델 성공을 구분한다.

구현은 전체 회귀, 생성 셀 결정성, resume와 규칙 fixture가 통과하면 완료다. 모델
연구는 공식 두 OOF fold가 끝나고 H1~H3 중 하나 이상이 자기 성능 gate를 통과해야
성공이다. 이후 전체 모델, lookup과 calibration state가 생성되고 행 독립성,
Python 3.11, 추론 시간·메모리 및 artifact 해시를 모두 통과해야 candidate delivery를
승인한다. 이 모든 조건이 끝나기 전에는 제출 ZIP을 만들지 않는다.
