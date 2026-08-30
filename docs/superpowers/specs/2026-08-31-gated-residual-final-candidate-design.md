# 최종 Gated Residual 제출 후보 설계

## 1. 목적

현재 규칙을 준수하는 제출 기준선은 Tree Expert E2이며 Public 점수는
`977.3809532715`다. Direct Expert 캠페인은 44개 시간 OOF 작업을 완료했지만,
직접 모델을 크게 섞은 조합이 2024 fold에서 악화돼 제출 후보로 승인되지 않았다.

이번 캠페인의 목적은 새 모델 계열을 다시 넓게 탐색하는 것이 아니다. 이미 확보한
E2와 Direct Expert OOF를 사용해 정규시즌 전문가의 유효한 부분만 작은 residual로
제한하고, 시간 순서가 보존된 계층 보정으로 안정성을 높인 뒤, 검증을 통과할 때만
최종 모델을 학습하는 것이다.

가용 예산은 Kaggle T4 x2 약 20시간이다. 이 캠페인의 GPU 사용 상한은 8시간으로
고정하고, 나머지는 실패 복구와 별도 후보를 위해 남긴다.

## 2. 근거와 기준선

입력 증거는 다음과 같다.

- E2 제출 archive SHA-256:
  `8bc33042b9258049f905df0c733b1153d1cedfac39cb0732b2a7e154629d779a`
- Direct Expert Stage A handoff SHA-256:
  `2894e29ff83291f95f72e7708d915161ea8a7a13fee10090be8e61872c65adae`
- Direct Expert Stage B handoff SHA-256:
  `e435f431e4d21036bc0524d14d73d0701e2cdca0ad4c1cb78c28f9017b066929`
- 공식 `train.csv` SHA-256:
  `d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff`
- 공식 `trackman_history.csv` SHA-256:
  `f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9`

Stage B의 28개 작업은 전부 완료됐고 학습 실패는 없었다. 실행 중 DataFrame dtype을
값과 함께 비교한 구현 오류 때문에 원본 decision은 중단됐지만, 저장된 CSV로 동일
결정을 재생한 결과 8개 recipe가 모두 정상 평가됐다.

가장 좋은 기존 recipe는 다음과 같았다.

```text
logit blend = E2 0.400 + D5 0.537 + D0 0.063
2022 gain = -0.0000273312
2023 gain = +0.0003895464
2024 gain = -0.0000544077
weighted gain = +0.0001007974
```

평균 개선과 bootstrap 하한은 양수였지만 최신 fold와 minimum-fold gate를 통과하지
못했다. 따라서 D5를 폐기하지는 않되, direct 비중 `0.60`을 훨씬 작은 행별 correction으로
축소하는 것이 다음 가설이다.

## 3. 범위

### 포함

- 기존 E2 OOF 및 봉인된 E2 제출 모델
- Direct Expert D0·D5의 2022~2024, seed `42/2026/3407` OOF
- 정규시즌 행에만 적용되는 logit residual
- cutoff-safe 투수·타자 표본 신뢰도
- 과거 OOF에서만 적합하는 global, game type, pitcher, batter, hand 계층 보정
- 구조 선택 후 2024 독립 확인
- 통과 시 D0·D5 3시드 full fit
- 행 독립성, 시간 누출, artifact binding, 추론 재현성 감사

### 제외

- D1·D6와 기각된 기존 recipe의 재학습
- 평가 데이터 전체에서 계산한 평균, 빈도, 순위, calibration
- 평가 행 사이 rolling, lag, 누적 통계
- 리더보드 점수를 이용한 강도 조정
- 승인 전 submission ZIP 생성
- OOF gate 실패 후 GPU full fit 강행

## 4. 최종 후보 구조

기준 확률을 `p0`, D0 확률을 `p_global`, D5 정규시즌 확률을 `p_r`라 한다. D5는
`game_type=R`에서만 사용하고 다른 행에는 `p_global`을 사용한다.

```text
p_direct(row) = p_r(row)       if game_type == R
                p_global(row)  otherwise

delta(row) = logit(p_direct(row)) - logit(p0(row))
p_shrunk(row) = sigmoid(logit(p0(row)) + alpha * reliability(row) * delta(row))
```

`alpha`는 direct 모델의 전체 강도다. `reliability`는 현재 행의 투수와 타자가 cutoff
이전 학습 데이터에서 얼마나 관측됐는지를 나타내며 `[0, 1]` 범위다.

```text
r_pitcher = n_pitcher / (n_pitcher + k_pitcher)
r_batter  = n_batter  / (n_batter  + k_batter)
reliability = 1[R] * sqrt(r_pitcher * r_batter)
```

R이 아닌 행의 residual 강도는 0이다. 선수 ID가 처음 등장한 경우 관측 수는 0이며
자동으로 E2에 완전히 backoff한다.

## 5. 계층 보정

Residual 이후 오차는 과거 OOF로만 적합한다. validation year `Y`에 적용할 보정표는
`oof_year < Y`인 행만 사용한다.

```text
effect(group) = sum(y - p_shrunk) / (sum(p_shrunk * (1 - p_shrunk)) + lambda)
logit(p_final) = logit(p_shrunk) + beta * effect(current_row)
```

계층은 다음 순서로 backoff한다.

1. global
2. game type
3. hand matchup
4. pitcher
5. batter

세부 계층 효과는 표본 수에 따라 상위 효과와 shrinkage한다. 현재 평가 행은 저장된
효과표를 조회할 뿐, 평가 데이터의 다른 행을 사용하지 않는다.

## 6. 후보 탐색과 시간 검증

전체 Cartesian product는 만들지 않는다. 다음 네 archetype만 비교한다.

1. `G0`: 고정 alpha의 R-only residual
2. `G1`: 선수 신뢰도 shrinkage가 있는 R-only residual
3. `G2`: G1 + global/game type calibration
4. `G3`: G1 + full hierarchical calibration

구조 후보 값은 다음으로 제한한다.

- `alpha`: `0.025`, `0.05`, `0.075`, `0.10`, `0.15`, `0.20`, `0.30`
- count shrinkage `k`: `25`, `100`, `400` (`k_pitcher=k_batter=k`로 묶어 선택)
- calibration `beta`: `0.05`, `0.10`, `0.20`
- hierarchy lambda: `100`, `500`, `2000`

탐색 순서는 다음과 같다.

1. 2022와 2023 OOF에서 alpha와 count shrinkage를 선택한다.
2. 선택 목적은 weighted gain을 최대화하되 worst-fold 음수에 두 배 패널티를 준다.
3. beta와 lambda는 2022 OOF로 만든 보정표를 2023에 적용한 결과로 선택한다.
4. archetype별 상위 한 개만 동결한다.
5. 동결한 최대 네 후보를 2024에 한 번 평가한다.
6. 2024 결과를 보고 alpha, k, beta, lambda를 다시 맞추지 않는다.

2022 calibration은 자료가 없으므로 적용하지 않는다. 2023 보정표는 2022 OOF로,
2024 보정표는 2022~2023 OOF로 적합한다. 최종 배포 보정표는 2022~2024 OOF만으로
고정한다.

## 7. 승인 gate

다음 조건을 모두 통과한 후보만 `accepted`로 기록한다.

- 3-fold weighted Brier gain `>= 0.00005`
- 2024 gain `>= 0`
- minimum fold gain `>= -0.00003`
- 투수 cluster bootstrap 95% lower bound `>= 0`
- 세 시드 중 최소 두 개의 weighted gain `>= 0`
- 세 시드 중 최소 두 개의 2024 gain `>= 0`
- 5,000행 이상 사전 지정 segment 최대 회귀 `<= 0.00030`
- 확률이 모두 유한하고 `[0, 1]` 범위

후보가 경계값을 소수점 오차 범위에서 넘지 못하면 완화하지 않는다. 모든 후보가
실패하면 캠페인은 `completed_no_candidate`로 종료하고 GPU 학습과 delivery 생성을
수행하지 않는다.

## 8. Full fit과 배포 구성

승인 후보가 있을 때만 D0와 D5를 공식 전체 학습 데이터로 다시 적합한다.

- 구조: CatBoost depth `10`, 최대 `2400` iterations
- seed: `42`, `2026`, `3407`
- D0 3개 + D5 3개 + 기존 E2 3개 = 최대 9개 모델
- iteration 수: 해당 구조의 세 fold best iteration 중앙값
- 피처 상태와 count table: 공식 학습 데이터에서 고정
- calibration table: 승인된 OOF 2022~2024에서 고정

추론 시 각 평가 행은 해당 행의 입력, 봉인된 모델, 고정 count/calibration table만
사용한다. `game_type`, 투수, 타자, 손잡이는 현재 행을 분기하거나 저장된 표를 조회하는
용도로만 사용한다.

## 9. Kaggle 실행 설계

사용자는 T4 x2를 선택하고 한 셀을 한 번 Save Version한다.

```text
P0 입력·해시·GPU·디스크 preflight
P1 기존 OOF 후보 탐색과 2024 확인
P2 승인 gate
P3 통과 시에만 D0/D5 3시드 full fit
P4 행 독립성·추론·artifact 감사
P5 review/handoff/delivery 생성
```

예상 시간은 다음과 같다.

- 후보 탐색과 gate: 30~60분
- 승인 후보 full fit: 4~6시간
- 감사와 artifact 생성: 30~60분
- 전체 상한: 8시간

입력은 공식 데이터와 하나의 compact final-candidate input Dataset으로 제한한다. Compact
input은 E2 OOF·모델, Stage A/B OOF와 원본 manifest를 포함한다. Kaggle의 재귀 ZIP
해제를 막기 위해 중첩 archive는 `.bin`으로 저장하고, 검증 후 `/kaggle/working`에서만
원래 ZIP으로 복구한다.

## 10. 산출물과 제출 제한

항상 생성:

- `gated_residual_final_review.zip`
- `gated_residual_final_handoff.zip`

승인 후보가 있을 때만 생성:

- `gated_residual_final_delivery.zip`

Kaggle 셀은 공식 제출 ZIP을 만들지 않는다. Delivery를 로컬에서 다시 검증하고
acceptance evidence, 규칙 감사, 모델 수·용량·실행시간 gate가 모두 통과한 뒤 별도
패키징 단계에서만 제출물을 만든다.

## 11. 복구와 실패 처리

- 모든 job과 phase는 원자적 state 파일로 기록한다.
- OOF 분석이 끝난 뒤 handoff를 만들고, full fit 모델 하나가 끝날 때마다 상태를 갱신한다.
- 같은 code/contract/input binding의 handoff만 재개한다.
- Kaggle Dataset의 ZIP·해제 디렉터리 두 형태를 실제 fixture로 검증한다.
- 임베드 런타임은 빈 Python 경로에서 import하는 테스트를 통과해야 한다.
- 셀은 `FINAL_CANDIDATE_STAGE`, code SHA, input SHA, GPU, phase, job, decision, artifact
  경로를 로그로 남긴다.
- 셀 파일 크기는 1MB 미만이어야 한다.
- 중단 시 완료 모델과 결정 증거를 포함한 handoff 하나만 다운로드 대상으로 만든다.

## 12. 성공 정의

이 설계의 성공은 단순히 Kaggle 셀이 끝나는 것이 아니다. 다음이 모두 충족돼야 한다.

1. 승인 gate를 통과한 후보가 존재한다.
2. full-fit 모델과 고정 전처리 상태가 생성된다.
3. 행 독립성과 시간 누출 감사에 통과한다.
4. delivery의 모든 member SHA-256과 binding이 일치한다.
5. 별도 제출 패키징에 필요한 acceptance evidence가 완전하다.

OOF 개선은 실제 리더보드 상승을 보장하지 않지만, 기존보다 개선됐다는 검증 증거가
없는 후보를 제출하지 않도록 보장한다.
