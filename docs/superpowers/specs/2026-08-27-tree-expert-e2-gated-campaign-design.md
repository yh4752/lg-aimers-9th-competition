# Tree Expert E2 단일 실행·단계별 Gate 설계

## 1. 목적

E2는 E1에서 승격된 `c1_anchor_residual`과 `c2_trackman_residual`을 과거 시간
fold와 다중 seed에서 확인하고, 근거가 충분할 때만 전체 데이터 모델 delivery를
만든다. Kaggle T4 x2의 한 번의 Save Version 안에서 구조 확인, seed 확인, TabM
결합 확인, 전체 학습과 추론 감사를 순서대로 수행한다.

E2는 DACON 제출 파일을 만들지 않는다. `submission.csv`, 제출 ZIP, 자동 업로드는
후속 로컬 감사 계획의 범위다.

## 2. E1 근거와 고정 판단

E1 handoff는 무결성 검사를 통과했고 다음 결과를 포함한다.

| 후보 | 2024 Brier | 기존 TabM 대비 개선 | E2 역할 |
|---|---:|---:|---|
| `c0_native_ctr` | 0.2479321043 | 0.0001786979 | 제외 |
| `c1_anchor_residual` | 0.2477889045 | 0.0003218977 | 주력 |
| `c2_trackman_residual` | 0.2478245921 | 0.0002862101 | 보조 확인 |
| `c3_failure_aware` | 없음 | 없음 | 라벨 gate 실패로 제외 |

`c1`의 투수 block bootstrap 95% 하한은 `0.0001420267`, `c2`의 하한은
`0.0001038564`로 모두 양수다. `c2`는 전체 성능이 `c1`보다 낮고 F 구간에서
`0.0001278567` 회귀했으므로 E2 동률 판정에서는 `c1`을 우선한다.

## 3. 입력과 identity binding

로컬 준비 도구는 아래 세 산출물을 검증해 하나의 `tree_expert_e2_input.zip`으로
봉인한다.

| 산출물 | SHA-256 |
|---|---|
| E1 handoff | `9e5b55d88715b7a76051d6e8497ffd19617a7a4b25fdafeb7f5f5af16b72fffc` |
| E1 review member | `85a2dcf1a01a4c665c46d075571a4ffb8f463a981f49c22a2f3d73824ae4c2d2` |
| E1 resume member | `66065feae86c0c252fdce1b33cfb0fb56c5f86811a7fc9a40b8c0e2a17af8d3c` |
| Stage C delivery | `f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a` |
| Stage C review member | `461c4422cebdc7383577ad6c53b044bfe3d9d15b667383e454591135e4242d2c` |
| 872점 TabM 제출 | `ee4d6324eb8d1afec08157526db473834620aabac9647a865e9ff6a6cd1345be` |
| TabM seed 3407 weights | `940c358c7e4af258ffec957a8ea42a438f3e639ffc6b5b6f777f77f45db9c945` |

E2 계약은 E1 decision의 `status=completed`와 승격 ID
`c1_anchor_residual`, `c2_trackman_residual`도 정확히 묶는다. 공식 데이터는 E1과
동일한 다음 해시만 허용한다.

- `train.csv`: `d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff`
- `trackman_history.csv`: `f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9`

Kaggle 입력은 공식 데이터셋과 `tree_expert_e2_input` 두 개다. 재실행할 때만 세 번째
입력으로 이전 `tree_expert_e2_handoff.zip` 또는 그 안의 resume 하나를 추가한다.

## 4. 시간 fold와 기준 TabM

| fold | 학습 시즌 | 검증 시즌 | TrackMan cutoff |
|---|---|---:|---:|
| F1 | 2019~2021 | 2022 | 2021 |
| F2 | 2019~2022 | 2023 | 2022 |
| F3 | 2019~2023 | 2024 | 2023 |
| Final | 2019~2024 | 평가 데이터 | 2024 |

Stage C에는 seed 3407의 F2와 F3 OOF가 있으므로 해시와 행 정렬을 검증해 재사용한다.
F1 OOF는 없으므로 E2가 Stage C와 같은 TabM 설정으로 기준 job 하나를 먼저 실행한다.

- architecture: TabM
- numeric embedding: piecewise linear
- blocks: 4
- width: 512
- `k`: 32
- dropout: 0.1
- learning rate: 0.0006
- loss: BCE
- scheduler: plateau
- seed: 3407

기준 F1 job이 실패하거나 F2/F3 OOF identity가 다르면 CatBoost 비교를 시작하지 않는다.

## 5. 한 번 실행의 단계

### 5.1 Phase B0: 입력·기준값 준비

입력, 계약, 코드, 공식 데이터, E1, Stage C와 TabM 제출 해시를 확인한다. F1 TabM
기준 OOF를 생성하고 F2/F3 기준 OOF를 복원한다. 각 OOF는 공식 검증 행의 `row_id`,
target, 순서와 정확히 일치해야 한다.

### 5.2 Phase B1: 구조 확인

`c1`, `c2`를 seed 3407로 F1과 F2에서 학습한다. F3는 E1의 검증 예측을 재사용한다.
두 구조의 세 fold 결과를 다음 순서로 정렬한다.

1. 최악 fold 개선량 최대
2. 세 fold 행 수 가중 개선량 최대
3. F3 개선량 최대
4. 동률이면 TrackMan과 추가 lookup이 없는 `c1`

seed 확인으로 진행하려면 선택 구조가 모두 만족해야 한다.

- 세 fold 가중 Brier 개선 `>= 0.00010`
- F3 개선 `>= 0.00010`
- 최악 fold 회귀 `<= 0.00010`

통과하지 못하면 `rejected_structure`로 종료하고 full fit을 시작하지 않는다.

### 5.3 Phase B2: seed 확인

선택 구조만 seed `42`, `2026`으로 F1~F3에서 학습한다. seed 3407 결과와 합쳐 세
seed 확률 평균을 계산한다. 다음 조건을 모두 만족해야 한다.

- 모든 확률, Brier와 best iteration이 유한하고 범위가 정상이다.
- 어느 seed도 세 fold 가중 Brier가 기준보다 `0.00010` 넘게 회귀하지 않는다.
- 세 seed 평균의 가중 개선 방향이 seed 3407과 같다.
- 각 fold에서 최소 두 seed의 Brier가 해당 TabM 기준보다 나쁘지 않다.

실패하면 `rejected_seed_instability`로 종료한다.

### 5.4 Phase B3: TabM 결합 확인

후보는 사전에 고정한다.

- CatBoost 단독
- 확률 평균 TabM/CatBoost `90/10`, `80/20`, `70/30`
- logit 평균 TabM/CatBoost `90/10`, `80/20`, `70/30`

F1에서 최소 Brier 후보를 고르고 F2에 적용한다. F1+F2에서 최소 Brier 후보를 다시
고르고 그 정확한 후보를 F3에 적용한다. 동률은 CatBoost 단독, 더 높은 CatBoost
비중, 확률 평균, logit 평균 순으로 해결한다. F3 결과를 본 뒤 비율을 다시 바꾸지
않는다.

F1+F2에서 선택해 F3에 적용한 후보를 최종 predictor로 채택하려면 다음을 만족해야
한다.

- 순차 F2+F3 행 수 가중 Brier가 CatBoost 단독보다 `0.00001` 이상 개선된다.
- 어느 순차 fold에서도 CatBoost 단독보다 `0.00005` 넘게 회귀하지 않는다.
- F3에서 TabM 기준과 CatBoost 단독 둘 다보다 나쁘지 않다.

그렇지 않으면 최종 predictor는 CatBoost 단독이다. 혼합이 선택되면 872점 제출의
TabM 모델과 전처리 파일을 해시 그대로 delivery에 포함한다.

## 6. 최종 acceptance gate

전체 학습과 delivery의 기본 acceptance는 세 seed CatBoost 단독 ensemble의 세 fold
OOF로 판정한다. 이는 F1을 이용해 혼합 비율을 선택하므로 F1의 혼합 성능을 독립
검증값처럼 다시 사용하는 오류를 막는다. CatBoost 단독이 아래 gate를 통과한 뒤,
Phase B3의 순차 검증도 통과한 경우에만 delivery predictor를 TabM 혼합으로 바꾼다.
혼합이 B3 gate를 통과하지 못해도 CatBoost 단독 acceptance에는 영향을 주지 않는다.

### 성능

- 세 fold 행 수 가중 Brier 개선 `>= 0.00015`
- F3 개선 `>= 0.00010`
- 최악 fold 회귀 `<= 0.00005`
- 세 fold를 합친 투수 block bootstrap 개선량 95% 하한 `> 0`
- `game_type`, `game_month`, 투수·타자 OOV 세그먼트 최대 회귀 `<= 0.00050`

bootstrap cluster는 동일 선수의 시즌 간 상관을 보존하도록 `pitcher_id`로 묶고 seed
3407로 1,000회 계산한다. 세그먼트는 fold별과 전체를 모두 기록한다.

### 규칙·실행

- 입력과 출력 `row_id`가 정확히 일치하고 중복·누락·fallback이 없다.
- 출력 확률은 유한하고 `[1e-5, 1-1e-5]` 범위다.
- 단독 행, 역순, 셔플, 다른 배치 크기에서 같은 행 예측이 허용 오차 내 동일하다.
- 같은 행을 다른 평가 행과 함께 넣어도 예측이 같다.
- 평가 행 사이의 groupby, rolling, lag, 누적 상태가 없다.
- 공식 학습 행에서 target을 제거해 만든 245,789행 모의 입력 추론이 480초 이하다.
- peak RSS는 22GB 이하, peak GPU memory는 20GiB 이하다.

성능 등급은 `incremental`(`>=0.00015`), `competitive`(`>=0.00025`),
`breakthrough`(`>=0.00045`)로 별도 기록한다.

## 7. 전체 학습과 고정 전처리 상태

acceptance 이후에만 선택 구조를 공식 2019~2024 전체 학습 데이터로 seed 42, 2026,
3407 세 개 학습한다. seed별 iterations는 그 seed의 F1~F3 best iteration 중앙값에
1을 더한 값으로 고정하고 `[50, 400]`으로 제한한다. 전체 학습에는 평가 데이터와
Public 리더보드 결과를 사용하지 않는다.

delivery는 다음 상태를 명시적으로 저장한다.

- CatBoost `.cbm` 세 개와 파일별 SHA-256
- 피처 이름과 categorical 피처 이름의 정확한 순서
- 전체 학습 prior와 S1 투수·타자 snapshot
- 선택 구조가 `c2`일 때 cutoff 2024 TrackMan lookup과 mapping evidence
- anchor 공식과 clip 범위
- 선택된 TabM 결합 방식과 고정 비율
- 결합 시 기존 TabM 모델·전처리 파일과 원본 해시
- 학습 데이터, 계약, 코드, E1과 Stage C 계보 해시

S1과 TrackMan 상태는 평가 배치에서 다시 집계하지 않는 고정 lookup이다. 추론은 현재
행의 값과 저장된 lookup 조회만 사용한다.

## 8. 재시작과 자원 관리

T4 두 개에 동시에 job 하나씩만 배정한다. job은 독립 디렉터리에 원자적으로 결과를
기록하고, 같은 계약·코드·데이터·입력 해시가 맞을 때만 완료 job을 재사용한다.

- 벽시계 상한: 6시간
- 새 job 시작 중단: 종료 10분 전
- 안정 resume publication: job 종료 직후와 최대 10분 간격
- 브라우저 자동 다운로드: 없음
- Kaggle Output 최종 handoff: 하나

실패한 독립 job은 해당 후보만 실패시킨다. 기준 TabM, identity 검증, 선택 구조의
모든 seed 또는 artifact publication 실패는 캠페인을 fail-closed로 종료한다.

E1 실측 7분 45초와 기존 TabM 학습 증거를 기준으로 예상 시간은 45~90분이다. 이
수치는 보장이 아니며 계약상 안전 상한은 6시간이다.

## 9. 산출물

항상 하나의 최종 handoff를 쓴다.

```text
tree_expert_e2_handoff.zip
├── tree_expert_e2_review.zip
├── tree_expert_e2_resume.zip
├── tree_expert_e2_delivery.zip   # accepted일 때만 존재
└── tree_expert_e2.log
```

review에는 fold·seed·세그먼트·bootstrap·blend·자원·규칙 감사와 모든 결정 이유가
들어간다. resume에는 재개에 필요한 모델, snapshot과 상태가 들어간다. delivery는
accepted일 때만 생성되며 `review_only=false`, `submission_package=false`를 명시한다.

거절된 실행은 review와 resume만 포함하고 delivery 이름을 manifest에 남기지 않는다.
handoff와 내부 ZIP은 중복 경로, symlink, 위험 경로, 선언되지 않은 멤버와 해시 차이를
거부한다.

성공 로그는 다음 형식을 사용한다.

```text
TREE_E2_INPUTS_VERIFIED ...
TREE_E2_BASELINE_READY fold=2021->2022 ...
TREE_E2_JOB_START candidate=... fold=... seed=... gpu=...
TREE_E2_STRUCTURE_DECISION status=passed|rejected candidate=...
TREE_E2_SEED_DECISION status=passed|rejected ...
TREE_E2_BLEND_DECISION predictor=... ...
TREE_E2_ACCEPTANCE status=accepted|rejected reason=...
TREE_E2_FULL_FIT_START seed=...
TREE_E2_FULL_FIT_END seed=... model_sha256=...
TREE_E2_HANDOFF_READY path=<absolute-path> sha256=<sha256>
```

오류는 `TREE_EXPERT_ERROR stage=<stage> type=<type> message=<message>`로 끝난다.

## 10. 구현·검증 경계

Codex는 계약, 입력 검증, runner, artifact, Kaggle 한 셀 코드와 합성·소규모 회귀
테스트만 실행한다. 공식 데이터 전체 전처리, F1 TabM, GPU CatBoost 학습, 전체 학습,
추론 성능 측정과 handoff 생성은 사용자가 Kaggle에서 실행한다.

구현 완료 조건은 다음과 같다.

- E1 promoted ID와 모든 artifact/data hash가 fail-closed로 묶인다.
- 동적 phase 전이가 합성 fixture에서 gate별로 검증된다.
- rejected 상태에서는 full fit과 delivery writer를 호출할 수 없다.
- 재시작 시 완료 job만 재사용하고 active job snapshot을 이어갈 수 있다.
- 생성된 한 셀이 1MB 미만이고 내장 runtime만으로 import된다.
- E2 실행 목적, 입력, 예상 시간, 재실행과 반환 파일이 문서화된다.
- 기존 TabM, E1과 사용자 미커밋 파일은 수정하지 않는다.

## 11. 비목표

- `c0`, `c3` 재실험
- 새로운 DL 구조 탐색
- 1130점 제출물의 코드·모델·통계 복사
- Public 리더보드로 비율·seed·iterations 선택
- 테스트 행 관계를 사용하는 피처
- acceptance 이전 full fit 또는 delivery 생성
- 제출 ZIP 또는 자동 DACON 업로드
