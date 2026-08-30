# 직접 타깃 전문가 앙상블 설계

## 1. 목적

현재 규칙 준수 Public 최고점인 Tree Expert E2의 `977.3809532715`를 기준선으로
유지하면서, 고용량 CatBoost 전문가가 `control_success`를 직접 예측하는 새 캠페인을
구축한다. 이번 캠페인은 E2 확률을 고정 anchor로 두고 작은 residual이나 calibration을
덧붙이는 기존 계열을 반복하지 않는다.

가용 예산은 Kaggle T4 GPU 두 장, 총 22시간이다. 실행은 최대 두 번의 장시간
Kaggle Version으로 제한한다. 첫 실행은 전문가 탐색, 두 번째 실행은 독립 확인과 최종
학습에 사용한다. Public 제출은 검증을 통과한 두 후보 이내로 제한한다.

점수 상승은 보장하지 않는다. 목표는 남은 시간 안에 기존 실험과 구별되는 충분한
용량의 직접 학습 모델을 만들고, 시간 전이와 행 독립성 증거를 갖춘 후보만 제출 가능한
상태로 승격하는 것이다.

## 2. 설계 근거

### 2.1 현재 기준선

- E2 Public: `977.3809532715`
- E2 3-fold weighted gain: `0.0008375137`
- E2 worst-fold gain: `0.0003439904`
- E2는 현재 규칙 준수 기준선이며 새 캠페인의 OOF와 배포 비교 대상이다.

### 2.2 S4 결과에서 확인한 한계

`anchor_residual_hierarchical_handoff (2).zip`의 SHA-256은
`5a410548de99d5d9c56f9b0d1940d5080e9167eafd97a2c94c45e31deacb4911`이다.
아카이브는 손상되지 않았고 캠페인은 `completed_no_candidate`로 정상 종료됐다.

S4는 15개 full-chain 구조와 5개 다중 시드 확인 후보를 평가했지만 모두 기각됐다.
가장 나았던 `c00`도 2024 fold에서는 `+0.0001542066` 개선됐으나 weighted gain은
`-0.0000235216`, minimum fold gain은 `-0.0001823839`였다. Residual 강도를 진단상
`0.25`에서 `0.10`으로 낮추고 calibration을 제거하면 weighted gain이 약
`+0.0000283170`까지 회복되지만 기존 최소 개선량 `+0.00005`에는 못 미친다.

따라서 문제는 트리 depth가 얕다는 데 있지 않다. 고정 anchor의 오차를 모든 연도에서
같은 방향으로 고치는 residual target이 안정적이지 않았고, 계층 calibration 효과도
연도마다 방향이 달랐다. 다음 캠페인은 직접 타깃 모델, 학습창 전문가와 제한된 OOF
스태킹으로 구조를 바꾼다.

## 3. 범위

### 포함

- 공식 `train.csv`와 `trackman_history.csv`만 사용
- cutoff-safe 시즌별 투수·타자 snapshot
- 현재 행의 공식 as-of 값과 학습 데이터로 고정한 snapshot의 차이
- R/F 경기 유형 전문가
- 고용량 CatBoost 직접 분류와 직접 회귀
- 시간 OOF 구조 선택, 3시드 확인, 제한된 스태킹
- Kaggle T4 x2 병렬 학습, resume, review, delivery 산출물
- 행 독립성, 배포 재현성, 파일 무결성 검사

### 제외

- 평가 데이터 다른 행이나 전체 분포에서 만든 통계
- 평가 행 사이 rolling, lag, 누적, 빈도, 순위, 그룹 집계
- 외부 데이터, 2025 TrackMan, 실명 복원 데이터
- 품질 gate를 통과하지 못한 middle/reverse/other-failure 복원 라벨
- exact-pitch TrackMan 결합
- S4 residual/calibration 조합의 단순 재실행
- 승인 전 submission ZIP 생성

## 4. 규칙 준수 원칙

평가 행 하나의 예측은 다음 정보에만 의존한다.

1. 해당 행의 입력 열
2. 해당 행만으로 계산한 파생변수
3. 공식 학습 데이터로 사전 고정한 snapshot과 모델
4. 구조 OOF에서 사전 고정한 모델 및 앙상블 상수

현재 행의 `game_type`, 투수, 타자, 손잡이, 카운트와 공식 as-of 값을 저장된 학습
상태에 조회하는 것은 허용한다. R/F 전문가는 현재 행의 `game_type`으로만 분기한다.
평가 데이터에서 R/F 비율, 선수 빈도, 평균, 순위를 다시 계산하지 않는다.

리더보드 점수는 제출 완료 후 사전에 생성한 두 후보 중 최종 후보를 선택하는 데만
사용할 수 있다. 점수를 본 뒤 개별 평가 행의 값을 생성하거나 평가 데이터 분포에 맞춰
보정하지 않는다.

## 5. 입력과 결합

공식 입력은 파일명이 아니라 내부 구성과 해시로 식별한다.

- `train.csv` SHA-256:
  `d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff`
- `trackman_history.csv` SHA-256:
  `f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9`
- E2 handoff SHA-256:
  `4dd0c9012b9a59e5235b76d6fe1f6ded4df4b404cab0082c0f3084b8c384050f`

Kaggle Dataset이 ZIP으로 보이거나 이미 풀린 디렉터리로 보이는 두 경우를 모두
지원한다. 최상위 폴더 이름은 식별 근거로 사용하지 않는다. 같은 artifact kind가 두 개
발견되면 임의 선택하지 않고 학습 시작 전에 실패한다.

실제 평가 `test.csv`의 행 수나 정렬은 학습 시 가정하지 않는다. 로컬의 5행 테스트
fixture는 실행 경로 확인용일 뿐 평가 데이터 분포나 크기를 대표하지 않는다.

## 6. 피처 상태

### 6.1 행 로컬 피처

- 이닝, 공수, 카운트, 아웃, 주자, 점수 차, leverage
- 투수·타자·팀 ID와 손잡이
- 공식 투수·타자 as-of 성공률과 관측 수
- 최근 1·3·5경기 성공률 및 middle rate
- 공식 pitch-mix 누적률
- 결측 여부와 관측 신뢰도

### 6.2 범주형 상호작용

- 투수 × 타자
- 투수 × 타자 손잡이
- 투수 × 카운트
- 투수 × 주자 상태
- 투수 × 경기 유형
- 타자 × 투수 손잡이
- 타자 × 카운트
- 팀 대진 × 경기 유형
- 카운트 × 주자 상태
- 카운트 × 이닝 구간 × 점수 구간
- leverage 구간 × 주자 수 × 카운트

상호작용 문자열은 한 행 안의 값으로만 만든다. CatBoost CTR은 학습 데이터에서 모델
학습 과정으로 고정하며 추론 평가 데이터에서 다시 적합하지 않는다.

### 6.3 시즌 snapshot과 현재 시즌 변화

각 validation year `Y`에 대해 snapshot은 `season < Y`인 공식 학습 행으로만 만든다.
투수·타자·팀의 이전 시즌과 다중 시즌 count/rate를 보관하고 다음을 파생한다.

- 현재 as-of 누적 수와 cutoff snapshot 수의 차이
- 성공·middle·reverse·ball·strike 누적 count의 차이
- 현재 시즌 추정률과 이전 시즌률의 차이
- 최근 경기률과 장기·시즌률의 차이
- 관측 수별 shrinkage rate
- 선수 신규·저표본·팀 변경 표시

학습 OOF에서도 동일한 cutoff 규칙을 사용한다. validation 또는 test의 다른 행을
snapshot에 추가하지 않는다.

### 6.4 TrackMan

공식 학습 기간 TrackMan에서 cutoff 이전의 투수·타자 단위 요약값만 사용한다. 평균
구속, 무브먼트, 구종 비율, 표본 수와 결측 상태를 고정 상태로 저장한다. exact-pitch
결합은 기존 감사에서 coverage가 지나치게 낮았으므로 사용하지 않는다.

## 7. 직접 전문가 8개

모든 전문가는 `control_success`를 직접 학습한다. E2 확률은 입력 anchor나 residual
target으로 사용하지 않는다.

| ID | 학습 데이터 | 목적함수 | 역할 |
|---|---|---|---|
| D0 | 전체 prefix, 동일 가중 | Logloss | 전역 기준 전문가 |
| D1 | 전체 prefix, 연도 감쇠 `0.75` | Logloss | 완만한 최근성 |
| D2 | 전체 prefix, 연도 감쇠 `0.55` | Logloss | 강한 최근성 |
| D3 | validation 직전 두 시즌 | Logloss | 최근 window |
| D4 | 전체 prefix, 연도 감쇠 `0.55` | RMSE | Brier 직접 최적화 다양성 |
| D5 | R 행, 연도 감쇠 `0.55` | Logloss | 정규리그 전문가 |
| D6 | F 행, 연도 감쇠 `0.55` | Logloss | 퓨처스리그 전문가 |
| D7 | 전체 prefix, 고차 상호작용 | Logloss | 선수·상황 CTR 전문가 |

D5와 D6는 자신의 경기 유형 행에서만 적용한다. 반대 경기 유형에는 전역 모델을
사용한다. 전문가와 전역 모델의 혼합 비율은 구조 fold에서만 결정하고 현재 평가
행의 `game_type`으로 적용한다.

## 8. 모델 용량

### 구조 탐색

- seed: `3407`
- iterations: 최대 `1800`
- depth: `9`
- learning rate: `0.03`
- `max_ctr_complexity=3`
- `border_count=128`
- early stopping patience: `150`
- GPU 한 장당 한 job

### 확인과 최종 학습

- confirmation seeds: `42`, `2026`, `3407`
- iterations: 최대 `2400`
- depth: 최대 `10`
- 최종 iteration: 확인 fold best iteration 중앙값을 기반으로 고정하되 `2400`을 넘지 않음
- 최종 배포: 상위 세 구조 × 세 시드, 최대 9개 모델

한 후보가 GPU 메모리 부족으로 실패해도 파라미터를 조용히 바꾸어 재시도하지 않는다.
해당 후보만 실패로 기록하고 독립 후보를 계속 실행한다. 실행 전 소형 GPU smoke test로
CatBoost 설치, CUDA 접근, 범주형 열과 기본 메모리 경로를 확인한다.

## 9. 시간 OOF와 선택

### 구조 fold

- `2021 -> 2022`
- `2022 -> 2023`

8개 전문가의 seed 3407 결과만 사용해 다음 역할의 네 구조를 선택한다.

1. weighted Brier gain 최고
2. 가장 최근 구조 fold인 2023 gain 최고
3. E2 및 최고 후보와 오차 상관이 가장 낮은 양호 후보
4. direct RMSE 또는 R/F/high-CTR 중 아직 포함되지 않은 구조적 wildcard

선택 레시피를 파일과 해시로 고정한 뒤에만 2024 확인 결과를 계산한다.

### 독립 확인 fold

- `2023 -> 2024`

선택된 네 구조에 seed 42와 2026을 추가한다. 한 시드만 개선된 후보는 최종 후보가 될
수 없다.

OOF 학습은 구조 탐색 16개, 선택 후보의 seed 3407 기반 2024 확인 4개, 선택 후보의
추가 두 seed와 세 fold 확인 24개로 최대 44개 job이다. 승인 후 full fit은 최대 9개
job이며 OOF 결과와 별도로 기록한다.

## 10. 제한된 스태킹

구조 fold의 OOF 예측만 사용해 다음 후보를 만든다.

1. 최대 네 전문가의 확률 공간 비음수 convex blend
2. 최대 네 전문가의 logit 공간 비음수 blend
3. direct champion과 E2의 안전 blend

가중치 합은 1이며 각 활성 가중치는 최소 `0.05`다. 구조 fold에서 deterministic
grid와 coordinate refinement로 고정한다. 2024 결과를 확인한 뒤 가중치를 다시
적합하지 않는다.

전문가가 E2와 사실상 같은 오차를 만들면 다음 중 하나를 충족해야만 앙상블에 남긴다.

- E2 대비 단독 weighted gain `>= 0.00005`
- 기존 blend에 추가했을 때 sequential gain `>= 0.00002`
- absolute residual correlation `< 0.995`

## 11. 사전 고정 승인 gate

### 안정형

- 3-fold weighted gain `>= 0.00005`
- 2024 gain `>= 0`
- minimum fold gain `>= -0.00003`
- 최대 R/F segment regression `<= 0.00030`
- 투수 cluster paired bootstrap 95% lower bound `>= 0`
- 세 시드 중 두 개 이상 weighted non-worse

### 공격형

- 2024 gain `>= 0.00015`
- fold weights `0.15 / 0.25 / 0.60`의 recent-heavy gain `>= 0.00008`
- minimum fold gain `>= -0.00025`
- 최대 R/F segment regression `<= 0.00100`
- 2024 투수 cluster bootstrap 95% lower bound `>= -0.00005`
- 세 시드 중 두 개 이상 2024에서 개선

안정형 또는 공격형 gate 전체를 통과해야 `accepted`가 된다. 일부 조건만 통과한
후보는 `research_only` 또는 `rejected`이며 delivery와 제출 패키지를 만들 수 없다.
두 gate는 S4 결과를 다시 승인하기 위한 사후 완화가 아니다. S4 c00은 공격형의
recent-heavy gain 조건을 통과하지 못한다.

## 12. 실행 A: 구조 탐색

- 환경: Kaggle T4 x2
- 예상 시간: 10~11시간
- 입력: 공식 데이터, E2 handoff, 봉인된 캠페인 코드
- 작업: 입력 검증, GPU smoke test, 8구조 × 2 structure folds × seed 3407
- 출력: `direct_expert_stage_A_handoff.zip`

완료 job만 원자적으로 state에 기록한다. 중간 resume은 `/kaggle/working`에서 같은
안정 경로로 교체하고 자동 다운로드하지 않는다. 최종 handoff 하나만 Dataset으로
승격해 실행 B에 전달한다.

## 13. 실행 B: 확인과 최종 학습

- 환경: Kaggle T4 x2
- 예상 시간: 8~9시간
- 입력: 공식 데이터, 실행 A handoff
- 작업: 2024 확인, 추가 시드, 스태킹, gate, 승인 후보 full fit
- 출력:
  - `direct_expert_review.zip`
  - `direct_expert_handoff.zip`
  - 승인 후보가 있을 때만 `direct_expert_delivery.zip`

`submission.zip`은 만들지 않는다. Delivery를 로컬에서 다시 검증하고 현재 artifact
해시가 계약과 일치할 때 별도 제출 패키징 작업을 시작한다.

## 14. 복구와 오류 처리

- ZIP과 풀린 Kaggle Dataset 디렉터리를 동일하게 지원한다.
- archive member 집합, manifest, member SHA-256을 추출 전에 확인한다.
- path traversal, symlink, 암호화 member, 중복 member를 거부한다.
- data/code/contract/E2 hash가 다른 resume은 거부한다.
- candidate 실패는 다른 candidate를 막지 않는다.
- 입력·규칙·artifact binding 실패는 전체 캠페인을 즉시 중단한다.
- 시간 reserve에 들어가면 새 job을 시작하지 않고 현재 완료 state를 handoff로 만든다.
- handoff status가 incomplete이면 실행 B는 완료된 job만 재사용하고 필요한 독립 job만
  이어서 수행한다.

## 15. 로그 계약

다음 이벤트를 시간, phase, candidate, fold, seed, GPU와 함께 한 줄 JSON 또는 고정된
key-value 형식으로 출력한다.

```text
DIRECT_EXPERT_CODE_READY
DIRECT_EXPERT_INPUTS_VERIFIED
DIRECT_EXPERT_GPU_READY count=2
DIRECT_EXPERT_SMOKE_SUCCESS
DIRECT_EXPERT_PHASE_START phase=<screening|confirmation|stacking|full_fit>
DIRECT_EXPERT_JOB_START candidate=... fold=... seed=... gpu=...
DIRECT_EXPERT_TRAINING_PROGRESS candidate=... iteration=... best_brier=...
DIRECT_EXPERT_JOB_END candidate=... status=<completed|failed>
DIRECT_EXPERT_DECISION candidate=... status=<accepted|rejected|research_only>
DIRECT_EXPERT_HANDOFF_READY path=...
```

## 16. 테스트와 제출 전 검사

Codex는 로컬에서 공식 전체 학습을 실행하지 않는다. 다음을 합성·fixture 데이터로
검사한다.

- cutoff 이후 행이 snapshot이나 CTR 사전 상태에 들어가지 않음
- 감쇠 가중치와 최근 window 경계
- R/F 전문가 라우팅과 전역 fallback
- convex/logit 가중치 제약
- 두 승인 gate의 경계값
- 실패 후보 격리와 resume 재개
- ZIP과 풀린 디렉터리 입력의 동등성
- manifest·member·binding 변조 거부
- 승인되지 않은 후보 패키징 거부

Delivery에는 다음 동적 검사 결과를 포함한다.

- singleton
- reverse order
- deterministic shuffle
- 여러 batch size
- companion row
- 동일 feature를 가진 별도 audit row
- 고정 표본 반복 추론

동일한 `row_id`의 확률은 모든 경우 허용 오차 `1e-6` 안에서 같아야 한다. 소스 정적
검사에서는 제출 추론 경로의 평가 프레임 groupby, rolling, rank, global
mean/frequency와 네트워크·외부 프로세스 사용을 확인한다. 학습 orchestration의 격리된
worker 프로세스는 이 금지 대상이 아니다. 최종 모델 수, feature 순서, 라이브러리
버전, 추론 시간, 메모리와 공식 ZIP 구조도 제출 패키지 생성 전에 다시 검증한다.

## 17. Public 제출 정책

승인된 경우에만 다음 두 후보를 제출 대상으로 유지한다.

1. 직접 전문가 champion
2. 직접 전문가와 E2의 안전 blend

하루 1~2회 제출 제한과 남은 3일을 고려해 두 후보를 같은 날 연속적으로 무작정
제출하지 않는다. 첫 후보의 Public 결과를 기록한 뒤 두 번째 후보가 사전에 생성되고
규칙·artifact gate를 모두 통과했을 때만 다음 제출 기회를 사용한다. Public 결과로
새로운 평가 행 통계나 행별 보정값을 만들지 않는다.

## 18. 완료 조건

캠페인 구현은 다음이 모두 충족돼야 완료다.

1. 두 Kaggle 실행 셀이 각각 한 셀로 재현된다.
2. 로컬 합성·fixture 테스트와 생성 셀 parity 검사가 통과한다.
3. 실행 A handoff가 독립 검증 가능하다.
4. 실행 B review가 모든 OOF·seed·segment·bootstrap 결과를 포함한다.
5. 승인 후보가 없으면 delivery와 submission이 생성되지 않는다.
6. 승인 후보가 있으면 delivery가 모델·상태·추론 코드·요구 버전과 해시를 봉인한다.
7. 사용자가 전달할 파일과 성공·오류 로그가 실행 안내에 명시된다.
