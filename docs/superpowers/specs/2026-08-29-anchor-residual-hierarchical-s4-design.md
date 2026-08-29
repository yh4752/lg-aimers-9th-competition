# S4 공격형 Anchor·Residual·Hierarchical Calibration 설계

## 1. 목적

현재 규칙을 통과한 최고 제출은 Tree Expert E2이며 Public 점수는
`977.3809532715`다. E2 이후 단순 시간 가중, 기존 계층 보정, 실패 유형 전문가와
이종 트리 직접 보정은 각각의 검증 기준에서 탈락했다. 다음 실험은 모델 이름만 바꾸지
않고 아래 세 요소를 하나의 완성형 파이프라인으로 넓게 비교한다.

```text
seasonal/context anchor
→ residual correction
→ hierarchical calibration
```

이 캠페인의 목표는 각 요소의 단독 효과와 전체 결합 효과를 함께 측정하는 것이다.
Residual 단독 결과가 약하다는 이유만으로 calibration을 생략하지 않는다. 규칙 위반,
데이터 손상 또는 회복하기 어려운 대규모 회귀가 없는 대표 후보는 반드시 전체 결합까지
실행한다.

## 2. 기존 실험과의 차이

기존 `tree_hierarchical_residual_v1`의 C2도 E2 anchor, residual correction,
hierarchical calibration을 연결했다. C2는 calibration gap과 ECE를 개선했지만 C1 대비
incremental gain이 `-0.0000196808`이었고, 최악 fold와 segment가 악화돼 기각됐다.

S4는 그 구성을 반복하지 않는다.

- C0 하나만 쓰지 않고 최근 시즌과 감쇠 다중 시즌을 섞은 anchor를 넓게 비교한다.
- residual target을 각 anchor마다 다시 만든다.
- joint 모델과 R/F 전문가를 함께 본다.
- CatBoost뿐 아니라 XGBoost와 LightGBM도 바뀐 anchor와 피처에서 다시 평가한다.
- 전체 결합 후보를 최소 12개 남기고, 상위 구조는 다중 seed로 확인한다.
- 외부 1130점 사례에서 알려진 `recent 0.75 + multi 0.25 + decay 0.55`는 코드나
  저장 통계를 가져오지 않고 공식 학습 데이터에서 독립적으로 재현한다.

기존 C2와 S3의 기각은 해당 고정 구성에만 적용한다. 새 anchor와 새 residual target을
사용하는 S4까지 같은 결과라고 가정하지 않는다.

## 3. 실행 형태

사용자는 Kaggle Notebook에서 T4 GPU 2개를 선택하고 한 셀을 한 번 `Save Version`으로
실행한다. S4-A~D는 사용자가 나누어 실행하는 버전이 아니라 로그와 내부 상태를 위한
phase 이름이다.

```text
S4-A 입력·규칙 검사와 anchor 탐색
S4-B residual 전문가 학습
S4-C hierarchical calibration과 전체 결합 비교
S4-D 다중 seed 확인, 최종 판정과 handoff 생성
```

예상 벽시계 시간은 10~12시간이다. 두 GPU worker가 독립 학습 작업을 병렬 실행하고,
LightGBM 및 calibration 계산은 GPU 작업과 충돌하지 않는 범위에서 CPU를 사용한다.
실행 시간은 성능 판정 기준이 아니며, runtime guard는 Kaggle 종료 전에 상태와 최종
handoff를 안전하게 쓰기 위해서만 사용한다.

중간 수동 업로드와 다운로드는 없다. 작업 하나가 끝날 때마다 내부 상태를 원자적으로
갱신하지만 정상 실행에서는 마지막 `anchor_residual_hierarchical_handoff.zip` 하나만
사용자에게 전달한다.

## 4. 입력과 provenance

입력은 두 Kaggle Dataset으로 제한한다.

1. 공식 train·Trackman 데이터
2. 검증된 E2 OOF와 필요한 작은 계보 파일을 담은 S4 입력 번들

S4 입력 번들은 manifest, 파일 크기와 SHA-256을 고정한다. ZIP으로 연결되거나 Kaggle이
풀어 놓은 디렉터리로 연결된 경우를 모두 지원하되, 동일 artifact의 ZIP과 해제본이
동시에 발견돼도 논리적으로 하나만 선택한다. resume과 training input은 artifact kind로
구분하며 파일명만으로 분류하지 않는다.

외부 1130점 제출물의 모델, pickle, 학습 통계, OOF 또는 가중치는 입력에 포함하지
않는다. 공개된 구조 설명은 가설을 정하는 참고 자료로만 사용한다.

## 5. 시간 전이 검증

모든 후보는 다음 세 fold에서 같은 행과 같은 E2 기준 예측을 사용한다.

- 2021까지 학습 → 2022 검증
- 2022까지 학습 → 2023 검증
- 2023까지 학습 → 2024 검증

2022·2023 fold만 구조와 강도를 고르는 데 사용한다. 2024 fold는 후보 구조를 동결한
뒤 한 번 계산하며, 그 결과를 보고 decay, 혼합 비율, residual 강도나 calibration
강도를 다시 바꾸지 않는다. 2024 결과는 최종 통과 여부와 사전 tie-break에만 사용한다.

각 validation 행의 target은 해당 행의 anchor, residual 피처, calibration table을
만드는 데 쓰이지 않는다. Fold별 상태는 그 fold의 cutoff 이전 공식 학습 행만으로
fit한 뒤 고정한다.

## 6. S4-A: 공격형 anchor 탐색

### 6.1 최근·다중 시즌 모델

각 fold에서 다음 두 확률을 별도로 만든다.

- `p_recent`: cutoff 직전의 최근 시즌을 중심으로 학습한 모델
- `p_multi(d)`: cutoff 이전 모든 시즌을 사용하되, 오래된 시즌에
  `d ** season_age` 가중치를 준 모델

Anchor는 확률 공간의 고정 혼합으로 만든다.

```text
p_anchor = w_recent * p_recent + (1 - w_recent) * p_multi(d)
```

탐색 범위는 다음과 같다.

- `w_recent`: `0.60`, `0.75`, `0.90`
- decay `d`: `0.30`, `0.55`, `0.75`
- 고정 대조군: 기존 E2 확률

9개 시간 혼합과 E2 대조군을 기본 anchor로 둔다. `0.75/0.25, decay 0.55`는 외부
사례 재현 anchor로 표시하고 순위와 관계없이 전체 결합 확인 대상에 포함한다.

### 6.2 R/F와 context 변형

구조 fold에서 대표 시간 anchor 세 개에 대해 `game_type=R`과 `game_type=F` 전문가를
분리한 변형을 추가한다. 표본이 부족한 계층은 공통 모델로 backoff한다.

투수, 타자, 손잡이 matchup과 현재 투구 상황의 평활 통계는 cutoff 이전 학습 행에서만
만든다. 이 통계는 anchor 모델의 입력으로 사용하며, 평가 행 전체의 빈도나 평균을
사용하지 않는다.

기본 10개와 R/F 변형을 합쳐 약 10~13개의 anchor를 비교한다.

### 6.3 Coverage 선택

Residual 단계에는 최대 6개 anchor를 넘긴다. 단순 점수 상위 6개가 아니라 다음 역할을
보장한다.

1. E2 대조군
2. 외부 사례 재현 anchor
3. 구조 fold weighted Brier 최상위
4. 최악 fold가 가장 안정적인 anchor
5. E2와 오차 상관이 가장 낮은 anchor
6. 가장 좋은 R/F 전문가 anchor

역할이 중복되면 다음 순위 후보로 채운다. Anchor가 단독 기준을 조금 못 넘더라도 이
coverage 역할에 해당하면 residual과 calibration까지 진행한다.

## 7. S4-B: residual 전문가

각 anchor에 대해 target을 새로 만든다.

```text
residual_target = control_success - p_anchor
p_residual = clip(p_anchor + alpha * residual_model(x, p_anchor))
```

Residual model은 현재 행 피처, cutoff 이전에 고정한 통계와 `p_anchor`의 logit을
사용한다. Validation target이나 다른 validation 행의 예측은 입력에 넣지 않는다.

비교할 구조는 다음과 같다.

1. joint CatBoost residual
2. R/F 분리 CatBoost residual
3. XGBoost residual
4. LightGBM residual
5. 최근 시즌 residual과 다중 시즌 residual의 고정 혼합

Residual 강도 `alpha`는 `0.25`, `0.50`, `0.75`, `1.00`을 구조 fold에서 비교한다.
최적값이 `0.25` 또는 `1.00` 경계에 있으면 같은 실행 안에서 인접 확장값 하나를 추가해
경계 선택을 확인한다.

모델 용량은 smoke용 축소 설정을 사용하지 않는다. E2와 S3에서 실제 전체 fold를
처리한 깊이, tree 수와 early stopping 범위를 출발점으로 사용한다. OOM이나 단일 작업
실패는 해당 설정만 실패 처리하며 모델 계열 전체를 종료하지 않는다.

## 8. S4-C: hierarchical calibration과 전체 결합

Calibration은 `p_residual`에 학습 cutoff 이전의 오차 효과를 작게 더한다.

```text
logit(p_final) = logit(p_residual) + beta * hierarchical_effect(current_row)
```

효과표는 전역에서 세부 계층으로 내려가며 표본 수에 따라 상위 계층으로 backoff한다.

- global
- game type
- pitcher
- batter
- hand matchup과 허용된 현재 상황

Calibration 강도 `beta`는 `0.10`, `0.25`, `0.50`, `0.75`를 비교한다. Group count,
target mean과 residual mean은 fold training 행에서만 계산한다. Test나 validation 행끼리
통계를 공유하지 않는다.

전체 Cartesian product는 만들지 않는다. 대신 아래 archetype이 빠지지 않는 coverage
grid를 사용한다.

- E2 + residual + calibration
- 외부 사례 anchor + residual + calibration
- best overall anchor + CatBoost residual + calibration
- stable anchor + CatBoost residual + calibration
- diverse anchor + XGBoost residual + calibration
- diverse anchor + LightGBM residual + calibration
- R/F anchor + joint residual + calibration
- R/F anchor + R/F residual + calibration
- recent-heavy anchor 완성형
- multi-season-heavy anchor 완성형
- pitcher hierarchy 완성형
- batter/matchup hierarchy 완성형

최소 12개의 완성형 결과를 남긴다. 성능이 낮은 중간 요소도 누출, NaN, 확률 범위 위반
또는 구조 fold에서 `0.001`을 넘는 대규모 Brier 회귀가 없다면 대표 archetype 하나는
끝까지 계산한다.

## 9. S4-D: 다중 seed 확인과 판정

구조 fold에서 다음 Pareto 기준으로 완성형 상위 4개를 고른다.

- weighted Brier gain
- worst-fold gain
- calibration gap과 ECE
- 최대 segment regression
- E2와 residual correlation

외부 사례 anchor 전체 결합이 상위 4개에 없으면 확인 전용 다섯 번째 후보로 남긴다.
확인 후보는 seed `42`, `2026`, `3407`을 사용한다. Seed 3407 결과는 재사용하고 나머지
두 seed만 새로 학습한다.

최종 제출 적격 후보는 다음 기준을 모두 만족해야 한다.

- E2 대비 3-fold weighted Brier gain `>= 0.00005`
- 2023→2024 gain `>= 0`
- minimum fold gain `>= -0.00003`
- 5,000행 이상 사전 지정 segment의 최대 회귀 `<= 0.00030`
- pitcher-cluster bootstrap 95% gain 하한 `> 0`
- 세 seed 중 두 개 이상이 E2보다 나쁘지 않음
- calibration gap과 10-bin ECE가 pre-calibration 후보보다 모두 나쁘지 않음
- reverse, shuffle, rebatch와 singleton 행 독립성 최대 오차 `<= 1e-6`
- NaN, 무한대와 `[0, 1]` 밖의 확률이 없음

Residual correlation은 다양성 진단과 tie-break에 사용하되, 단독 완성형의 hard gate로
쓰지 않는다. 실제 Brier가 충분히 좋아지는 후보를 상관 하나만으로 버리지 않기
위해서다. 서로 다른 완성형을 blend할 때는 추가 gain과 오차 다양성을 별도 gate로
검사한다.

모든 hard gate를 통과한 후보가 여러 개면 3-fold weighted Brier가 가장 낮은 후보를
선택한다. 차이가 `0.00002` 이하라면 구조가 단순하고 모델 파일이 작은 후보를 고른다.
Public 점수는 구조, 강도나 tie-break를 다시 맞추는 데 사용하지 않는다.

## 10. 실행 예산과 우선순위

단일 Save Version의 목표 벽시계 시간은 10~12시간이다.

- 0~2시간: 입력 검증, fold 상태와 anchor 모델
- 2~7시간: residual 구조 작업을 두 GPU에서 병렬 실행
- 7~9시간: calibration coverage와 전체 결합 판정
- 9~11시간: 상위 구조 추가 seed 확인
- 마지막 1시간: acceptance, 선택 후보 full fit 가능 여부와 handoff 생성

새 학습 작업을 시작하지 않는 deadline guard와 artifact 생성용 reserve를 둔다. 시간이
부족하면 순위가 낮은 추가 seed나 blend를 생략하지만, 최소 12개 전체 결합 결과와 E2,
외부 사례 anchor의 비교는 먼저 끝내도록 job priority를 고정한다.

후보가 gate를 통과하고 시간이 남으면 공식 train만으로 full-fit model delivery를 만든다.
시간이 부족하면 `accepted_review_ready`로 종료하고 모델 학습은 다음 별도 실행으로
넘긴다. 어느 경우에도 S4 캠페인 자체는 DACON 제출 ZIP을 만들지 않는다.

## 11. 상태, 로그와 산출물

모든 로그는 한 줄 JSON 또는 고정 prefix로 남긴다.

```text
S4_CODE_READY
S4_INPUTS_VERIFIED
S4_GPU_READY
S4_PHASE_START phase=<anchor|residual|calibration|confirmation>
S4_JOB_START candidate=<id> gpu=<id>
S4_TRAINING_PROGRESS candidate=<id> ...
S4_JOB_END candidate=<id> status=<completed|failed>
S4_DECISION candidate=<id> status=<accepted|rejected|research_only>
S4_HANDOFF_READY path=/kaggle/working/anchor_residual_hierarchical_handoff.zip
S4_SUCCESS
```

내부 state에는 completed, failed, skipped job과 후보 판정을 기록한다. State 파일과
checkpoint는 임시 파일에 먼저 쓴 뒤 rename해 중간 쓰기 손상을 막는다.

최종 handoff에는 다음만 포함한다.

- campaign manifest와 bindings
- 전체 후보 요약표
- anchor, residual, calibration 및 full-chain decision
- 작은 aggregate diagnostics
- 필요한 OOF와 resume 상태
- 실행 로그
- 통과했을 때만 model delivery

원본 데이터와 평가 데이터는 넣지 않는다. 정상 실행에서는 중간 ZIP을 자동으로 여러
개 다운로드하지 않는다. 예외가 발생하면 `finally` 경로에서 현재 상태를 담은 emergency
handoff 하나만 `/kaggle/working`에 만든다.

## 12. 대회 규칙 안전선

- 모든 target 통계는 fold training cutoff 이전 행에서만 fit한다.
- 평가 행 전체의 평균, 빈도, 순위, rolling, lag와 누적 통계를 사용하지 않는다.
- 현재 행의 `game_type`, 투수, 타자, 손잡이와 투구 전 상황만 저장된 학습 상태에
  조회한다.
- R/F 전문가는 현재 행의 `game_type`으로 분기하며 다른 평가 행의 구성 비율을 보지
  않는다.
- 추론 순서와 batch 크기를 바꿔도 동일한 확률을 반환해야 한다.
- 외부 통신, 외부 모델과 평가 정답을 사용하지 않는다.
- 학습·검증 artifact의 SHA-256이 달라지면 acceptance와 model delivery를 만들지 않는다.
- 후보 상태, acceptance evidence와 현재 해시가 모두 통과하기 전에는 제출 패키징
  진입점을 호출할 수 없다.

## 13. 오류 처리

- 입력 후보가 없거나 여러 논리 artifact가 충돌하면 학습 전에 중단한다.
- 한 worker의 OOM이나 모델 오류는 해당 job만 실패 처리하고 다른 GPU 작업은 계속한다.
- 완료 checkpoint는 load 검증과 binding 검사를 통과해야 재사용한다.
- code 또는 contract SHA가 바뀐 resume은 명시적인 호환 목록이 없으면 거부한다.
- 최종 판정 직전에 전체 state와 decision schema를 다시 검증한다.
- 후보가 모두 탈락해도 `completed_no_candidate` review handoff를 정상 결과로 만든다.

## 14. 구현 및 검증 범위

Codex는 다음만 로컬에서 수행한다.

- 계약과 후보 생성 로직 작성
- 작은 합성 temporal fixture 테스트
- cutoff 누출, 행 독립성과 backoff 테스트
- anchor·residual·calibration 수식 단위 테스트
- dual-GPU scheduler와 deadline guard 테스트
- resume, manifest, hash와 ZIP/해제 디렉터리 탐색 테스트
- decision serialization과 emergency handoff 테스트
- Kaggle 한 셀 렌더링, Python 문법·import와 1MB 미만 source 검사

공식 전체 데이터의 anchor 탐색, GPU 학습, OOF 생성, full fit과 최종 handoff 평가는
사용자가 Kaggle에서 실행한다. Codex의 smoke 결과는 성능 근거로 사용하지 않는다.

## 15. 완료 기준

구현 완료는 학습 성공을 뜻하지 않는다. 다음 조건을 만족해야 사용자 실행 준비가 끝난다.

1. 합성 테스트와 저장소 계약 테스트가 통과한다.
2. 한 셀 코드가 공식 데이터와 S4 입력을 정확히 하나씩 찾는다.
3. 두 GPU가 서로 다른 job을 실행하고 job 상태가 중복 기록되지 않는다.
4. 최소 12개 full-chain archetype이 계획에 포함됐음을 정적으로 확인한다.
5. 기존 C2와 S3를 그대로 반복하지 않는 anchor·target binding이 확인된다.
6. 정상·실패 경로 모두 handoff 하나를 생성한다.
7. 제출 ZIP 생성 코드가 S4 runtime에 포함되지 않는다.

사용자 실행 후에는 최종 handoff의 manifest, decision, OOF 수치와 해시를 검증한다.
Submission-eligible 판정이 확인된 경우에만 별도의 제출물 설계를 시작한다.
