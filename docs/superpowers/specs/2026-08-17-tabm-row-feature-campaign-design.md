# TabM 행 단위 파생변수 캠페인 설계

## 배경과 목표

규칙을 지킨 첫 TabM 제출은 Public `872.3920184667`을 기록했다. Stage C의 seed
42·2026·3407 예측을 다시 확인한 결과, 단순 평균은 seed 3407 단독보다 시간 전이
OOF Brier가 `0.000318` 이상 나빠졌다. 따라서 추가 seed 전체 학습은 중단하고, 현재
단일 모델이 원본 열에서 바로 학습하기 어려운 행 내부 관계를 명시적으로 제공한다.

목표는 `TabM + dl_standard + hand_matchup + seed 3407` 기준선보다 재현 가능한
Brier 개선을 찾는 것이다. 모델 구조, optimizer와 기본 전처리는 바꾸지 않는다.
피처 묶음 하나의 효과를 먼저 분리하고, 통과한 묶음만 제한적으로 결합한다.

## 규칙 경계

2026년 8월 17일 다시 확인한 공식 공지는 평가 행 A의 예측이 다음 정보만 사용해야
한다고 명시한다.

- 행 A에 들어 있는 입력값
- 행 A의 값만으로 계산한 파생변수
- 공식 학습 데이터와 그 데이터에서 미리 고정한 상태

평가 자료 안의 다른 행을 이용한 빈도·순위·집계·rolling·lag·보정은 금지된다.
이번 캠페인의 여섯 묶음은 한 행의 값만 사용한다. 수치 결측 대체와 표준화 상태는
fold 학습 구간에서만 적합하고, 평가 시에는 그대로 적용한다. `control_success`는
모델 학습과 검증 지표 계산 외에는 피처 함수에 전달하지 않는다.

공식 근거는 다음 두 공지다.

- [평가 데이터 독립 예측 원칙 재안내](https://dacon.io/competitions/official/236743/talkboard/417123?page=1&dtype=recent)
- [대회 FAQ](https://dacon.io/competitions/official/236743/talkboard/417082?page=1&dtype=recent)

이 단계는 연구용 review와 resume ZIP만 만든다. 제출 ZIP, 평가 예측과 리더보드
제출은 만들지 않는다. 향후 제출 후보가 생기면 당일 공식 규칙을 다시 확인한다.

## 고정 기준선

- 모델: Stage C 최종 P2 TabM
- 수치 임베딩: `piecewise_linear`
- 손실: BCE
- scheduler: plateau
- learning rate: `0.0006`
- weight decay: `0.0001`
- effective batch size: `4096`
- micro batch size: `512`
- 전처리: `dl_standard + hand_matchup`
- 전체 검증 seed: `3407`

Proxy에서는 피처 효과와 seed 방향성을 함께 보기 위해 seed `42`, `3407`을 각각
사용한다. 각 seed의 후보는 같은 seed 기준선과 같은 표본, epoch 예산으로 짝비교한다.
동일 seed를 두 번 반복하는 방식은 seed 안정성을 측정하지 못하므로 사용하지 않는다.

## 피처 묶음

모든 범주 결합은 결측을 명시적인 `__MISSING__` 값으로 바꾼 뒤 `_`로 연결한다.
연속형 파생값은 원본 중 하나라도 결측이면 결측을 유지하고, 기존 fold-fit median과
표준화 경로에 맡긴다.

### 1. `count_context`

- `count_state`: `balls_before` × `strikes_before`
- `count_out_state`: `balls_before` × `strikes_before` × `outs_before`
- `base_out_state`: `base_state` × `outs_before`

볼·스트라이크·아웃과 주자 상태는 원본에도 있지만, TabM이 작은 이산 상태의 결합을
직접 다시 학습하지 않아도 되도록 낮은 cardinality 범주로 제공한다.

### 2. `pressure_context`

- `inning_bucket`: `1~3`, `4~6`, `7~9`, `10 이상`, 결측
- `pitcher_score_bucket`: 투수팀 점수 차 기준 `-4 이하`, `-3~-2`, `-1~1`,
  `2~3`, `4 이상`, 결측
- `leverage_bucket`: `li < 0.7`, `0.7 <= li < 1.5`, `li >= 1.5`, 결측
- `pressure_state`: 위 세 범주의 결합
- `pitcher_team_win_expectancy`: 초공격이면 home, 말공격이면 away 승리 기대값

경계는 실행 전에 고정한 야구 문맥 구간이다. 학습·검증·평가 분포를 보고 다시
조절하지 않는다.

### 3. `hand_state_interactions`

- `hand_count_state`: `hand_matchup` × `count_state`
- `hand_base_state`: `hand_matchup` × `base_state`
- `game_hand_matchup`: `game_type` × `hand_matchup`

기존 `hand_matchup`이 두 시간 fold와 모든 확인 segment에서 개선됐으므로, 그 효과가
카운트·주자·경기 유형에 따라 달라지는지를 별도 묶음으로 확인한다.

### 4. `pitcher_batter_gap`

- 투수 성공률 - 타자 성공률
- 투수 middle rate - 타자 middle rate
- `log1p(asof_pitcher_n) - log1p(asof_batter_n)`

횟수는 음수를 거부한 뒤 `log1p`를 적용한다. 이 묶음은 선수별 새 통계를 만들지
않고 공식 열에 이미 들어 있는 as-of 값의 행 내부 차이만 계산한다.

### 5. `recent_trend`

- 최근 1경기 성공률 - 최근 5경기 성공률
- 최근 3경기 성공률 - 최근 5경기 성공률
- 최근 1경기 성공률 - 투수 장기 성공률
- 최근 1경기 middle rate - 최근 5경기 middle rate
- 최근 3경기 middle rate - 최근 5경기 middle rate
- 최근 1경기 middle rate - 투수 장기 middle rate

최근 기록을 다시 rolling하지 않는다. 공식 데이터에 이미 제공된 as-of 1·3·5경기
열끼리만 뺀다.

### 6. `pitchmix_shape`

fastball·breaking·offspeed 세 비율을 한 행에서 음수가 없도록 검사한다. 관측된 세
값의 합이 0보다 클 때만 합이 1이 되도록 행 내부 정규화하고 다음을 계산한다.

- 최대 비율
- 최소 비율
- 가장 큰 비율과 두 번째 비율의 차이
- `-sum(p * log(p)) / log(3)` 정규화 엔트로피
- fastball-breaking, fastball-offspeed, breaking-offspeed 차이

한 값이라도 결측이거나 합이 0이면 이 묶음의 파생 연속값을 모두 결측으로 둔다.
다른 행의 구종 분포는 사용하지 않는다.

## 단계별 실행

### Stage P: 두 seed proxy

2023년까지의 학습 행에서 기존 결정론적 400,000행 표본을 만들고 2024년 전체
253,507행을 검증한다. seed 42와 3407마다 기준선 하나와 여섯 단일 묶음을 실행하므로
총 14개 작업이다.

- 최대 8 epoch
- 최소 3 epoch
- patience 3
- 후보별 epoch 체크포인트
- 한 Colab 세션 최대 10,800초
- 종료 900초 전부터 새 작업을 시작하지 않고 bundle을 만든다

각 후보의 proxy delta는 같은 seed 기준선 Brier를 뺀 값이다. 다음 조건을 만족하면
강한 생존 후보로 분류한다.

- 두 seed 평균 delta가 `-0.00003` 이하
- 어느 seed에서도 delta가 `+0.00005`보다 크지 않음

강한 후보는 개수 제한 없이 전체 OOF로 보낸다. 강한 후보가 놓칠 수 있는 작은 양의
방향을 보완하기 위해, 나머지 중 두 seed 평균 delta가 음수인 최상위 한 묶음도 안전
후보로 보낸다. 어떤 후보도 평균을 개선하지 못하면 피처 캠페인을 종료한다.

### Stage V: 전체 시간 전이 OOF

생존 후보를 seed 3407로 다음 두 fold에서 검증한다.

- `2022->2023`: 검증 245,525행
- `2023->2024`: 검증 253,507행

기준선은 해시가 고정된 기존 Stage C seed 3407 예측을 재사용한다. 후보는 기준선과
같은 최대 40 epoch, 최소 3 epoch, patience 10을 사용한다. 세션당 작업 수를 강제로
늘리지 않고 3시간 경계에서 끊어 여러 resume 세션으로 이어 간다.

재사용할 입력 identity는 다음 값과 모두 일치해야 한다.

- Stage C review SHA-256:
  `461c4422cebdc7383577ad6c53b044bfe3d9d15b667383e454591135e4242d2c`
- Stage C campaign config SHA-256:
  `5fd4845eeed60e311911e30fdff4090b511bf7485c6ee0540c6458a525b8e9c3`
- 공식 train CSV SHA-256:
  `d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff`

최종 승격 조건은 다음 세 가지를 모두 만족하는 것이다.

- 499,032행 가중 Brier 개선이 `0.00003` 이상
- 각 fold의 Brier 악화가 `0.00003` 이하
- 1,000행 이상 사전 정의 segment의 최대 Brier 악화가 `0.00050` 이하

Segment는 `game_type`, hand matchup, 투수·타자 ID known/OOV, `li >= 1.5`,
7회 이후, 주자 2루 또는 3루 존재 여부다. known/OOV 사전은 각 fold 학습 ID에서만
만들고 검증 행끼리 집계하지 않는다.

### Stage C: 제한된 결합

단일 묶음이 둘 이상 승격할 때만 수행한다. 가중 Brier가 가장 낮은 단일 묶음에서
시작하고, 승격 후보 중 상위 세 묶음만 결합 탐색에 참여한다. 남은 묶음을 하나씩
추가해 전체 두 fold를 다시 학습하고, 현재 조합보다 `0.00003` 이상 개선하면서 fold와
segment guardrail을 모두 지킨 추가만 채택한다. 최대 두 번 추가한 뒤 종료한다.

이 방식은 여섯 묶음의 모든 부분집합을 탐색하지 않으면서도 단일 피처 사이의 보완
효과를 확인한다. 결합 선택과 경계는 Public 점수를 보지 않고 OOF에서 닫는다.

## 산출물과 중단 복구

각 세션은 다음을 출력한다.

- 후보별 fold·segment Brier와 기준선 delta
- 후보별 예측 CSV와 SHA-256
- epoch 체크포인트와 optimizer/scheduler 상태
- 코드·계약·공식 입력·이전 resume 해시
- stage review ZIP
- 다음 세션용 resume ZIP
- 사람이 확인할 수 있는 진행 로그와 종료 요약

같은 identity의 완료 작업은 재사용한다. epoch가 끝난 작업은 다음 epoch부터 재개한다.
시간 제한에 걸린 작업은 `inconclusive`로 기록하고 기각으로 바꾸지 않는다. 입력이나
코드 해시가 다르면 기존 체크포인트를 재사용하지 않는다. ZIP은 임시 경로에서 검증을
마친 뒤 최종 이름으로 원자적으로 게시한다.

## 오류 처리

- 필수 원본 열 누락, 중복 출력 열과 알 수 없는 피처 묶음은 학습 전에 중단한다.
- 음수 count, 음수 pitchmix 비율과 비유한 파생값은 후보 실패로 기록한다.
- 범주 cardinality와 출력 열 순서가 fit 상태와 다르면 transform을 거부한다.
- 예측 `row_id` 누락·중복·정답 불일치와 `[0, 1]` 밖 확률은 판정을 막는다.
- GPU 미탐지, deadline과 checkpoint 불일치는 명시적인 stage 오류 로그를 남긴다.
- 일부 후보 실패는 그 후보만 막고 독립 후보와 정상 resume 생성을 막지 않는다.

## 검증 계획

Codex는 전체 자료 학습을 실행하지 않고 다음 fixture·정적 검사를 담당한다.

- 여섯 묶음의 정확한 열과 수치 계산
- 결측, 잘못된 count와 pitchmix 비율 거부
- fit/transform 열 순서와 상태 고정
- 단독 행·역순·shuffle·batch 분할에서 같은 파생값
- 평가 행 추가·삭제가 기존 행 파생값에 영향을 주지 않음
- 피처 transform 경로에서 target 열 접근 금지
- 두 seed proxy delta와 생존 규칙
- fold·segment gate와 결합 전진 선택
- checkpoint/resume identity와 deadline 종료
- review/resume ZIP 멤버와 SHA-256 검증
- 저장소 전체 pytest, compileall과 diff 검사

사용자는 Colab T4 한 장에서 공식 전체 자료 학습을 실행하고 review와 resume ZIP을
전달한다. Stage P는 한 세션 완료를 목표로 하지만 3시간을 넘기지 않는다. Stage V와
Stage C는 후보 수에 따라 여러 3시간 세션으로 나눈다.

## 성공과 중단 기준

- Stage P 생존 후보가 없으면 행 단위 피처 실험을 종료하고 CatBoost OOF blend로
  넘어간다.
- Stage V 승격 후보가 없으면 현재 seed 3407 제출 후보를 유지한다.
- 단일 또는 결합 후보가 최종 gate를 통과하면 별도 전체 학습 후보 설계를 연다.
- 연구 review 통과만으로 제출물을 만들거나 기존 제출 후보를 덮어쓰지 않는다.
