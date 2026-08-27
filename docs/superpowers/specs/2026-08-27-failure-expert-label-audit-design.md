# 실패 유형 전문가 라벨 감사 설계

## 1. 목적

기존 977점 CatBoost 모델을 기준선으로 유지하면서 `middle`, `reverse`,
`other_failure` 전문가를 학습할 근거가 있는지 먼저 확인한다. 이번 단계는 공식
학습 데이터에서 실패 유형 라벨을 시간 순서대로 복원하고 신뢰도와 표본 수만
감사한다. 모델 학습, 평가 데이터 추론, 제출 파일 생성은 하지 않는다.

과거 `c3_failure_aware`는 네 클래스를 한 모델에서 동시에 다뤘고, 하나의 클래스나
전체 coverage가 gate를 통과하지 못하면 후보 전체가 생략됐다. 이번 감사는 같은
라벨 복원 원칙을 유지하되 유형별 one-vs-rest 가능성을 독립적으로 판정한다. 한
유형의 표본 부족이 다른 유형의 가능성까지 막지 않도록 하기 위함이다.

`wild`는 공식 학습 데이터에서 직접 복원할 수 있는 누적률 열이 없다. 따라서 이를
추정하거나 다른 실패를 `wild`로 부르지 않고, 남은 실패는 의미를 확장하지 않은
`other_failure`로 기록한다.

## 2. 범위와 실행 조건

- 실행 환경: Kaggle CPU
- 입력: 공식 `lg-aimers-9th-data` 하나
- 예상 시간: 10~30분
- 네트워크: 필요 없음
- 출력: `failure_expert_label_audit_review.zip` 하나
- 재실행: 입력과 코드가 같으면 처음부터 결정론적으로 다시 실행
- 제외: GPU 학습, full fit, 평가 데이터, submission ZIP, Public 점수 기반 선택

실행 코드는 한 셀로 제공하며, 공식 `train.csv`와 `trackman_history.csv`의 고정
SHA-256을 모두 확인한다. TrackMan 파일은 데이터셋 identity 확인에만 사용하고
라벨 복원에는 사용하지 않는다.

## 3. 시간 cutoff

다음 네 prefix를 서로 독립적으로 감사한다.

| 감사 ID | 사용 가능한 학습 시즌 | 후속 검증 용도 |
|---|---|---|
| A1 | 2019~2021 | 2022 시간 전이 |
| A2 | 2019~2022 | 2023 시간 전이 |
| A3 | 2019~2023 | 2024 시간 전이 |
| A4 | 2019~2024 | 조건부 전체 학습 가능성 |

각 prefix는 cutoff 이후 행을 읽지 않는다. A1~A3의 판정이 모두 통과한 유형만 후속
OOF 전문가 후보가 될 수 있으며, A4도 통과해야 최종 전체 학습을 허용한다.

## 4. 라벨 복원

각 prefix에서 같은 투수의 행을 `asof_pitcher_n`과 원래 행 위치로 안정 정렬한다.
현재 행과 다음 행이 모두 유일하고 다음 `asof_pitcher_n`이 정확히 1 증가할 때만
연결한다. 현재 누적 건수 `n`, 현재 누적률 `r`, 다음 누적률 `r_next`에서 다음과
같이 한 투구의 변화량을 계산한다.

```text
delta = (n + 1) * r_next - n * r
```

`success`, `middle`, `reverse` 변화량이 각각 0 또는 1에서 고정 허용오차 `0.02`
이내일 때만 이진 변화량으로 인정한다. 허용오차는 감사 결과를 보고 바꾸지 않는다.
복원한 success는 현재 행의 공식 `control_success`와 일치해야 한다.

라벨은 다음과 같이 정한다.

- `success`: 공식 성공이며 middle과 reverse 변화가 모두 0
- `middle`: 공식 실패이며 middle만 1
- `reverse`: 공식 실패이며 reverse만 1
- `other_failure`: 공식 실패이며 middle과 reverse가 모두 0

middle과 reverse가 동시에 1인 행, 공식 성공인데 실패 변화가 잡힌 행, 누적 건수가
건너뛴 행, 중복 누적 건수와 비이진 변화량은 학습 가능 행에서 제외하고 사유별로
집계한다.

## 5. 감사 지표와 고정 gate

각 prefix는 다음 공통 품질 gate를 모두 통과해야 한다.

- 전체 원본 행 대비 최종 라벨 coverage `>= 0.98`
- 연결 행의 세 변화량이 모두 이진으로 복원되는 비율 `>= 0.999`
- 복원 success와 공식 `control_success` 일치율 `>= 0.999`
- 실패 행 중 middle/reverse 동시 양성률 `<= 0.001`

공통 품질 gate가 실패하면 해당 prefix의 모든 유형을 `ineligible`로 판정한다. 공통
gate가 통과한 뒤 각 유형을 독립 판정한다.

- 양성 행 수 `>= 5,000`
- 음성 행 수 `>= 5,000`
- A1~A4 모든 prefix에서 위 두 기준 충족

`middle`, `reverse`, `other_failure` 중 조건을 만족한 유형만 후속 one-vs-rest
전문가 후보로 허용한다. 어떤 유형도 통과하지 못하면 실패 유형 전문가 계열을
종료한다. 이 단계에서는 gate를 완화하거나 특정 시즌만 제외하지 않는다.

## 6. 진단 산출물

review ZIP에는 다음 파일만 포함한다.

- `audit_summary.json`: 전체 상태, 유형별 eligibility와 근거
- `cutoffs/A1.json`~`A4.json`: 공통 지표, 클래스 수, 제외 사유 수
- `class_counts.csv`: cutoff·시즌·유형별 행 수
- `exclusion_counts.csv`: cutoff·시즌·제외 사유별 행 수
- `audit.log`: 단계별 시작·완료 로그
- `manifest.json`: 코드, 계약, 공식 데이터와 모든 멤버의 SHA-256 및 크기

행 단위 라벨이나 원본 데이터는 ZIP에 넣지 않는다. 감사 결과만으로 원본 학습 행을
복원할 수 없도록 집계값만 저장한다.

## 7. 로그와 오류 처리

한 셀은 다음 안정 prefix를 출력한다.

```text
FAIL_AUDIT_CODE_READY
FAIL_AUDIT_DATA_VERIFIED
FAIL_AUDIT_CUTOFF_START cutoff=A1
FAIL_AUDIT_CUTOFF_RESULT cutoff=A1 status=<passed|failed>
FAIL_AUDIT_DECISION type=<middle|reverse|other_failure> status=<eligible|ineligible>
FAIL_AUDIT_SUCCESS review=<path>
FAIL_AUDIT_ERROR stage=<stage> type=<type> message=<message>
```

입력 해시, 스키마, cutoff 경계, 행 identity 또는 산출물 검증이 실패하면 review를
성공으로 표시하지 않는다. 임시 ZIP은 최종 검증 후 원자적으로 교체한다.

## 8. 대회 규칙 경계

- 공식 학습 데이터만 사용한다.
- 평가 데이터와 sample submission을 읽지 않는다.
- 라벨은 각 cutoff 안의 과거 학습 행에서만 복원한다.
- 평가 행 간 집계, 순위, 빈도, rolling과 추론 순서 의존성은 없다.
- Public 리더보드 점수를 gate나 라벨 정의에 사용하지 않는다.
- 이번 산출물은 review-only이며 모델과 제출 진입점을 포함하지 않는다.

## 9. 코드 구조와 검증

새 코드는 기존 E1/RF 실행기와 분리한다.

- `failure_audit_contract.json`: cutoff, tolerance와 gate
- `failure_audit.py`: 라벨 복원 및 집계
- `failure_audit_artifacts.py`: review ZIP과 manifest 검증
- `failure_audit_kaggle.py`: 런타임 봉인과 입력 탐색
- `KAGGLE_FAILURE_AUDIT_CELL.py`: 생성된 복사용 한 셀
- `prepare` 단계는 만들지 않는다. 공식 데이터셋만 직접 읽는다.

합성 데이터 테스트로 성공·middle·reverse·other_failure, 중복, 건너뜀, overlap,
비이진 변화량과 cutoff 누출 차단을 검증한다. 생성 셀은 결정론, 1MB 제한, 문법,
공식 해시 확인과 review-only 보장을 검사한다. 기존 failure label, E1, E2와 RF
테스트도 회귀 실행한다.

## 10. 완료 조건과 다음 단계

감사 성공은 모델 성능 개선을 뜻하지 않는다. 최소 한 유형이 A1~A4에서
`eligible`이어야만 별도의 실패 전문가 OOF 설계를 시작할 수 있다. 다음 설계에서는
통과한 유형만 고정하고, 기존 977점 기준선에 `0.05`, `0.10`, `0.15`의 작은 보정만
시간 전이 Brier로 평가한다. 모든 유형이 `ineligible`이면 이 계열은 종료하고 이종
잔차 앙상블로 이동한다.
