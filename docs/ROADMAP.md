# 대회 로드맵

## 운영 원칙

- 목표는 누출 없는 검증으로 확인한 성능 향상이다.
- 비용은 예상 시간과 실행 순서를 정하는 정보이며 비용만으로 후보를 제외하지 않는다.
- `rejected`와 `failed`는 해당 후보의 패키지만 막고 독립 후보를 막지 않는다.
- 모델 코드, Colab, 패키징은 한 번에 합치지 않고 각각 작은 이전 계획으로 검증한다.
- Public 결과는 확인 신호이지 사후 가중치 최적화의 정답으로 사용하지 않는다.

## 현재 완료

- R9 anchor와 세 acceptance evidence 고정
- FwFM standalone·F blend·F80 기각 및 계열 종료 감사
- TabM residual 기각
- Calibration 고정 세 변형 감사: exact variants rejected, family open
- XGBoost v3: exploratory accepted
- XGBoost original preprocessing rescue: verified ready
- XGBoost 공격적 구조·seed ensemble: Public `820.9583317093`
- 새 저장소의 실행 계약, 장부와 작은 판정 evidence 정리

## 다음 순서

### 1. R9 기반과 공통 검증 코드 이전

Preflight, 시간 전이 OOF 검증, 행 독립성, evidence와 package gate를 먼저 옮긴다.
새 저장소에서 작은 fixture가 통과하기 전에는 다른 후보 코드의 기준 경로를 바꾸지
않는다.

### 2. Calibration family 재설계

기존 세 변형의 평균 개선과 실패 gate를 진단 근거로만 사용한다. 다음 후보는 segment,
파라미터 범위와 수용 기준을 실행 전에 고정한다. 기존 결과를 본 뒤 탐색 grid를
연속적으로 넓히지 않는다.

### 3. XGBoost 코드와 Colab 이전

V3와 rescue의 체크포인트·해시 계약을 보존해 새 경로로 옮긴다. 두 실험은 검증
프로토콜이 다르므로 R9과 단순 Brier 순위를 만들지 않는다. 제출 여부는 package
gate와 현재 artifact를 다시 확인한 뒤 별도로 결정한다.

다음 CatBoost 혼합 검토는 두 모델의 정렬된 OOF가 같은 행·검증 프로토콜인지 먼저
확인하고, 혼합 비율과 grid를 실행 전에 고정한다. 이는 다음 검토 후보이며 현재
Public 점수에 맞춘 사후 가중치 선택은 하지 않는다.

### 4. 동료 저장소 연구 선별 검토

동료 저장소의 R25 TabM 잔차와 R32 예측 분모 보정은 장기 후보로 검토한다. 전체
코드를 복사하지 않고 다음 질문에 답할 수 있는 최소 기능만 가져온다.

- 우리 R9·데이터 스키마와 같은가?
- 검증 행 독립성과 시간 cutoff를 재현할 수 있는가?
- 기존 실패한 TabM residual과 목표·입력·결합 방식이 실질적으로 다른가?
- 사전에 고정한 여러 시즌 전이에서 방향이 유지되는가?

출처는 `castle9612/lg_aimers_9th`의 관련 README와 round 보고서에 연결한다.

## 보류

- 전체 과거 코드와 모든 Colab의 일괄 이전
- 자동 제출과 제출 ZIP 대량 아카이브 이전
- 실험 웹 대시보드 또는 자동 문서 생성기
- 검증되지 않은 외부 모델을 비용만 보고 제외하거나 최신성만 보고 우선하는 결정
