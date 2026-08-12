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

## 다음 주력 순서

### 1. 독립 DL 탐색

트리 예측에 의존하지 않는 독립 DL 구조와 넓은 용량·최적화 범위를 먼저 탐색한다.
후보 배열 순서로 64개를 소진하지 않고, 전체 데이터의 큰 후보를 입력 표현과 모델
계열 사이에 교차 배치한다. 비용과 시간은 순서 안내에만 사용하며 후보 크기를
제한하지 않는다.

- 독립 DL 캠페인 코드: `code_ready`
- Colab Pro·Drive 실행: `waiting_for_user_run`
- 기존 TabM `raw_typed` P1 Brier `0.248291636714`, P2 Brier
  `0.248141183204`는 보존하고 재실행하지 않는다.
- 다음 실행은 TabM P2의 다른 세 입력 표현, TabM P3, 대형 ResNet·FT-Transformer,
  TabICLv2 순서다.
- TabR는 fold-local key cache를 재사용하는 확장성 수정 후 실행 순서에 들어간다.
- TabICLv2는 정확한 대회 사용 가능성을 확인하기 전까지 `research_only`다.
- 개정 캠페인 결과가 돌아오기 전에는 독립 DL 계열의 성능 결론을 기록하지 않는다.

### 2. 유망 후보 심화와 DL 내부 앙상블

초기 결과가 유망한 후보에 충분한 학습 시간, 여러 seed와 필요한 체크포인트를
적용한다. 단독 성능과 fold 안정성을 확인하고 상위 구조·seed의 DL 내부 앙상블을
평가한다.

### 3. 정렬된 OOF 기반 ML+DL 앙상블

살아남은 DL 후보와 CatBoost·XGBoost의 동일 행·동일 검증 프로토콜 OOF를
정렬한다. 단독 점수뿐 아니라 오차 다양성과 앙상블 기여도를 함께 평가한다.

## 보조 트랙

- R9 preflight·시간 전이 OOF 검증·행 독립성·package gate 이전
- 열린 calibration family 재설계
- XGBoost 체크포인트·해시 계약과 Colab 이전
- 동료 저장소 연구의 출처·누출·행 독립성 검토 후 선별 이식

보조 트랙은 독립 DL 준비나 사용자 GPU 실행을 지연시키지 않는 범위에서 진행한다.

Calibration은 기존 세 변형의 평균 개선과 실패 gate를 진단 근거로 사용하고, 다음
후보의 segment·범위·수용 기준은 실행 전에 고정한다. XGBoost v3와 rescue는
검증 프로토콜이 다르므로 R9과 단순 Brier 순위를 만들지 않고 체크포인트·해시
계약을 보존한다.

동료 저장소의 R25 TabM 잔차와 R32 예측 분모 보정은 장기 후보로 검토한다. 전체
코드를 복사하지 않고 우리 스키마와의 일치, 시간 cutoff와 행 독립성, 기존 실패와의
실질적 차이, 여러 시즌 전이에서의 방향성을 확인한 기능만 출처와 함께 선별 이식한다.

## 보류

- 전체 과거 코드와 모든 Colab의 일괄 이전
- 자동 제출과 제출 ZIP 대량 아카이브 이전
- 실험 웹 대시보드 또는 자동 문서 생성기
- 검증되지 않은 외부 모델을 비용만 보고 제외하거나 최신성만 보고 우선하는 결정
