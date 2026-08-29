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
- XGBoost 공격적 구조·seed ensemble: Public `820.9583317093`, 평가 분포 평균 이동
  보정 때문에 현재 규칙 재검토 전 패키지 차단
- 새 저장소의 실행 계약, 장부와 작은 판정 evidence 정리
- 규칙 준수 TabM 단일 모델: Public `872.3920184667`, 제출·추론 gate 통과
- Tree Expert E2: 세 시간 fold와 행 독립성 감사를 통과, Public
  `977.3809532715`
- E2 이후 T3 시간 가중, 계층 잔차, 실패 유형 라벨 후보 기각
- XGBoost·LightGBM 이종 트리 S3: 여섯 작업 완료, 두 직접 잔차 후보 기각

## 다음 주력 순서

### 1. 독립 DL 탐색

TabM은 유효한 독립 축으로 남긴다. 다만 더 긴 학습과 단순 seed 평균은 이미 이득이
없었다. 다음 독립 DL은 같은 모델을 늘리기보다 `count_context`,
`pressure_context`, `recent_trend`, `pitchmix_shape`처럼 같은 행과 학습 구간에서만
만든 정보를 하나씩 비교한다. TabR·TabICLv2 같은 새 구조는 대회 사용 가능성과
실행 자원을 먼저 확인한 뒤 별도 캠페인으로 다룬다.

### 2. DL 내부 앙상블

시간 전이 OOF에서 단일 후보보다 나은 유망 후보만 DL 앙상블 대상으로 올린다. 추가 seed와
모델 이름을 늘리는 것이 아니라 오차 상관이 실제로 낮은지 확인하고, 고정 가중치와
구성원을 manifest에 남긴다.

### 3. 정렬된 OOF 기반 ML+DL 앙상블

독립 DL을 같은 행과 같은 시간 fold에서 CatBoost E2와 다시 비교한다. 단독 Brier뿐
아니라 오차 다양성과 혼합의 추가 gain을 함께 평가한다. 검증 때와 배포 때의 모델
상태가 다르면 이전 CatBoost·TabM 혼합처럼 제출을 차단한다.

### 4. 시즌·상황 정보 적격성 감사

E2와 같은 입력으로 모델 이름만 바꾼 S3는 오차 상관이 `0.998`을 넘었다. 다음에는
모델보다 입력의 차이를 먼저 만든다. 시즌별 선수 스냅샷, 경기 유형과 현재 투구 상황을
공식 학습 자료에서 시간 cutoff를 지켜 만들 수 있는지 확인한다. 평가 데이터 전체의
평균·빈도·순서에 의존하는 값은 후보에서 제외한다.

### 5. Anchor와 잔차 보정 재설계

적격성 감사를 통과한 정보만 E2 anchor에 하나씩 더한다. 기존 계층 잔차 후보 C1·C2와
같은 정의를 반복하지 않고, 어떤 정보가 새 오차를 만드는지 단일 ablation으로 먼저
확인한다. 평균 gain뿐 아니라 최악 시즌, 구간 회귀와 기존 E2 잔차 상관을 함께 본다.

### 6. Calibration과 전문가 분기

새로운 OOF 신호가 확인된 뒤에만 계층 calibration이나 경기 유형별 전문가를 붙인다.
보정 계수와 분기 규칙은 시간 전이 OOF에서 고정하며 Public 결과로 다시 맞추지 않는다.
실패 유형 라벨은 현재 정의가 부적합하므로 같은 라벨로 모델부터 다시 학습하지 않는다.

### 7. 제출 후보 확인

세 시간 fold, bootstrap, seed 안정성, 행 순서·배치 독립성과 전체 추론 리허설을 모두
통과한 후보만 E2와 비교한다. acceptance와 현재 모델·코드 해시가 일치하기 전에는
제출 ZIP을 만들지 않는다.


## 보조 트랙

- R9 preflight·시간 전이 OOF 검증·행 독립성·package gate 이전
- 열린 calibration family 재설계
- XGBoost 체크포인트·해시 계약과 Colab 이전
- 동료 저장소 연구의 출처·누출·행 독립성 검토 후 선별 이식

보조 트랙은 독립 DL 준비나 사용자 GPU 실행을 지연시키지 않는 범위에서 진행한다.

모든 트랙은 `competition_rules` 정책과 후보별 `experiment_contract.json`을 먼저
통과한다. 과거 XGBoost ZIP과 평균 이동 후처리는 새 후보의 성능 참고 자료일 뿐
현재 제출 후보나 ensemble 구성요소로 재사용하지 않는다.

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
