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

## 다음 주력 순서

### 1. 독립 DL 탐색

현재 독립 DL의 첫 작업은 기존 TabM seed OOF 앙상블 감사다.

Stage C의 seed 42, 2026, 3407 두 시간 fold 예측으로 단순 평균의 이득을 먼저
계산한다. 새 모델을 학습하지 않으며, 가중 평균 Brier `0.00003` 이상 개선과 fold
악화 `0.00003` 이하를 동시에 만족할 때만 추가 seed를 전체 학습한다.

그다음 `count_context`, `pressure_context`, `recent_trend`, `pitchmix_shape`를 각각
단일 ablation으로 비교한다. 평가 행 집계나 순서를 사용하지 않고, fold-fit 상태와
같은 행 값만 사용한다. 승급 후보만 결합해 다시 확인한다.

- 독립 DL 캠페인 코드: `code_ready`
- Colab Pro·Drive 실행: `waiting_for_user_run`
- 기존 TabM `raw_typed` P1 Brier `0.248291636714`, P2 Brier
  `0.248141183204`는 보존하고 재실행하지 않는다.
- TabICLv2는 정확한 대회 사용 가능성을 확인하기 전까지 `research_only`다.

### 2. DL 내부 앙상블

OOF 감사를 통과한 유망 후보만 기존 단일 모델과 분리된 Version E로 전체 학습한다.
추가 seed, 모델 member와 고정 가중치를 별도 manifest에 기록하고 기존 Version D
제출물은 수정하지 않는다.

### 3. 정렬된 OOF 기반 ML+DL 앙상블

규칙을 지키는 CatBoost를 TabM과 동일한 행, 동일한 두 시간 fold로 다시 검증한다.
TabM 비중 `0.70`, `0.80`, `0.90`만 비교하고 단독 점수뿐 아니라 오차 다양성과
앙상블 기여도를 함께 평가한다.

### 4. OOF 기반 보정

앙상블과 파생변수를 먼저 확정한 뒤 전역 shrinkage 한 종류만 검토한다. 보정 계수는
OOF에서 고정하며 평가 데이터의 평균 확률이나 성공률에 맞추지 않는다.

### 5. DL 계열 전체 장기 탐색

FT-Transformer와 TabNet sentinel은 현재 비교에서 TabM보다 낮았다. TabR,
TabICLv2와 다른 대형 구조는 새로운 근거나 자원이 생길 때 별도 캠페인으로 다룬다.
현재 seed 앙상블, 파생변수와 규칙 준수 blend를 지연시키지 않는다.

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
