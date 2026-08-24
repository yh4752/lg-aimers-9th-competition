# Temporal T2-C 재현성 확인과 최종 제출 모델 설계

## 1. 목적과 종료 조건

T2-B에서 `S1`은 세 temporal fold 모두 전체 Brier를 개선했지만, 2023년
`game_type=F` 구간에서 큰 퇴행이 발생해 사전 정의한 세그먼트 기준을 통과하지
못했다. T2-C는 `game_type=F` 행에서 고정 T1 anchor로 되돌리는 안전 게이트가
새 seed에서도 재현되는지 확인한다.

이 연구는 다음 두 결과 중 하나로 끝난다.

- T2-C 통과: 고정된 3-seed 최종 모델을 학습하고 제출 산출물을 검증한다.
- T2-C 실패: S1 계열 연구를 중단하며 다른 게이트나 가중치를 추가 탐색하지 않는다.

T2-C 이후에는 같은 OOF를 이용한 추가 피처·가중치 탐색을 하지 않는다. 전체 잔여
GPU 예산은 검증과 최종 학습을 합쳐 5~9시간으로 제한한다.

## 2. 고정 근거와 계보

- T2-B 입력 SHA-256:
  `9f5e9a67f98979aeb7fc199ff5d330eb7e53057fc57babfc61ba1ddbccc93891`
- T2-B handoff SHA-256:
  `86b118b02ad0d48ad147499c7bcc1a6d9af28f82701d1964de13b659a73c7d32`
- T2-B review SHA-256:
  `ccaf8c7a3aa31e968358045ef45df7d06f3fd466289197050791370d3131248b`
- T2-B resume SHA-256:
  `02c33b25c36b95c7315619e94e106be04b463422adcfb4232210e9d2411b0076`
- T2-B 상태: 완료 10개, 대기 0개, 실패 0개
- 고정 T1 설정: multi decay `0.55`, recent weight `0.50`, logit blend,
  anchor beta `0`

원래 T2-B 결정은 `promoted=[]`이며 이를 바꾸지 않는다. 안전 게이트 후보는 T2-B
결과를 보고 정의한 새 후보이므로 `posthoc_candidate`로 기록하고 T2-C의 독립 seed
확인을 통과해야만 최종 학습 권한을 얻는다.

## 3. T2-C 후보

후보 ID는 `s1_game_type_f_fallback_v1`로 고정한다.

- recent expert: `base+S1`, TabM p2
- multi expert: T1의 고정 `base` multi 예측
- non-F 행: recent S1와 base multi를 `0.50:0.50` logit 결합
- `game_type=F` 행: T1 anchor 사용
- 모델 설정: T2-B와 동일
- 신규 seed: `42`, `2026`
- 기존 참고 seed: `3407`
- temporal folds: `2021→2022`, `2022→2023`, `2023→2024`

게이트는 문자열 정규화나 전체 평가 집합 통계를 사용하지 않는다. 학습 때 확정된
스키마에 따라 각 행의 `game_type` 값이 정확히 `F`일 때만 anchor로 전환한다. 결측,
미등록 값, 다른 값은 non-F 경로를 사용한다.

## 4. T2-C 학습 구성

새 작업은 2개 seed와 3개 fold의 곱인 정확히 6개다.

| seed | 학습 시즌 | 검증 시즌 |
|---:|---:|---:|
| 42 | 2021 | 2022 |
| 42 | 2022 | 2023 |
| 42 | 2023 | 2024 |
| 2026 | 2021 | 2022 |
| 2026 | 2022 | 2023 |
| 2026 | 2023 | 2024 |

각 학습은 검증 연도 직전 한 시즌만 사용한다. S1 fitting 문맥은 검증 연도보다 이전인
공식 train 행으로 제한한다. history와 ID 매핑도 같은 cutoff를 지킨다. validation
target은 전처리, 모델 fitting, 게이트 생성에 사용하지 않는다.

## 5. 판정

각 seed의 세 fold를 먼저 평가하고, 세 seed의 recent S1 logit을 평균한 ensemble도
별도로 평가한다. ensemble은 recent expert 내부에서만 seed를 결합한 뒤 고정 base
multi와 `0.50:0.50`으로 결합한다.

T2-C는 다음 조건을 모두 만족해야 통과한다.

- 새 seed `42`, `2026` 각각의 행 수 가중 Brier 개선이 `0` 초과
- 3-seed ensemble의 행 수 가중 개선이 `0.00005` 이상
- ensemble의 2024 개선이 `0.00003` 이상
- ensemble의 3-fold pitcher-block bootstrap 하한이 `0` 초과
- ensemble의 최대 eligible segment regression이 `0.00050` 이하
- ensemble의 어느 fold도 Brier 퇴행이 `0.00003`을 초과하지 않음
- 모든 작업과 모든 행의 증거가 완전함

통과와 실패 외에 예산 때문에 작업이 덜 끝난 경우는 `budget_inconclusive`로 기록하며
실패로 간주하지 않는다. 불완전 상태에서는 최종 학습을 시작하지 않는다.

## 6. 실행과 복구

- 환경: Kaggle Tesla T4 두 장
- T2-C 벽시계 제한: 2시간
- 종료 25분 전부터 새 작업을 시작하지 않음
- 마지막 10분은 artifact 생성에 예약
- 두 GPU에 최대 두 작업을 병렬 배치
- 완료 작업은 semantic training identity 확인 후 재사용
- 진행 중 작업은 checkpoint와 epoch 상태로 복구
- 주기적 자동 다운로드 없음
- 정상 또는 오류 종료 시 handoff ZIP 하나만 생성

정상 산출물은 `temporal_t2c_handoff.zip`, 오류 또는 시간 종료 산출물은
`temporal_t2c_emergency_handoff.zip`이다. 둘 다 review, resume, stage summary,
run log를 포함한다.

## 7. 최종 학습

T2-C가 통과한 경우에만 3개 seed(`3407`, `42`, `2026`)로 다음 전문가를 full-data
학습한다.

| 전문가 | 학습 행 | 피처 | sample weight |
|---|---|---|---|
| base recent | 2024 시즌 | base | 모두 1 |
| base multi | 2020~2024 시즌 | base | 연도당 `0.55` 감쇠 |
| S1 recent | 2024 시즌 | base+S1 | 모두 1 |

총 학습 작업은 9개다. 각 전문가 안에서 세 seed의 logit을 평균한다. 최종 확률은
다음과 같이 계산한다.

- `anchor = logit_blend(base_recent, base_multi, recent_weight=0.50)`
- `candidate = logit_blend(S1_recent, base_multi, recent_weight=0.50)`
- `game_type=F`이면 `anchor`, 아니면 `candidate`

최종 학습에서는 train으로 전처리 상태를 확정한 뒤 test 각 행을 독립적으로 변환한다.
test의 ID 빈도, 범주 비율, 확률 분포, 행 간 집계는 fitting이나 분기 결정에 사용하지
않는다.

## 8. 제출 허용 게이트

제출 패키지 생성기는 다음 조건을 모두 확인하기 전에는 파일을 만들지 않는다.

- T2-C 상태가 `promoted`이고 모든 SHA-256 계보가 현재 입력과 일치
- 9개 full-data checkpoint가 모두 완료되고 training identity가 일치
- 모델 설정, seed, 학습 시즌, 감쇠율, 피처 상태가 설계와 일치
- 모든 전처리 상태가 train-only fitting임을 검증
- test 행 순서와 batch 크기를 바꿔도 row_id별 예측이 허용 오차 내 동일
- 예측이 유한한 `[0, 1]` 확률이며 sample submission과 행·키·컬럼이 정확히 일치
- 제출 코드가 네트워크, 외부 학습 데이터, 평가 행 간 통계를 사용하지 않음

T2-C 실패 후보는 그 후보의 최종 학습과 패키징만 차단한다. 다른 독립 제출 후보의
준비는 막지 않는다.

## 9. 검증

- 작업표 밖 seed, fold, 피처, 모델 설정을 거부하는 단위 테스트
- cutoff 이후 행이 feature fitting에 들어가지 않는 회귀 테스트
- 정확히 6개 T2-C 작업과 최대 9개 최종 작업을 확인하는 테스트
- F/non-F 행별 게이트 테스트와 행 순서 독립성 테스트
- seed ensemble의 결합 순서를 고정하는 수치 테스트
- tampered input, 다른 T2-B 계보, 다른 checkpoint identity 거부 테스트
- compact 완료 작업 재사용과 진행 중 checkpoint 복구 테스트
- 정상·오류 경로에서 단일 handoff만 생성하는 테스트
- Kaggle 셀 1MB 미만, deterministic generation, embedded runtime import 테스트
- 제출 패키징 진입점의 모든 허용 게이트 fail-closed 테스트

## 10. 예상 시간

- T2-C 6개 작업: 약 1~2시간
- 최종 9개 모델: 약 3~6시간
- 추론, 독립성 검사, 산출물 검증: 약 1시간
- 합계: 약 5~9시간

실측이 예상보다 빠르더라도 추가 탐색으로 예산을 채우지 않는다. 정해진 판정과 최종
학습만 완료하면 제출 준비를 종료한다.
