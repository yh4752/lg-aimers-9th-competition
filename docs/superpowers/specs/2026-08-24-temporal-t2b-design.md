# Temporal Portfolio T2-B 설계

## 1. 목적

T2-A에서 승격된 `S1`, `P3`, `P2`를 과거 시간 fold에서 확인하고, 상호 보완적인
피처 조합 하나를 선택한다. 이 단계는 피처의 시간 안정성을 검증하는 실험이며 모델
구조, 감쇠율, 전문가 비중, anchor, seed는 바꾸지 않는다.

T2-B는 제출 파일이나 평가 데이터 예측을 만들지 않는다. 결과물은 T3 입력 후보와
재시작용 산출물뿐이다.

## 2. 고정 입력과 계보

입력 생성기는 다음 자료를 모두 검증한 뒤 하나의 `temporal_t2b_input.zip`을 만든다.

- 공식 `train.csv`, `test.csv`, `trackman_history.csv`, `sample_submission.csv`
- T1 review SHA-256:
  `59a34ac423f666a981dbc5d6b974b97c06366dde104dab7bf88bca1aceb68176`
- T1 결정 SHA-256:
  `d2b425344c119b2f1a9cf038545d8ca5a68c082ab71ff253ac763761cdb2f602`
- T2-A handoff SHA-256:
  `185608e8dc04676155463142ef2c44ddb2feed6d107aa3a621f72878f80f7da2`
- T2-A 완료 상태: 완료 8개, 대기 0개, 실패 0개, 생략 2개
- T2-A 승격 후보:
  `S1/both_experts`, `P3/recent_only`, `P2/recent_only`

입력 ZIP에는 T1의 고정 baseline과 다중 시즌 예측을 검증 연도별로 저장한다.

- `t1_anchor_2022.csv`, `t1_anchor_2023.csv`, `t1_anchor_2024.csv`
- `t1_multi_2022.csv`, `t1_multi_2023.csv`, `t1_multi_2024.csv`
- `t1_decision.json`
- `t2a_decision.json`
- `manifest.json`

모든 멤버의 크기와 SHA-256, 공식 데이터 행 identity, T1/T2-A 계보를 manifest에
기록한다. 하나라도 다르면 학습을 시작하지 않는다.

## 3. 고정 모델 설정

- 모델: TabM `p2`
- 수치 임베딩: piecewise-linear
- 손실: BCE
- 최적화: AdamW와 plateau scheduler
- seed: `3407`
- 최대 epoch 수: `12`
- patience: `3`
- 최근 시즌 전문가만 신규 학습
- T1 다중 시즌 전문가: 감쇠율 `0.55`로 생성된 기존 OOF 예측을 고정
- 최근 전문가 비중: `0.50`
- 결합 방식: logit
- anchor beta: `0`

T2-A에서 확인된 S1 다중 시즌 개선은 보조 증거로 보존한다. 그러나 과거 fold에서
동일한 다중 시즌 S1 모델을 확인하지 않으므로 T2-B의 다중 fold champion 판정에는
섞지 않는다. 이 제한으로 서로 다른 실험 설정을 같은 후보처럼 비교하는 오류를 막는다.

## 4. 10개 학습 구성

### Phase F: 단일 피처 과거 fold 확인, 6개

| 피처 | 학습 시즌 | 검증 시즌 |
|---|---:|---:|
| S1 | 2021 | 2022 |
| P3 | 2021 | 2022 |
| P2 | 2021 | 2022 |
| S1 | 2022 | 2023 |
| P3 | 2022 | 2023 |
| P2 | 2022 | 2023 |

각 예측은 같은 검증 연도의 고정 T1 다중 시즌 예측과 `0.50:0.50` logit 결합한 뒤
고정 T1 baseline과 비교한다. T2-A의 2024 recent-only 증거를 세 번째 fold로
결합해 `S1`, `P3`, `P2`의 다중 fold 지표를 계산한다.

### Phase C: 최신 fold 조합 확인, 2개

- `S1+P3`, 2023→2024
- `S1+P2`, 2023→2024

조합은 각 단일 모델의 확률을 사후 평균하는 방식이 아니다. `base`와 두 피처군을
동시에 입력한 TabM 하나를 새로 학습한다. 두 조합 중 아래 조건을 만족하면서 Brier
개선이 큰 하나를 선택한다.

- Brier 개선 `0.00003` 이상
- pitcher block bootstrap 하한 `0` 초과
- 최대 segment regression `0.00050` 이하

동률이면 `S1+P3`를 선택한다. 두 조합이 모두 조건을 통과하지 못하면 Phase H를
생략하고 단일 피처 결과만 T2-B 판정에 사용한다.

### Phase H: 선택 조합 과거 fold 확인, 최대 2개

- 선택 조합, 2021→2022
- 선택 조합, 2022→2023

따라서 전체 신규 학습 수는 최대 `6 + 2 + 2 = 10`이다.

## 5. 최종 판정

단일 피처 세 개와 선택 조합 하나를 동일한 세 fold에서 평가한다. 연도별 Brier
개선, 최신 fold 개선, 행 수 가중 개선, bootstrap 하한, segment regression을
기록한다.

Champion 조건은 기존 temporal portfolio 계약을 그대로 사용한다.

- 행 수 가중 개선 `0.00005` 이상
- 최신 fold 개선 `0.00003` 이상
- 세 fold를 합친 bootstrap 하한 `0` 초과
- 최대 segment regression `0.00050` 이하
- 어느 fold에서도 Brier 퇴행이 `0.00003`을 넘지 않음

Champion이 없더라도 기존 exploratory 조건을 충족한 후보는 별도로 기록한다.
Champion과 exploratory를 합쳐 최대 세 개를 T3 입력 후보로 보낸다. 불완전한
증거는 탈락이 아니라 `budget_inconclusive`로 기록한다.

## 6. 실행과 복구

- 환경: Kaggle Tesla T4 두 장
- 벽시계 제한: 4시간
- 종료 25분 전부터 새 작업 시작 금지
- 종료 10분은 산출물 생성에 예약
- 완료 작업은 compact 결과로 검증 후 재사용
- 진행 중 작업은 checkpoint와 epoch 상태로 복구
- 정상 종료: `temporal_t2b_handoff.zip` 하나 다운로드
- 오류 또는 시간 종료: `temporal_t2b_emergency_handoff.zip` 하나 다운로드
- 주기적인 자동 다운로드는 하지 않음

한 셀은 공식 데이터, `temporal_t2b_input.zip`, 선택적인 T2-B resume 또는
handoff를 `/kaggle/input`에서 자동 탐색한다. 동일 종류가 둘 이상이면 임의로
선택하지 않고 오류로 종료한다.

## 7. 로그 계약

사용자가 진행 상황을 판단할 수 있도록 다음 로그를 남긴다.

- `T2B_CODE_READY`
- `T2B_DEPENDENCIES_READY`
- `T2B_GPU_READY`
- `T2B_INPUTS_VERIFIED`
- `T2B_RESUME_READY`
- `T2B_PHASE_START phase=F|C|H`
- `T2B_JOB_START`, `T2B_JOB_REUSED`, `T2B_JOB_COMPLETE`
- `T2B_COMBINATION_SELECTED`
- `T2B_DECISION`
- `T2B_STAGE_RESULT`
- `T2B_HANDOFF_READY`
- `T2B_PORTFOLIO_ERROR`
- `T2B_EMERGENCY_HANDOFF_READY`

## 8. 대회 규칙 경계

- 피처 fitting은 각 fold의 검증 시즌보다 이전인 공식 train과 history만 사용한다.
- 검증 시즌 target은 모델 fitting, 전처리 통계, ID 매핑에 사용하지 않는다.
- `test.csv`의 행 분포, ID 빈도, 확률 분포로 후보를 선택하지 않는다.
- T2-B에서는 full-data 학습, test 추론, 제출 ZIP 생성을 금지한다.
- 최종 제출 코드에서도 각 평가 행은 다른 평가 행과 독립적으로 예측해야 한다.

## 9. 검증 기준

- 입력 위·변조, 다른 T1/T2-A 계보, 중복 입력을 거부한다.
- fold cutoff 이후 행이 feature fitting 또는 매핑에 들어가면 테스트가 실패한다.
- 작업 수가 10개를 넘으면 계획 단계에서 거부한다.
- 조합 선택은 동일한 증거에서 항상 같은 결과를 낸다.
- resume의 모델 identity가 현재 설정과 다르면 재사용하지 않고 오류로 종료한다.
- 생성 셀은 1MB 미만이며 두 번 생성한 바이트가 같아야 한다.
- 정상·오류 경로 모두 최종 다운로드는 handoff ZIP 하나만 요청한다.
