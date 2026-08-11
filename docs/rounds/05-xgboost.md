# XGBoost v3와 original preprocessing rescue

## 가설

시간 전이와 train-only 피처 계약을 유지한 XGBoost 구조·시즌 가중치 탐색이 기존
XGBoost보다 안정적인 후보를 만들 수 있다. 별도로 원본 전처리의 더 단순한 구조도
안전한 탐색 제출 후보가 될 수 있다.

## 검증 프로토콜

XGBoost v3는 네 개 역사 시간 fold에서 설정을 선택한 뒤 2024를 한 번 확인했다.
Rescue는 2023→2024에서 depthwise와 lossguide를 비교한 뒤 선택 구성을 full fit하고
ZIP 구조, CRC와 격리 CPU 추론을 검사했다. 이 값은 R9의 세 fold와 직접 순위를
매기지 않는다.

## 결과

V3 선택 후보의 4-fold 평균 Brier는 `0.24701737756648098`, 2024 holdout은
`0.248274358430683`이다. Rescue의 `lossguide_l31`은 2024
`0.24826687414041645`였고 일곱 기술 gate를 통과했다.

## 판정

V3는 `accepted_for_exploratory_submission`, rescue는 `verified_ready`다. 둘 다
Public 개선이나 공식 제출 완료를 의미하지 않는다.

## 배운 점

선택 fold와 최종 holdout을 분리하고 중단 복구 산출물에 데이터·행·코드 해시를
묶으면 장시간 Colab 탐색도 판정 가능한 evidence로 남길 수 있다.

## 다음 결정

코드와 Colab은 별도 이전 계획으로 검증한다. 실제 제출 후보화는 현재 artifact
해시와 package gate를 다시 확인한 뒤 별도 승인한다.

## 근거

- [XGBoost v3 exploratory acceptance](../../reports/acceptances/xgboost_v3_exploratory_acceptance.json)
- [Original preprocessing rescue acceptance](../../reports/acceptances/xgboost_original_preproc_rescue_acceptance.json)
- [Aggressive capacity Public result](../../reports/acceptances/xgboost_aggressive_capacity_public_result.json)

## 공격적 구조 탐색의 Public 결과

최종 후보는 `depthwise_d6`와 `lossguide_l63` 각각 seed 42·2026의 네 모델을
평균했다. 선택된 rounds는 순서대로 119, 134, 119, 106이며 확률 배율 `1.05`와
`linear_extrapolated` 시즌 평균 보정을 적용했다.

| 단계 | 로컬 점수 |
|---|---:|
| 네 모델 단순 평균 | `714.8814792915847` |
| 선형 시즌 평균 보정 | `750.5344761643662` |
| 확률 배율 1.05 | `752.5433411090132` |

최종 Public은 `820.9583317093`이었다. `depthwise_d8`, `lossguide_l127`,
`lossguide_l255`도 탐색했지만 d6·l63보다 낮았다. 결론은 깊은 트리를 미리 막는
것이 아니라 충분히 탐색하고 실제 fold·ensemble·보정 결과로 선택해야 한다는 것이다.
