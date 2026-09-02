---
title: "TabM 872점에서 Tree Expert E2 977점까지, 더 큰 모델보다 잔차 보정이 나았던 이유"
date: "2026-09-02"
type: "deep-dive"
project: "lg-aimers-9th-competition"
tags: ["AI", "Evaluation", "Performance", "Architecture"]
summary: "확정 전처리와 TabM 제출 이후, 과거 선수 기록으로 만든 기준 확률과 CatBoost 잔차 보정을 결합해 Public 977.3809532715를 만든 과정을 정리합니다."
featured: false
draft: false
canonicalProjectPath: "docs/blog/2026-09-02-tabm-to-tree-expert-e2.md"
sourceRepository: "https://github.com/yh4752/lg-aimers-9th-competition"
---

> 한 줄 요약: TabM을 무작정 키우는 대신 과거 선수 기록으로 기준 확률을 만들고, 반복되는 오차만 CatBoost로 고쳤다.

## 문제

첫 TabM 제출은 Public `872.3920184667`을 기록했다. 첫 CatBoost 제출보다 점수는 올랐지만 상위권과의 차이는 컸다. 처음 떠오른 해법은 학습 epoch를 늘리거나 같은 모델을 여러 seed로 학습해 평균하는 것이었다.

로그를 다시 보니 단순 학습 부족으로 설명하기 어려웠다. 최대 43 epoch까지 확인한 실험에서도 가장 좋은 checkpoint는 2~3 epoch에 형성됐다. 같은 TabM을 오래 돌리는 것보다 어떤 오차를 놓쳤는지 찾는 편이 나았다.

## 전처리 Stage 1~5

전처리는 정규화와 파생변수를 한꺼번에 넣지 않고 다섯 단계로 나눠 범위를 좁혔다.

1. TabM, FT-Transformer, TabNet과 CatBoost를 비교했다.
2. 수치 변환과 선수별 평활화를 하나씩 바꿨다.
3. 과거 집계, 결측 표시, ID 빈도와 같은 행 상호작용을 확인했다.
4. 좋아진 후보를 다른 시간 전이와 CatBoost에서도 검사했다.
5. 전체 학습 데이터에서 기본 전처리와 최종 후보를 다시 비교했다.

비교한 딥러닝 모델 중 TabM이 가장 좋았다. Yeo-Johnson 변환과 단독 선수 평활화는 기본 구성을 이기지 못했다. 반면 현재 행의 투수 손잡이와 타자 손잡이를 결합한 `hand_matchup`은 두 시간 전이와 CatBoost에서 같은 개선 방향을 보였다.

전체 학습 비교에서 기본 TabM Brier는 `0.2483070113`, `hand_matchup`을 더한 값은 `0.2480974092`였다. 최종 전처리는 `dl_standard + hand_matchup`으로 고정했다. `hand_matchup`은 평가 데이터 전체에서 만든 통계가 아니라 현재 행의 두 값만 조합하므로 행 독립성 규칙에도 맞았다.

## TabM 첫 제출

확정 전처리와 TabM P2를 전체 데이터로 3 epoch 학습했다. 3 epoch는 시간이 부족해 임의로 자른 숫자가 아니다. 이전 시간 전이 검증에서 최적 checkpoint가 이 구간에 모인다는 근거가 있었다.

seed를 늘리면 안정적일 것이라는 가설도 확인했다. 그러나 TabM seed 평균은 단일 seed 3407보다 Brier가 `0.000318` 이상 나빠졌다. 여기서 “같은 모델을 더 오래, 더 많이” 돌리는 실험은 멈췄다. TabM 계열 전체를 포기한 것이 아니라 이 구성의 반복만 중단했다.

## 선택지

| 선택지 | 기대한 점 | 확인한 한계 |
| --- | --- | --- |
| TabM epoch 확대 | 별도 구조 없이 학습을 더 진행할 수 있다. | 최적 checkpoint가 이미 초반에 형성됐다. |
| TabM 다중 seed 평균 | 난수에 따른 흔들림을 줄일 수 있다. | 고정 검증에서 단일 seed보다 나빠졌다. |
| TabM과 CatBoost 확률 혼합 | 서로 다른 오차를 평균으로 줄일 수 있다. | 검증 모델과 배포 모델의 트리 수를 맞추지 못하면 같은 결과를 재현할 수 없다. |
| Anchor 잔차 보정 | 선수의 과거 성공률을 출발점으로 두고 반복 오차에 집중한다. | Anchor가 틀린 방향을 보정 모델이 그대로 따라갈 수 있어 시간 전이 검사가 필요하다. |

## 왜 E2로 바꿨나

여기에는 이름이 비슷해 헷갈리기 쉬운 두 가지 기준이 있었다. TabM은 E2가 실제로 나아졌는지 판단하기 위한 **비교 기준 모델**이었다. 반면 E2 내부의 **anchor**는 TabM 예측값이 아니었다. 예측 시점보다 과거인 학습 데이터에서 투수의 통산 성공률, 해당 시즌 성공률과 최근 기록을 계산하고, 표본이 적을 때 전체 평균 쪽으로 완만하게 줄여 만든 출발 확률이었다.

Tree Expert E2의 CatBoost는 정답 전체를 처음부터 예측하지 않고 `실제 정답 - anchor`를 학습했다. 추론할 때는 anchor에 CatBoost가 예상한 잔차를 더했다. 시험 답안을 새로 쓰기보다 과거 기록으로 만든 첫 답안에서 반복되는 실수만 고치는 방식이다.

CatBoost를 고른 이유는 선수 ID와 경기 상황처럼 범주형 상호작용을 다루기 좋았기 때문이다. seed 42, 2026, 3407의 세 모델을 학습해 평균했다. 앞서 TabM에서는 seed 평균이 나빴지만, E2에서는 세 seed 평균이 같은 시간 전이 검증에서 이득을 보였다. 모델 종류가 아니라 실제 검증 결과를 기준으로 남겼다.

## 검증

E2는 2021→2022, 2022→2023, 2023→2024의 세 시간 전이에서 같은 TabM 기준 예측과 비교했다. 가중 평균 Brier 개선은 `0.0008375137`이었고 가장 약한 fold도 `0.0003439904` 좋아졌다. Bootstrap 하한은 `0.0006692336`이었다. 평균만 좋아지고 특정 시즌이 나빠지는 후보가 아니었다.

제출 전에는 245,789개 행으로 순서 반전, 무작위 섞기, 재배치와 한 행 추론을 비교했다. 같은 행의 최대 예측 차이는 `0`이었다. 평가 파일의 다른 행을 보지 않는지도 성능 검증과 따로 확인했다.

승인된 제출의 Public 점수는 `977.3809532715`였다. TabM 제출보다 약 104.99점 높지만 Public 점수 차이만으로 잔차 보정의 일반적인 우월성을 주장하지는 않는다. 로컬에서는 E2의 동일 시간 전이 비교가 판단 근거였고 Public은 그 판단 뒤에 한 번 확인한 결과다.

## 기각한 후속 후보

E2 뒤에도 여러 방향을 시험했다.

- 앞서 확인한 TabM seed 평균은 단일 seed 3407보다 나빠 기각했다.
- 최근 시즌과 감쇠한 과거 시즌을 나눈 T3는 가중 개선이 `-0.0000505772`였고 두 구조 선택 fold가 모두 나빠졌다.
- 계층 잔차 후보는 일부 평균 개선이 있었지만 최악 fold와 seed 일관성 기준을 통과하지 못했다.
- XGBoost와 LightGBM 잔차 보정 S3는 평균적으로 아주 조금 좋아졌지만 최소 개선량에 못 미쳤고 E2와의 예측 상관도 `0.998`을 넘었다.

여기서 닫은 것은 실험에 쓴 고정 구성이다. T3가 기각됐다고 시간 피처 전체가 쓸모없는 것은 아니다. S3 결과도 XGBoost와 LightGBM 계열 전체의 실패로 넓혀 해석하지 않았다.

## 트레이드오프

Anchor 잔차 보정은 과거 기록으로 설명되는 기준 확률을 출발점으로 유지한다. 작은 데이터 변화에 전체 예측이 뒤집힐 위험도 줄어든다. 반면 anchor가 보지 못한 새로운 패턴을 크게 바꾸기 어렵고 통계 산식과 보정 모델의 계보를 함께 관리해야 한다.

세 seed 평균은 파일 크기와 추론 시간을 늘린다. 이 비용을 허용한 이유는 세 시간 전이와 독립성 검사에서 이득이 확인됐기 때문이다. 배포 시에는 모델, 피처 순서, 전처리와 anchor 해시가 검증 때와 같아야 했다.

## 내가 이해한 것

점수가 낮으면 먼저 모델을 더 크게 만들어야 한다고 생각하기 쉽다. 이번에는 오히려 문제를 나눴을 때 가장 큰 개선이 나왔다. 기존 확률이 잘 맞히는 부분은 그대로 두고 범주형 상황에서 반복되는 실수만 다른 모델이 맡았다.

앙상블의 수보다 역할 분담이 더 중요하다는 점도 배웠다. 비슷한 모델을 늘리면 같은 오차를 반복한다. 새 모델이 기존 모델과 다른 실수를 고치는지는 같은 OOF 행에서 직접 확인해야 했다.

## Codex에게 맡긴 것과 내가 검토한 것

Codex는 전처리 캠페인과 E2 학습·복구 코드, 검증 gate, 제출 전 행 독립성 검사 작성을 지원했다. 나는 Kaggle과 Colab에서 전체 실험을 실행하고 로그와 산출물을 회수했다.

어떤 후보를 계속 실행할지, GPU 시간을 어디에 쓸지, 실제로 제출할 파일이 무엇인지는 대화 속에서 내가 결정했다. 다만 OOF 계산과 행 독립성 검사의 기술적 타당성을 혼자 감사한 수준은 아니었고 Codex의 설명과 검사 결과에 의존한 부분이 컸다. 내가 외부에서 직접 확인한 최종 결과는 DACON Public `977.3809532715`였다. 실행 책임과 기술적 검증 수준을 같은 것으로 과장하지 않는 것이 이 기록의 전제다.

## 코드에서 다시 볼 지점

| 자료 | 다시 볼 내용 |
| --- | --- |
| [전처리 연구 기록](https://github.com/yh4752/lg-aimers-9th-competition/blob/main/docs/rounds/06-budgeted-preprocessing-campaign.md) | Stage 1~5의 승격과 기각 흐름 |
| [TabM 첫 제출 기록](https://github.com/yh4752/lg-aimers-9th-competition/blob/main/docs/rounds/07-tabm-first-submission.md) | 3 epoch 선택과 제출 무결성 |
| [E2 계약](https://github.com/yh4752/lg-aimers-9th-competition/blob/main/experiments/tree_expert/e2_contract.json) | Fold, seed와 수용 기준 |
| [E2 학습 코드](https://github.com/yh4752/lg-aimers-9th-competition/blob/main/experiments/tree_expert/e2_training.py) | Anchor 잔차를 CatBoost가 학습하는 방식 |
| [E2 추론 코드](https://github.com/yh4752/lg-aimers-9th-competition/blob/main/experiments/tree_expert/e2_inference.py) | 저장 모델과 anchor를 행별로 결합하는 방식 |
| [실험 장부](https://github.com/yh4752/lg-aimers-9th-competition/blob/main/reports/EXPERIMENT_LEDGER.md) | 후속 후보를 해당 기준에서 닫은 근거 |

## 면접에서 설명할 수 있어야 할 질문

1. TabM의 epoch를 늘리지 않기로 한 근거는 무엇인가?
2. `hand_matchup`이 대회 행 독립성 규칙을 지키는 이유는 무엇인가?
3. TabM 비교 기준과 E2 내부 anchor는 어떻게 다른가?
4. TabM seed 평균은 기각하고 E2 seed 평균은 채택한 이유는 무엇인가?
5. 평균 개선과 최악 fold 개선을 함께 본 이유는 무엇인가?
6. E2의 Public 상승만으로 잔차 보정이 항상 낫다고 말할 수 없는 이유는 무엇인가?
