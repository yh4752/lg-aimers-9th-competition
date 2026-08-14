# TabM 첫 공식 제출

## 제출 모델

확정 전처리 `dl_standard + hand_matchup`과 TabM P2를 결합한 단일 모델을
DACON에 제출했다. 모델은 공식 학습 데이터 전체를 3 epoch 학습했으며 seed는
3407이다. 3 epoch는 실행 시간이 부족해 임의로 중단한 값이 아니다. 앞선 시간 전이
검증에서 최대 43 epoch까지 확인했을 때 최적 checkpoint가 2~3 epoch에 형성된 결과를
전체 학습에 옮긴 것이다.

| 항목 | 값 |
|---|---:|
| 후보 | `tabm_hand_matchup_version_d_seed3407_v1` |
| Public | `872.3920184667` |
| 수료 기준 | `549.51` |
| 기준 대비 여유 | `+322.8820184667` |
| 2023→2024 Brier | `0.2481108023` |
| 2022→2023 Brier | `0.2508657359` |
| 제출 ZIP SHA-256 | `bcea66999322721e60016761ce041ca151c71b034d0e4122f0b2852f5b06697a` |

모델과 제출 ZIP은 Git에 넣지 않는다. 위 해시는 실제 제출 파일과 이 기록을 연결하기
위한 식별자다.

## 점수 해석

공식 점수는 다음 Brier Skill Score다.

```text
Score = max(0, 100000 × (1 - model_brier / mean_rate_brier))
```

따라서 `872.3920184667`은 평가 자료의 평균 성공률만 예측하는 기준보다 Brier가
약 `0.872392%` 낮다는 뜻이다. 리더보드 1위가 1,100점을 초과한다면 점수 차이는
최소 `227.6079815333`점이다. 평균 성공률 Brier는 최대 `0.25`이므로 1,100점까지
필요한 절대 Brier 감소량은 최대 약 `0.00056902`다. 화면의 점수 차이에 비해 실제
오차 차이는 작지만, 이미 낮은 Brier에서 이 정도를 안정적으로 줄이는 일은 쉽지 않다.

## 무엇을 확인했나

- 제출 ZIP의 설치, 모델 로드, 추론과 `submission.csv` 생성이 평가 서버에서 끝났다.
- TabM은 기존 CatBoost Public `828.9963889533`과 XGBoost Public
  `820.9583317093`보다 높은 점수를 기록했다.
- 평가 행의 평균, 빈도, 순서나 다른 평가 행을 사용하지 않는 독립 추론 모델이다.
- 전처리 상태, 범주 사전과 수치 bin은 공식 학습 데이터에서 미리 고정했다.
- 현재 제출은 TabM 한 구조, seed 하나, `hand_matchup` 한 파생변수만 사용한다.

이 결과는 TabM 방향이 유효하다는 신호다. 반면 한 번의 Public 결과만으로 특정
seed나 후처리 가중치를 선택하는 근거로 사용하지 않는다.

## 다음 결정

같은 모델의 epoch를 늘리는 실험은 우선순위가 아니다. 전체 행 검증에서 43 epoch를
확인했지만 최적 시점은 초반이었다. 다음 연구는 기존 시간 전이 예측을 재사용한 seed
앙상블 사전 평가부터 시작한다. 그다음 행 단위 파생변수, 규칙을 지키는 CatBoost
블렌딩, OOF 기반 보정을 순서대로 검토한다.

세부 승급 기준과 중단 조건은
[TabM 점수 개선 설계](../superpowers/specs/2026-08-15-tabm-score-improvement-design.md)에
고정한다.

## 근거

- [최종 제출물 검증](../TABM_SUBMISSION_VALIDATION.md)
- [전처리 Stage 1~5](06-budgeted-preprocessing-campaign.md)
- [Public 결과 기록](../../reports/acceptances/tabm_hand_matchup_public_result.json)
- [공식 평가 안내](https://dacon.io/competitions/official/236743/overview/evaluation)
