# R9 calibration

## 가설

R9의 특히 큰 game_type=F calibration gap을 시간 순서로 fit한 단순 보정으로 줄이면
모델을 다시 학습하지 않고 Brier를 개선할 수 있다.

## 검증 프로토콜

R9 세 시간 fold의 fit 구간에서만 보정 파라미터를 구하고 다음 시즌에 적용했다.
Global temperature, global logit affine, game-type temperature 세 변형과 전체,
fold, F/R 지표를 사전에 고정한 gate로 감사했다.

## 결과

세 변형 모두 전체 Brier를 낮췄다. 최선 `game_type_temperature`는
`0.24789890676984772`이고 F Brier는 `0.2474564827036286`이다. 하지만 global
calibration gap과 최대 fold delta gate를 통과하지 못했다.

## 판정

고정한 세 변형은 `rejected`. Calibration family 전체는 닫지 않는다.

## 배운 점

보정의 평균 개선과 calibration gap 개선은 같은 조건이 아니다. Segment별 개선도
연도 fold 안정성을 넘어서는 증거가 필요하다.

## 다음 결정

다음 calibration 후보는 결과를 보기 전에 segment와 파라미터 범위를 고정하고,
이전 세 후보의 결과로 탐색 범위를 사후 확장하지 않는다.

## 근거

- [Calibration rejection](../../reports/rejections/calibration_blending_rejection.json)
