# TabM residual

## 가설

TabM이 R9 확률의 bounded logit residual만 학습하면 단독 모델보다 안정적으로 다른
함수 방향을 추가할 수 있다.

## 검증 프로토콜

2022→2023과 2023→2024 meta OOF 499,032행에서 전체, fold, F/non-F와
fold×segment를 같은 R9 행과 비교했다.

## 결과

TabM residual Brier `0.2548630286197798`은 같은 행의 R9
`0.250711786685226`보다 `0.004151241934553795` 나빴다. 모든 확인 segment가
악화됐고 최악은 `valid_2023:F`의 `+0.02862235669164276`였다.

## 판정

이 residual 구성은 `rejected`다.

## 배운 점

모델 계열의 표현력이 높아도 입력, 목표와 결합 방식이 맞지 않으면 강한 anchor를
보정하지 못한다. 한 residual 구성의 실패는 TabM standalone이나 다른 목표의
TabM 후보를 자동 기각하지 않는다.

## 다음 결정

같은 residual 구성은 반복하지 않는다. 독립 TabM 후보는 별도 사전 설계와 gate가
있을 때만 실행한다.

## 근거

- [TabM residual rejection](../../reports/rejections/tabm_residual_rejection.json)
- 동료 저장소의 [Round 25 deep-learning 기록](https://github.com/castle9612/lg_aimers_9th/blob/main/reports/round25_deep_learning.md)
