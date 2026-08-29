# 실험 라운드

라운드는 모델 이름을 다시 번호 매기는 체계가 아니라, 하나의 가설을 검증하고 다음
판단을 내린 의사결정 묶음이다. R9 같은 기존 모델명은 그대로 유지한다.

전체 프로젝트를 처음 보는 독자는 [실험 여정](../EXPERIMENT_JOURNEY.md)부터 읽으면
된다. 문제를 발견하고 가설을 세운 뒤 채택·폐기한 흐름을 쉬운 말로 설명한다. 정확한
수치와 산출물 식별값은 [실험 장부](../../reports/EXPERIMENT_LEDGER.md)가 기준이다.

## 초기 라운드 기록

- [R9 검증 기반](01-r9-foundation.md)
- [FwFM 단독·제한 blend와 계열 종료](02-fwfm.md)
- [TabM residual](03-tabm-residual.md)
- [R9 calibration](04-calibration.md)
- [XGBoost v3와 original preprocessing rescue](05-xgboost.md)
- [예산 제한 전처리 캠페인: Stage 1~5](06-budgeted-preprocessing-campaign.md)
- [TabM 첫 공식 제출](07-tabm-first-submission.md)

## 이후 캠페인

TabM seed·행 단위 파생변수, CatBoost·TabM 배포 정렬, Tree Expert E2·T3, 계층
보정과 실패 유형 라벨 감사는 여러 실행 환경과 재개 번들을 오간 장기 캠페인이다.
번호를 억지로 이어 붙이지 않고 [실험 여정](../EXPERIMENT_JOURNEY.md)에서 의사결정
순서로 설명한다. 완료 여부와 판정 수치는 [실험 장부](../../reports/EXPERIMENT_LEDGER.md)에
한 번만 기록한다.
