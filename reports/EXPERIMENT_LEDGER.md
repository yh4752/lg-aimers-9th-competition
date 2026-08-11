# 실험 장부

완료된 실행과 판정을 시간순으로 기록한다. Brier는 낮을수록 좋지만 서로 다른 검증
프로토콜의 값을 하나의 순위처럼 비교하지 않는다. Public 열은 실제 DACON 제출
점수가 확인된 경우에만 채운다.

| 순서 | 실험 ID | 모델 계열 | 검증 프로토콜 | 핵심 결과 | 판정 | Public | 근거 |
|---:|---|---|---|---|---|---:|---|
| 1 | `catboost_smooth_v1` | CatBoost | 2023→2024 | Brier `0.248033`, BSS `710.109689` | 기준선 | `828.9963889533` | 기존 proxy ladder 기록 |
| 2 | `catboost_proxy_01_no_smoothing` | CatBoost | 2023→2024 quick gate | 기준선 대비 Brier `+0.000145` | rejected | — | 기존 proxy ladder 기록 |
| 3 | `catboost_proxy_02_k125` | CatBoost | 2023→2024 quick gate | 기준선 대비 Brier `+0.000030` | rejected | — | 기존 proxy ladder 기록 |
| 4 | `catboost_proxy_03_iter800` | CatBoost | 2023→2024 quick gate | 기준선 대비 Brier `+0.000130` | rejected | — | 기존 proxy ladder 기록 |
| 5 | `round9_temporal_oof` | R9 CatBoost anchor | 2021→2022, 2022→2023, 2023→2024 | 746,504행, Brier `0.24825099524638927` | passed | — | [OOF acceptance](acceptances/round9_temporal_oof_acceptance.json) |
| 6 | `fwfm_standalone` | FwFM | R9과 동일한 3-fold·746,504행 | Brier `0.2501797846421676` | rejected | `88.5362742196` | [rejection](rejections/fwfm_standalone_rejection.json) |
| 7 | `r9_fwfm_game_type_f_blend` | R9 + FwFM | game_type=F 제한 grid | 최저 전체 Brier `0.24725145521757802`, worst fold delta `0.000534024610467615` | rejected | — | [rejection](rejections/r9_fwfm_game_type_f_blend_rejection.json) |
| 8 | `r9_fwfm_game_type_f_blend_w080` | R9 + FwFM | FwFM 80% 고정 F blend | 전체 Brier `0.24728849890644408`, 2022·2024 F fold 악화 | rejected | — | [rejection](rejections/r9_fwfm_game_type_f_blend_w080_rejection.json) |
| 9 | `fwfm_failure_boundary_exit_audit` | FwFM 진단 | 검증 OOF 읽기 전용 segment 감사 | `stable_signal_count=0` | diagnostic, family closed | — | [diagnostic](diagnostics/fwfm_failure_boundary_exit_audit.json) |
| 10 | `tabm_residual` | R9 + TabM residual | 2022→2023, 2023→2024 meta OOF | 499,032행, R9 대비 Brier `+0.004151241934553795` | rejected | — | [rejection](rejections/tabm_residual_rejection.json) |
| 11 | `calibration_blending` | R9 calibration | R9 3-fold·746,504행 | best `game_type_temperature` Brier `0.24789890676984772`, 두 gate 실패 | exact variants rejected, family open | — | [rejection](rejections/calibration_blending_rejection.json) |
| 12 | `xgboost_score_push_v3` | XGBoost | 4개 역사 시간 fold 선택 후 2024 holdout | 4-fold 평균 Brier `0.24701737756648098`, 2024 `0.248274358430683` | exploratory accepted | — | [acceptance](acceptances/xgboost_v3_exploratory_acceptance.json) |
| 13 | `xgboost_original_preproc_rescue` | XGBoost | 2023→2024 holdout 후 full fit | `lossguide_l31`, 2024 Brier `0.24826687414041645`, 기술 gate 7개 PASS | verified ready | — | [acceptance](acceptances/xgboost_original_preproc_rescue_acceptance.json) |

## 현재 결론

- R9은 누출 없는 공통 비교 기반이다.
- FwFM standalone, 제한 blend와 종료 감사가 모두 끝나 FwFM 계열은 닫혔다.
- TabM residual 후보만 기각됐으며 다른 TabM 설계를 자동으로 막지 않는다.
- Calibration의 고정된 세 변형은 기각됐지만 segment-aware family는 열려 있다.
- XGBoost v3와 rescue는 각각 탐색 수용·기술 준비 상태이며 Public 개선을 의미하지
  않는다.
- 대용량 OOF, 모델과 ZIP은 Google Drive에 두고 이 장부는 작은 evidence와 실행
  ID로 원본을 식별한다.
