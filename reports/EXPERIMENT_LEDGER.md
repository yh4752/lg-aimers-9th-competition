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
| 14 | `xgboost_aggressive_capacity_v1` | XGBoost | 역사 fold 구조 탐색·최근 시즌 ensemble | depthwise d6·lossguide l63 4-member, local Brier `0.2479270213638507` | public scored; rules quarantine | `820.9583317093` | [result](acceptances/xgboost_aggressive_capacity_public_result.json) |
| 15 | `tabm_hand_matchup_version_d_seed3407_v1` | TabM | 2022→2023·2023→2024 후 전체 학습 | 최신 Brier `0.2481108023`, 이전 Brier `0.2508657359`, 전체 학습 3 epoch | accepted; public scored | `872.3920184667` | [result](acceptances/tabm_hand_matchup_public_result.json) |
| 16 | `tabm_seed_ensemble` | TabM seed 평균 | 2022→2023·2023→2024 OOF 499,032행 | seed 3407 Brier `0.2494662366`; 두 평균 모두 `0.000318` 이상 악화 | rejected; keep seed 3407 | — | [rejection](rejections/tabm_seed_ensemble_rejection.json) |
| 17 | `tree_expert_e2_c1_catboost` | anchor residual CatBoost 3개 seed | 2021→2022·2022→2023·2023→2024, 전체 245,789행 독립성 감사 | weighted gain `0.0008375137`, worst fold gain `0.0003439904`, bootstrap lower `0.0006692336`; 행 독립성 최대 오차 `0` | accepted; public scored | `977.3809532715` | E2 handoff SHA-256 `4dd0c901…384050f` |
| 18 | `tree_expert_t3_temporal_dual_v1` | 최근 시즌·감쇠 다중 시즌 CatBoost 잔차 전문가 | 2021→2022·2022→2023 구조 선택, 2023→2024 확인 | 선택 decay `0.75`, 최근 비중 `0.70`; weighted gain `-0.0000505772`, 두 fold 악화, 최대 segment 회귀 `0.0015859681` | rejected; structure gate failed | — | T3 handoff SHA-256 `5157f1d0…d680b` |
| 19 | `catboost_tabm_fixed_blend_v1` | TabM·CatBoost 고정 확률 혼합 | 2022→2023·2023→2024 정렬 OOF | TabM 70%·CatBoost 30%; Brier `0.2494662366`→`0.2492874755`, gain `0.0001787611`, 두 fold 개선 | OOF blend passed; deployment pending | — | delivery SHA-256 `ab7ca41e…dcfa` |
| 20 | `catboost_deployment_alignment_v1` | 고정 트리 수 CatBoost·TabM 70:30 | 2022→2023·2023→2024 배포 정렬 | 트리 수 `4/32/64/128/192/296/400` 모두 실패; 4트리는 전체 gain `0.0000320407`이나 최근 fold `+0.0001464537` 악화 | deployment blocked | — | review SHA-256 `fc65693a…41e8` |
| 21 | `tree_hierarchical_residual_v1` | E2 anchor + 계층 잔차·보정 | 2021→2022·2022→2023 구조, 2023→2024 확인 | C1 gain `0.0000629719`이나 최소 fold `-0.0000083590`·seed 일관성 실패; C2도 네 gate 실패 | completed, no candidate; C0 fallback | — | handoff SHA-256 `fb6200aa…c896` |
| 22 | `failure_expert_label_audit_v1` | 실패 유형 라벨 CPU 감사 | 2021·2022·2023·2024 cutoff | coverage `0.9644~0.9650`, middle·reverse overlap `0.0716~0.0744`; 세 유형 모두 공통 gate 실패 | ineligible; no model training | — | review SHA-256 `0e03e2b5…ba8c` |
| 23 | `tree_hetero_residual_s3_v1` | E2 anchor + XGBoost·LightGBM 잔차 보정 | 2021→2022·2022→2023 구조 선택, 2023→2024 확인 | XGBoost gain `0.0000058661`·상관 `0.9985597`, LightGBM gain `0.0000125015`·상관 `0.9987254`; 두 후보 모두 2022→2023 악화 | rejected; fixed direct residual structure closed | — | [rejection](rejections/tree_hetero_residual_s3_rejection.json) |
| 24 | `tree_privileged_trackman_teacher_v1` | TrackMan exact-pitch teacher 적격성 감사 | 전체·최신 시즌 exact match coverage | 전체 약 `0.39%`, 최신 약 `0.29%`; 요구 기준 `30%/20%`에 크게 미달 | diagnostic; exact-match distillation closed | — | handoff SHA-256 `5754124…ddd2` |
| 25 | `tree_privileged_profile_p_only_v1` | E2 anchor + rolling target profile CatBoost | 2021→2022·2022→2023·2023→2024 | 공식 weighted gain `0.0000202678`, latest `0.0000224496`, 3 folds 개선; 최소 gain `0.00005` 미달 | rejected; profile definition must change | — | handoff SHA-256 `b89c7757…bd8fc` |
| 26 | `failure_regime_e3_v1` | E2 anchor + 전역·최근·R/F 성공 전문가 + middle·wild·reverse 전문가 + 행 단위 gate | 2021→2022·2022→2023 구조 선택, 2023→2024 확인, 3 seed 계획 | OOF 작업 63개 중 53개 완료: screening 14, confirmation 7, extra seed 32. Kaggle Version 70이 `33,677.4초` 뒤 메모리 부족으로 종료돼 판정·전체 학습·감사에 도달하지 못함 | failed; resource memory, 제출 패키지 차단 | — | [failure evidence](evidence/failure_regime_e3_20260902.json) |

## 원본 증거 재확보가 필요한 실행

아래 실행은 대화상 완료 기록이 있으나 현재 로컬에서 최종 review 또는 handoff 원본을
검증하지 못했다. 결과를 추정해 장부에 넣지 않고 원본을 다시 확보할 때까지 보류한다.

| 실험 | 필요한 증거 |
|---|---|
| `temporal_portfolio_t1` | review 또는 handoff bundle |
| `temporal_portfolio_t2a` | review 또는 handoff bundle |
| `temporal_portfolio_t2b` | review 또는 handoff bundle |
| `temporal_portfolio_t2c` | review 또는 handoff bundle |
| `tree_expert_rf` | review 또는 handoff bundle |
| `anchor_residual_hierarchical_s4` | review 또는 handoff bundle |

## 현재 결론

- R9은 누출 없는 공통 비교 기반이다. FwFM은 단독·제한 blend·종료 감사까지 끝나
  계열을 닫았고, calibration의 고정 세 변형은 기각하되 다른 독립 구성은 열어 뒀다.
- XGBoost 공격적 탐색은 Public `820.9583317093`을 기록했지만 평가 예측 평균 이동
  보정이 현재 독립 예측 규칙과 맞지 않는다. 점수와 교훈만 보존하고 모델·후처리는
  재사용하거나 패키징하지 않는다.
- 확정 전처리의 단일 TabM은 Public `872.3920184667`을 기록했다. 최대 43 epoch
  검증의 최적 시점은 2~3 epoch였고, OOF seed 평균도 단일 seed보다 나빠 같은 모델을
  더 오래 학습하거나 seed만 늘리는 방향은 종료했다.
- 현재 최고 제출은 Tree Expert E2다. 세 시간 fold에서 모두 좋아졌고 CatBoost seed
  42·2026·3407 평균이 승인됐다. 245,789행 reverse·shuffle·rebatch·singleton
  감사의 최대 예측 차이는 `0`이었다. 제출 ZIP SHA-256은
  `8bc33042b9258049f905df0c733b1153d1cedfac39cb0732b2a7e154629d779a`,
  Public 점수는 `977.3809532715`다.
- E2 이후 T3 시간 가중, 계층 잔차와 실패 유형 라벨은 각자의 사전 gate에서 탈락했다.
  CatBoost·TabM 70:30은 정렬 OOF에서 좋아졌지만 배포와 같은 트리 수로 다시 맞추지
  못해 제출을 차단했다. 이 판정은 E2를 취소하지 않으며 해당 추가 구성만 막는다.
- S3의 XGBoost와 LightGBM은 평균적으로 아주 조금 좋아졌지만 최소 gain `0.00005`에
  못 미쳤고, E2 잔차와의 상관도 둘 다 `0.998`을 넘었다. 현재 피처와 직접 확률 보정
  구조는 닫되 두 모델 계열 전체의 실패로 확대하지 않는다. 다음 후보는 E2와 다른
  오차를 만들 수 있는 시즌·상황 정보부터 다시 설계한다.
- TrackMan exact-pitch teacher는 매칭 coverage가 전체 약 `0.39%`에 불과해 모델
  학습 전에 계열을 닫았다. rolling target profile P-only 후보는 세 fold 모두 양의
  방향이었지만 공식 weighted gain이 기준의 약 40.5%에 그쳤다. 별도 3-seed 평균
  진단도 `0.0000418138`로 기준에 못 미쳤고 2023 fold gain은 `0.0000045861`이었다.
  같은 profile 정의와 exact-match 구성을 반복하지 않는다.
- E3는 과거 failure audit의 수치를 숨기지 않고 해석을 바꾼 후보였다. 기존 감사의
  `7.16~7.44%` middle·reverse 중첩을 삭제 사유가 아니라 두 현상이 함께 나타나는
  신호로 보존했고, `wild`는 성공도 middle도 reverse도 아닌 실패로 정의했다.
  실제 실행에서는 OOF 작업 63개 중 53개까지 끝났지만 약 9시간 21분 뒤 메모리 부족으로
  종료됐다. 저장 상태는 `extra_seeds`였고 decision·full fit·audit는 시작하지 못했다.
  따라서 부분 Brier로 성능을 주장하지 않으며 이 후보는 `rejected`가 아닌 `failed`다.
  다음 반복은 모델 가설보다 먼저 파일럿 실측으로 peak RAM과 작업 시간을 계산하고,
  단계 사이 모델 풀 해제와 단일 최신 resume 보존을 계약에 포함해야 한다.
- 대용량 OOF, 모델과 ZIP은 Google Drive에 두고 이 장부는 작은 evidence와 실행 ID,
  SHA-256으로 원본을 식별한다.

기계 판독 가능한 전체 기록은 [experiment registry](experiment_registry.json), 비교
그룹별 감사와 다음 방향은 [실험 증거 재감사](EXPERIMENT_RESET_AUDIT.md)를 기준으로
한다.
