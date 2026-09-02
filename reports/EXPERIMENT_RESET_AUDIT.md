# 실험 증거 재감사

## 결론

확인된 실험은 **26건**, 원본 결과가 없어 보류한 실행은 **6건**이다. 서로 다른 `comparison_group`의 Brier를 한 순위로 합치지 않았으며, 실행 실패와 성능 기각도 분리했다.

현재 최고 규칙 준수 Public 결과는 Tree Expert E2다. E2 이후 후보는 일부 양의 OOF 신호를 보였지만 최소 개선량, 시즌 안정성, 배포 정렬 또는 오차 다양성 중 하나 이상을 통과하지 못했다.

## 실제 Public 제출

| 실험 | Public | 증거 | 규칙 상태 |
|---|---:|---|---|
| `catboost_smooth_v1` | `828.9963889533` | C | unknown |
| `fwfm_standalone` | `88.5362742196` | C | unknown |
| `xgboost_aggressive_capacity_v1` | `820.9583317093` | B | quarantined |
| `tabm_hand_matchup_version_d_seed3407_v1` | `872.3920184667` | B | passed |
| `tree_expert_e2_c1_catboost` | `977.3809532715` | B | passed |

Public 점수는 OOF Brier와 다른 척도이며, 이 5건으로 점수 환산식이나 사후 가중치를 맞추지 않는다.

## 비교 가능한 OOF 그룹

아래 표는 그룹 내부 결과만 비교하기 위한 것이다. 그룹 사이의 Brier 크기는 학습 행과 fold가 다를 수 있어 직접 순위를 의미하지 않는다.

### `comparison_group=catboost_2024_holdout`

| 실험 | 상태 | Brier | weighted gain | worst fold | latest fold | residual corr | 증거 |
|---|---|---:|---:|---:|---:|---:|---|
| `catboost_smooth_v1` | public_scored | 0.248033 | — | — | — | — | C |

### `comparison_group=catboost_2024_quick_gate`

| 실험 | 상태 | Brier | weighted gain | worst fold | latest fold | residual corr | 증거 |
|---|---|---:|---:|---:|---:|---:|---|
| `catboost_proxy_01_no_smoothing` | rejected | — | -0.000145 | — | — | — | C |
| `catboost_proxy_02_k125` | rejected | — | -3e-05 | — | — | — | C |
| `catboost_proxy_03_iter800` | rejected | — | -0.00013 | — | — | — | C |

### `comparison_group=e2_temporal_3fold`

| 실험 | 상태 | Brier | weighted gain | worst fold | latest fold | residual corr | 증거 |
|---|---|---:|---:|---:|---:|---:|---|
| `tree_expert_e2_c1_catboost` | public_scored | — | 0.0008375137 | 0.0003439904 | — | — | B |
| `tree_expert_t3_temporal_dual_v1` | rejected | — | -5.05772e-05 | — | — | — | B |
| `tree_hierarchical_residual_v1` | rejected | — | 6.29719e-05 | -8.359e-06 | — | — | B |
| `tree_hetero_residual_s3_v1` | rejected | — | 1.25014545316763e-05 | -1.36291450071454e-05 | 1.70650261923988e-05 | 0.998725379488123 | B |
| `tree_privileged_profile_p_only_v1` | rejected | — | 2.02677513555682e-05 | — | 2.24495818452464e-05 | — | A |
| `failure_regime_e3_v1` | failed | — | — | — | — | — | A |

### `comparison_group=failure_label_cutoff_audit`

| 실험 | 상태 | Brier | weighted gain | worst fold | latest fold | residual corr | 증거 |
|---|---|---:|---:|---:|---:|---:|---|
| `failure_expert_label_audit_v1` | diagnostic | — | — | — | — | — | B |

### `comparison_group=fwfm_exit_audit`

| 실험 | 상태 | Brier | weighted gain | worst fold | latest fold | residual corr | 증거 |
|---|---|---:|---:|---:|---:|---:|---|
| `fwfm_failure_boundary_exit_audit` | diagnostic | — | — | — | — | — | B |

### `comparison_group=round9_meta_2fold`

| 실험 | 상태 | Brier | weighted gain | worst fold | latest fold | residual corr | 증거 |
|---|---|---:|---:|---:|---:|---:|---|
| `tabm_residual` | rejected | — | -0.0041512419345538 | — | — | — | B |

### `comparison_group=round9_temporal_3fold`

| 실험 | 상태 | Brier | weighted gain | worst fold | latest fold | residual corr | 증거 |
|---|---|---:|---:|---:|---:|---:|---|
| `calibration_blending` | rejected | 0.247898906769848 | 0.00035208847654155 | — | — | — | B |
| `fwfm_standalone` | public_scored | 0.250179784642168 | -0.00192878939577833 | — | — | — | C |
| `r9_fwfm_game_type_f_blend` | rejected | 0.247251455217578 | 0.00099954002881125 | -0.000534024610467615 | — | — | B |
| `r9_fwfm_game_type_f_blend_w080` | rejected | 0.247288498906444 | 0.00096249633994519 | — | — | — | B |
| `round9_temporal_oof` | accepted | 0.248250995246389 | — | — | — | — | B |

### `comparison_group=tabm_catboost_temporal_2fold`

| 실험 | 상태 | Brier | weighted gain | worst fold | latest fold | residual corr | 증거 |
|---|---|---:|---:|---:|---:|---:|---|
| `catboost_tabm_fixed_blend_v1` | accepted | 0.2492874755 | 0.0001787611 | — | — | — | B |
| `catboost_deployment_alignment_v1` | rejected | — | 3.20407e-05 | — | -0.0001464537 | — | B |

### `comparison_group=tabm_temporal_2fold`

| 실험 | 상태 | Brier | weighted gain | worst fold | latest fold | residual corr | 증거 |
|---|---|---:|---:|---:|---:|---:|---|
| `tabm_hand_matchup_version_d_seed3407_v1` | public_scored | — | — | — | — | — | B |
| `tabm_seed_ensemble` | rejected | 0.2494662366 | -0.000318 | — | — | — | B |

### `comparison_group=trackman_exact_match_eligibility`

| 실험 | 상태 | Brier | weighted gain | worst fold | latest fold | residual corr | 증거 |
|---|---|---:|---:|---:|---:|---:|---|
| `tree_privileged_trackman_teacher_v1` | diagnostic | — | — | — | — | — | A |

### `comparison_group=xgboost_2024_holdout`

| 실험 | 상태 | Brier | weighted gain | worst fold | latest fold | residual corr | 증거 |
|---|---|---:|---:|---:|---:|---:|---|
| `xgboost_original_preproc_rescue` | accepted | 0.248266874140416 | — | — | — | — | B |

### `comparison_group=xgboost_aggressive_public`

| 실험 | 상태 | Brier | weighted gain | worst fold | latest fold | residual corr | 증거 |
|---|---|---:|---:|---:|---:|---:|---|
| `xgboost_aggressive_capacity_v1` | public_scored | 0.247927021363851 | — | — | — | — | B |

### `comparison_group=xgboost_historical_4fold`

| 실험 | 상태 | Brier | weighted gain | worst fold | latest fold | residual corr | 증거 |
|---|---|---:|---:|---:|---:|---:|---|
| `xgboost_score_push_v3` | accepted | 0.247017377566481 | — | — | — | — | B |

## 반복 기각 원인

| 원인 | 건수 |
|---|---:|
| `data_eligibility` | 3 |
| `deployment_alignment` | 1 |
| `diversity` | 1 |
| `instability` | 5 |
| `performance` | 7 |
| `rule_quarantine` | 1 |
| `runtime` | 1 |

## 증거가 부족한 실행

이 항목은 결과를 추정하지 않는다. 원본 review 또는 handoff를 다시 확보한 뒤 검증 장부에 추가한다.

| 실험 | 부족한 증거 | 사유 |
|---|---|---|
| `anchor_residual_hierarchical_s4` | review or handoff bundle | result bundle is not currently available |
| `temporal_portfolio_t1` | review or handoff bundle | result bundle is not currently available |
| `temporal_portfolio_t2a` | review or handoff bundle | result bundle is not currently available |
| `temporal_portfolio_t2b` | review or handoff bundle | result bundle is not currently available |
| `temporal_portfolio_t2c` | review or handoff bundle | result bundle is not currently available |
| `tree_expert_rf` | review or handoff bundle | result bundle is not currently available |

## 닫을 계열과 다시 정의할 계열

- 모든 검토 변형을 닫은 계열: `failure_expert`, `fwfm`, `trackman_teacher`
- 유지 기준선 없이 구조를 다시 정의할 계열: `calibration`, `xgboost_lightgbm`
- 다음 비교의 기준으로 유지할 실험: `catboost_smooth_v1`, `round9_temporal_oof`, `tabm_hand_matchup_version_d_seed3407_v1`, `tree_expert_e2_c1_catboost`, `xgboost_original_preproc_rescue`, `xgboost_score_push_v3`
- 같은 모델 계열 안에서도 변형별 판정이 다르면 계열 전체를 닫지 않고 registry의 `repeat_policy`를 따른다.

## 다음 단일 캠페인

다음은 작은 확률 보정이나 단일 feature 추가가 아니라 **구조적으로 깊은 캠페인**으로 설계한다. 여기서 깊다는 말은 트리 depth만 크게 만든다는 뜻이 아니다.

1. 학습 데이터에서 cutoff를 지켜 만든 시즌별 선수·상황 snapshot과 시간 감쇠 anchor
2. 정규 시즌 `R`과 `F`를 나누는 전문가
3. 공식 학습 데이터로 안정적으로 정의되는 상황 전문가
4. 충분한 용량의 CatBoost 다중 seed
5. E2와 실제 오차 다양성이 확인될 때만 추가하는 XGBoost·LightGBM residual
6. 구조 fold에서 강도를 고정한 residual correction과 마지막 계층 calibration

이 구성은 탐색 깊이를 높이기 위한 것이며 점수를 보장하지 않는다. 먼저 저비용 적격성 검사로 입력 신호와 2023 fold 병목을 확인하고, 통과한 구조에만 Kaggle T4 x2 장시간 예산을 배정한다.
