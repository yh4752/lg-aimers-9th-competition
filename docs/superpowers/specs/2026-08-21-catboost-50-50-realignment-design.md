# CatBoost 50:50 배포 정렬 재검증 설계

**작성일:** 2026-08-21
**대회:** DACON 236743, LG Aimers 9기 Phase 2
**상태:** 구현 전 승인 설계

## 목적

OOF reset audit에서 확인된 `0.50 × TabM + 0.50 × CatBoost`를 배포 가능한
단일 후보로 옮길 수 있는지 검증한다. 기존 두 시간 폴드만으로 tree 수를 고르지
않고, 더 과거인 2022 검증 폴드를 새로 추가한다. 세 폴드에서 같은 CatBoost tree
수와 같은 blend weight가 모두 작동할 때만 전체 학습 후보를 만든다.

이 작업은 제출물을 만들지 않는다. 검증 gate를 통과하면 공식 학습 데이터 전체로
CatBoost 모델을 학습한 review-only delivery를 만들고, 제출 후보 승급과 패키징은
그 산출물을 다시 검토한 뒤 별도 단계로 진행한다.

기존 `0.70 × TabM + 0.30 × CatBoost` 배포 실험은 실패 기록과 산출물을 그대로
보존한다. 이 설계는 기존 계약을 수정하거나 실패를 성공으로 재해석하지 않는 별도
`deployment_alignment_v2` 실험이다.

## 현재 근거

2026-08-21 OOF reset audit 결과는 다음 산출물에 봉인되어 있다.

```text
artifact: oof_reset_audit_results.zip
sha256: 0740e5904edd1bad7c4a11c9ca5953e3650e04e8b5b9c3cdc340612ae865409a
decision: DIVERSE_BLEND
selected practical weight: 0.50 TabM + 0.50 CatBoost
```

`catboost_hand_matchup_seed_42`와 TabM의 50:50 조합은 기존 두 폴드에서 모두
TabM 단독보다 좋아졌다.

| 검증 fold | TabM 대비 Brier 개선량 |
|---|---:|
| 2019~2022 학습 → 2023 검증 | 0.00017567 |
| 2019~2023 학습 → 2024 검증 | 0.00018444 |

행 수 가중 개선량은 `0.00018013`, 최신 fold block bootstrap 하한은
`0.00009216`, eligible segment 최대 악화는 `0.00049369`였다. audit gate의
segment 한도 `0.00075` 안쪽이다.

하지만 원래 CatBoost early stopping의 최적 tree 수는 2023 검증에서 약 4,
2024 검증에서 약 296으로 크게 달랐다. fold마다 다른 tree 수를 사용하는 OOF
결과는 공식 학습 데이터 전체에 적용할 하나의 tree 수를 직접 지정하지 못한다.
기존 400-tree fold model의 prefix를 확인한 결과 4는 최신 fold를 악화시키고,
32 이상은 과거 fold를 악화시키는 경향이 있었다. 따라서 아직 확인하지 않은
`8~28` 구간을 좁은 고정 grid로 검증한다.

이 구간은 기존 결과를 본 뒤 정한 것이므로 완전히 독립적인 최초 가설 검정은 아니다.
사후 선택 위험을 줄이기 위해 구현 전에 grid와 gate를 이 문서에 봉인하고, 새 2022
fold를 추가하며, fold별 설정 변경을 금지한다.

## 실험 단위와 코드 경계

새 코드는 `experiments/catboost_50_50_realign/` 아래의 독립 패키지로 만든다.
기존 `experiments/catboost_deployment/`, TabM 제출 runtime, 기존 제출 후보와
Kaggle·Colab 셀은 수정하지 않는다. 새 패키지의 책임은 다음 네 가지뿐이다.

1. 로컬 입력 handoff 작성과 재귀 검증
2. 세 시간 폴드의 고정 50:50 배포 정렬 평가
3. gate 통과 시 CatBoost 전체 학습
4. review/resume delivery 작성과 검증

test.csv, sample_submission.csv, hidden 평가 정보, leaderboard 점수는 입력으로 받지
않는다. 추론 코드나 submission ZIP writer도 이 패키지에 두지 않는다.

## 입력 handoff

사용자가 여러 대형 파일을 Colab에서 반복해 올리지 않도록 로컬 준비 도구가 필요한
입력을 먼저 검증하고 ZIP 하나로 묶는다. 준비 도구의 출력은 다음 하나다.

```text
artifacts/catboost_50_50_realign_input.zip
```

원본은 아래 다섯 개다.

| 원본 | 고정 SHA-256 | 용도 |
|---|---|---|
| `catboost_tabm_blend_input.zip` | `43557583fbd78efc0d3d83ab7ad17f701a1bd63f2cd423234d1dbee30006fe00` | 공식 학습 데이터와 Trackman history |
| `tabm_colab_stage_C_delivery.zip` | `f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a` | 기존 두 fold TabM OOF와 고정 설정 |
| `catboost_deployment_emergency_1eef761337e3.zip` | `1eef761337e34054973a36fc4b098e278094f5fc7472b2b12240fc6f87be5f9f` | 기존 두 fold 400-tree model과 전처리 상태 |
| `catboost_deployment_review.zip` | `fc65693aa9439bdbd808fcd29c421c1cfcfcb23abebd1a2c6512109b034641e8` | 기존 prediction과 model evidence 연결 |
| `oof_reset_audit_results.zip` | `0740e5904edd1bad7c4a11c9ca5953e3650e04e8b5b9c3cdc340612ae865409a` | 50:50 선택 근거와 gate 결과 |

준비 도구는 파일명으로 신뢰하지 않는다. ZIP member allowlist, 크기와 압축비,
manifest, 모든 member hash, artifact kind와 identity를 확인한다. 특히 다음 연결이
모두 맞아야 출력 ZIP을 만든다.

- emergency resume의 fold prediction SHA와 review가 가리키는 prediction SHA
- review의 source data·code·contract binding
- audit inventory의 네 source artifact SHA
- audit의 고정 weight `0.50`, 두 fold 양의 개선, bootstrap과 segment 결과
- 공식 train과 Trackman history의 기존 identity

입력 handoff에는 검증에 필요한 공식 학습 데이터, 기존 fold model·state·prediction,
감사 decision과 provenance만 넣는다. 원본 ZIP 자체, 테스트 데이터, 제출 파일,
불필요한 로그 사본은 중첩하지 않는다.

## 시간 폴드와 모델 설정

세 fold는 다음과 같이 고정한다.

```text
F1: 2019~2021 학습 → 2022 검증   # 신규 학습
F2: 2019~2022 학습 → 2023 검증   # 기존 검증 산출물 재사용
F3: 2019~2023 학습 → 2024 검증   # 기존 검증 산출물 재사용
```

F1 TabM은 Stage C에서 승급한 다음 설정을 그대로 사용한다.

```text
candidate: a__p2__piecewise_linear__bce__plateau
seed: 3407
k: 32
width: 512
blocks: 4
dropout: 0.10
numeric embedding: piecewise_linear
loss: BCE
optimizer schedule: plateau
learning rate: 0.0006
```

전처리도 Stage C에 봉인된 전처리를 사용한다. F1 검증 데이터는 early stopping과
best checkpoint 선택에만 사용하고, 그 fold 밖의 모델이나 통계를 고르는 데 쓰지
않는다.

F1 CatBoost는 audit의 `catboost_hand_matchup_seed_42` 설정을 그대로 사용해 최대
400 trees까지 한 번 학습한다. 기존 F2·F3 400-tree model도 같은 config·feature
contract·CatBoost version인지 검증한 뒤 재사용한다. 세 model에서 아래 prefix의
prediction만 계산한다.

```text
4, 8, 12, 16, 20, 24, 28, 32
```

prefix마다 별도 모델을 학습하지 않는다. 동일한 400-tree model에 `ntree_end`를
적용한다. grid, seed, feature bundle, fold 또는 blend weight를 실행 결과에 따라
늘리거나 바꾸지 않는다.

## 50:50 평가와 gate

각 fold와 각 prefix에서 다음 예측을 계산한다.

```text
p_blend = 0.50 * p_tabm + 0.50 * clip(p_catboost, 0, 1)
gain = brier(p_tabm, y) - brier(p_blend, y)
```

동일한 tree 수 하나가 다음 조건을 모두 만족해야 한다.

1. F1, F2, F3의 `gain`이 각각 `0`보다 큼
2. 세 fold 검증 행 수 가중 gain이 `0.00003` 이상
3. 최신 F3의 block bootstrap 95% 신뢰구간 하한이 `0` 이상
4. 세 fold를 합친 eligible segment의 최대 Brier 악화가 `0.00075` 이하
5. 모든 prediction의 행 ID·label·fold provenance와 유한성·범위가 검증됨

threshold 비교는 binary float의 경계 흔들림을 피하도록 decimal 문자열에서 만든
`Decimal`로 수행한다. block bootstrap seed, block 정의, segment 최소 표본 수와
eligible 규칙은 OOF reset audit 계약을 그대로 사용한다.

통과 후보가 여러 개면 다음 순서로 하나를 선택한다.

1. `min(F1 gain, F2 gain, F3 gain)`이 큰 후보
2. 동률이면 행 수 가중 gain이 큰 후보
3. 다시 동률이면 tree 수가 작은 후보

비교 동률 허용오차와 정렬 순서를 계약 파일에 명시한다. 실행 후 사람 판단으로
순위를 뒤집지 않는다.

하나도 통과하지 못하면 상태는 `deployment_blocked`다. 이 경우 전체 학습 model,
승급 decision 또는 제출 가능 artifact를 만들지 않는다. review와 재현 가능한
resume만 남긴다.

## 전체 학습

배포 정렬 gate를 통과한 경우에만 같은 실행에서 CatBoost를 공식 학습 데이터 전체로
한 번 학습한다. feature bundle, seed 42, hyperparameter와 선택된 tree 수는 OOF
후보와 정확히 같아야 한다. 전체 학습에는 validation set, early stopping 또는
`use_best_model`을 사용하지 않는다.

TabM은 기존에 검증된 전체 학습 model bytes를 그대로 재사용한다. 이 실행에서
TabM 전체 모델을 다시 학습하거나 두 모델의 weight를 재조정하지 않는다. 전체 학습
delivery는 추후 고정 TabM과 결합할 CatBoost model과 다음 evidence를 담는다.

```text
decision/alignment_decision.json
decision/fold_metrics.json
decision/bootstrap_metrics.json
decision/segment_metrics.json
frozen_catboost/model.cbm
frozen_catboost/preprocessing_state.json
frozen_catboost/inference_manifest.json
logs/campaign.log
policy/policy.json
manifest.json
```

정상 완료 파일명은 `catboost_50_50_realign_delivery.zip`이다. delivery에는 test
prediction, submission.csv, test schema에서 fit한 통계, optimizer나 불필요한
snapshot을 넣지 않는다.

## Colab T4 실행과 복구

실행용 셀은 한 개이며, 사용자는 `catboost_50_50_realign_input.zip` 하나만 직접
업로드한다. GitHub, Drive, 외부 URL에서 코드나 데이터를 읽지 않는다. 같은 Colab
runtime에서 셀을 다시 실행하면 검증된 업로드 cache를 재사용한다.

세션 제한은 업로드와 dependency 확인을 포함해 3시간으로 둔다. 예상 T4 실행 시간은
약 1~2시간이다. 새 job은 남은 안전 시간이 부족하면 시작하지 않는다.

- TabM은 epoch가 끝날 때 model·optimizer·scheduler·RNG를 원자적으로 저장
- CatBoost는 native snapshot과 현재 prefix 평가 상태를 저장
- state와 artifact는 임시 경로에서 검증한 뒤 원자적으로 latest pointer를 교체
- 손상되거나 identity가 다른 resume은 이전 정상 snapshot을 덮어쓰지 못함
- resume에는 absolute code·contract·data·history·job identity를 기록

브라우저 자동 다운로드는 반복 epoch마다 하지 않는다. 다음 시점만 허용한다.

1. 신규 F1의 TabM과 CatBoost가 모두 끝났을 때
2. 최종 성공했을 때
3. 오류 또는 deadline 안전 종료 시

따라서 정상 실행의 중간 emergency 다운로드는 한 번, 최종 delivery와 resume을
포함해도 다운로드 요청 수를 작게 유지한다. Colab runtime 안의 latest resume은
checkpoint가 진전될 때마다 갱신되므로 같은 runtime의 셀 재실행은 다운로드 파일
없이도 이어서 실행할 수 있다. runtime 자체가 사라진 뒤 재개하려면 가장 최근에
내려받은 resume을 입력 handoff와 함께 올린다.

핵심 로그 marker는 다음과 같이 고정한다.

```text
REALIGN_CODE_READY
REALIGN_GPU_READY
REALIGN_INPUTS_VERIFIED
REALIGN_JOB_START
REALIGN_TRAINING_PROGRESS
REALIGN_CHECKPOINT_READY
REALIGN_PREFIX_RESULT
REALIGN_DECISION
REALIGN_FULL_TRAINING_START
REALIGN_CAMPAIGN_SUCCESS
REALIGN_DOWNLOAD_REQUESTED
REALIGN_ERROR
```

`REALIGN_CAMPAIGN_SUCCESS status=completed`가 없으면 정상 완료로 간주하지 않는다.

## 규칙과 독립 예측 경계

이 단계는 공식 train과 과거 Trackman history만 사용한다. 평가 데이터나 리더보드
점수는 model, preprocessing, tree 수, weight 또는 calibration을 고르는 데 사용하지
않는다. `hand_matchup`은 현재 행과 공식 학습 데이터로 고정된 상태만 사용하며,
평가 batch의 다른 행에서 빈도·평균·순위·rolling·lag를 계산하지 않는다.

전체 학습 preprocessing state에는 category vocabulary, missing marker, source/output
schema와 학습 데이터 identity를 봉인한다. 추후 평가 시 unknown category는 고정 OOV
경로로 처리하고 평가 데이터에서 vocabulary를 다시 fit하지 않는다. 행 하나를 넣은
예측과 같은 행을 임의의 batch에 넣은 예측이 같아야 한다.

## 로컬 검증 범위

Codex는 전체 데이터나 GPU 학습을 실행하지 않는다. 구현 시 작은 synthetic fixture로
다음을 검증한다.

- ZIP traversal, duplicate member, symlink, 크기·압축비와 hash 변조 차단
- 다섯 source artifact와 audit decision의 재귀 identity 연결
- 세 fold row ID·label·prediction 정렬
- 정확한 8개 prefix와 고정 50:50 외 설정 거부
- Brier, block bootstrap, segment gate와 deterministic tie-break
- gate 실패 시 full training과 delivery 작성 차단
- checkpoint 중단·재개 및 stale/snapshot 변조 차단
- Colab 셀 1 MB 미만, embedded runtime inventory와 재생성 결정성
- 코드와 산출물에서 test, submission, Drive, GitHub, network source 접근 부재
- 기존 CatBoost deployment와 TabM campaign의 회귀 테스트

실제 공식 데이터 학습, T4 시간, model size와 최종 수치는 사용자가 Colab에서 실행해
받은 delivery로 검토한다.

## Acceptance와 다음 단계

이 캠페인의 성공은 제출 승인을 뜻하지 않는다. 다음을 모두 확인한 뒤에만 별도
candidate 승급 설계를 시작한다.

- `alignment_decision.status == "promoted"`
- 선택 tree 수가 고정 grid 중 하나이며 세 fold gate를 모두 통과
- 전체 학습 model·preprocessing·manifest의 재귀 hash 일치
- 실행 code·contract·data·history identity 일치
- Colab 최종 marker와 review metrics 일치
- 현재 대회 규칙과 평가 환경 재확인

그다음 단계에서만 기존 고정 TabM model과 새 CatBoost model을 50:50으로 묶고,
행 독립성·평가 서버 자원·설치 시간·추론 시간 감사를 수행한다. 그 acceptance까지
통과하기 전에는 submission package 생성 경로를 열지 않는다.

## 실패 처리와 비목표

- 입력·hash·schema 불일치: GPU 학습 전에 실패
- 새 fold 학습 중단: 최신 검증 resume만 게시하고 후보 판단 보류
- prefix gate 실패: `deployment_blocked`, 전체 학습 금지
- 전체 학습 또는 delivery 검증 실패: candidate import 금지
- deadline 도달: 기존 정상 resume을 보존하고 안전 종료
- 실패한 50:50 후보는 기존 독립 TabM 후보의 기록과 사용 가능성을 막지 않음

비목표는 TabM 구조 재탐색, CatBoost hyperparameter 탐색, blend weight 탐색,
calibration, 평가 데이터 추론, leaderboard 제출, submission ZIP 생성이다.
