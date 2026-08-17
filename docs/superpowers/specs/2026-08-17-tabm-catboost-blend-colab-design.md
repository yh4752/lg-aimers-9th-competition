# TabM·CatBoost 시간 전이 블렌드 Colab 설계

## 배경

Stage P에서 `dl_standard + hand_matchup` TabM을 기준으로 여섯 가지 행 단위 피처를
검증했지만 승급 후보가 없었다. 따라서 행 피처 조합을 더 넓히지 않고, 미리 정한 다음
순서인 이질 모델 블렌드로 넘어간다.

기존 Stage C delivery에는 최종 선택된 `single_s3407` TabM의 `2022→2023`,
`2023→2024` 예측이 모두 들어 있다. 검증된 예측을 다시 만들기 위해 TabM을 재학습하지
않고, 같은 행과 같은 시간 fold에서 새 CatBoost OOF만 만든다.

이 단계의 목적은 CatBoost 단독 최고점을 찾는 것이 아니다. CatBoost가 TabM과 다른
오차를 내서 고정 비율 블렌드에 도움이 되는지 확인하는 것이 목적이다.

## 확정한 접근

### 채택: 기존 TabM OOF 재사용 + CatBoost 한 구성

- TabM 기준선: Stage C에서 봉인된 `single_s3407`
- CatBoost seed: `42`
- CatBoost 전처리: `tree_native + hand_matchup`
- 시간 fold: `2022→2023`, `2023→2024`
- 블렌드의 TabM 비중: `0.70`, `0.80`, `0.90`

CatBoost의 역할은 모델 다양성 확인이므로 첫 실행에서 seed 탐색, 깊이 탐색 또는 학습률
탐색을 섞지 않는다. 블렌드가 통과한 뒤에만 별도 설계로 CatBoost 안정성을 확인한다.

### 보류: CatBoost 두 seed 동시 실행

seed 평균이 분산을 줄일 수 있지만 GPU 비용이 약 두 배가 되고, CatBoost seed와 blend
비중을 동시에 선택하게 된다. 첫 이질 모델 검증에서 필요한 범위보다 넓다.

### 기각: TabM과 CatBoost 동시 재학습

Stage C의 두 fold TabM 예측이 이미 검증되어 있다. 동일한 TabM을 다시 학습하면 비용은
늘지만 CatBoost의 추가 기여를 판단하는 근거는 좋아지지 않는다.

## 입력과 신뢰 경계

Colab 셀은 직접 업로드만 사용한다. Drive, GitHub와 외부 URL에서 코드나 데이터를
받지 않는다.

첫 실행의 필수 입력은 다음 두 개다.

1. 공식 학습 데이터만 담은 입력 ZIP
2. `tabm_colab_stage_C_delivery.zip`

재개 실행에서는 `catboost_tabm_blend_resume.zip` 한 개를 추가로 올릴 수 있다.
Stage C delivery는 다음 조건을 모두 만족해야 한다.

- outer delivery manifest와 모든 member SHA-256 일치
- inner Stage C review bundle 검증 성공
- `stage_complete=true`
- `selected_predictor=single_s3407`
- seed `3407`, 두 fold prediction 존재
- prediction 파일의 `row_id`, target, prediction schema와 행 수 일치

현재 확인한 Stage C delivery SHA-256은
`f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a`다. 구현은
파일명만 신뢰하지 않고 manifest와 내부 hash를 다시 검증한다.

공식 입력에서는 학습에 필요한 세 파일만 허용한다. 테스트 데이터,
`sample_submission.csv`, 제출 결과와 이전 Public prediction은 입력으로 받지 않는다.

## CatBoost 학습

두 job을 순서대로 실행한다.

1. 2022년까지 학습하고 2023년 검증
2. 2023년까지 학습하고 2024년 검증

기존 CatBoost 전처리 연구에서 안정적으로 개선된 `tree_native + hand_matchup`을 쓴다.
`hand_matchup`은 같은 행에 있는 투수·타자 손잡이 정보만 조합하며 target이나 다른 검증
행에 의존하지 않는다. 범주 사전과 모든 학습 상태는 각 fold의 학습 구간에서만 만든다.

초기 고정 설정은 다음과 같다.

| 설정 | 값 |
|---|---:|
| model | `CatBoostRegressor` |
| loss / eval metric | `RMSE` / `RMSE` |
| iterations | `400` |
| depth | `7` |
| learning rate | `0.05` |
| border count | `128` |
| bootstrap | `Bayesian` |
| bagging temperature | `1.0` |
| l2 leaf reg | `3.0` |
| model size reg | `0.5` |
| max CTR complexity | `1` |
| seed | `42` |
| task type | `GPU` |
| overfitting detector | `Iter`, wait `50` |

snapshot을 쓰기 위해서만 `allow_writing_files=True`로 두고 `train_dir`은 해당 run의
격리된 작업 폴더로 제한한다. 그 밖의 CatBoost 분석 파일은 review bundle에 넣지 않는다.

## 정렬과 평가

각 fold에서 TabM과 CatBoost는 같은 검증 행을 사용한다. 다음 조건 중 하나라도 다르면
점수를 계산하지 않고 실패한다.

- `row_id`의 값, 순서와 유일성
- target의 값과 결측 상태
- prediction의 길이, 유한성, `[0, 1]` 범위
- fold 학습·검증 연도

비교 후보는 아래 네 개뿐이다.

- TabM 단독
- `0.90 × TabM + 0.10 × CatBoost`
- `0.80 × TabM + 0.20 × CatBoost`
- `0.70 × TabM + 0.30 × CatBoost`

연속 가중치 최적화, Public 점수 기반 선택과 사후 grid 확장은 하지 않는다. 후보마다
아래 값을 기록한다.

- fold별 Brier
- 검증 행 수 가중 평균 Brier
- TabM 대비 fold별·가중 평균 delta
- TabM·CatBoost prediction Pearson correlation
- 두 모델 residual Pearson correlation
- 사전에 정한 구간별 Brier와 행 수

승급 조건은 기존 공통 계약을 그대로 쓴다.

- 검증 행 수 가중 평균 Brier가 TabM보다 `0.00003` 이상 낮음
- 두 fold 중 어느 하나도 TabM보다 `0.00003`를 초과해 악화하지 않음

여러 가중치가 통과하면 가중 평균 Brier가 가장 낮은 후보 하나만 선택한다. 동률은
TabM 비중이 높은 후보를 우선한다. 구간 결과는 진단용으로 남기며 실행 후 임의의 차단
기준을 만들지 않는다.

## 중단 복구

CatBoost 공식 snapshot 기능을 사용한다.

- `save_snapshot=True`
- 내부 snapshot 간격 300초
- 같은 데이터, 코드, 설정과 snapshot 파일일 때만 재개
- 완료된 fold는 다시 학습하지 않음
- 진행 중 fold는 마지막 검증된 snapshot 이후부터 재개

브라우저 자동 다운로드는 내부 snapshot마다 실행하지 않는다. 약 20분마다 snapshot이
실제로 갱신됐을 때만 emergency resume ZIP을 내려받고, 각 fold 완료 직후에도 한 번
내려받는다. 정상 완료 시 최종 review와 resume만 내려받는다. 이 방식으로 이전 Stage P처럼
다운로드 요청이 계속 쌓이는 문제를 피하면서, 세션이 끊겼을 때 잃는 학습 시간을 약
20분 이내로 제한한다.

resume bundle은 다음 identity를 봉인한다.

- 공식 입력 파일별 SHA-256과 입력 manifest SHA-256
- Stage C delivery 및 inner review manifest SHA-256
- embedded code와 dependency lock SHA-256
- CatBoost 설정과 fold identity
- 완료 결과, prediction과 model SHA-256
- 진행 중 snapshot SHA-256과 생성 시각

identity가 하나라도 다르면 완료 결과나 snapshot을 재사용하지 않고 명확한 오류로
중단한다. 손상된 resume가 독립적인 새 실행을 막지는 않는다.

## Colab 셀과 로그

사용자에게는 1MB 미만의 복사용 Python 셀 하나를 제공한다. 셀에는 필요한 프로젝트
런타임의 고정 inventory만 포함하며 repository clone이나 package source 다운로드를
하지 않는다. CatBoost 설치가 필요한 경우 버전을 고정하고 설치 결과를 로그와 manifest에
남긴다.

주요 로그 marker는 다음과 같다.

- `BLEND_CODE_READY`
- `BLEND_INPUTS_VERIFIED`
- `BLEND_GPU_READY`
- `BLEND_RESUME_READY`
- `CATBOOST_JOB_START`
- `CATBOOST_PROGRESS`
- `CATBOOST_SNAPSHOT_READY`
- `CATBOOST_FOLD_RESULT`
- `BLEND_WEIGHT_RESULT`
- `BLEND_DECISION`
- `BLEND_BUNDLE_SUCCESS`
- `BLEND_DOWNLOAD_REQUESTED`
- `BLEND_ERROR`

오류는 stage, exception type과 짧은 message를 함께 출력한다. 정상 완료 전에 review ZIP을
성공 산출물처럼 표시하지 않는다.

## 시간과 자원

- 가속기: Colab Tesla T4 한 장
- 예상 시간: 약 45분~2시간
- absolute session deadline: 업로드와 설치를 포함해 3시간
- GPU job 수: CatBoost 두 fold
- TabM 재학습: 없음

deadline이 가까워지면 새 fold를 시작하지 않고 최신 검증 snapshot 또는 완료 결과로
resume bundle을 만든 뒤 종료한다.

## 산출물과 다음 단계

정상 완료 시 두 파일을 만든다.

- `catboost_tabm_blend_review.zip`
- `catboost_tabm_blend_resume.zip`

review에는 OOF prediction, fold/가중 평균 metric, 상관 분석, blend 판정, 로그와 manifest를
넣는다. resume에는 완료 모델과 진행 중 snapshot 등 재개에 필요한 자료만 넣는다.

두 ZIP은 연구 산출물이며 제출 파일이나 전체 학습 모델이 아니다. blend가 승급 조건을
통과하면 선택된 고정 비율만 사용해 전체 학습·추론·제출 후보를 별도 설계한다. 통과하지
못하면 CatBoost를 제출 후보에 붙이지 않고 현재 TabM을 유지한다.

## 대회 규칙 보호

- 테스트 데이터와 sample submission을 읽지 않는다.
- 검증 행 전체의 평균, 빈도, 순위, 분위수로 feature나 prediction을 바꾸지 않는다.
- 평가 행 사이에서 상태를 전달하지 않는다.
- Public 점수로 seed, 설정 또는 blend 비중을 고르지 않는다.
- 이 단계에서 full-data inference와 제출 ZIP을 만들지 않는다.
- review 통과만으로 기존 제출 후보나 repository 산출물을 덮어쓰지 않는다.
