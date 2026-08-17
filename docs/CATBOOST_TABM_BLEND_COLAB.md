# CatBoost·TabM 블렌드 Colab 실행 안내

이 실험은 TabM을 다시 학습하지 않는다. Stage C에서 확정한 `single_s3407`의
시간 전이 예측을 그대로 쓰고, CatBoost가 서로 다른 오차를 보이는지만 확인한다.
연구용 OOF 실험이므로 이 단계에서 제출 파일이나 전체 학습 모델은 만들지 않는다.

## 무엇을 학습하나

CatBoost job은 두 개이며 한 번에 하나씩 실행된다.

1. 2022년까지 학습하고 2023년을 검증
2. 2023년까지 학습하고 2024년을 검증

전처리는 `tree_native + hand_matchup`, seed는 `42`로 고정했다. 모델은
`CatBoostRegressor`이고 주요 설정은 400 iteration, depth 7, learning rate 0.05,
GPU, RMSE다. 실행 후 아래 네 후보만 비교한다.

- TabM 단독
- TabM 90% + CatBoost 10%
- TabM 80% + CatBoost 20%
- TabM 70% + CatBoost 30%

가중치는 실행 결과에 맞춰 추가로 찾지 않는다. 검증 행 수 가중 Brier가 TabM보다
`0.00003` 이상 좋아지고, 어느 fold에서도 `0.00003`을 초과해 나빠지지 않아야
승급한다. Public 점수나 평가 데이터는 선택에 쓰지 않는다.

## 1. 입력 ZIP 만들기

터미널에서 아래 명령을 그대로 실행한다.

```bash
cd /Users/yonghyun/Documents/lg-aimers-9th-competition

artifacts/tabm_submission_python311/bin/python \
  tools/prepare_catboost_tabm_blend_input.py \
  --data-dir /Users/yonghyun/Documents/kaggle-lg-aimers-9th-data-upload \
  --output artifacts/catboost_tabm_blend_input.zip
```

성공하면 다음 형식의 한 줄이 나온다.

```text
CATBOOST_BLEND_INPUT_READY path=... sha256=... size_bytes=...
```

입력 ZIP에는 `train.csv`, `trackman_history.csv`, 두 파일을 묶는 manifest만 들어간다.
`test.csv`와 제출 양식은 받지 않는다. 출력 파일이 이미 있으면 덮어쓰지 않으므로,
공식 데이터가 바뀌지 않았다면 기존 파일을 그대로 쓰면 된다.

## 2. Colab에서 실행하기

Colab 가속기를 T4 GPU로 설정한다. 아래 파일을 열어 전체 내용을 빈 셀 하나에
복사한다.

```text
experiments/catboost_tabm_blend/COLAB_CATBOOST_TABM_BLEND_CELL.py
```

첫 실행에서는 업로드 창에 정확히 두 파일을 넣는다.

```text
catboost_tabm_blend_input.zip
tabm_colab_stage_C_delivery.zip
```

중단된 실행을 이어갈 때만 최신 `catboost_tabm_blend_resume.zip`을 세 번째 파일로
추가한다. 파일명은 편의용이며, 셀은 ZIP 내부 manifest와 SHA-256으로 종류를 다시
확인한다. Stage C delivery도 기존에 검증한 정확한 산출물이어야 한다.

셀은 업로드 전에 3시간 제한을 시작한다. CatBoost 1.2.10을 설치하고 T4 이상의
CUDA 장치를 확인한 다음 두 fold를 순서대로 처리한다. 보통 45분~2시간을 예상한다.

## 중단과 재실행

CatBoost는 300초마다 native snapshot을 갱신한다. 셀은 snapshot이 실제로 바뀐
경우에만 20분 간격으로 검증된 emergency resume을 내려받는다. fold가 끝나면
시간 간격과 관계없이 새 resume을 한 번 더 내려받는다. 같은 snapshot을 반복해서
받지 않기 때문에 다운로드 창이 계속 쌓이지 않는다.

세션이 끊기면 가장 번호가 크거나 가장 늦게 받은 resume 하나만 보관하면 된다.
새 Colab 세션에서 입력 ZIP, Stage C delivery와 함께 올리면 완료된 fold는 재사용하고
진행 중이던 fold는 CatBoost snapshot부터 이어 간다. 코드, 설정, 공식 입력 또는
Stage C 해시가 달라지면 잘못된 재개를 막기 위해 실행 전에 중단한다.

## 확인할 로그

초기 설정이 정상이라면 다음 순서의 marker를 볼 수 있다.

```text
BLEND_CODE_READY
BLEND_DEPENDENCIES_READY catboost=1.2.10
BLEND_GPU_READY
BLEND_INPUTS_VERIFIED count=2
BLEND_RESUME_READY source=fresh
CATBOOST_JOB_START job=...
CATBOOST_PROGRESS job=... iteration=...
CATBOOST_FOLD_RESULT job=... brier=...
BLEND_WEIGHT_RESULT tabm_weight=...
BLEND_DECISION selected_tabm_weight=... reason=...
BLEND_DELIVERY_READY path=... sha256=...
```

`BLEND_INPUTS_VERIFIED`의 입력 수는 재개할 때 3이 된다. 오류가 나면 아래 한 줄과
가장 최근에 받은 emergency resume ZIP을 함께 전달한다.

```text
BLEND_ERROR stage=<stage> type=<type> message=<message>
```

## 완료 후 전달할 파일

정상 완료 때 자동으로 받는 파일은 다음 하나다.

```text
catboost_tabm_blend_delivery.zip
```

delivery 안에는 campaign log, review ZIP, resume ZIP, delivery manifest가 들어 있다.
이 파일을 그대로 전달하면 내부 해시와 Brier를 다시 검증한 뒤 결과를
`blend_promoted`, `blend_rejected`, `blend_incomplete` 중 하나로 판정한다.

`blend_promoted`가 확인되기 전에는 전체 데이터 CatBoost 학습이나 제출 후보 제작으로
넘어가지 않는다. 이 delivery 자체도 DACON 제출물이 아니다.
