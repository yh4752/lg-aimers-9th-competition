# CatBoost 고정 트리 수 확인 및 전체 학습

이 실행은 앞선 OOF 블렌드에서 정한 `TabM 0.70 + CatBoost 0.30`을 그대로 둔 채,
실전에 쓸 CatBoost 트리 수 하나를 고르는 과정이다. 가중치를 다시 탐색하지 않는다.
두 시간 전이 fold에서 같은 트리 수가 기준을 통과한 경우에만 전체 학습을 시작한다.

여기서 만드는 파일은 검토용 학습 산출물이다. `test.csv`를 읽거나 추론하지 않으며,
제출 ZIP도 만들지 않는다.

## 준비할 파일

새 실행에서는 아래 ZIP 3개를 한 번에 선택한다. 파일명보다 ZIP 내부 구조와 해시를
먼저 확인하므로 이름을 바꿔도 되지만, 원본 파일을 그대로 쓰는 편이 안전하다.

1. `catboost_tabm_blend_input.zip`
2. `tabm_colab_stage_C_delivery.zip`
3. `catboost_tabm_blend_delivery.zip`

중단된 작업을 이어갈 때는 여기에 가장 최근에 받은
`catboost_deployment_emergency_*.zip`을 하나만 추가한다. 예전 Stage C resume나
CatBoost·TabM 블렌드 resume는 이 자리에 넣지 않는다.

## Colab에서 실행하는 순서

1. 새 Colab 노트북을 열고 런타임 가속기를 T4 GPU로 설정한다.
2. [COLAB_CATBOOST_DEPLOYMENT_CELL.py](../experiments/catboost_deployment/COLAB_CATBOOST_DEPLOYMENT_CELL.py)의
   내용을 전부 복사해 셀 하나에 붙여 넣는다.
3. 셀을 실행한다. 파일 선택 창이 열리면 준비한 ZIP 3개를 동시에 고른다.
4. 다운로드가 시작돼도 셀은 계속 실행 중일 수 있다. 최종 상태가 출력될 때까지
   브라우저 탭과 런타임을 닫지 않는다.

대체로 10~30분 정도 걸리지만 Colab 상태에 따라 더 길어질 수 있다. 코드는 업로드와
설치 시간을 포함해 시작 시점부터 3시간이 지나면 새 작업을 시작하지 않는다.

## 실제로 하는 일

먼저 2022→2023, 2023→2024 두 fold에서 400-tree CatBoost를 한 번씩 학습한다.
각 모델의 `4, 32, 64, 128, 192, 296, 400` tree 지점 예측을 꺼내 고정 70:30
블렌드의 Brier Score를 계산한다.

후보가 통과하려면 다음 두 조건을 모두 만족해야 한다.

- 전체 행 가중 Brier 개선량이 `0.00003` 이상
- 어느 fold에서도 기준 TabM보다 `0.00003` 넘게 나빠지지 않음

통과한 후보 중 행 가중 Brier가 가장 낮은 트리 수를 고른다. 차이가 `1e-12` 이내면
더 작은 모델을 택한다. 통과 후보가 없으면 `deployment_blocked`로 끝나며 전체
CatBoost는 학습하지 않는다.

## 로그 확인

정상 시작 시 아래 순서의 로그를 볼 수 있다.

```text
DEPLOY_CODE_READY
DEPLOY_DEPENDENCIES_READY catboost=1.2.10
DEPLOY_GPU_READY
DEPLOY_INPUTS_VERIFIED count=3
DEPLOY_RESUME_READY source=fresh
DEPLOY_JOB_START job=align_2022_2023
CATBOOST_DEPLOY_PROGRESS
DEPLOY_JOB_START job=align_2023_2024
DEPLOY_ALIGNMENT_DECISION status=deployment_aligned selected_tree_count=<N>
DEPLOY_JOB_START job=full_2024
DEPLOY_BUNDLE_SUCCESS status=full_training_complete
DEPLOY_DELIVERY_READY path=<path> sha256=<sha256>
```

`DEPLOY_ALIGNMENT_DECISION status=deployment_blocked`가 나오면 오류가 아니다. 사전에
정한 기준을 통과한 트리 수가 없다는 뜻이다.

오류는 한 줄로도 식별할 수 있게 다음 형식으로 남는다.

```text
DEPLOY_ERROR stage=<stage> type=<type> message=<message>
```

문의할 때는 이 줄만 자르지 말고 바로 위아래 로그도 함께 보낸다.

## 중단과 resume

CatBoost 자체 snapshot은 5분마다 갱신한다. 브라우저 다운로드는 snapshot이 바뀐
상태에서 20분 간격으로 요청하며, fold나 전체 학습이 끝났을 때도 검증된 resume를
내려받는다. 여러 ZIP이 다운로드되는 것은 정상이다. 가장 번호가 큰 파일이 아니라
다운로드 시각이 가장 최근인 검증 파일을 보관한다.

런타임이 끊기면 새 Colab 세션에서 셀을 다시 실행한다. 원본 입력 ZIP 3개와 최신
deployment resume 1개를 같은 파일 선택 창에서 올리면 된다. 완료된 fold는 재사용하고,
진행 중이던 CatBoost는 native snapshot에서 이어간다.

## 끝난 뒤 전달할 파일

상태에 따라 전달 파일이 다르다.

- `full_training_complete`: `catboost_full_training_delivery.zip`
- `deployment_blocked`: `catboost_deployment_review.zip`과 같은 시점의 resume ZIP
- `deployment_incomplete`: 가장 최근 deployment resume ZIP
- `DEPLOY_ERROR`: 가장 최근 deployment resume ZIP과 오류 전후 로그

`catboost_full_training_delivery.zip` 안에는 고정 CatBoost 모델, 전처리 상태,
두 fold 예측과 판정 근거가 들어 있다. 이 파일을 다시 검증해 해시가 맞아야 다음
단계인 TabM+CatBoost 독립 추론 검증을 설계한다. 이 단계가 끝나기 전에는 제출물을
만들지 않는다.

## 자주 생기는 문제

- `source count must be three or four`: 파일을 따로 올렸거나 개수가 맞지 않는다.
  새 실행은 3개, 재개는 4개를 한 번에 선택한다.
- `duplicate source kind`: 같은 종류의 ZIP을 두 번 넣었다.
- `unknown source archive`: Stage P resume, 예전 blend resume, 결과와 무관한 ZIP이
  섞였을 가능성이 크다.
- `one CUDA GPU is required`: 런타임을 GPU로 바꾼 뒤 처음부터 다시 실행한다.
- `blend source SHA-256 differs`: 앞서 검증한 CatBoost·TabM 블렌드 결과가 아닌 파일이다.

원본 데이터, 모델과 결과 ZIP은 Git에 올리지 않는다.
