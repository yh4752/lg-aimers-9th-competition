# XGBoost·LightGBM 잔차 보정 S3

이 단계는 기존 CatBoost OOF 예측을 버리지 않고, XGBoost와 LightGBM의 서로 다른 오차를 5~15%만 보정 성분으로 쓸 가치가 있는지 확인한다. 검증용 연구 단계이므로 제출 ZIP이나 최종 학습 모델을 만들지 않는다.

> 실행 결과: 여섯 작업은 모두 정상 완료됐고 두 후보는 기각됐다. XGBoost와
> LightGBM 모두 최소 개선량과 오차 다양성 기준을 넘지 못했다. 판정 근거는
> [S3 rejection](../reports/rejections/tree_hetero_residual_s3_rejection.json)에 있다.

## Kaggle 입력

Kaggle Notebook에 다음 두 데이터셋을 연결한다.

1. 공식 데이터 `lg-aimers-9th-data`
2. 로컬의 `artifacts/tree_expert_t3_input.zip`을 Kaggle Dataset으로 올린 입력

중단된 실행을 이어갈 때만 이전 `tree_hetero_handoff.zip`을 별도 Kaggle Dataset으로 추가한다. 안에 풀린 `tree_hetero_resume.zip`을 코드가 자동으로 찾는다. 원본 resume와 handoff를 동시에 추가하면 resume이 두 개로 잡히므로 하나만 둔다.

## 실행

Kaggle 가속기는 `GPU T4 x2`를 선택한다. 저장소의
`experiments/tree_expert/KAGGLE_HETERO_CELL.py` 전체를 Notebook의 한 셀에 붙여
넣고 `Save Version`으로 실행한다. 인터넷 연결은 패키지 버전 설치가 필요할 수 있으므로
켜 둔다.

예상 시간은 새 실행 기준 약 4~8시간이다. 구조 후보가 모두 탈락하면 더 일찍 끝나며, 통과한 모델 계열만 두 개 추가 시드로 확인한다. 작업 하나가 끝날 때마다 재개 상태를 갱신한다. 제한 시간 직전에는 새 작업을 시작하지 않는다.

정상 시작 로그는 다음 순서다.

```text
TREE_HETERO_CODE_READY
TREE_HETERO_DEPENDENCIES_READY
TREE_HETERO_GPU_READY
TREE_HETERO_INPUTS_VERIFIED
TREE_HETERO_JOB_START
```

완료 후 Kaggle Output에서 `tree_hetero_handoff.zip` 하나를 내려받아 전달한다. 성공 여부는 아래 로그로 확인한다.

```text
TREE_HETERO_HANDOFF_READY path=/kaggle/working/tree_hetero_handoff.zip
TREE_HETERO_SUCCESS
```

`status=incomplete`여도 실패가 아니다. 같은 handoff를 입력으로 추가해 다시 실행하면 완료된 작업은 재사용한다. `TREE_HETERO_ERROR`가 나오면 해당 줄과 전체 로그를 함께 전달한다.

## 판정 기준

- 2022·2023 OOF만으로 보정 가중치를 고르고 2024 OOF는 한 번만 확인한다.
- 전체 가중 Brier 개선, 최근 폴드 비열화, 최악 폴드, 세그먼트, 부트스트랩, 오차 다양성을 모두 통과해야 한다.
- 구조를 통과한 모델만 시드 `42`, `2026`을 추가한다.
- 두 모델 계열이 모두 통과해도 50:50 잔차 혼합이 최고 단일 모델보다 `0.00002` 이상 좋아야 채택한다.
- 어떤 결과도 평가 행끼리 통계나 순서 정보를 공유하지 않는다.
