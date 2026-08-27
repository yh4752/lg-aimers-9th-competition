# 계층형 잔차·보정 실험

이 실험은 기존 Tree Expert E2를 버리고 새로 시작하는 작업이 아니다. E2의 예측을 `C0`로 고정하고, 선수·상황별 과거 기록을 이용한 CatBoost 잔차 모델 `C1`을 얹는다. `C2`는 C1의 확률을 과거 오차표로 한 번 더 보정하는 후보다.

현재 제공하는 Kaggle 셀은 H1까지만 실행한다. H1 결과를 확인하기 전에는 H2 학습이나 제출 파일을 만들지 않는다.

## 후보 이름

- C0: 검증을 통과한 기존 E2 CatBoost 앙상블
- C1: C0 확률과 과거 시즌 계층 통계를 입력으로 쓰는 잔차 CatBoost 3시드 앙상블
- C2: C1에 투수·타자·경기 상황별 과거 오차 보정을 더한 후보

평가 행에서 평균이나 빈도를 다시 계산하지 않는다. 계층 통계와 보정표는 학습 데이터에서 미리 만들고, 추론할 때는 현재 행의 값으로 조회만 한다. 행 순서, 배치 크기, 다른 평가 행의 존재가 예측을 바꾸지 않아야 한다.

## 단계별 작업

### H1: 계층 강도 선택

2020년까지 학습하고 2021년을 예측하는 E2 기준 폴드를 새로 만든다. 이 결과와 이미 검증된 2022~2024 E2 예측을 합쳐 네 시즌 OOF를 완성한다.

이후 계층 통계의 수축 강도 세 가지를 비교한다.

- `hc_strong`: 선수·상황 통계를 많이 줄여서 쓴다.
- `hc_balanced`: 중간 강도다.
- `hc_light`: 관측값을 비교적 적극적으로 반영한다.

2022·2023 폴드의 행 가중 Brier가 가장 낮은 강도를 고른다. 동점이면 strong, balanced, light 순서다. 선택된 강도만 2024에서 한 번 더 확인한다. 2024 결과가 나쁘다고 다른 강도로 다시 고르지는 않는다.

예상 시간은 T4 두 장 기준 최대 6시간이다. 실제 시간은 새 2021 TabM 기준 모델의 조기 종료 시점에 따라 달라진다.

입력은 다음 두 개다.

1. 공식 데이터셋: 루트에 `train.csv`, `trackman_history.csv`
2. `tree_hierarchical_input.zip`을 Kaggle Dataset으로 올린 것

완료되면 Kaggle Output에 `tree_hierarchical_handoff.zip` 하나가 남는다. H1이 끝났으면 로그에 아래 문구가 나온다.

```text
TREE_HC_SUCCESS stage=H1 next=H2 handoff=/kaggle/working/tree_hierarchical_handoff.zip
```

시간이 부족하면 `next=H1`, `status=incomplete`인 재개 ZIP이 만들어진다. 같은 입력과 그 handoff를 다음 Save Version의 Input으로 추가하면 완료된 작업은 다시 돌리지 않는다.

오류가 나면 `TREE_HC_ERROR` 한 줄과 마지막 로그 80줄을 함께 확인한다. 오류 시점까지 상태 파일을 만들 수 있었다면 같은 이름의 긴급 handoff도 Output에 남긴다.

### H2: C1·C2 독립 판정

H1 handoff를 검증한 뒤에만 진행한다. 선택된 강도로 나머지 두 시드를 학습하고 C1을 판정한다. C2의 보정 강도는 2022·2023에서 정한 뒤 2024를 확인용으로만 쓴다. C1이 탈락해도 C2는 별도 기준으로 판정한다.

H2가 끝나기 전에는 모델 delivery나 제출 파일을 만들지 않는다.

### H3: 전체 학습과 추론 검사

H2에서 C1 또는 C2가 통과했을 때만 실행한다. 각 시드의 세 temporal fold 최적 반복 수 중앙값에 1을 더하고 50~800 범위로 제한한다. 2021~2024 OOF 잔차로 최종 C1 모델을 학습한 뒤 245,789행 추론 시간, 모델 크기, 행 독립성을 검사한다.

H3 산출물도 모델 delivery일 뿐 제출 ZIP은 아니다. 제출 코드는 대회 형식과 현재 모델 해시를 다시 확인한 뒤 별도 단계에서 만든다.

## H1 입력 만들기

저장소 루트에서 아래 명령을 한 번 실행한다.

```bash
artifacts/tabm_submission_python311/bin/python \
  tools/prepare_tree_hc_input.py \
  --e2-handoff "/Users/yonghyun/Downloads/tree_expert_e2_handoff (1).zip" \
  --output artifacts/tree_hierarchical_input.zip
```

성공 문구는 `TREE_HC_INPUT_READY`다. 이 ZIP과 공식 데이터셋을 Kaggle Dataset으로 등록한다. Notebook Input에는 두 데이터셋만 두고, 완료된 H1을 이어서 돌릴 때만 이전 `tree_hierarchical_handoff.zip` 데이터셋을 하나 더 추가한다.

## Kaggle 실행

가속기를 `GPU T4 x2`로 고른다. [KAGGLE_HC_CELL.py](../experiments/tree_expert/KAGGLE_HC_CELL.py) 전체를 빈 Notebook의 셀 하나에 붙여 넣고 Save Version을 실행한다. 인터넷은 꺼도 된다. 코드는 GitHub나 Drive에서 파일을 받지 않는다.

실행 중에는 다음 로그를 보면 된다.

- `TREE_HC_CODE_READY`: 내장 코드 검증 완료
- `TREE_HC_INPUTS_FOUND`: 공식 데이터와 HC 입력 탐색 완료
- `TREE_HC_GPU_READY`: T4 확인 완료
- `TREE_HC_JOB_START`, `TREE_HC_JOB_END`: 작업 시작과 종료
- `TREE_HC_DECISION`: 선택된 계층 강도
- `TREE_HC_SUCCESS`: handoff 생성 완료
- `TREE_HC_ERROR`: 실행 실패

Codex에 전달할 것은 `tree_hierarchical_handoff.zip` 하나와 마지막 로그 블록이다.

## 결과를 읽는 법

H1은 점수를 바로 올리는 학습이 아니라, 계층 정보를 어느 정도 믿을지 정하는 단계다. strong이 선택되면 선수별 표본이 아직 불안정하다는 뜻이고 light가 선택되면 과거 선수 기록의 신호가 비교적 강하다는 뜻이다. 어느 쪽이 선택돼도 2024 확인 성능과 세부 구간 악화를 본 뒤 H2로 넘어간다.

후보가 탈락해도 실험 실패는 아니다. 등록한 시간 분할과 기준에서 C0보다 안정적으로 낫다는 증거를 얻지 못했다는 뜻이다. 이 경우 기존 E2는 그대로 보존되고, 탈락한 후보만 제출 경로에서 제외한다.
