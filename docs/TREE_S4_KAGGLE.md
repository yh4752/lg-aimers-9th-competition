# S4 앵커·잔차·계층 보정 실험 실행 안내

이 실험은 한 모델만 오래 학습하는 방식이 아닙니다. 최근 시즌과 여러 시즌을 섞은 기준 확률(anchor), 그 기준값의 오차를 보완하는 모델(residual correction), 선수·손잡이·경기 유형별 확률 보정(hierarchical calibration)을 하나의 파이프라인으로 묶어 비교합니다.

최소 12개의 완전한 파이프라인을 끝까지 비교하고, 구조 검증을 통과한 상위 후보는 시드 3개로 다시 확인합니다. 2024 시즌 결과는 모델 종류나 혼합 비율을 고르는 데 사용하지 않고 마지막 합격 여부를 확인할 때만 사용합니다.

## 1. 로컬에서 Kaggle 입력 만들기

프로젝트 최상위 폴더에서 아래 명령을 실행합니다. `E2_HANDOFF_PATH`에는 이전에 받은, 합격 상태의 `tree_expert_e2_handoff` 파일 경로를 넣습니다.

```bash
artifacts/tabm_submission_python311/bin/python \
  tools/prepare_tree_s4_input.py \
  --e2-handoff "E2_HANDOFF_PATH" \
  --output artifacts/tree_s4_input.zip
```

정상 출력은 다음 형식입니다.

```text
TREE_S4_INPUT_READY path=.../artifacts/tree_s4_input.zip sha256=...
```

같은 출력 경로에 파일이 이미 있으면 덮어쓰지 않습니다. 기존 파일을 그대로 사용하거나 새 파일명을 지정하면 됩니다.

## 2. Kaggle 입력 구성

Notebook의 Input에는 아래 두 데이터셋만 필요합니다.

1. 공식 데이터 `lg-aimers-9th-data`
2. 방금 만든 `tree_s4_input.zip`을 올린 데이터셋

이전 실행이 `incomplete`로 끝났을 때만, 결과로 받은 `anchor_residual_hierarchical_handoff.zip`을 별도 Kaggle Dataset으로 추가합니다. ZIP 형태와 압축 해제된 폴더 형태를 모두 인식하며, 동일한 파일이 두 형태로 보여도 하나로 처리합니다.

## 3. Notebook 실행

1. Accelerator를 **GPU T4 x2**로 설정합니다.
2. 인터넷 연결은 패키지 버전이 다를 때 설치할 수 있도록 켜 둡니다.
3. [KAGGLE_S4_CELL.py](../experiments/tree_expert/KAGGLE_S4_CELL.py)의 전체 내용을 Notebook 셀 하나에 복사합니다.
4. **Save Version → Save & Run All**을 한 번 실행합니다.

최초 실행 예상 시간은 **10~12시간**입니다. Kaggle 환경과 조기 종료 시점에 따라 달라질 수 있습니다. 셀은 종료 1시간 전부터 새 학습을 시작하지 않고 결과 ZIP을 만드는 시간을 확보합니다.

## 4. 확인할 로그

시작할 때 아래 로그가 순서대로 나옵니다.

```text
S4_CODE_READY ...
S4_DEPENDENCIES_READY ...
S4_INPUTS_VERIFIED ...
S4_GPU_READY count=2 ...
```

학습 중에는 작업별로 다음 로그가 반복됩니다.

```text
S4_JOB_START job=... phase=... gpu=...
S4_JOB_END job=... status=completed
S4_PHASE_COMPLETE phase=... next=...
```

정상 종료 표시는 다음 두 문구입니다.

```text
S4_HANDOFF_READY path=/kaggle/working/anchor_residual_hierarchical_handoff.zip
S4_SUCCESS status=completed ...
```

`S4_SUCCESS status=incomplete`이거나 오류가 나더라도 `S4_HANDOFF_READY`가 보이면 끝난 작업은 재개 ZIP에 보존됩니다. 다음 실행에 그 handoff를 Input으로 추가하면 완료된 작업을 다시 학습하지 않습니다.

## 5. 실행 후 전달할 파일

Kaggle Output에서 아래 파일 하나만 내려받아 전달하면 됩니다.

```text
anchor_residual_hierarchical_handoff.zip
```

이 파일에는 검토 결과와 재개 상태가 함께 들어 있습니다. 실행 중간마다 여러 ZIP을 내려받을 필요는 없습니다.

이 캠페인은 후보를 검증하는 단계이므로 **DACON 제출 ZIP을 만들지 않습니다**. handoff를 검토해 실제 합격 조건과 파일 해시가 모두 맞는 후보만 별도의 제출물 제작 단계로 넘깁니다.
