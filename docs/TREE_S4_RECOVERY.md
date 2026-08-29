# S4 중단 실험 복구 안내

## 목적

이 절차는 저장 공간 부족으로 중단된 S4 실험을 처음부터 다시 학습하지 않고 이어서 실행하기 위한 것입니다. 기존 handoff의 파일 해시와 내부 manifest를 검증한 뒤, 이미 끝난 14개 full-chain 결과와 다음 단계에 필요한 데이터만 남긴 복구 입력을 만듭니다.

이 단계는 후보 검증을 계속하는 연구 단계입니다. **DACON 제출 ZIP을 만들지 않습니다.** 제출 파일은 acceptance gate를 통과한 후보가 확인된 뒤 별도로 제작합니다.

## 1. 로컬에서 복구 입력 만들기

필요한 파일은 이전 실행에서 받은 `anchor_residual_hierarchical_handoff.zip`입니다. 압축을 풀지 말고 ZIP 그대로 사용합니다. 임시 파일까지 고려해 로컬 디스크에 **10 GB 이상**의 여유 공간을 확보합니다.

저장소 최상위 폴더에서 다음 명령을 실행합니다.

```bash
artifacts/tabm_submission_python311/bin/python \
  tools/prepare_tree_s4_recovery_input.py \
  --source-handoff /path/to/anchor_residual_hierarchical_handoff.zip \
  --output artifacts/tree_s4_recovery_input.zip
```

예상 시간은 저장 장치 속도에 따라 약 10~40분입니다. 3.8GB 원본과 내부 ZIP을 전부 해시 검증하고 다시 압축하기 때문입니다. 출력 파일이 이미 있으면 덮어쓰지 않고 멈추므로 재실행도 안전합니다.

성공 시 마지막에 다음 두 줄이 나옵니다.

```text
TREE_S4_RECOVERY_SOURCE_VERIFIED sha256=...
TREE_S4_RECOVERY_INPUT_READY path=... sha256=... retained_bytes=... dropped_bytes=...
```

## 2. Kaggle 입력 구성

Kaggle Dataset으로 다음 세 항목만 추가합니다.

1. 공식 데이터 `lg-aimers-9th-data`
2. 기존 S4 기본 입력 `tree_s4_input`
3. 새로 만든 `tree_s4_recovery_input.zip`을 올린 Dataset

이전의 `anchor_residual_hierarchical_handoff` Dataset은 함께 추가하지 않습니다. 일반 handoff와 recovery input이 동시에 발견되면 안전을 위해 실행이 중단됩니다. Kaggle에서 ZIP이 폴더 형태로 자동 해제되어 보여도 정상적으로 인식합니다.

## 3. Kaggle에서 실행

Accelerator를 **T4 x2**로 설정하고, [KAGGLE_S4_CELL.py](../experiments/tree_expert/KAGGLE_S4_CELL.py)의 전체 내용을 노트북 한 셀에 복사합니다. 그 셀 하나만 둔 상태에서 **Save Version → Save & Run All**을 실행합니다.

복구 입력 검증이 끝나면 다음 로그가 순서대로 보여야 합니다.

```text
S4_CODE_READY ...
S4_INPUTS_VERIFIED ...
S4_GPU_READY count=2 ...
S4_RECOVERY_READY ... phase=full_chains completed_full_chains=14
S4_JOB_START job=full_chains__14 ...
```

첫 미완료 작업은 `full_chains__14`입니다. 이후 후보 선택과 confirmation 검증을 진행합니다. 예상 GPU 시간은 약 30분~2시간이며, Kaggle 혼잡도와 CatBoost 실행 편차에 따라 더 길어질 수 있습니다.

저장 시 다음 로그로 남은 공간과 스냅샷 상태를 확인할 수 있습니다.

```text
S4_DISK_STATUS free_bytes=... estimated_peak_bytes=...
S4_SNAPSHOT_READY path=... size_bytes=...
```

공간이 부족한 순간에는 학습을 바로 실패시키지 않고 아래처럼 주기 저장을 건너뜁니다. 기존의 정상 handoff가 있으면 그대로 보존합니다.

```text
S4_SNAPSHOT_SKIPPED reason=insufficient_space
```

## 4. 실행 종료 후 전달할 파일

Kaggle Output에서 다음 파일 하나를 다운로드해 전달합니다.

```text
anchor_residual_hierarchical_handoff.zip
```

정상 종료 시 `S4_SUCCESS`와 `S4_HANDOFF_READY`가 출력됩니다. 오류가 발생하더라도 `S4_HANDOFF_READY`가 있다면 해당 ZIP과 마지막 오류 로그를 함께 전달합니다. 같은 recovery input으로 다시 실행하면 저장된 상태 기준으로 이미 완료된 작업을 건너뜁니다.
