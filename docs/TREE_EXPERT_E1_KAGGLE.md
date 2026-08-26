# Tree Expert E1 Kaggle 실행 안내

E1은 기존 TabM 제출물을 대체하는 최종 학습이 아니다. 2023년까지 학습하고
2024년을 검증해, CatBoost 기반 구조 네 가지 중 후속 실험 가치가 있는 후보를
고르는 최신 시즌 검증 실험이다. 평가 데이터와 제출 파일은 읽거나 만들지 않는다.

## 1. 로컬에서 입력 파일 만들기

아래 명령을 그대로 실행한다. Stage C 파일의 해시와 내부 구조가 계약값과 다르면
입력 ZIP을 만들지 않고 오류로 종료한다.

```bash
cd /Users/yonghyun/Documents/lg-aimers-9th-competition
artifacts/tabm_submission_python311/bin/python \
  tools/prepare_tree_expert_e1_input.py \
  --stage-c-delivery /Users/yonghyun/Downloads/tabm_colab_stage_C_delivery.zip \
  --output artifacts/tree_expert_e1_input.zip
```

성공하면 다음 형식의 한 줄이 출력된다.

```text
TREE_E1_INPUT_READY path=<absolute-path> sha256=<sha256> size_bytes=<bytes>
```

## 2. Kaggle 입력 구성

Kaggle Notebook의 Data에 다음 두 데이터셋을 추가한다.

1. 공식 데이터셋 `lg-aimers-9th-data`
2. 방금 만든 `tree_expert_e1_input.zip`을 업로드해 만든 데이터셋

Kaggle이 ZIP을 자동으로 풀어 파일들이 보이는 상태여도 괜찮다. 코드가 압축 ZIP과
풀린 디렉터리를 모두 인식한다. 이전 실행을 이어갈 때만 세 번째 데이터셋으로
`tree_expert_e1_handoff.zip` 또는 그 안의 `tree_expert_e1_resume.zip`을 추가한다.
resume과 handoff를 동시에 추가하면 중복 입력으로 중단되므로 하나만 넣는다.

Accelerator는 반드시 `GPU T4 x2`로 선택한다. Internet은 패키지 설치가 필요한
경우에만 켠다. `catboost==1.2.10`이 이미 있으면 설치를 건너뛴다.

## 3. 한 셀 실행

[KAGGLE_E1_CELL.py](../experiments/tree_expert/KAGGLE_E1_CELL.py)의 전체 내용을
Kaggle 코드 셀 하나에 복사하고 `Save Version`으로 실행한다. 예상 시간은
약 3~4시간이며 Kaggle 부하와 조기 종료 여부에 따라 달라질 수 있다.

정상 시작 로그는 다음 순서다.

```text
TREE_E1_CODE_READY ...
TREE_E1_DEPENDENCIES_READY catboost=1.2.10
TREE_E1_INPUTS_VERIFIED ...
TREE_E1_GPU_READY device_count=2
TREE_E1_JOB_START ...
```

작업별 종료 후 `TREE_E1_JOB_END`, 최종 판정 때 `TREE_E1_DECISION`이 남는다.
완료 시 마지막 핵심 로그는 아래 형식이다.

```text
TREE_E1_HANDOFF_READY path=<absolute-path> sha256=<sha256>
```

결과 파일은 Kaggle Output의 다음 위치에 하나로 모인다.

```text
/kaggle/working/tree_expert_e1/bundles/tree_expert_e1_handoff.zip
```

## 4. 중단과 재실행

handoff ZIP에는 검토 자료와 재실행 상태가 함께 들어 있다. 세션이 중단되면 마지막
handoff를 Kaggle 데이터셋으로 올리고 같은 셀을 다시 실행한다. 입력·계약·코드
해시가 모두 일치할 때만 완료된 작업을 재사용하고, 일치하지 않으면 안전하게
오류로 중단한다. 따라서 다른 버전의 입력을 섞어서 억지로 이어서 실행하지 않는다.

오류가 나면 다음 전체 로그 한 줄을 전달한다.

```text
TREE_EXPERT_ERROR stage=<stage> type=<type> message=<message>
```

그 아래 `TREE_E1_EMERGENCY_HANDOFF path=<path>`가 있으면 해당 파일도 보관한다.

## 5. 실행 후 전달할 것

다음 두 가지만 전달하면 된다.

- `tree_expert_e1_handoff.zip`
- 전체 `TREE_E1_HANDOFF_READY path=<absolute-path> sha256=<sha256>` 로그 한 줄

이 handoff는 검증 결과를 분석하기 위한 연구 산출물이며 DACON 제출물이 아니다.
후보 승인과 규칙·산출물 검증이 끝나기 전에는 제출 ZIP을 만들지 않는다.
