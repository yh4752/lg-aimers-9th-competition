# Tree Expert E2 Kaggle 실행 안내

E2는 앞서 통과한 두 트리 구조를 세 개의 시간 폴드와 세 개의 시드로 재검증하는 실험이다. 모든 검증 게이트를 통과하면 추론용 모델 전달 파일까지 만들지만, DACON 제출 파일을 만들지 않는다.

## 1. 로컬 입력 만들기

```bash
cd /Users/yonghyun/Documents/lg-aimers-9th-competition

artifacts/tabm_submission_python311/bin/python \
  tools/prepare_tree_expert_e2_input.py \
  --e1-handoff /Users/yonghyun/Downloads/tree_expert_e1_handoff.zip \
  --stage-c-delivery /Users/yonghyun/Downloads/tabm_colab_stage_C_delivery.zip \
  --tabm-submission artifacts/tabm_submission_version_d_superseded_data_only/submit.zip \
  --output artifacts/tree_expert_e2_input.zip
```

이 작업은 GPU를 쓰지 않으며 보통 1–3분 걸린다. 지정한 출력 파일만 원자적으로 교체하므로 같은 명령을 다시 실행해도 안전하다.

## 2. Kaggle 입력과 실행

Kaggle Notebook에 다음 데이터셋을 연결한다.

1. `lg-aimers-9th-data`
2. `tree_expert_e2_input.zip`으로 만든 데이터셋
3. 재실행할 때만 직전 `tree_expert_e2_handoff.zip` 또는 그 안의 resume ZIP

가속기는 `GPU T4 x2`를 선택한다. `experiments/tree_expert/KAGGLE_E2_CELL.py` 전체를 한 셀에 붙여 실행하거나 Save Version을 사용한다. 정상적인 예상 시간은 45–90분이고, 하드 캡은 6시간이다. 첫 안정 산출물 게시 전에는 Output Data가 비어 보일 수 있다.

주요 로그는 `TREE_E2_CODE_READY`, `TREE_E2_DEPENDENCIES_READY`, `TREE_E2_INPUTS_FOUND`, `TREE_E2_GPU_READY`, 각 B0–B3 단계 로그, 마지막 `TREE_E2_HANDOFF_READY` 순서로 나온다.

완료되면 아래 파일 하나만 내려받아 전달하면 된다.

```text
/kaggle/working/tree_expert_e2/bundles/tree_expert_e2_handoff.zip
```

마지막이 `TREE_EXPERT_ERROR`라면 전체 로그와 가장 최근 resume ZIP도 같이 전달한다. `TREE_E2_HANDOFF_READY`에 `delivery=yes`가 표시되더라도 이는 모델 전달 파일일 뿐이다. 별도의 로컬 규칙·형식 감사를 통과하기 전에는 제출물로 취급하지 않는다.
