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

가속기는 `GPU T4 x2`를 선택한다. `experiments/tree_expert/KAGGLE_E2_CELL.py` 전체를 한 셀에 붙여 실행하거나 Save Version을 사용한다. 빠른 환경에서는 45–90분, 보통은 약 1.5–3시간을 예상하며 하드 캡은 6시간이다. 첫 안정 산출물 게시 전에는 Output Data가 비어 보일 수 있다.

주요 로그는 `TREE_E2_CODE_READY`, `TREE_E2_DEPENDENCIES_READY`, `TREE_E2_INPUTS_FOUND`, `TREE_E2_GPU_READY`, 각 B0–B3 단계 로그, 마지막 `TREE_E2_HANDOFF_READY` 순서로 나온다.

완료되면 아래 파일 하나만 내려받아 전달하면 된다.

```text
/kaggle/working/tree_expert_e2/bundles/tree_expert_e2_handoff.zip
```

마지막이 `TREE_EXPERT_ERROR`라면 전체 로그와 가장 최근 resume ZIP도 같이 전달한다. `TREE_E2_HANDOFF_READY`에 `delivery=yes`가 표시되더라도 이는 모델 전달 파일일 뿐이다. 별도의 로컬 규칙·형식 감사를 통과하기 전에는 제출물로 취급하지 않는다.

## 3. 2026-08-27 감사 오류 복구

`TreeFeatureError: S1 transform season differs from valid_year`로 끝난 실행은 학습 실패가 아니다. 승인된 세 모델과 검증 결과는 직전 handoff에 남아 있으므로 전체 학습을 다시 하지 않는다.

새 Kaggle Version에는 아래 세 데이터셋만 연결한다.

1. `lg-aimers-9th-data`
2. 기존 `tree_expert_e2_input` 데이터셋
3. 실패한 `tree_expert_e2_handoff.zip`으로 만든 데이터셋

Kaggle 화면에서 handoff 아래에 `tree_expert_e2_resume/`, `tree_expert_e2_review/`, `handoff_manifest.json`이 보이는 것은 정상이다. Kaggle이 중첩 ZIP까지 자동으로 푼 형태이며, 현재 셀은 내부 resume을 별도 입력으로 중복 계산하지 않고 원본 SHA-256과 같은 ZIP으로 복원한다. 내부 폴더를 삭제하거나 다시 압축하지 않는다.

이 문서와 같은 커밋에서 다시 생성한 `experiments/tree_expert/KAGGLE_E2_CELL.py`를 한 셀로 실행한다. 복구 코드는 이전 코드 해시, 정확한 오류 문구, 승인 결정, 세 모델의 해시가 모두 일치할 때만 감사 단계로 되돌아간다. B0-B3 검증과 full-fit 학습은 재사용하며 보통 5–15분 안에 끝난다.

정상 복구의 마지막 로그는 다음 형태다.

```text
TREE_E2_HANDOFF_READY ... status=accepted delivery=yes
```

완료 후 새로 생성된 `tree_expert_e2_handoff.zip` 하나만 전달한다. 이전 handoff와 새 handoff를 동시에 다음 실행 입력으로 연결하면 중복 resume으로 판정될 수 있으므로 연결하지 않는다.
