# OOF 리셋 감사 실행 안내

이 도구는 새 모델을 학습하지 않는다. 지금까지 만든 시간 OOF를 같은 행끼리 다시
맞춰 성능, 보정, 시간 변화와 모델 간 차이를 확인한다. 평가 데이터나 제출 파일은
읽지 않으며 결과만 보고 모델을 자동 승인하지도 않는다.

## 준비할 파일

가능한 파일만 넣으면 된다. 파일 이름이 달라도 내부 manifest로 구분한다.

- `tabm_colab_stage_C_delivery.zip`
- `tabm_row_feature_stage_P_delivery.zip`
- `catboost_tabm_blend_delivery.zip`
- `catboost_deployment_review.zip`
- `hierarchical_tabm_review.zip`
- 원본 시간 OOF를 별도 규격으로 묶은 격리 XGBoost ZIP(있을 때만)

`submit_xgboost_v3.zip`은 OOF가 없는 제출 파일이므로 넣지 않는다. 행 피처 결과나
원본 XGBoost OOF가 없다면 억지로 만들 필요도 없다. 보고서에
`missing_evidence`로 남는다.

## 실행 명령

프로젝트 폴더에서 다음 명령을 실행한다. 실제로 가진 ZIP에 해당하는
`--artifact` 줄만 남기고, 같은 옵션을 파일마다 반복한다.

```bash
cd /Users/yonghyun/Documents/lg-aimers-9th-competition

artifacts/tabm_submission_python311/bin/python \
  tools/run_oof_reset_audit.py \
  --artifact /Users/yonghyun/Downloads/tabm_colab_stage_C_delivery.zip \
  --artifact /Users/yonghyun/Downloads/tabm_row_feature_stage_P_delivery.zip \
  --artifact /Users/yonghyun/Downloads/catboost_tabm_blend_delivery.zip \
  --artifact /Users/yonghyun/Downloads/catboost_deployment_review.zip \
  --artifact /Users/yonghyun/Downloads/hierarchical_tabm_review.zip \
  --output-root artifacts/oof_reset_audit_runs
```

CPU만 사용한다. 전체 OOF 크기에 따라 약 5~15분, 메모리는 약 2~6GB를 예상한다.
학습이나 추론은 하지 않는다. 다시 실행하면 기존 결과를 덮지 않고 새로운
`oof_reset_audit_*` 폴더를 만든다.

정상 종료 로그는 다음 형태다.

```text
OOF_RESET_AUDIT_SUCCESS output_dir=<절대경로> result_zip=<절대경로> sha256=<64자리 해시>
```

실패하면 다음 형태가 나온다.

```text
OOF_RESET_AUDIT_ERROR stage=<inputs 또는 reports> type=<오류종류> message=<원인>
```

성공 후에는 `oof_reset_audit_results.zip` 하나만 전달하면 된다. 이 ZIP에는 작은
표와 감사 요약만 들어가며 원본 OOF, 학습 데이터, 모델, 체크포인트와 제출 파일은
포함되지 않는다.
