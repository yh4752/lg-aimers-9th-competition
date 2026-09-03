# 직접 타깃 전문가 앙상블 실행 안내

이 실험은 기존 E2 모델의 오차를 조금 수정하는 방식이 아니라, 성공 여부를 직접
학습하는 CatBoost 전문가 8개를 비교한다. 전체 학습은 사용자가 Kaggle에서 실행하며,
로컬 준비 작업과 테스트에는 GPU가 필요하지 않다.

## 실행 전 준비

로컬에서 `tools/prepare_direct_expert_input.py`를 한 번 실행해
`direct_expert_input.zip`을 만든다. 이 파일은 S4에서 검증된 2021~2024 E2 OOF와
Public 977점 제출에 실제 사용한 E2 3시드 모델을 해시로 묶은 입력이다. 다음 명령에서
`<S4_HANDOFF>`만 실제 S4 handoff 경로로 바꾼다.

```bash
cd /path/to/lg-aimers-9th-competition

artifacts/tabm_submission_python311/bin/python \
  tools/prepare_direct_expert_input.py \
  --s4-handoff "<S4_HANDOFF>" \
  --e2-submission artifacts/tree_expert_e2_submission_20260827/catboost_3seed_v1.zip \
  --e2-receipt artifacts/tree_expert_e2_submission_20260827/submission_receipt.json \
  --output artifacts/direct_expert_input.zip
```

S4 handoff의 SHA-256은
`5a410548de99d5d9c56f9b0d1940d5080e9167eafd97a2c94c45e31deacb4911`, E2 모델은
`8bc33042b9258049f905df0c733b1153d1cedfac39cb0732b2a7e154629d779a`여야 한다.
`DIRECT_EXPERT_INPUT_READY`가 출력되기 전에는 Kaggle Dataset으로 올리지 않는다.

## Version A: 구조 탐색

- 가속기: `T4 x2`
- Internet: CatBoost 1.2.10이 없으면 설치할 수 있도록 켠다.
- 입력 Dataset: 공식 데이터, `direct_expert_input.zip`
- 실행 코드: `experiments/direct_expert/KAGGLE_STAGE_A_CELL.py` 전체를 한 셀에 복사
- 예상 시간: 10~11시간

Save Version으로 전체 실행한다. 정상 시작 로그는
`DIRECT_EXPERT_INPUTS_VERIFIED`, `DIRECT_EXPERT_GPU_READY count=2`,
`DIRECT_EXPERT_SMOKE_SUCCESS` 순서다. 완료 후 Output에서
`direct_expert_stage_A_handoff.zip` 하나를 내려받는다. 성공 로그는
`DIRECT_EXPERT_HANDOFF_READY path=...`다.

시간 제한이나 후보별 실패로 `incomplete` handoff가 생기면 새 Version의 Input에 그
handoff를 하나만 추가하고 같은 Stage A 셀을 다시 실행한다. 완료된 OOF 예측은 해시를
검사해 재사용하고 빠진 작업만 이어서 실행한다.

Kaggle이 ZIP을 자동으로 풀어 폴더로 표시해도 정상이다. 동일한 ZIP과 그 압축 해제본이
동시에 보이면 manifest가 같은 한 개의 논리 산출물로 처리한다. 서로 다른 Stage A
handoff 두 개를 동시에 넣으면 모호성을 막기 위해 즉시 중단한다.

## Version B: 독립 확인과 최종 학습

- 가속기: `T4 x2`
- 입력 Dataset: 공식 데이터, 같은 `direct_expert_input.zip`, Version A handoff
- 실행 코드: `experiments/direct_expert/KAGGLE_STAGE_B_CELL.py` 전체를 한 셀에 복사
- 예상 시간: 8~9시간

항상 `direct_expert_review.zip`과 `direct_expert_handoff.zip`이 생성된다. 모든 gate와
행 독립성 검사를 통과한 경우에만 `direct_expert_delivery.zip`도 생성된다. 확인할
로그는 다음과 같다.

```text
DIRECT_EXPERT_REVIEW_READY path=...
DIRECT_EXPERT_HANDOFF_READY path=...
DIRECT_EXPERT_DELIVERY_READY path=...   # 승인 후보가 있을 때만
```

Version B가 중단되면 마지막 `direct_expert_handoff.zip`을 새 Version의 Input에 하나만
추가해 같은 Stage B 셀을 다시 실행한다. 완료된 seed/fold 결과는 다시 학습하지 않는다.
승인됐지만 종료까지 두 시간 미만인 경우에는 무리하게 최종학습을 시작하지 않고
`accepted_pending_full_fit` handoff를 만든다. 그 handoff로 새 Version을 실행하면 OOF를
재학습하지 않고 최종학습부터 이어간다.

각 Version은 Output을 Kaggle Dataset으로 저장한 뒤 다음 Version의 Input으로 붙인다.
셀은 브라우저 자동 다운로드를 하지 않는다. 오류가 나면 마지막
`DIRECT_EXPERT_...` 로그와 traceback 전체, 생성된 handoff가 있으면 그 ZIP만 전달한다.

이 단계에서는 제출 패키지를 만들지 않는다. Delivery를 로컬에서 다시 검증해 후보
상태, 승인 증거, 감사 결과와 현재 해시가 모두 일치한 뒤에만 별도의 제출물 제작을
시작한다.
