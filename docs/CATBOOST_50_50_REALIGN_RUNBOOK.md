# CatBoost 50:50 배포 정렬 재검증 실행 안내

이번 실행의 목적은 모델 종류를 다시 넓게 탐색하는 것이 아니다. 이미 확보한 TabM과
CatBoost를 50:50으로 섞었을 때 세 연도 OOF에서 개선이 반복되는지 다시 확인하고,
통과한 경우에만 CatBoost 전체 학습 모델을 만든다.

대회 규칙상 평가 데이터는 읽지도 않고 학습 통계에 쓰지도 않는다. 이 단계에서 만드는
`delivery.zip`도 제출 파일이 아니다. 최종 제출 코드에 붙일 수 있는 CatBoost 모델과
검증 기록만 담는다.

## 1. 로컬 입력 ZIP 만들기

터미널에서 아래 명령을 그대로 실행한다. 출력 파일이 이미 있으면 덮어쓰지 않고
중단하므로, 예전에 만든 파일이 있다면 이름을 바꾸거나 다른 위치를 지정한다.

```bash
cd /Users/yonghyun/Documents/lg-aimers-9th-competition

artifacts/tabm_submission_python311/bin/python \
  tools/prepare_catboost_50_50_realign_input.py \
  --training-input /Users/yonghyun/Documents/lg-aimers-9th-competition/artifacts/catboost_tabm_blend_input.zip \
  --stage-c-delivery /Users/yonghyun/Downloads/tabm_colab_stage_C_delivery.zip \
  --deployment-resume /Users/yonghyun/Downloads/catboost_deployment_emergency_1eef761337e3.zip \
  --deployment-review /Users/yonghyun/Downloads/catboost_deployment_review.zip \
  --oof-audit /Users/yonghyun/Documents/lg-aimers-9th-competition/artifacts/oof_reset_audit_runs/oof_reset_audit_20260821T130502Z_06e0e656/oof_reset_audit_results.zip \
  --output /Users/yonghyun/Documents/lg-aimers-9th-competition/artifacts/catboost_50_50_realign_input.zip
```

성공하면 마지막에 다음 형식의 로그가 나온다.

```text
REALIGN_INPUT_READY path=.../catboost_50_50_realign_input.zip sha256=<64자리> size_bytes=<숫자>
```

원본 다섯 개의 SHA-256과 ZIP 내부 파일까지 확인하므로 몇 분 정도 걸릴 수 있다.
`REALIGN_INPUT_READY`가 나오기 전에는 Colab으로 넘어가지 않는다.

## 2. Colab에서 실행하기

1. 새 Colab 노트북을 열고 런타임을 `T4 GPU`로 설정한다.
2. [COLAB_CATBOOST_50_50_REALIGN_CELL.py](../experiments/catboost_50_50_realign/COLAB_CATBOOST_50_50_REALIGN_CELL.py)의 내용을 전부 복사해 셀 하나에 붙여 넣는다.
3. 셀을 실행하고 `catboost_50_50_realign_input.zip` 하나만 올린다.

셀은 Drive나 GitHub에서 파일을 가져오지 않는다. 필요한 코드는 셀 안에 들어 있으며,
공식 train과 기존 OOF 산출물도 방금 만든 입력 ZIP에서만 읽는다.

T4 한 장 기준 예상 시간은 1~2시간이다. 업로드와 라이브러리 확인 시간을 포함해
세션 한도는 3시간으로 잡혀 있다. 실제 시간은 Colab 상태와 2022 TabM 조기 종료
시점에 따라 달라질 수 있다.

## 3. 정상적으로 보이는 로그

초반에는 아래 두 줄을 확인한다.

```text
REALIGN_GPU_READY name=Tesla T4
REALIGN_UPLOAD_CACHE_REUSED count=1   # 같은 런타임에서 다시 실행했을 때만 표시
```

학습 중에는 CatBoost 진행 로그와 기존 TabM worker 로그가 이어진다. F1 두 개가 모두
끝나면 resume ZIP 하나를 먼저 내려받는다.

```text
REALIGN_DOWNLOAD_REQUESTED phase=f1_complete path=...
```

이 파일은 중간 보험이다. 다운로드가 끝나도 셀은 계속 실행되므로 중지하지 않는다.

최종 판정은 둘 중 하나다.

```text
REALIGN_CAMPAIGN_SUCCESS status=completed selected_tree_count=<4|8|12|16|20|24|28|32>
```

또는

```text
REALIGN_CAMPAIGN_SUCCESS status=deployment_blocked selected_tree_count=none
```

`completed`이면 review, resume, delivery가 차례로 내려온다. 브라우저 다운로드 요청은
F1 중간 파일을 포함해 모두 네 번이다.

```text
catboost_50_50_realign_review.zip
catboost_50_50_realign_resume.zip
catboost_50_50_realign_delivery.zip
```

`deployment_blocked`이면 review와 resume만 내려온다. 이때 delivery가 없는 것은 오류가
아니다. 사전에 정한 OOF 기준을 통과하지 못해 전체 학습을 막은 정상 결과다.

## 4. 중단됐을 때 다시 시작하는 법

학습 중에는 5분마다 검증된 emergency resume를 Colab 내부에 한 개만 보관한다.
평소에는 계속 다운로드하지 않는다. 오류나 시간 제한이 생겼을 때만 최신 파일을 한 번
내려받는다.

```text
REALIGN_DOWNLOAD_REQUESTED phase=emergency_resume path=...
REALIGN_ERROR stage=campaign type=<오류 종류> message=<내용>
```

같은 Colab 런타임에서 셀을 다시 실행하면 입력 ZIP과 방금 받은 resume 경로를 자동으로
재사용한다. `REALIGN_UPLOAD_CACHE_REUSED count=2`가 나오면 정상이다. 런타임이 완전히
초기화됐다면 입력 ZIP과 최신 resume ZIP, 두 파일을 한 번에 선택해 올린다.

resume 검증에 실패하면 예전 결과를 억지로 이어 붙이지 않는다. `artifact bindings`나
`SHA-256 differs`가 보이면 해당 오류 로그와 사용한 파일명을 그대로 전달한다.

## 5. 실행이 끝난 뒤 전달할 것

다음 파일과 로그를 이 대화에 올린다.

- 최종 review ZIP
- 최종 resume ZIP
- delivery ZIP (`completed`일 때만)
- `REALIGN_CAMPAIGN_SUCCESS` 또는 `REALIGN_ERROR`가 나온 줄부터 마지막 줄까지

F1 중간 resume와 emergency resume는 최종 resume가 정상적으로 내려왔다면 따로 보낼
필요가 없다. 다만 최종 파일이 나오지 않은 채 런타임이 끝났다면 가장 최근 resume를
보내면 된다.

이 단계의 delivery는 대회 제출 ZIP이 아니다. 결과를 받은 뒤 OOF 수치와 세그먼트
게이트를 다시 확인하고, 승인된 모델만 기존 TabM 제출 코드에 붙이는 작업을 별도로
진행한다.
