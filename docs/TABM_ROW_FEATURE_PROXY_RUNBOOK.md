# TabM 행 단위 파생변수 Stage P 실행 안내

Stage P는 동결한 Stage C TabM 기준선과 여섯 개의 행 단위 파생변수
묶음을 비교하는 proxy 실험이다. seed `42`와 `3407`에서 각각 같은 표본과
epoch 예산으로 짝을 맞춰 Brier 차이를 본다. 결과는 후속 연구 방향을
고르기 위한 review/resume 증거일 뿐이다. 제출 파일이 아니며 이 단계에서
평가 예측이나 제출 ZIP을 만들지 않는다.

## 비교하는 피처 묶음

| 묶음 | 한 행에서 더하는 정보 |
|---|---|
| `count_context` | 볼·스트라이크·아웃 카운트와 주자 상태의 결합 |
| `pressure_context` | 이닝, 투수팀 점수 차, leverage 구간과 투수팀 승리 기대값 |
| `hand_state_interactions` | `hand_matchup`과 카운트·주자 상태·경기 유형의 결합 |
| `pitcher_batter_gap` | 공식 as-of 투수·타자 성공률, middle rate, 관측 횟수의 차이 |
| `recent_trend` | 공식 as-of 1·3·5경기 지표와 투수 장기 지표의 차이 |
| `pitchmix_shape` | fastball·breaking·offspeed 비율의 행 내부 정규화, 엔트로피와 비율 차이 |

여섯 묶음은 모두 현재 행에 있는 값만 읽는다. 순서를 바꾸거나 다른 행을
추가하고 배치를 나눠도 기존 행의 파생값은 같다. 결측 대체와 표준화
상태는 fold 학습 구간에서만 fit하고 검증 행에는 고정해 적용한다.
`control_success`는 피처 함수에 넘기지 않고 학습·지표 계산에서만 쓴다.

입력 준비 도구는 `train.csv`와 `trackman_history.csv`만 담는다. `test.csv`와
`sample_submission.csv`가 있어도 ZIP에 넣지 않으며 Colab 런타임도 이를
읽지 않는다. 평가 행이나 평가 분포에서 집계·순위·rolling·보정값을
만들지도 않는다. 이 구조에서는 test 데이터, test 분포, target 정보가 피처로
새어 들어갈 경로가 없다.

## 입력 준비

로컬의 데이터 디렉터리에 `train.csv`와 `trackman_history.csv`가 있어야 한다.
저장소 루트에서 아래 명령을 실행한다.

```bash
artifacts/tabm_submission_python311/bin/python \
  tools/prepare_tabm_row_feature_colab_input.py \
  --data-dir /Users/yonghyun/Documents/kaggle-lg-aimers-9th-data-upload \
  --output artifacts/tabm_row_feature_input.zip
```

공식 train SHA-256과 두 CSV의 크기·SHA-256을 확인한 뒤
`tabm_row_feature_input.zip`을 만든다. 기존 출력은 기본으로 덮어쓰지
않는다. 성공 시 마지막 줄은 다음 형태다.

```text
ROW_FEATURE_INPUT_READY path=... sha256=... size_bytes=...
```

## Colab에서 실행

1. Colab 런타임의 accelerator를 T4 GPU 한 장으로 설정한다.
2. [COLAB_ROW_FEATURE_PROXY_CELL.py](../experiments/tabm_campaign/COLAB_ROW_FEATURE_PROXY_CELL.py)
   전체를 빈 Colab 셀 하나에 복사해 실행한다.
3. 업로드 창이 열리면 `tabm_row_feature_input.zip`을 올린다. 이전 세션을
   이어갈 때만 가장 최근의 유효한 Stage P resume ZIP을 같이 올린다.

업로드는 데이터 ZIP 정확히 하나와 Stage P resume ZIP 0개 또는 1개여야
한다. Drive를 mount하거나 GitHub에서 코드를 받지 않는 direct-upload 셀이다.
Stage A–D resume, 다른 계약·코드·입력에 묶인 resume는 GPU 작업 전에
거부한다.

세션 제한은 셀 시작 시각부터 `10800`초이다. 업로드와 패키지 준비 시간도
여기에 포함되며 마지막 `900`초 구간에는 새 작업을 시작하지 않고
증거 ZIP을 마무리한다. early stopping에 따라 한 번에 끝날 수도 있지만
3시간 세션이 하나에서 여러 번 필요할 수 있다.

## 작업 순서와 로그

2023년까지의 학습 행에서 결정론적 `400000`행 표본을 뽑고 2024년 전체
`253507`행을 검증한다. 기준선 두 개를 먼저 실행한 뒤 같은 묶음의 seed 쌍을 붙여
총 14개 작업을 아래 순서로 진행한다.

```text
rfp__baseline__s42
rfp__baseline__s3407
rfp__count_context__s42
rfp__count_context__s3407
rfp__pressure_context__s42
rfp__pressure_context__s3407
rfp__hand_state_interactions__s42
rfp__hand_state_interactions__s3407
rfp__pitcher_batter_gap__s42
rfp__pitcher_batter_gap__s3407
rfp__recent_trend__s42
rfp__recent_trend__s3407
rfp__pitchmix_shape__s42
rfp__pitchmix_shape__s3407
```

초기화가 정상이면 다음 마커가 순서대로 보인다.

```text
ROW_FEATURE_CODE_READY sha256=... size_bytes=...
ROW_FEATURE_INPUTS_VERIFIED manifest_sha256=... resume=none|...
ROW_FEATURE_DEPENDENCIES_READY
ROW_FEATURE_GPU_READY device_count=... name=...
ROW_FEATURE_STAGE_SELECTED version=P
```

학습 중에는 `JOB_START`, `TRAINING_PROGRESS`, `EPOCH_CHECKPOINTED`가 이어진다.
정상적으로 증거 묶음과 전달 ZIP을 완성하면 마지막에 아래 두 줄이
출력된다.

```text
ROW_FEATURE_BUNDLE_SUCCESS version=P review=... resume=...
ROW_FEATURE_DELIVERY_READY path=... sha256=...
```

## 중단될 때와 재실행

완료한 작업은 job·계약·코드·캐시·예측·체크포인트 해시가 모두 맞을 때만
재사용한다. 진행 중이던 작업은 모델, optimizer, scheduler, epoch와
job·계약·캐시 바인딩이 일치할 때 마지막으로 완료한 epoch 체크포인트부터
이어서 학습한다. 일치하지 않는 자료를 추측해 재사용하지 않는다.

각 후보가 끝나면 검증한 emergency resume를 다운로드한다. 작업이 진행 중일
때는 600초마다 새 epoch 체크포인트를 포함한 snapshot을 시도하며
다음 snapshot이 검증되기 전까지 최근 유효본을 유지한다. Colab이 끝나거나 오류로
멈추면 가장 최근의 유효한 Stage P emergency resume ZIP을 저장한 뒤 다음
세션에서 데이터 ZIP과 함께 올린다.

오류는 다음 형태로 남는다.

```text
ROW_FEATURE_ERROR stage=<stage> type=<type> message=<message>
```

실패했을 때는 이 줄을 포함한 오류 로그와 마지막으로 다운로드된 유효한
emergency resume ZIP을 같이 보내야 한다. resume 다운로드 자체가 실패하면
`ROW_FEATURE_ERROR stage=emergency_resume ...`도 빼지 말고 전달한다.

## 결과 파일과 판정

정상 종료 후 반환할 파일은 `tabm_row_feature_stage_P_delivery.zip`이다.
안에는 `tabm_row_feature_stage_P_review_bundle.zip`,
`tabm_row_feature_stage_P_resume_bundle.zip`, `row_feature_proxy.log`,
`delivery_manifest.json`이 든다. 마지막 파일은 두 bundle의 SHA-256을 런타임·입력·
계약 바인딩과 함께 기록한다. review bundle은 완료한 후보의 Brier·예측 증거와
proxy 판정을 담고 resume bundle은 완료 결과와 다음 실행에 필요한 상태·체크포인트를
담는다. 14개 작업이 끝나지 않았다면 판정은 `incomplete`이며 미완료 후보를
기각으로 바꾸지 않는다. 한 후보의 실패는 그 후보만 제외하고 다른 후보의
실행과 resume 생성은 계속한다. 단, 기준선이 실패하면 판정은 막힌다.

각 묶음의 delta는 같은 seed 기준선의 Brier를 뺀 값이다. 두 seed 평균 delta가
`-0.00003` 이하이고 더 나쁜 seed의 delta도 `+0.00005` 이하면 strong
survivor다. 이 조건을 통과한 묶음은 개수 제한 없이 모두 남긴다. 나머지
묶음 중 평균 delta가 음수인 최상위 하나는 safety survivor로 남긴다.
나머지 후보가 모두 평균을 개선하지 못하면 safety survivor는 없다.

survivor가 나와도 Stage P는 후속 Stage V 설계를 열 수 있는 proxy 근거일 뿐이다.
전체 데이터 최종 학습이나 제출물 생성을 허용하지 않는 연구·검토용 단계다.

성공했을 때는 `tabm_row_feature_stage_P_delivery.zip`을 그대로 반환한다. 오류로
끝났다면 전체 오류 로그와 가장 최근의 유효한 emergency resume ZIP이 필요하다.
파일명을 바꾸거나 제출 ZIP으로 사용하지 않는다.
