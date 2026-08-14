# TabM Stage C Colab 복구 실행 안내

## 목적

Kaggle에서 완료하지 못한 Stage C 작업 한 개를 Colab T4 GPU에서 이어서
실행한다. 완료된 기존 실험 20개의 결과는 그대로 보존하며, Kaggle에서
중단된 마지막 작업만 Colab 환경에서 처음부터 다시 학습한다.

이 과정은 연구용 review 및 resume 번들을 생성한다. 대회 제출 파일은
생성하지 않는다.

## 준비 파일

`artifacts/tabm_colab_stage_c_handoff` 디렉터리에서 다음 파일을 사용한다.

- `COLAB_STAGE_C_RECOVERY_CELL.py`
- `lg-aimers-9th-data.zip`
- `tabm_search_stage_C_resume_bundle.zip`
- `handoff_manifest.json`

`handoff_manifest.json`은 나머지 세 파일의 크기와 SHA-256을 기록한다.
파일 이름을 바꾸거나 ZIP의 내용을 다시 압축하지 않는다.

## 첫 실행

1. 새 Colab 노트북을 연다.
2. `런타임 > 런타임 유형 변경`에서 `T4 GPU`를 선택한다.
3. `COLAB_STAGE_C_RECOVERY_CELL.py`의 전체 내용을 한 셀에 복사한다.
4. 셀 상단의 `UPLOAD_EMERGENCY_SNAPSHOT = False`를 유지한다.
5. 셀을 실행한다.
6. `COLAB_UPLOAD_REQUIRED filename=lg-aimers-9th-data.zip`이 출력되면
   해당 데이터 ZIP만 업로드한다.
7. `COLAB_UPLOAD_REQUIRED filename=tabm_search_stage_C_resume_bundle.zip`이
   출력되면 해당 resume ZIP만 업로드한다.
8. 로그에서 다음 항목을 확인한다.

```text
COLAB_CODE_READY ...
COLAB_INPUTS_VERIFIED ...
COLAB_GPU_READY device_count=1 name=Tesla T4
COLAB_RESUME_READY source=stage_C_base epoch=0 ...
STAGE_SELECTED version=C
JOB_START job=c_final__a__p2__piecewise_linear__bce__plateau__s42__s3407__tr2022__va2023 gpu=0 ...
```

데이터 업로드, 압축 해제, 전처리 캐시 생성을 포함한 예상 시간은 약
20~70분이다. Colab 상태와 업로드 속도에 따라 더 오래 걸릴 수 있다.

## 실행 중 확인할 로그

정상 학습 중에는 다음 로그가 반복된다.

```text
TRAINING_PROGRESS ...
VALIDATION_PROGRESS ...
EPOCH_CHECKPOINTED ...
```

약 20분마다 새 epoch 체크포인트가 있으면 다음 로그와 함께 비상 복구
ZIP의 다운로드가 요청된다.

```text
EMERGENCY_SNAPSHOT_READY epoch=<n> path=<path> sha256=<sha256>
EMERGENCY_DOWNLOAD_REQUESTED path=<path>
```

`EMERGENCY_DOWNLOAD_REQUESTED`는 브라우저에 다운로드를 요청했다는 뜻이다.
로컬 컴퓨터에 저장됐다는 보장은 아니므로 Downloads 디렉터리에 파일이
실제로 있는지 직접 확인한다. 이전 비상 복구 ZIP은 삭제하지 않는다.

## 같은 Colab VM에서 셀이 중단된 경우

런타임이 유지되고 있다면 같은 셀을 다시 실행한다. 업로드 파일과 검증된
epoch 체크포인트가 `/content/tabm_stage_c_recovery`에 남아 있으므로 입력
검증과 완료된 epoch를 재사용한다.

기존 학습 프로세스가 아직 살아 있으면 중복 실행을 거부한다. 이 경우 기존
프로세스가 종료되거나 인터럽트 정리가 끝난 뒤 다시 실행한다.

## Colab VM이 초기화된 경우

1. 새 T4 GPU 런타임을 연다.
2. 셀 상단을 `UPLOAD_EMERGENCY_SNAPSHOT = True`로 변경한다.
3. 셀을 실행하고 데이터 ZIP과 base resume ZIP을 차례대로 업로드한다.
4. 비상 복구 ZIP 업로드 요청이 나오면 로컬 컴퓨터에 저장된 가장 높은
   epoch의 `tabm_colab_emergency_epoch_*.zip` 하나만 업로드한다.
5. 다음 로그를 확인한다.

```text
COLAB_RESUME_READY source=emergency epoch=<다음 epoch> ...
```

환경, 코드, 데이터, 캐시 또는 체크포인트 해시가 다르면 학습을 시작하기
전에 중단된다. 이 경우 다른 비상 복구 ZIP을 임의로 사용하지 말고 오류
로그를 전달한다.

## 최종 성공

Stage C가 완성되면 다음 로그가 출력된다.

```text
COLAB_DELIVERY_READY path=/content/tabm_stage_c_recovery/tabm_colab_stage_C_delivery.zip sha256=<sha256>
COLAB_DOWNLOAD_REQUESTED path=/content/tabm_stage_c_recovery/tabm_colab_stage_C_delivery.zip
```

Downloads 디렉터리에서 `tabm_colab_stage_C_delivery.zip`을 확인한 뒤 그
파일 하나만 Codex에 전달한다. ZIP에는 다음 연구 산출물이 들어 있다.

```text
tabm_colab_stage_C_delivery.zip
├── tabm_search_stage_C_review_bundle.zip
├── tabm_search_stage_C_resume_bundle.zip
├── colab_stage_C.log
└── delivery_manifest.json
```

## 오류 발생 시

오류의 마지막 표식은 다음과 같다.

```text
COLAB_STAGE_C_ERROR stage=<stage> type=<type> message=<message>
```

이 표식부터 traceback 마지막 줄까지의 로그를 복사해 전달한다. VM이
초기화되기 전이라면 `/content/tabm_stage_c_recovery`의 파일을 삭제하지
않고 그대로 둔다.
