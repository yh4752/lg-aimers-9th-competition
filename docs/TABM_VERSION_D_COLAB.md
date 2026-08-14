# TabM Version D Colab 실행 안내

## 목적

이 셀은 Stage C에서 선택된 `single_s3407` TabM을 공식 2019~2024 학습 데이터
전체로 3 epoch 학습하고, 대회 규칙 및 평가 자원 검토를 거친 review-only 번들을
만든다. 제출 파일은 만들지 않는다.

- 실행 환경: Google Colab, NVIDIA T4 GPU 1개
- 예상 시간: 약 20~45분
- 전체 학습: 사용자가 Colab에서 실행
- Codex 로컬 검증: 문법, fixture 테스트와 정적 검사만 수행

## 준비 파일

파일 이름을 다음과 정확히 맞춘다.

1. `lg-aimers-9th-data.zip`
2. `tabm_colab_stage_C_delivery.zip`
3. 복구할 때만 emergency 또는 frozen ZIP 1개

공식 데이터 ZIP SHA-256은
`0a1df39e82ed6621629378650c7b938c21bee56f0d57e414c61882f6de409176`,
Stage C delivery SHA-256은
`f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a`다.
셀이 두 값을 직접 확인하므로 잘못된 파일은 학습 전에 거부된다.

실행 셀은
`experiments/tabm_campaign/COLAB_VERSION_D_REVIEW_CELL.py`다. 파일 전체를 복사해
Colab 코드 셀 하나에 붙여 넣는다. 현재 셀 SHA-256은
`4af85f6110328d0622e109553e0d1f13ec6a847a9f4095756c48342cb5e08672`이며
크기는 125,811바이트다.

## 처음 실행

1. Colab에서 `런타임` → `런타임 유형 변경` → `T4 GPU`를 선택한다.
2. 셀 상단의 `RECOVERY_MODE = "none"`을 그대로 둔다.
3. 셀 전체를 실행한다.
4. 요청 순서에 맞춰 공식 데이터 ZIP과 Stage C delivery를 각각 업로드한다.
5. 브라우저가 여러 파일을 내려받도록 허용한다.

정상 진행 중에는 epoch마다 다음 형식의 로그가 나온다.

```text
FINAL_TRAINING_PROGRESS seed=3407 epoch=1/3
VERSION_D_EMERGENCY_SNAPSHOT_READY epoch=1 path=...
```

epoch 1~3이 끝날 때 emergency ZIP이 하나씩 다운로드된다. 학습이 끝나면
`tabm_version_D_frozen_model.zip`이 추가로 다운로드되고, 검토가 모두 통과하면
`tabm_hand_matchup_stage_D_review_delivery.zip`이 다운로드된다.

최종 성공 로그는 다음 순서다.

```text
VERSION_D_INDEPENDENCE_PASSED
VERSION_D_SCALE_GATE_PASSED rows=245789 seconds=...
VERSION_D_REVIEW_READY path=...
VERSION_D_DELIVERY_READY path=...
VERSION_D_DOWNLOAD_REQUESTED kind=delivery path=...
```

## 중단 후 재실행

같은 Colab VM이 유지되고 있다면 `RECOVERY_MODE = "none"`인 상태로 셀을 다시
실행한다. `/content/tabm_version_d`의 마지막 완료 checkpoint를 검증하고 다음
epoch부터 이어서 실행한다. 중단된 epoch는 처음부터 다시 실행한다.

VM이 초기화됐다면 새 세션에서 공식 데이터 ZIP과 Stage C delivery를 다시
업로드해야 한다.

### Emergency ZIP으로 복구

가장 높은 epoch 번호의 emergency ZIP을 사용한다.

```python
RECOVERY_MODE = "emergency"
```

공식 데이터와 Stage C delivery 다음에
`tabm_version_D_emergency_epoch_NNN_<hash>.zip`을 업로드한다. 완료된 epoch 다음부터
학습한다.

### Frozen model로 복구

이미 `tabm_version_D_frozen_model.zip`을 받았다면 다음과 같이 설정한다.

```python
RECOVERY_MODE = "frozen"
```

공식 데이터, Stage C delivery와 frozen ZIP을 업로드한다. 학습은 생략하고 독립성,
의존성, 시간, GPU, RAM과 artifact 크기 검토부터 실행한다.

## 다운로드 파일 관리

브라우저에는 emergency 3개, frozen model 1개, 최종 delivery 1개의 다운로드 요청이
나올 수 있다. 가장 최신 emergency ZIP만 복구에 필요하지만, 실행이 끝날 때까지는
모두 보관한다. frozen model과 최종 delivery도 보관한다.

브라우저가 자동 다운로드를 막더라도 로그에 함께 표시되는 파일 링크로 직접 받을 수
있다. 오류가 발생하면 셀은 마지막으로 완성된 emergency 또는 frozen ZIP을 다시
내려받도록 요청한다.

## Codex에 전달할 파일

성공했다면 다음 파일 하나만 전달한다.

```text
tabm_hand_matchup_stage_D_review_delivery.zip
```

오류가 발생했다면 다음 두 가지를 전달한다.

1. 전체 `VERSION_D_ERROR` 로그
2. 마지막으로 다운로드된 emergency 또는 frozen ZIP

## 이 단계에서 만들지 않는 것

Version D는 대회 제출 ZIP, 제출 실행 스크립트, 제출 의존성 파일 또는 제출용 모델
디렉터리를 만들지 않는다. 최종 delivery 검증, Python 3.11.15 호환성 확인과 제출
직전 공식 규칙 재검토가 끝난 뒤 별도 패키징 단계로 넘어간다.
