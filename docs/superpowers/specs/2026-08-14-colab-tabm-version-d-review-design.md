# Colab TabM Version D 최종 검토 설계

## 1. 목적과 범위

이 작업은 검증된 Stage C 결과에서 선택된 단일 TabM 모델을 공식 학습 데이터
전체에 다시 학습하고, 평가 서버에서 사용할 수 있는 동결 추론 산출물과 규칙 검토
증거를 만든다. 실행 환경은 Google Colab의 단일 NVIDIA T4이며 Google Drive와
GitHub에 의존하지 않는다.

이 단계는 `submit.zip`을 만들지 않는다. 최종 출력은 이후 제출 패키징 심사에
입력할 검토 전용 번들이다. 제출 패키징은 이 번들의 무결성, 실제 평가 환경 모사,
최신 공식 규칙 재검토가 모두 통과한 뒤 별도 작업으로 진행한다.

## 2. 고정 입력과 모델 결정

### 2.1 입력

- 공식 데이터 ZIP: 루트에 `train.csv`, `trackman_history.csv`, `test.csv`,
  `sample_submission.csv`만 포함한다.
- Stage C delivery:
  `tabm_colab_stage_C_delivery.zip`
- 승인된 Stage C delivery SHA-256:
  `f054e8e819d1053299fef4749a32e923db9136e20fc9c12cda79bbc7ef8c484a`
- 승인된 Stage C 상태: Version C, `stage_complete=true`, 21개 결과가 모두
  `completed`여야 한다.

셀은 입력 ZIP의 멤버 이름, 크기, SHA-256, Stage C 내부 manifest, review와 resume의
계보 및 상태 일치를 검증한다. 검증값이 하나라도 다르면 학습 전에 중단한다.

### 2.2 최종 후보

- predictor: `single_s3407`
- preprocessing: `dl_standard + hand_matchup`
- model: TabM P2
- `k=32`, `width=512`, `blocks=4`, `dropout=0.1`
- numeric embedding: `piecewise_linear`
- loss: BCE
- learning rate: `0.0006`
- weight decay: `0.0001`
- seed: `3407`
- final members: 1

Stage C의 3-seed 평균은 최신 fold에서 소폭 개선됐지만 시간 가중 temporal Brier가
단일 seed 3407보다 나빴다. one-cycle sentinel은 최신 fold 회귀와 segment gate
실패로 제외됐다. Version D는 이 결정을 다시 탐색하거나 변경하지 않는다.

## 3. 공식 규칙 계약

2026-08-14에 다음 공식 페이지를 다시 확인했다.

- 대회 규칙: <https://dacon.io/competitions/official/236743/overview/rules>
- 평가 및 코드 제출 안내:
  <https://dacon.io/competitions/official/236743/overview/evaluation>
- 평가 데이터 독립 예측 재안내:
  <https://dacon.io/competitions/official/236743/talkboard/417123?page=1&dtype=recent>

Version D는 다음 계약을 강제한다.

1. 전처리 상태, 범주 사전, 결측 대치값, 표준화 통계와 piecewise-linear bin은
   공식 `train.csv`만으로 적합한다.
2. 모델 학습은 공식 2019~2024 학습 행만 사용한다.
3. `test.csv`는 학습과 모델 선택이 끝나고 모델 상태가 동결된 뒤에만 읽는다.
4. `hand_matchup`은 같은 행의 `pitcher_hand`와 `batter_hand`만 결합한다.
5. test 행 사이의 집계, 빈도, 평균, 순위, rolling, lag, 보정 및 상태 갱신을
   금지한다.
6. 전체 배치, 역순, 셔플, 여러 batch size와 단일 행 추론의 확률이 허용 오차 안에서
   같고, 추론 전후 모델 상태 해시가 같아야 한다.
7. 외부 데이터, 외부 API, 추론 중 인터넷 다운로드를 사용하지 않는다.
8. Stage D 셀과 검토 번들에는 제출 패키지 생성 경로를 포함하지 않는다.

공식 평가 환경은 Ubuntu 22.04.5, Python 3.11.15, NVIDIA L4 22.4 GiB,
6 vCPU, RAM 28 GB다. 패키지 설치와 추론은 각각 600초 안에 끝나야 하며 제출 ZIP은
10 GB, 압축 해제 후 파일은 32 GB 이하여야 한다. Version D는 더 보수적인 480초
내부 제한, GPU allocation 20 GiB, process RSS 22 GB, 동결 산출물 2 GB 제한을
적용한다.

## 4. 전체 학습 정책

### 4.1 고정 epoch

최종 학습은 3 epoch로 고정한다. 선택된 seed 3407의 Stage C 최적 epoch 수는 최신
fold 4회와 이전 fold 1회다. 최신 fold에 0.7, 이전 fold에 0.3의 가중치를 적용하면
`0.7 × 4 + 0.3 × 1 = 3.1`이므로 가장 가까운 정수 3회를 사용한다.

기존의 `round(median([4, 1]))`는 Python의 짝수 반올림으로 2를 반환한다. 이 값은
데이터 근거가 아니라 언어의 반올림 규칙에 좌우되므로 사용하지 않는다. Version D는
Stage C 결과를 본 뒤 epoch를 다시 조정하지 않으며 3회를 계약에 고정한다.

### 4.2 학습률 정책

Stage C에서 선택된 scheduler 이름은 `plateau`지만 전체 학습에는 검증 fold가 없다.
평가 데이터나 임의의 내부 지표로 scheduler를 움직이지 않고 3 epoch 동안 학습률
`0.0006`을 고정한다. 이는 검증 데이터 의존성을 제거하고 초기 plateau 구간의 동작을
재현한다.

## 5. 구성 요소

### 5.1 Version D 계약과 검증기

Stage C delivery 해시, 선택 predictor, 후보 설정, seed, epoch, 공식 데이터 해시와
정책 해시를 하나의 Version D 계약으로 고정한다. 계약과 실행 코드의 SHA-256은 결과
manifest에 기록한다.

### 5.2 재시작 가능한 전체 학습기

학습기는 전처리와 numeric embedding 상태를 공식 학습 데이터에서 한 번 적합한 뒤
epoch 경계마다 다음 상태를 원자적으로 저장한다.

- 모델 state dict
- optimizer와 AMP scaler 상태
- 완료 epoch와 seed
- 전처리 및 numeric embedding 상태 해시
- 입력, 계약, 학습 코드 해시
- RNG 상태

중단 시 마지막으로 완전히 저장된 epoch 다음부터 재개한다. 진행 중이던 epoch는
재실행한다. 최종 동결 산출물에는 optimizer, scaler와 RNG 상태를 포함하지 않는다.

### 5.3 동결 추론 산출물

학습 완료 직후 다음 파일을 묶은 모델 snapshot을 먼저 생성하고 다운로드를 요청한다.

- 모델 가중치
- `preprocessing_state.json`
- numeric embedding state
- `inference_manifest.json`
- 각 파일의 SHA-256

모델 snapshot이 있으면 Colab VM이 초기화돼도 학습을 반복하지 않고 규칙 및 성능
검증만 다시 실행할 수 있다.

### 5.4 규칙 및 평가 환경 검토기

동결된 모델에 대해 다음을 순서대로 실행한다.

1. 추론 소스 정적 검사
2. Colab의 현재 Python으로 만든 일회용 가상 환경에서 요구 패키지 설치 및 import 검사
3. 5개 행 동결 추론 smoke test
4. test 행 순서, 셔플, singleton 및 batch-size 독립성 검사
5. 245,789행 synthetic scale 추론
6. 설치 시간, 추론 시간, GPU, RAM과 산출물 크기 검사
7. 추론 전후 manifest 및 모델 상태 해시 불변 검사

독립성 검사는 모델이나 전처리를 변경하지 않는다. 검사 결과를 이용한 보정,
재학습, hyperparameter 변경도 금지한다.

### 5.5 Colab 한 셀 실행기

사용자는 생성된 Python 파일 전체를 Colab 코드 셀 하나에 붙여 넣는다. 셀은 Drive나
GitHub에 접근하지 않고 필요한 파일을 하나씩 직접 요청한다. 로그는 즉시 출력하고
브라우저 다운로드를 요청한다.

## 6. 데이터 흐름

1. 내장 코드 압축 해제 및 코드 SHA-256 확인
2. 공식 데이터 ZIP과 Stage C delivery 직접 업로드
3. 모든 입력과 Stage C 결정 검증
4. T4 1개, CUDA와 라이브러리 버전 확인
5. 공식 학습 데이터만으로 전처리 적합
6. 단일 seed 3407 모델 3 epoch 학습
7. epoch 경계 emergency snapshot 생성 및 다운로드
8. 동결 모델 snapshot 생성 및 다운로드
9. 동결 이후에만 test 입력을 사용해 독립성과 평가 환경 검사
10. Stage D review bundle과 실행 로그를 delivery ZIP으로 묶음
11. 최종 delivery ZIP 다운로드

## 7. 출력과 로그 계약

정상 로그에는 최소한 다음 marker가 있어야 한다.

- `VERSION_D_CODE_READY`
- `VERSION_D_INPUTS_VERIFIED`
- `VERSION_D_GPU_READY`
- `FINAL_TRAINING_PROGRESS seed=3407 epoch=N/3`
- `VERSION_D_EMERGENCY_SNAPSHOT_READY`
- `VERSION_D_FROZEN_MODEL_READY`
- `VERSION_D_INDEPENDENCE_PASSED`
- `VERSION_D_SCALE_GATE_PASSED`
- `VERSION_D_REVIEW_READY`
- `VERSION_D_DELIVERY_READY`
- `VERSION_D_DOWNLOAD_REQUESTED`

오류는 `VERSION_D_ERROR stage=<stage> type=<type> message=<message>` 형식으로
출력한다. 부분 결과를 최종 성공으로 표시하지 않는다.

최종 출력은 `tabm_hand_matchup_stage_D_review_delivery.zip`이며 다음을 포함한다.

- `tabm_hand_matchup_final_review_bundle.zip`
- 전체 실행 로그
- Version D delivery manifest
- 입력, 코드, 계약, 정책 및 내부 산출물 해시

review bundle은 제출물이 아니며 archive 루트에 `script.py`, `requirements.txt` 또는
`model/` 제출 구조를 만들지 않는다.

## 8. 재실행과 복구

- 같은 VM에서 다시 실행하면 검증된 기존 입력과 마지막 완료 epoch를 재사용한다.
- VM 초기화 후에는 공식 데이터 ZIP, Stage C delivery와 가장 최신 emergency snapshot을
  업로드한다.
- frozen-model snapshot이 있으면 emergency snapshot보다 우선하며 학습을 생략한다.
- snapshot의 계약, 코드, 입력, 환경 또는 epoch 해시가 다르면 사용하지 않는다.
- 같은 epoch를 주장하는 서로 다른 snapshot이 있으면 충돌로 중단한다.

## 9. 자원 판단

Stage C의 선택 모델은 T4에서 GPU reserved 약 1 GB, process RSS 약 4 GB를 사용했고
약 25만 행 검증 추론은 약 6.7초였다. Version D는 같은 단일 모델과 전처리를 사용하고
3 epoch만 학습하므로 T4 15 GB에 충분한 여유가 있다. 예상 총 실행 시간은 데이터
업로드, 전처리, 학습, 별도 환경 검사와 scale 추론을 합쳐 약 20~45분이다.

평가 서버의 L4는 T4보다 빠르고 VRAM도 크지만 성능을 가정해 통과시키지 않는다.
Version D는 T4에서 480초 추론 제한을 통과해야만 review-ready로 인정한다.

## 10. 테스트와 승인 조건

Codex는 전체 데이터 학습을 실행하지 않는다. 다음 경량 검증만 수행한다.

- 계약 및 입력 변조 fixture 테스트
- Stage C 선택값과 해시 불일치 거부 테스트
- train-only fit 및 test-before-freeze 거부 테스트
- epoch snapshot 원자성, 재개 및 충돌 테스트
- frozen snapshot 재사용 테스트
- 행 독립성 fixture 테스트
- source gate와 제출 패키지 경로 부재 테스트
- Colab 셀 문법, 크기, 결정성 및 필수 로그 테스트
- 기존 전체 경량 테스트 회귀 검사

사용자가 Colab에서 실행한 뒤 다음 조건을 모두 만족해야 다음 패키징 설계로 넘어간다.

1. Stage D delivery 검증 성공
2. review bundle lineage와 모든 파일 해시 일치
3. 독립성 검사 통과
4. T4 scale gate 480초 이내
5. GPU, RAM, artifact 크기 제한 통과
6. 별도 환경 의존성 및 5행 추론 통과
7. 학습 입력이 공식 train 데이터로 제한됐다는 증거 확인
8. 실행에 사용한 Python, CUDA와 핵심 패키지 버전 기록
9. 제출 패키징 단계에서 Python 3.11.15 호환성 재검증 예정 상태 유지
10. 제출 직전 최신 공식 규칙 재검토 예정 상태 유지

## 11. 제외 범위

- `submit.zip`, `script.py`, `requirements.txt`, 제출용 `model/` 생성
- 리더보드 제출 또는 외부 전송
- test 분포를 이용한 보정, calibration 또는 feature 생성
- 다른 seed, ensemble, 모델 또는 전처리 재탐색
- Colab Drive mount, GitHub clone 또는 외부 모델 다운로드
