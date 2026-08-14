# TabM 제출 전 검증 안내

지금 만드는 파일은 DACON에 바로 올리는 제출 ZIP이 아니다. 최종 모델이
평가 서버에서 제대로 열리는지, 전처리가 행마다 독립적으로 동작하는지,
제한 시간 안에 추론을 마칠 수 있는지 확인하기 위한 Kaggle 실행 묶음이다.

검증 결과를 확인하기 전에는 제출 ZIP을 만들 수 없게 막아 두었다. 실수로
검증용 파일을 DACON에 제출하거나, 아직 확인하지 않은 모델을 패키징하는
일을 피하기 위해서다.

## 이번에 검증하는 모델

후보 ID는 `tabm_hand_matchup_version_d_seed3407_v1`이다.

- 모델: TabM 한 개
- seed: 3407
- 학습 epoch: 3
- 학습 범위: 공식 train 2019~2024년, 총 1,475,092행
- 전처리: `raw_typed + dl_standard + hand_matchup`
- 숫자 임베딩: 학습 데이터에서 고정한 piecewise-linear bin
- 모델 디렉터리 SHA-256:
  `f5ef58341a8e2e3c630a38da74dad666666f9f39923c220f556a9c20c737474a`
- 검증용 `script.py` SHA-256:
  `fd4225b3cee8427f16ca54a00adb49417ad5914abf6fbb40e25686debfd303f8`

전처리에는 평가 데이터 전체의 평균, 빈도, 순위, 그룹 통계가 들어가지
않는다. `hand_matchup`은 한 행 안의 `pitcher_hand`와 `batter_hand`만 합쳐서
만든다. 평균·표준편차·결측치 대체값·범주 사전·piecewise bin은 모두 공식
학습 데이터에서 미리 계산한 값이다.

## handoff 폴더 안의 파일

로컬 생성 결과는 다음 폴더다.

```text
artifacts/tabm_submission_validation_handoff_v2/
├── candidate/
│   ├── candidate_manifest.json
│   └── model/
│       ├── inference_manifest.json
│       ├── numeric_embedding_0.json
│       ├── preprocessing_state.json
│       └── tabm_member_0_seed_3407.pt
├── runtime/
│   ├── competition_rules/
│   └── submission/
├── handoff_manifest.json
└── requirements.txt
```

각 파일의 역할은 이렇다.

- `tabm_member_0_seed_3407.pt`: 학습이 끝난 TabM 가중치
- `preprocessing_state.json`: 학습 데이터에서 고정한 전처리 값과 범주 사전
- `numeric_embedding_0.json`: 숫자 피처별 piecewise-linear bin 경계
- `inference_manifest.json`: 모델 구조, seed, epoch, 각 파일 해시
- `candidate_manifest.json`: 원본 Version D 전달물부터 이어지는 모델 계보
- `handoff_manifest.json`: 업로드 폴더 전체의 파일 목록과 해시
- `runtime/`: 검증할 추론 코드와 규칙 검사 코드
- `requirements.txt`: 실제 제출 시 추가 설치할 패키지 두 개

이 폴더에는 학습 데이터, 평가 예측 CSV, 제출용 ZIP이 없다.

현재 handoff manifest SHA-256은
`f635864a5c65f73b2e14887e7e00a0bc8766f89065734a0edc88050ee9e848dc`이고,
Kaggle 셀 SHA-256은
`9c6f9dfd808318165c2ba5df15d41cd23760e91d45dea6e833dc4dcdb60bc3df`이다.
업로드 전후 파일이 바뀌지 않았는지 확인할 때 쓰면 된다.

## 왜 환경을 두 개 쓰는가

DACON 평가 서버는 Python 3.11.15와 PyTorch 2.7.1을 사용한다. 반면 현재
Kaggle이나 Colab GPU 런타임은 Python 3.12와 더 최신 PyTorch를 쓰는 경우가
있다. GPU 런타임 하나만 확인하면 공식 환경 호환성을 증명할 수 없다.

그래서 한 번의 Kaggle 실행 안에서 검사를 둘로 나눈다.

1. 별도 Python 3.11.15 환경에서 PyTorch 2.7.1 CPU 버전으로 실제
   체크포인트를 연다. 공식 5행의 예측값도 저장한다.
2. Kaggle T4 GPU에서 같은 5행을 예측한다. CPU 환경 결과와 오차
   `1e-6` 이내로 같은지 확인한다.

그다음 T4에서 행 순서와 배치 크기를 바꿔도 결과가 같은지 검사하고,
245,789행 규모에서 실행 시간과 메모리를 잰다.

## 245,789행의 정확한 의미

로컬 공식 `test.csv`는 5행짜리 실행 예시다. 실제 245,789행 평가 데이터는
DACON 평가 서버 안에서만 제공되기 때문에 제출 전에 읽을 수 없다.

성능 검사는 공식 5행을 고정된 순서로 반복하고 `row_id`만 새로 붙인
245,789행짜리 scale fixture를 사용한다. 이 결과로 확인할 수 있는 것은
추론 시간과 RAM·VRAM 사용량이다. 리더보드 점수나 숨은 데이터 전체의
통계적 성질을 확인하는 검사가 아니다. 검증 보고서에도
`official_sample_plus_synthetic_scale`이라고 그대로 기록한다.

실제 평가 실행 중에는 해시로 선택한 8개 행을 한 행씩 다시 예측한다.
전체 배치에서 얻은 값과 다르면 `submission.csv`를 쓰지 않고 실패한다.

## Kaggle에서 실행하는 방법

필요한 설정은 다음과 같다.

- Accelerator: T4 GPU 한 개
- Internet: On
- Kaggle Input 1: `tabm_submission_validation_handoff` 폴더를 올린 비공개 데이터셋
- Kaggle Input 2: 기존 `lg-aimers-9th-data` 데이터셋

T4 두 개를 선택해도 검증기는 한 개만 쓴다. 이번 검사는 병렬 학습이
아니라 단일 제출 환경을 확인하는 작업이기 때문이다.

1. 새 Kaggle Notebook을 만든다.
2. 위의 두 데이터셋을 Input에 추가한다.
3. Internet을 켜고 Accelerator를 T4로 설정한다.
4. `experiments/tabm_campaign/KAGGLE_SUBMISSION_VALIDATION_CELL.py`의 내용을
   코드 셀 하나에 전부 복사한다.
5. 셀을 실행한다. 결과 파일을 보존하려면 `Save Version`으로 실행해도 된다.

셀은 GitHub에서 코드를 받지 않고 Drive도 연결하지 않는다. Input 폴더의
manifest와 모든 파일 해시를 먼저 확인한 뒤 실행한다.

## 예상 시간

보통 15~40분 정도로 잡으면 된다.

- Python 3.11.15와 CPU PyTorch 설치: 대략 5~20분
- 정확 환경 5행 검사: 수 분 이내
- T4 독립성 검사와 245,789행 성능 검사: 대략 5~15분

네트워크와 Kaggle 패키지 캐시 상태에 따라 더 걸릴 수 있다. 이 작업에는
모델 재학습이 없으므로, 이전처럼 몇 시간짜리 epoch를 다시 돌리지는 않는다.

## 정상 로그

처음에는 아래 로그가 차례로 나온다.

```text
VALIDATION_HANDOFF_FOUND path=...
VALIDATION_DATA_FOUND path=...
EXACT_ENV_READY python=...
VALIDATION_CODE_READY candidate=tabm_hand_matchup_version_d_seed3407_v1 runtime_sha256=...
VALIDATION_INPUTS_VERIFIED rows=5 scope=official_sample_plus_synthetic_scale model_sha256=...
VALIDATION_ENVIRONMENT_READY exact_python=3.11.15 exact_torch=2.7.1+cpu ...
VALIDATION_PROGRESS phase=...
VALIDATION_OUTPUT_VERIFIED rows=245789 output_sha256=... elapsed_seconds=...
VALIDATION_GATES_PASSED audit_scope=official_sample_plus_synthetic_scale hidden_rows_inspected=False
VALIDATION_SUCCESS review=... resume=... review_sha256=... resume_sha256=...
```

`hidden_rows_inspected=False`는 실패가 아니다. 숨은 평가 데이터를 미리 본
것처럼 기록하지 않기 위한 정상 표시다.

오류가 나면 다음 형식으로 끝난다.

```text
VALIDATION_ERROR stage=<단계> type=<오류 종류> message=<원인>
VALIDATION_DIAGNOSTICS_READY path=... sha256=...
```

이 경우 diagnostics ZIP과 마지막 `VALIDATION_ERROR` 줄을 보내면 된다. 오류가
초기 폴더 생성보다 먼저 발생해 diagnostics ZIP이 만들어지지 않았다면 화면의
전체 로그를 보내면 된다.

## 실행이 끝나면 보낼 파일

Kaggle `/kaggle/working`에 다음 두 파일이 생긴다.

- `tabm_submission_validation_review_<시간>.zip`
- `tabm_submission_validation_resume_<시간>.zip`

우선 review ZIP과 마지막 로그를 보내면 된다. resume ZIP은 검증 근거를
보존하는 파일이므로 같이 내려받아 두는 편이 좋다. 두 ZIP 모두 모델을
포함하지 않으며 DACON 제출 파일도 아니다.

파일명에 `resume`이 들어가지만 이번 검증에서 학습 epoch를 이어 가는 용도는
아니다. 검사가 중간에 끊기면 셀을 다시 실행해야 한다. 다만 모델은 이미
학습이 끝난 상태라 재학습하지 않으며, 다시 하는 일은 환경 설치와 제출 전
검사뿐이다.

review ZIP을 받은 뒤 확인할 항목은 다음과 같다.

- Python 3.11.15와 PyTorch 2.7.1에서 체크포인트가 열렸는가
- 정확 환경 CPU와 T4 GPU의 5행 예측이 허용 오차 안에서 같은가
- 5행을 뒤집거나 잘라서 예측해도 행별 결과가 같은가
- 245,789행 scale fixture가 480초 안에 끝났는가
- RAM 22 GiB, VRAM 20 GiB 안전선을 넘지 않았는가
- 출력 컬럼, 행 수, ID 순서, 확률 범위가 맞는가
- 보고서 identity가 현재 모델·전처리·코드 해시와 모두 일치하는가

이 항목이 전부 통과한 뒤에만 승인 기록을 만들고 최종 제출 ZIP 작업으로
넘어간다.

## 최종 제출 ZIP과의 차이

나중에 승인까지 끝난 최종 파일은 공식 예제처럼 아래 세 항목만 가진다.

```text
submit.zip
├── model/
├── script.py
└── requirements.txt
```

`script.py`는 `./data/test.csv`와 `./data/sample_submission.csv`만 읽고
`./output/submission.csv`만 만든다. 평가 데이터로 전처리를 다시 맞추거나
다른 평가 행의 통계를 계산하지 않는다. 누락 행에 임의 확률을 채우는
동작도 없다. ID나 스키마가 다르면 제출 파일을 만들지 않고 즉시 멈춘다.

현재 단계에서는 이 최종 ZIP을 만들지 않는다. 검증 결과가 아직 없기
때문이다.

## 대회 규칙과 연결해서 보면

공식 FAQ에 따르면 별도의 최종 제출 선택 절차가 없고, 규칙을 지킨 제출 중
가장 높은 점수가 순위에 반영된다. 점수가 낮거나 나중에 쓰이지 않는
제출도 규칙을 지켜야 하며 제출 이력이 검토될 수 있다.

공식 학습 데이터로 미리 계산한 통계와 ID 매핑은 사용할 수 있다. 반대로
평가 데이터 전체의 평균·빈도·순위·그룹 통계나 이전 평가 행을 이용한
피처는 사용할 수 없다. 이번 스크립트는 후자를 만들지 않으며, 저장된
전처리 상태가 추론 중 바뀌는지도 확인한다.

확인한 공식 자료는 [평가 및 코드 제출 안내](https://dacon.io/competitions/official/236743/overview/evaluation),
[평가 데이터 독립 예측 원칙 공지](https://dacon.io/competitions/official/236743/talkboard/417123?page=1&dtype=recent),
[대회 FAQ와 운영진 답변](https://dacon.io/competitions/official/236743/talkboard/417082?page=1&dtype=recent)이다.
규칙과 공지는 바뀔 수 있으므로 실제 제출 직전에는 세 페이지를 다시
확인해야 한다.
