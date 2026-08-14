# TabM 최종 제출물 설명

## 현재 상태

Version D TabM 후보의 최종 제출 파일을 로컬에 생성했다. 아직 DACON에
업로드하지 않았으며, 실제 제출할 파일은 아래 하나다.

```text
artifacts/tabm_submission_version_d/submit.zip
```

파일의 SHA-256은 다음과 같다.

```text
bcea66999322721e60016761ce041ca151c71b034d0e4122f0b2852f5b06697a
```

이 값이 다르면 이 문서에서 검증한 파일과 같은 제출물이 아니다. 같은 폴더의
`submission_receipt.json`과 `evidence/`는 검증 기록이며 DACON에는 올리지 않는다.

## 어떤 모델인가

- 후보 ID: `tabm_hand_matchup_version_d_seed3407_v1`
- 모델: 단일 TabM
- seed: 3407
- 전체 학습 epoch: 3
- 학습 데이터: 주최 측이 제공한 2019~2024년 `train.csv`
- 전처리: `raw_typed + dl_standard + hand_matchup`
- 숫자 임베딩: 공식 학습 데이터로 고정한 piecewise-linear bin
- 모델 SHA-256:
  `f5ef58341a8e2e3c630a38da74dad666666f9f39923c220f556a9c20c737474a`

Stage C 시간 분할 검증에서 seed 3407 단일 모델이 선택됐다. 최신 분할
`2023→2024`의 Brier는 `0.2481108023`, 이전 분할 `2022→2023`의 Brier는
`0.2508657359`였다. 이 수치는 모델 선택 근거이며 리더보드 점수를 의미하지
않는다.

## ZIP 안에는 무엇이 들어 있는가

최상위에는 DACON이 요구하는 세 항목만 있다.

```text
submit.zip
├── script.py
├── requirements.txt
└── model/
    ├── inference_manifest.json
    ├── numeric_embedding_0.json
    ├── preprocessing_state.json
    └── tabm_member_0_seed_3407.pt
```

압축 크기는 `9,557,546`바이트, 압축 해제 후 파일 합계는
`10,495,143`바이트다. 공식 제한인 압축 10GB와 압축 해제 후 32GB보다
충분히 작다.

각 파일의 역할은 다음과 같다.

- `script.py`: 입력 검증, 전처리, TabM 추론, 독립성 canary, 결과 저장
- `requirements.txt`: 추가 설치가 필요한 `tabm==0.0.3`과
  `rtdl-num-embeddings==0.0.12`
- `preprocessing_state.json`: 공식 학습 데이터에서 고정한 결측치 대체값,
  평균·표준편차, 범주 사전
- `numeric_embedding_0.json`: 숫자 피처별 piecewise-linear 경계
- `tabm_member_0_seed_3407.pt`: 학습이 끝난 모델 가중치
- 두 manifest: 모델 구조, 계보와 파일별 SHA-256

PyTorch, pandas, NumPy는 평가 서버 기본 버전을 그대로 사용하므로
`requirements.txt`에 다시 설치하도록 적지 않았다.

## 실행되면 무슨 일이 일어나는가

`script.py`의 실행 흐름은 다음과 같다.

1. 내장된 후보 ID와 모델 파일 SHA-256을 확인한다.
2. 공식 예제의 `data/`에서 입력을 찾고, 없으면 평가 안내에 적힌 `open/`을
   확인한다.
3. `test.csv`와 `sample_submission.csv`의 스키마, `row_id` 누락·중복,
   두 파일의 ID 집합을 확인한다.
4. 저장된 학습 전처리 상태와 모델을 읽는다.
5. 각 행의 성공 확률을 예측한다.
6. 해시로 고정한 최대 8행을 한 행씩 다시 예측해 전체 배치 결과와
   `1e-6` 이내로 같은지 확인한다.
7. sample submission의 순서로 `output/submission.csv`를 원자적으로 쓴다.

입력 경로를 두 개 지원하는 이유는 운영진 공식 추론 예제가 `./data`를
사용하는 반면 평가 안내 본문에는 `open/`이라고 적혀 있기 때문이다. 어느
경로를 선택하더라도 모델 입력이나 예측식은 같고, 다른 평가 행의 값이나
분포는 사용하지 않는다.

## 대회 규칙을 어떻게 지키는가

전처리는 평가 데이터 전체에 맞춰 다시 학습되지 않는다. 평균, 표준편차,
결측치 대체값, 범주 사전과 bin 경계는 모두 공식 학습 데이터에서 미리
계산해 모델과 함께 저장했다.

`hand_matchup`은 같은 행에 있는 `pitcher_hand`와 `batter_hand`만 조합한다.
다음 동작은 추론 코드에 없다.

- 평가 데이터 전체의 평균·빈도·순위·분위수 계산
- 선수, 팀, 월, 경기 단위의 평가 행 집계
- 평가 행 사이의 rolling, lag, 누적 통계
- 평가 데이터로 scaler, encoder 또는 bin 재학습
- 앞선 평가 행의 결과를 다음 행에 전달
- 외부 API 호출이나 외부 데이터 다운로드

따라서 같은 행을 단독으로 넣거나 다른 평가 행과 함께 넣어도 예측값이
같아야 한다. 실제 T4 검사에서 최대 차이는
`2.9802322387695312e-08`이었다.

## 무엇을 실제로 검증했는가

### 기존 T4 전체 규모 증거

Stage D에서 현재와 같은 모델 파일로 다음을 확인했다.

- GPU: Tesla T4 한 장
- 처리 행 수: 245,789행 규모 fixture
- 추론 시간: `11.938824569`초
- 최대 프로세스 RAM: `4,092,870,656`바이트
- 최대 GPU 할당 메모리: `973,573,120`바이트
- 행 독립성 최대 차이: `2.9802322387695312e-08`
- 패키지 설치 시간: `1.7805408`초

평가 서버의 L4는 직접 사용하지 않았기 때문에 L4 실측이라고 기록하지
않는다. 대신 더 작은 VRAM의 T4에서 공식 10분 제한보다 훨씬 짧게 끝난
증거를 현재 모델 해시와 결속했다.

### 현재 최종 코드의 정확 환경 검사

Python.org 공식 소스로 Python 3.11.15를 빌드하고 아래 버전을 사용했다.

- Python 3.11.15
- PyTorch 2.7.1 CPU
- pandas 2.0.3
- NumPy 1.26.4
- TabM 0.0.3
- rtdl-num-embeddings 0.0.12

공식 5행 샘플을 현재 최종 `script.py`의 predictor로 예측한 값은 다음과
같다.

```text
0.42751315
0.43481126
0.45881426
0.49110359
0.48840734
```

Stage D T4의 같은 5행 예측과 모두 절대 오차 `1e-6` 이내로 일치했다.
현재 코드로 행 순서 변경, 단일 행 추론과 상태 불변성 검사도 다시 통과했다.

### 자동 게이트

다음 게이트가 모두 `true`인 경우에만 ZIP을 생성했다.

- temporal validation
- performance
- provenance와 파일 해시
- row independence
- evaluator runtime과 메모리
- pretrained artifact license: 외부 사전학습 모델 없음
- 당일 공식 규정 재검토

최종 runtime SHA-256은
`feae1a931ea374926308a2bf0d9a655df3885b7c83a5e29b0be4e3be530f3863`이다.

## 검증 기록

```text
artifacts/tabm_submission_version_d/
├── submit.zip
├── submission_receipt.json
└── evidence/
    ├── acceptance.json
    ├── gpu_evidence.json
    ├── python_probe.json
    ├── runtime_benchmark.json
    └── full_audit/
```

`submission_receipt.json`은 후보 ID와 최종 ZIP 해시를 기록한다. `evidence/`는
어떤 실험 결과와 환경 검사를 근거로 패키징했는지 확인하기 위한 로컬 자료다.
이 파일들은 ZIP 안에 넣으면 안 된다.

`artifacts/tabm_submission_version_d_superseded_data_only/`는 입력 경로 보강
전에 생성한 구버전이다. 제출 대상으로 사용하지 않는다.

## 이 검증이 보장하지 않는 것

- 숨겨진 평가 데이터 245,789행을 미리 읽거나 분석하지 않았다.
- 숨겨진 정답이나 리더보드 점수를 알 수 없다.
- 로컬 검증 통과가 리더보드 점수나 본선 진출을 보장하지 않는다.
- 실제 제출 서버의 일시적 설치·스케줄링 오류까지 없다고 보장할 수는 없다.

이 제한은 의도적이다. 숨겨진 평가 데이터의 분포를 이용하지 않는 것이
대회 규칙을 지키는 핵심 조건이다.

## 제출할 때 확인할 것

1. DACON 제출 화면에는
   `artifacts/tabm_submission_version_d/submit.zip` 하나만 올린다.
2. 업로드 전 SHA-256이 이 문서의 값과 같은지 확인한다.
3. 제출 횟수는 하루 5회 제한이므로 다른 ZIP과 혼동하지 않는다.
4. 설치 오류가 아니라 `script.py` 실행 이후 오류는 제출 횟수에 반영될 수
   있으므로, 서버 로그와 결과를 보존한다.
5. 리더보드 결과가 나온 뒤 제출 ID, 점수와 서버 로그를 별도로 기록한다.

코드는 제출만 준비하며 웹사이트에 자동 업로드하지 않는다.

## 확인한 공식 자료

- [대회 규칙](https://dacon.io/competitions/official/236743/overview/rules)
- [평가 및 코드 제출 안내](https://dacon.io/competitions/official/236743/overview/evaluation)
- [평가 데이터 독립 예측 원칙](https://dacon.io/competitions/official/236743/talkboard/417123?page=1&dtype=recent)
- [대회 FAQ 및 운영진 답변](https://dacon.io/competitions/official/236743/talkboard/417082?page=1&dtype=recent)
- [운영진 RandomForest 추론 예제](https://dacon.io/competitions/official/236743/codeshare/14146?page=1&dtype=recent)

규정 검토 시각과 정책 해시는
`reports/rules/2026-08-15-final-policy-review.json`에 기록했다.
