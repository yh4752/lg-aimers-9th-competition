# 독립 DL 프런티어 캠페인 개정 설계

## 목적

기존 독립 DL 캠페인의 누출 방지, 전체 규모 학습, 체크포인트 재개와 산출물 계약은
유지한다. 다만 후보 64개를 설정 배열 순서대로 실행하는 방식은 중단하고, 강한
대형 후보를 모델 계열과 입력 표현 사이에 교차 배치한다. 최신 표 foundation model을
별도 연구 트랙으로 추가하고, 대규모 데이터에서 실행 불가능한 현재 TabR 검색은
수정되기 전까지 본 학습 순서에 넣지 않는다.

Codex는 코드와 작은 합성 검증만 수행한다. 공식 데이터 전처리, GPU 학습, OOF와
장시간 재개 실행은 사용자가 Colab Pro에서 수행한다.

## 이번 개정에서 유지하는 것

- 공식 전체 학습 행과 전체 검증 시즌을 사용하는 `2023→2024` 본 탐색
- `2021→2022`, `2022→2023`, `2023→2024` 시간 전이 확인
- fold 학습 구간에서만 fit하는 전처리·집계·범주 사전
- 후보·fold·seed·epoch 체크포인트와 중단 후 재개
- OOM 시 micro-batch, gradient accumulation, AMP와 activation checkpoint 조정
- 단독 Brier뿐 아니라 segment, 오차 다양성과 사전 고정 blend 기여도 평가
- 대용량 산출물은 Drive에 두고 Git에는 코드·설정·작은 판정 근거만 기록

Smoke, OOM, 실행 시간, 비용과 단일 설정 실패는 모델 계열의 성능 판정 근거로
사용하지 않는다.

## 변경하는 것

### 명시적인 실행 순서

후보 ID를 설정 파일의 명시적인 파동에 배치한다. 모델 계열 배열의 중첩 반복문
순서에 실행 우선순위를 맡기지 않는다.

첫 파동은 다음 순서로 실행한다.

1. 이미 완료된 `TabM raw_typed P1·P2`를 기록하고 재실행하지 않는다.
2. `TabM P2`를 `engineered`, `entity_context`, `trackman_augmented`에 실행한다.
3. `TabM P3`를 네 입력 표현에 실행해 용량 증가 방향을 확인한다.
4. `MLP/ResNet P3`, `FT-Transformer P3`를 네 입력 표현에 교차 실행한다.
5. `TabICLv2`를 동일한 `2023→2024` 행 경계로 실행한다.
6. 확장성 수정이 확인된 뒤 `TabR P2·P3`를 네 입력 표현에 실행한다.

이 순서는 작은 proxy로 큰 후보를 제거하는 successive halving이 아니다. P2·P3는
처음부터 전체 데이터와 본 epoch를 사용하는 대형 후보다. P4와 boundary expansion은
각 계열의 P2·P3 결과를 기다리되, 최선이 모델 크기·epoch·retrieval 경계에 있거나
학습 곡선이 마지막 checkpoint까지 개선되면 반드시 이어서 실행한다.

### TabR 확장성 수정

현재 구현은 매 query batch마다 전체 학습 context를 chunk 순회하면서 candidate key를
반복 인코딩한다. 약 120만 학습 행에서는 검색 비용이 후보 학습을 지배하므로, GPU를
늘리는 것만으로 해결되지 않는다.

TabR 본 실험 전에 다음을 만족해야 한다.

- 학습 context의 retrieval key를 batch 외부에서 생성하고 재사용한다.
- key cache는 모델 파라미터 갱신 이후 명시된 주기에만 갱신한다.
- 검증 추론은 고정된 최종 key cache를 재사용한다.
- self-neighbor는 `row_id` 또는 fold-local row index로 제외한다.
- retrieval context는 현재 fold의 학습 행과 정답만 포함한다.
- 전체 context와 query를 작은 합성 fixture로 비교해 chunk 결과가 exact top-k 기준과
  일치함을 확인한다.

FAISS 또는 공식 TabR 검색 구현을 사용할 수 있다. 구현 선택은 정확한 fold 경계와
재현 가능한 top-k를 만족하는 가장 단순한 방식으로 한다. 범용 검색 프레임워크는
만들지 않는다.

### 2026년 프런티어 모델 트랙

첫 구현 대상은 로컬 가중치로 실행할 수 있고 BSD-3-Clause인 `TabICLv2`다. 기존
feature view 전체를 억지로 숫자 행렬로 평탄화하지 않고, 모델이 요구하는 수치형·범주형
입력 계약을 별도 adapter에서 처리한다. 학습/검증 시즌 분리, row ID, target과 출력
확률 스키마는 기존 캠페인과 동일하게 유지한다.

다음 후보는 실행 가능성과 대회 규정을 확인한 뒤 같은 인터페이스에 추가한다.

- `TabPFN-3`: 로컬 가중치 접근, 실제 배정 GPU 메모리와 제출 사용권이 확인될 때
- `TabH2O`: 공식 재현 가능한 코드와 가중치가 확인될 때
- `TabFM`: 현재 라이선스상 연구용 OOF만 허용하고 제출 후보로 표시하지 않는다

외부 API만으로 추론하는 모델은 사용하지 않는다. 정확한 대회 규정 또는 주최자
확인 전에는 foundation model 결과를 `research_only`로 기록하며 제출 패키징 대상이
될 수 없다. 규정 확인은 성능 탐색 자체를 막지 않고 제출 전환만 막는다.

## 실행 환경

주 실행 환경은 Colab Pro와 Google Drive다. Colab Pro가 특정 GPU를 보장한다고
가정하지 않는다. 노트북은 실제 배정 GPU, VRAM과 CUDA를 출력하고 설정에 기록한다.

- 단일 GPU가 배정되면 해당 GPU를 최대한 사용하는 micro-batch와 gradient
  accumulation으로 실행한다.
- 여러 GPU가 보이더라도 코드가 명시적으로 지원하지 않으면 다중 GPU라고 표시하지
  않는다.
- 다중 GPU 지원은 실제 배정이 반복되고 단일 GPU 병목이 확인된 trainable model에만
  추가한다. foundation model의 자체 device/offload 정책을 덮어쓰지 않는다.
- checkpoint와 완료 manifest는 Drive에 직접 기록해 런타임 종료 후에도 남긴다.

Kaggle은 보조 실행 환경으로 유지하지만, Save & Run All 중단 시 작업 디렉터리
보존을 전제로 재개하지 않는다.

## 평가와 다음 파동

모든 후보는 동일한 행 경계를 사용하고 다음 산출물을 남긴다.

- `row_id`, fold, season, target, probability를 포함한 OOF CSV
- 모델·feature view·seed·fold·best epoch·하드웨어·입력 및 코드 해시 manifest
- 전체·fold·필수 segment Brier와 예측 평균
- R9, XGBoost v3, CatBoost 및 다른 DL 후보와 정렬 가능한 예측 해시

P3 또는 P4가 P2보다 나빠도 그 사실만으로 계열을 닫지 않는다. 다음 중 하나를
만족하면 3-fold·multi-seed 심화 후보로 유지한다.

- 동일 프로토콜의 강한 기준선에 근접하거나 개선
- 강한 기준선과 사전 고정 blend grid에서 개선
- 특정 fold 또는 segment에서 반복되는 개선
- 낮은 오차 상관으로 ensemble에 기여
- 모델 계열 또는 입력 표현의 Pareto 상위 후보

Public 점수는 확인 신호로만 기록하고 후보 가중치의 사후 최적화 정답으로 사용하지
않는다.

## 코드 변경 범위

기존 구조 안에서 다음만 변경한다.

- `experiments/independent_dl/configs/campaign_v1.json`: 명시적 실행 파동과
  프런티어 후보 계약
- `experiments/independent_dl/contracts.py`, `campaign.py`: 설정 순서 보존과
  연구 전용 후보 상태
- `experiments/independent_dl/models/tabr.py`: 재사용 가능한 fold-local key cache
- `experiments/independent_dl/models/tabicl_v2.py`: TabICLv2 adapter
- `experiments/independent_dl/training.py`: 필요한 adapter 경계와 실제 하드웨어 기록
- `notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb`: Colab Pro·Drive 실행과 재개
- 위 동작에 직접 대응하는 기존 테스트와 소수의 새 fixture 테스트

새 저장소, 웹 대시보드, 자동 제출, 범용 HPO 플랫폼, foundation model 공통
프레임워크와 이번 실행에 필요 없는 자동화는 만들지 않는다.

## 사용자 실행 인계

노트북은 위에서 아래로 실행 가능한 완결된 `.ipynb`로 제공한다. 사용자가 확인해야
할 것은 데이터 경로와 Drive 출력 경로뿐이다. 실행 전 셀은 다음을 출력한다.

- 현재 Git 커밋
- 실제 GPU 이름·개수·VRAM
- Python, CUDA, PyTorch와 선택 모델 패키지 버전
- 새 실행인지 기존 checkpoint 재개인지
- 다음 후보 ID와 모델·feature view·예상 epoch

각 후보 완료 시 Brier, best epoch, 산출물 경로를 즉시 출력한다. 실패 시 다음 후보로
넘어가기 전에 원래 traceback과 실패 후보 ID를 남긴다. 사용자는 성공 시
`campaign_manifest.json`, `candidate_results.jsonl`, `campaign_summary.json`과 상위
후보 지표를 전달하고, 실패 시 첫 원본 traceback을 전달한다.

## 완료 기준

1. 기존 P1·P2 결과를 덮어쓰지 않고 명시적 파동 순서로 재개된다.
2. TabM의 다른 입력 표현과 P3, MLP/ResNet P3, FT-Transformer P3가 큰 본 후보로
   등록된다.
3. TabR는 key cache exactness와 fold 격리 fixture를 통과한 뒤에만 본 실행된다.
4. TabICLv2 후보가 기존 OOF 스키마로 결과를 남기며 규정 미확인 시
   `research_only`로 기록된다.
5. 실제 GPU·VRAM과 단일/다중 GPU 사용 여부가 manifest에 기록된다.
6. 중단 후 Colab Pro에서 완료 후보를 건너뛰고 마지막 checkpoint부터 재개된다.
7. Codex는 공식 데이터·GPU 본 학습을 실행하지 않고 사용자가 노트북으로 수행한다.
8. 제출 파일이나 자동 제출 코드는 추가하지 않는다.
