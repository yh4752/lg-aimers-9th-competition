# TabR T4 Retrieval and Observability Design

## 목적

현재 TabR은 약 122만 학습 행 전체를 매 배치마다 PyTorch 완전탐색하여 첫 배치도 현실적인 시간 안에 끝내지 못한다. 모델 용량과 retrieval 수를 축소하지 않고 검색 경로를 고쳐, Tesla T4에서 P1 후보를 4~6시간 안에 실행 가능하게 만든다. 사용자는 Colab 화면과 Drive 로그에서 실제 진행 여부를 확인할 수 있어야 한다.

## 고정 조건

- P1의 `retrieval=32`, `width=256`, `blocks=3`과 학습 데이터 범위는 유지한다.
- 검색 후보는 현재 시간 fold의 학습 행만 사용한다.
- validation/test 행은 서로를 검색 후보로 사용하지 않는다.
- 출력 행을 단독·역순·셔플·다른 배치 크기로 추론해도 같은 확률이어야 한다.
- Codex는 코드와 작은 합성 테스트만 실행한다. 전체 데이터 GPU 학습은 사용자가 실행한다.
- 이번 변경은 TabR 검색과 관측성에 한정하며 다른 모델 구조를 바꾸지 않는다.

## 검색 구조

공식 TabR 구현이 사용하는 FAISS 검색 방식을 따른다. 학습 임베딩을 배치로 계산해 학습 행 전용 인덱스를 만들고, 각 query에서 필요한 이웃 수보다 넓은 후보를 검색한 뒤 현재 PyTorch 거리식으로 정확 재정렬한다. 학습 query 자신의 행 ID는 최종 이웃에서 제외한다.

초기 epoch에는 모델 변화에 맞춰 인덱스를 갱신한다. 이후 검색 이웃을 고정해 반복적인 전체 검색 비용을 제거한다. 인덱스와 고정 이웃은 현재 후보 체크포인트에 귀속되며 다른 fold·후보·모델에서 재사용하지 않는다. 재개 시 체크포인트 identity가 일치하지 않으면 다시 만든다.

FAISS GPU를 사용할 수 없으면 느린 전체 완전탐색으로 자동 후퇴하지 않는다. 명확한 오류로 중단해 GPU 시간 낭비를 막는다.

참조:

- 공식 구현: `yandex-research/tabular-dl-tabr`, commit `17baa9082506f8e7a0f8d11bb1e08212926a1507`
- 논문: <https://openreview.net/pdf?id=rhgIgTSSxW>

## 관측 가능한 실행 로그

같은 구조의 한 줄 JSON heartbeat를 Colab 표준 출력과 후보 폴더의 `progress.jsonl`에 동시에 기록한다. 각 줄은 즉시 flush한다.

필수 이벤트:

- `TABR_CONTEXT_ENCODING_PROGRESS`: 인코딩한 학습 행 수, 전체 행 수, rows/sec, ETA
- `TABR_INDEX_READY`: 인덱스 종류, 벡터 수, 차원, 생성 시간, GPU 메모리
- `TABR_SEARCH_PROGRESS`: 현재 epoch/batch, 검색한 query 수, queries/sec, ETA
- `TRAINING_PROGRESS`: epoch/batch, loss, 처리 행 수, rows/sec, GPU allocated/reserved/peak memory
- `EPOCH_CHECKPOINTED`: validation Brier, best Brier, epoch 시간, 체크포인트 경로
- `TABR_CONTEXTS_FROZEN`: 고정 시점과 고정 이웃 shape/hash
- `TRAINING_TIME_BUDGET_REACHED` 또는 원본 오류 traceback

긴 연산은 행 chunk마다 진행량을 갱신하며, 화면 로그는 최대 60초 간격을 넘기지 않는다. 별도 생존 메시지만 출력하지 않고 완료 행 수와 처리량을 함께 출력한다. 처리량이 0이거나 같은 완료 행 수가 5분 이상 유지되면 `TABR_STALL_WARNING`을 출력한다. 첫 `TABR_INDEX_READY` 또는 실질 처리 로그가 10분 안에 나오지 않으면 안전하게 실패시킨다.

`progress.jsonl`은 append-only이며 재개 시 기존 기록을 보존한다. 각 이벤트에는 UTC 시각, candidate ID, fold, process ID와 checkpoint epoch를 포함해 이전 실행과 현재 실행을 구분한다.

## 재개와 실패 처리

- epoch가 완료된 뒤 기존 체크포인트를 원자적으로 저장한다.
- Colab 중단 후에는 마지막 완료 epoch부터 재개한다.
- RNG, 모델, optimizer, scheduler와 검색 identity를 함께 검증한다.
- 인덱스 생성 실패, FAISS 미설치, GPU 메모리 부족, 정체 감지는 후보 실패로 기록하되 다른 후보의 결과를 훼손하지 않는다.
- 실패한 정확한 후보 ID만 명시적 재시도 옵션으로 되돌릴 수 있다.

## 검증 기준

작은 합성 데이터로 다음을 자동 검증한다.

- FAISS 후보를 정확 거리로 재정렬한 결과가 작은 전체 완전탐색 결과와 같다.
- self-neighbor가 제외되고 학습 fold 밖 행은 인덱스에 들어가지 않는다.
- singleton/reverse/shuffle/rebatch 예측이 일치한다.
- heartbeat가 진행량·처리량·GPU 메모리·ETA를 포함하고 `progress.jsonl`에도 동일하게 기록된다.
- 정체와 10분 초기 진행 제한이 fail-closed로 작동한다.
- 체크포인트 재개 시 잘못된 검색 identity는 거부하고 올바른 identity는 재사용한다.

실제 성공 판정은 사용자가 T4에서 실행해 다음을 확인한 뒤 내린다.

- 10분 이내 인덱스 또는 실질 진행 로그 출력
- 로그가 최대 60초 간격으로 계속 갱신
- P1 후보 전체가 4~6시간 안에 완료 또는 시간 예산 경계에서 재개 가능한 체크포인트 생성
- 유한한 2024 Brier와 규칙 준수 evidence 생성

이번 단계는 성능 evidence만 생성하며 제출 CSV나 ZIP을 만들지 않는다.
