# Independent DL T4 실행 로그 안내

이 문서는 Colab에서 TabM, MLP/ResNet, FT-Transformer, TabR 후보가 실제로 진행 중인지 확인하는 기준만 정리한다. 학습과 검증 evidence만 생성하며 제출 CSV나 ZIP은 만들지 않는다.

## 로그 위치

각 후보 폴더의 다음 파일에 화면 출력과 같은 JSON 이벤트가 한 줄씩 계속 추가된다.

```text
<output_root>/candidates/<candidate_id>/progress.jsonl
```

런타임이 끊겨도 기존 줄은 보존된다. `run_id`와 `process_id`로 이전 실행과 재개 실행을 구분할 수 있다.

## 정상 진행 표시

- `CANDIDATE_RUNTIME_READY`: 모델, GPU, 학습·검증 행 수와 배치 설정 준비 완료
- `TRAINING_PROGRESS`: epoch, batch, loss, 완료 행 수, 처리량, ETA와 GPU 메모리
- `VALIDATION_PROGRESS`: 검증 완료 행 수, 처리량, ETA와 GPU 메모리
- `EPOCH_CHECKPOINTED`: 검증 Brier와 마지막 완료 epoch 체크포인트 저장 완료
- `CANDIDATE_COMPLETED`: 후보 학습과 검증 완료
- `TRAINING_TIME_BUDGET_REACHED`: 현재 세션을 안전하게 끝내고 마지막 완료 epoch부터 재개 가능
- `CANDIDATE_FAILED`: 원본 예외 형식, 메시지와 traceback 기록

TabR에는 다음 표시가 추가된다.

- `TABR_CONTEXT_ENCODING_PROGRESS`: fold 학습 행 임베딩 진행
- `TABR_INDEX_READY`: FAISS GPU IVF-Flat 인덱스 준비 완료
- `TABR_SEARCH_PROGRESS`: 검색 또는 학습 행 고정 이웃 생성 진행
- `TABR_CONTEXTS_FROZEN`: epoch 0 뒤 고정 이웃 파일의 shape와 SHA-256 기록

첫 실질 진행은 10분 안에 나타나야 한다. 완료 행 수가 5분 동안 증가하지 않으면 `TRAINING_STALL_WARNING`이 기록된다. 화면의 정상 진행 출력 간격은 최대 60초를 목표로 하며, 한 GPU chunk 자체가 실행 중인 동안에는 가짜 생존 메시지를 출력하지 않는다.

## 안전한 재실행

- Colab이 끊기면 같은 후보 실행 셀을 다시 실행한다.
- 완료된 epoch의 `checkpoint.pt`가 있으면 다음 epoch부터 재개한다.
- TabR의 `frozen_neighbors.npy`는 검색 정책, 모델 설정, fold 학습 행 ID와 SHA-256이 모두 일치할 때만 재사용한다.
- 일치하지 않으면 완료된 모델 epoch는 유지하고 검색 상태만 다시 만든다.
- FAISS GPU가 없으면 전체 PyTorch 완전탐색으로 후퇴하지 않고 즉시 실패한다.

FT-Transformer에서 RNG 체크포인트 오류로 실패했던 정확한 후보는 다음 ID로만 명시적으로 재시도한다.

```text
ft_transformer__raw_typed__p3__s42
```

다른 실패 후보를 임의로 `pending`으로 바꾸거나 결과 파일을 덮어쓰지 않는다.
