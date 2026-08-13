# TabM Champion Kaggle 실행 안내

이 캠페인은 확정 전처리 `dl_standard + hand_matchup`에 TabM만 붙여 넓게 탐색한 뒤,
시간 전이 검증과 평가 서버 모사를 통과한 후보의 **검토 자료만** 만든다. 한 번의
Kaggle Save Version은 A–D 중 정확히 한 단계만 진행한다.

## 실행 전 확인

- Notebook accelerator: `GPU T4 x2`
- 필수 Input: 공식 `lg-aimers-9th-data` 하나
- 추가 Input:
  - A: 없음
  - B: `tabm_search_stage_A_resume_bundle.zip`
  - C: `tabm_search_stage_B_resume_bundle.zip`
  - D: `tabm_search_stage_C_resume_bundle.zip`
- 실행 코드: [KAGGLE_CELL.py](../experiments/tabm_campaign/KAGGLE_CELL.py)의 전체 내용을
  Kaggle 코드 셀 하나에 복사
- 인터넷은 Python 패키지 설치에만 필요하며 GitHub에는 접속하지 않는다.

resume ZIP이 두 개 이상 보이거나 공식 `train.csv`가 둘 이상 발견되면 모호한 입력을
추측하지 않고 중단한다. 이전 단계 review ZIP은 다음 단계 입력이 아니다.

## 단계별 목적과 최대 시간

| Save Version | 목적 | 최대 wall time | 결과 |
|---|---|---:|---|
| A | 24개 proxy 구조·임베딩·손실·scheduler 탐색 | 2시간 | A review + A resume |
| B | 생존 후보의 2024/2023 full temporal 검증 | 3시간 | B review + B resume |
| C | 9개 refinement, seed 안정성, ensemble 증거 | 3시간 | C review + C resume |
| D | 전체 train 고정 epoch 학습과 서버 규모 검증 | 2시간 | final review 하나 |

각 단계는 종료 10분 전부터 새 작업을 시작하지 않고 증거 파일을 마무리한다. 후보별
epoch 체크포인트는 설정·코드·전처리 캐시 해시와 함께 저장된다. 같은 입력으로 다시
실행하면 일치하는 완료 결과만 재사용하며, 다른 해시의 결과는 거부한다. 시간 제한에
도달한 작업은 `inconclusive`로 기록하므로 무한 pending 재시도가 발생하지 않는다.
한 단계가 시간 안에 충분한 증거를 모으지 못하면 다음 단계로 넘어가지 않고 같은
버전의 resume ZIP을 만든다. 그 ZIP을 다음 Save Version에 넣으면 완료 epoch부터 같은
버전을 이어간다. 따라서 정상 예상은 A–D 네 번이지만, 느린 세션은 한 단계를 여러
Save Version으로 나누어도 안전하다.

## 확인할 로그

시작할 때 다음 로그가 순서대로 보여야 한다.

```text
CODE_READY sha256=...
DEPENDENCIES_READY tabm=0.0.3 rtdl_num_embeddings=0.0.12
DATA_FOUND path=/kaggle/input/.../lg-aimers-9th-data/...
RESUME_FOUND path=None 또는 ...resume_bundle.zip
GPU_READY device_count=2 status=...
STAGE_SELECTED version=A|B|C|D wall_deadline_unix=...
```

학습 중에는 GPU별 독립 작업과 epoch 진행이 계속 출력된다.

```text
JOB_START job=... gpu=0|1 deadline_unix=...
WORKER[0:후보ID] TRAINING_PROGRESS candidate=... epoch=... batch=...
WORKER[1:후보ID] EPOCH_CHECKPOINTED candidate=... epoch=... brier=...
JOB_END job=... gpu=0|1 status=completed|inconclusive|failed
```

정상 종료의 마지막 핵심 로그는 다음과 같다.

```text
BUNDLE_SUCCESS version=A|B|C review=... resume=...
BUNDLE_SUCCESS version=D review=...tabm_hand_matchup_final_review_bundle.zip resume=None
```

오류가 나면 아래 한 줄과 그 직전 로그 100줄을 그대로 전달한다.

```text
TABM_CAMPAIGN_ERROR stage=... type=... message=...
```

## 전달할 파일

- A–C가 끝날 때: 해당 단계의 review ZIP과 resume ZIP 둘 다
- D가 끝날 때: `tabm_hand_matchup_final_review_bundle.zip`

review ZIP은 지표, 예측, 자원 사용량, 선택 판정과 규칙 검증 증거다. resume ZIP은
다음 Save Version을 이어가기 위한 상태와 필요한 체크포인트다. D 결과도 검토 전용이며
제출 파일이 아니다. 파일명을 바꾸거나 제출 ZIP으로 사용하지 않는다. 실제 제출물
생성은 후보 승인, 모든 게이트, 현재 아티팩트 해시와 당일 공식 규칙 재확인이 끝난 뒤
별도 작업으로만 진행한다.
