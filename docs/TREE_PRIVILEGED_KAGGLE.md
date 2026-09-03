# TrackMan 교사 + 계층 프로필 실험 실행 안내

이 실험은 기존 E2 제출 모델을 기준점으로 두고, 학습 데이터에서만 확인할 수 있는 현재 투구 TrackMan 정보와 이전 시즌의 계층 통계를 이용해 CatBoost 잔차 모델을 비교합니다. TrackMan 현재 투구 값은 교사 확률을 만드는 데만 쓰이며 최종 모델 입력에는 들어가지 않습니다. 평가 행끼리 평균·빈도·정렬·누적 통계를 계산하지 않으므로 평가 데이터 독립 예측 원칙을 지키도록 설계했습니다.

TrackMan 원본에서 야구 상태 범위를 벗어난 98행과 게임 내부 식별·메타데이터가 충돌하는 5경기는 어느 값을 정답으로 추정하지 않고 매칭 후보에서 제외합니다. 정상 2021 cutoff 기준으로 836,114행과 2,817경기가 남는 것을 실제 공식 데이터에서 확인했습니다. 투수 연결에는 피처 lookup이 아니라 원본 one-to-one ID mapping 계약을 사용하며, 결측치 때문에 pandas가 실수형으로 보관한 accepted ID는 정수성을 확인한 뒤 정수 ID로 복원합니다.

이 코드는 실험용 산출물을 만드는 코드입니다. DACON 제출 ZIP을 만들지 않으며, OOF 승인 기준과 행 독립성 검사를 통과한 경우에만 모델 delivery를 handoff 안에 포함합니다.

## 1. 로컬에서 Kaggle 입력 만들기

저장소 루트에서 아래 명령을 실행합니다.

```bash
artifacts/tabm_submission_python311/bin/python \
  tools/prepare_tree_privileged_input.py \
  --e2-handoff "/path/to/Downloads/tree_expert_e2_handoff (1).zip" \
  --output artifacts/tree_privileged_input.zip
```

성공 로그는 다음 형식입니다.

```text
TREE_PRIV_INPUT_READY path=<절대 경로> sha256=<64자리 해시> size_bytes=<크기>
```

생성된 `tree_privileged_input.zip`을 Kaggle Dataset으로 한 번 등록합니다. ZIP은 Kaggle에서 폴더로 풀려도 실행 코드가 인식합니다.

## 2. Kaggle Notebook 입력 구성

새 실행에는 Dataset 두 개만 추가합니다.

1. 공식 데이터 `lg-aimers-9th-data`
2. 위에서 만든 `tree_privileged_input`

중단된 실행을 이어갈 때만 최신 `tree_privileged_handoff.zip`을 Kaggle Dataset으로 등록해 세 번째 입력으로 추가합니다. handoff 내부의 `*.bundle` 파일은 이름을 바꾸거나 풀지 않습니다. 동일 handoff의 복사본을 두 개 추가하면 입력 중복으로 거부됩니다.

가속기는 반드시 `GPU T4 x2`로 설정합니다. 실행 셀은 [KAGGLE_CELL.py](../experiments/tree_privileged/KAGGLE_CELL.py)의 전체 내용을 복사해 한 셀로 실행합니다. Notebook의 인터넷 설정은 필요한 패키지 설치가 끝날 때까지 켜 둡니다. Kaggle에 정확한 버전이 이미 있으면 추가 다운로드 없이 넘어갑니다.

## 3. 시간과 재실행

- 목적: P/D/PD 구조 선별, 최대 2개 다중 시드 확인, 승인 후보 full fit, 행 독립성 검사
- 예상 시간: 약 7~10.5시간
- 자원: Tesla T4 두 장, CPU/RAM은 Kaggle 기본 할당
- 새 학습 시작 제한: 종료 2시간 전
- 산출물 예약 시간: 마지막 75분

완료된 job은 `predictions.csv`, 지표, best iteration으로 재사용합니다. 재개용 resume에는 불필요한 OOF 모델 바이트를 넣지 않아 handoff 크기를 줄였습니다. 세션이 정상적으로 시간 예산 경계에 도달하면 `paused` handoff를 만들며, 다음 실행에서 그 handoff를 추가하면 완료된 job을 다시 학습하지 않습니다.

Kaggle 프로세스 자체가 강제 종료되어 마지막 handoff 로그까지 나오지 않은 경우에는 `/kaggle/working/tree_privileged/<코드 해시>/bundles/`의 최신 handoff가 Output에 남았는지 먼저 확인합니다. 파일이 없다면 마지막으로 완료 처리된 job 이후부터 다시 실행될 수 있습니다. 코드 해시별로 작업 폴더가 분리되므로 같은 커널에서 새 셀을 실행해도 이전 코드의 중간 산출물을 잘못 재사용하지 않습니다.

## 4. 확인할 로그와 전달할 파일

정상 시작 시 다음 로그가 순서대로 보입니다.

```text
TREE_PRIV_DEPENDENCIES_READY ...
TREE_PRIV_CODE_READY ...
TREE_PRIV_INPUTS_VERIFIED ...
TREE_PRIV_GPU_READY count=2 names=Tesla T4 | Tesla T4
```

학습 중에는 다음 접두사가 표시됩니다.

```text
TREE_PRIV_MATCH_AUDIT ...
TREE_PRIV_TEACHER_JOB_START ...
TREE_PRIV_TEACHER_JOB_END ...
TREE_PRIV_CANDIDATE_JOB_START ...
TREE_PRIV_CANDIDATE_JOB_END ...
TREE_PRIV_DECISION ...
```

정상 종료의 핵심 로그는 다음 두 줄입니다.

```text
TREE_PRIV_HANDOFF_READY path=/kaggle/working/tree_privileged/bundles/tree_privileged_handoff.zip sha256=<64자리 해시>
TREE_PRIV_CAMPAIGN_SUCCESS status=<accepted|rejected|paused|failed> phase=<단계> accepted=<후보 또는 none>
```

오류는 다음 형식입니다.

```text
TREE_PRIV_ERROR stage=<단계> type=<오류 종류> message=<내용>
```

실행 후에는 Kaggle Output에서 아래 파일 하나만 내려받아 전달하면 됩니다.

```text
tree_privileged_handoff.zip
```

`accepted`는 이 실험 내부의 OOF·안전성 기준을 통과했다는 뜻이지, DACON 점수나 제출 적합성이 자동으로 확정됐다는 뜻은 아닙니다. handoff를 검토한 뒤 별도의 요청에서 대회 제출 패키지를 만들어야 합니다.

## 검증 기록

- 생성 셀 SHA-256: `be6a5829b660cc379e92c97440f9849b4341d38d335f6bb89a4d57a1df9fcecf`
- 계약 SHA-256: `5a1bbf9b9672750242de28fd390ce9281d75a01c6ebae72addc2feae81b824c4`
- 새 캠페인 테스트: `36 passed`
- 재사용 LUPI/E2/S4 회귀 테스트: `214 passed`
- 전체 저장소: `2677 passed`, 작업 전부터 존재한 동일한 5개 실패 유지
