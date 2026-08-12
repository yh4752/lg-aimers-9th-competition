# Kaggle 독립 DL 후보 실행·결과 전달 설계

## 목적

GitHub 연결 없이 Kaggle의 비공개 공식 Dataset과 임베디드 실행 코드만 사용해 독립
DL 후보를 하나씩 학습한다. 각 실행 후 사람 검토에 필요한 결과를 후보별 handoff
ZIP으로 내려받아 대화에 첨부한다. 선택된 후보의 전체 학습과 제출 ZIP 생성은 OOF
분석과 acceptance 이후 별도 단계로 진행한다.

## 실행 구조

노트북은 다음 셀로 나눈다.

1. 공통 준비: `/kaggle/input`에서 네 공식 CSV가 같은 폴더에 있는 유일한 Dataset을
   찾고, 임베디드 코드와 고정 패키지를 `/kaggle/working`에 준비한다. GitHub,
   token, clone과 네트워크 코드 다운로드는 사용하지 않는다.
2. 상태 확인: 기존 `campaign_manifest.json`이 있으면 계열별 완료·실패·남은 후보와
   다음 후보를 읽기 전용으로 표시한다.
3. 계열별 실행: TabM, MLP/ResNet, FT-Transformer, TabR, TabICLv2 셀을 나누고
   각 셀은 해당 계열의 미완료 후보 하나만 실행한다. checkpoint와 캠페인 산출물은
   `/kaggle/working/independent_dl_campaign_v1`에 유지한다.
4. 결과 전달: 방금 완료된 후보 ID를 명시적으로 선택해 후보별 handoff ZIP 하나를
   만든다.

실행 단위만 후보 하나로 제한한다. P1~P4, 네 입력 표현, 최대 400 epoch, boundary
expansion, 다중 fold·seed confirmation과 TabICLv2 연구 후보는 기존 캠페인 그대로
보존한다.

## 공식 데이터와 재개

입력 폴더는 `train.csv`, `test.csv`, `sample_submission.csv`,
`trackman_history.csv` 네 파일이 모두 있는 폴더만 후보로 인정한다. 없거나 두 곳
이상이면 학습 전에 중단하고 발견 경로를 출력한다.

Kaggle 세션 내부 재실행은 `/kaggle/working/independent_dl_campaign_v1`을 그대로
사용한다. 새 세션에서 이어가려면 이전 Notebook Output 또는 그 출력으로 만든 비공개
Dataset을 Input에 연결한다. 원본과 재개 후보가 여러 개면 자동 선택하지 않는다.
완료 후보는 해시가 일치할 때 건너뛰고, 중단 후보는 epoch checkpoint부터 재개한다.

## 전달 ZIP

`/kaggle/working/codex_handoffs/<candidate_id>_handoff.zip`에 다음만 넣는다.

- 캠페인 ID·protocol·코드 번들 SHA-256·패키지 버전·GPU/VRAM
- 해당 후보의 전체 설정과 manifest 상태
- `metrics.json`과 그 SHA-256
- `predictions.csv` OOF와 그 SHA-256
- 후보 실행 로그 또는 manifest에 기록된 전체 실패 이유
- ZIP 내부 파일 목록, 크기와 SHA-256을 담은 `handoff_manifest.json`

checkpoint, feature cache, 원본 데이터, test 예측과 제출 파일은 handoff ZIP에 넣지
않는다. ZIP 생성 전에 manifest에 기록된 metrics·predictions 해시와 현재 파일을 다시
검사한다. 완료되지 않았거나 ID가 없거나 해시가 다르면 ZIP을 만들지 않는다. 기존
동일 경로 ZIP은 덮어쓰지 않는다.

후보 하나의 OOF를 포함하므로 ZIP은 대체로 약 5~15MB로 예상하지만 문자열 길이와
압축률에 따라 달라질 수 있다. 실제 크기와 SHA-256을 마지막 셀에서 출력한다.

## 분석 이후 흐름

사용자는 handoff ZIP을 이 대화에 첨부한다. Codex는 단독 Brier, season·game_type
segment, calibration, 다른 후보와의 행 정렬·오차 상관 및 앙상블 이득을 분석한다.
그 결과로 전체 학습 후보, seed와 앙상블 가중치를 선택한다.

선택 전에는 전체 train 재학습, test 추론 또는 제출 ZIP 코드를 제공하지 않는다.
선택 후 별도 명세에서 전체 train 재학습, 행 독립 추론, acceptance evidence와 현재
해시 검사를 통과한 Kaggle 제출 패키징 셀을 추가한다. 실제 제출은 사용자가 수동으로
수행한다.

## 설명과 검증

각 실행 셀은 목적, 입력, 후보 하나 실행, 예상 시간, checkpoint 재개, 정상 완료
문구, 오류 전달 방법과 비제출 조건을 한국어로 설명한다.

구현 검증은 다음 작은 검사로 제한한다.

- GitHub·token·clone이 없는 임베디드 Kaggle 노트북 계약
- 공식 데이터 유일성 및 이전 checkpoint 입력의 모호성 거부
- 계열별 `--max-candidates 1` 명령
- 완료·ID·경로·해시 검사 전에는 ZIP을 만들지 않는 테스트
- ZIP 허용 파일 목록과 재생성 거부 테스트
- 노트북 JSON, import, 작은 fixture 테스트

Codex는 공식 전체 데이터 전처리, GPU 학습과 대용량 ZIP 생성을 실행하지 않는다.
사용자가 Kaggle에서 실행한다. 노트북 구현 후에도 요청 전에는 push하지 않는다.
