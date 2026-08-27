# Tree Expert E2 Expanded Handoff 입력 복구 설계

## 목적

Kaggle이 `tree_expert_e2_handoff.zip`과 그 안의 `tree_expert_e2_resume.zip`을 모두 폴더로 자동 해제한 경우에도 이를 하나의 복구 입력으로 인식한다. 학습, 검증, 승인, 제출 패키징 로직은 변경하지 않는다.

## 입력 선택

- `tree_expert_e2_handoff_v1` 디렉터리를 최상위 복구 입력으로 선택한다.
- 그 handoff 디렉터리 아래에 있는 `tree_expert_e2_resume_v1` 디렉터리는 동일 산출물의 구성 요소이므로 별도 후보에서 제외한다.
- 서로 다른 경로에 존재하는 독립 handoff 또는 resume 두 개는 기존처럼 모호한 입력으로 거부한다.

## Resume 복원과 검증

- 확장된 handoff 안에 원본 `tree_expert_e2_resume.zip` 파일이 있으면 그 파일을 사용한다.
- ZIP 대신 `tree_expert_e2_resume/` 디렉터리가 있으면 기존 산출물과 동일한 파일 순서, 타임스탬프, 권한, 압축 설정으로 결정적 ZIP을 복원한다.
- 복원된 ZIP은 기존에 고정한 전체 SHA-256과 artifact manifest 검증을 그대로 통과해야 한다. 해시가 다르면 복구를 중단하며 우회하지 않는다.

## 오류 처리

- handoff 아래 resume이 없거나 둘 이상이면 명시적으로 실패한다.
- 심볼릭 링크, 중복 멤버, 경로 이탈은 허용하지 않는다.
- 원본 handoff와 독립 resume 데이터셋을 동시에 연결한 경우에는 계속 `resume_count` 오류를 낸다.

## 검증

- 화면과 같은 완전 확장형 handoff가 한 후보로 탐색되는 실패 재현 테스트를 먼저 추가한다.
- 확장 resume을 다시 만든 ZIP의 SHA-256이 원본 resume과 일치하는지 실제 handoff fixture로 검증한다.
- 독립 resume 두 개의 거부, 일반 ZIP handoff, resume 없는 신규 실행 동작을 회귀 테스트한다.
- Kaggle 한 셀을 다시 생성하고 1MB 제한, 문법, 결정성을 확인한다.
