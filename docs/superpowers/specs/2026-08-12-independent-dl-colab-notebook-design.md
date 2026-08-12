# Independent DL Colab Notebook Design

## 목적

사용자가 GitHub에서 Colab으로 바로 열어 독립 DL 캠페인을 실행할 수 있는 단일
노트북을 제공한다. 기존 `experiments/independent_dl/COLAB.md`의 검증된 실행 셀을
그대로 사용하며 새로운 실행 경로나 자동화를 만들지 않는다.

## 산출물

- `notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb`
- 노트북 구조를 확인하는 작은 정적 테스트

노트북은 안내 Markdown 셀 하나와 완결된 Python 코드 셀 하나로 구성한다. Python
셀은 기존 `COLAB.md` 코드와 바이트 단위로 동일해야 한다. Colab GPU 메타데이터를
포함하고 출력 및 실행 횟수는 비운 상태로 저장한다.

## 실행 경계

- 사용자가 Colab에서 공식 데이터와 T4 GPU로 실행한다.
- 재실행은 동일한 Drive 결과 경로에서 checkpoint를 이어받는다.
- 노트북은 제출 CSV·ZIP을 만들지 않는다.
- 기존 `COLAB.md`, 학습 코드, 캠페인 계약과 패키지 목록은 변경하지 않는다.
- 노트북 생성기나 추가 CLI는 만들지 않는다.

## 검증

정적 테스트는 다음만 확인한다.

- nbformat 4의 유효한 JSON이다.
- 셀 순서가 Markdown 1개, 코드 1개다.
- 코드 셀이 `COLAB.md`의 유일한 Python 코드 블록과 정확히 같다.
- Colab 이름과 GPU 가속기 메타데이터가 있다.
- 코드 셀의 출력과 실행 횟수가 비어 있다.

공식 데이터, 패키지 설치, GPU 학습은 Codex가 실행하지 않는다.
