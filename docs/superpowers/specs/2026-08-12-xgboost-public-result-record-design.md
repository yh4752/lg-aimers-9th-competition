# XGBoost Public 결과 기록 설계

## 목적

Run `a0b99dd0e7eb41fba2b5ff729b11aeb1`의 검증된 XGBoost 앙상블과 DACON
Public `820.9583317093`을 기존 기록 구조에 최소한으로 추가한다.

## 변경 범위

- 작은 요약 JSON `reports/acceptances/xgboost_aggressive_capacity_public_result.json`
  하나를 추가한다.
- `README.md`, `reports/EXPERIMENT_LEDGER.md`, 기존
  `docs/rounds/05-xgboost.md`, `docs/ROADMAP.md`를 짧게 갱신한다.
- `AGENTS.md`에 최소 수정·중복 금지 원칙을 추가한다.
- 기존 repository contract 테스트에 핵심 값과 링크 검사만 추가한다.

## 원본 경계

실행 영수증, 후보 CSV, 비교 보고서와 제출 ZIP은 Google Drive를 원본으로
유지한다. Git에는 스크린샷, CSV, ZIP이나 모델을 복사하지 않는다. 요약 JSON은
아래 값을 Drive 근거에 연결한다.

- Public: `820.9583317093`
- 제출 파일: `submit_xgboost_v3.zip`
- 제출 시각: `2026-08-11 23:52:02`
- ZIP SHA-256: `f81b5df770733898535d9a1d4a019b7c339aa72d7cbce0cb67a3e49ccb41f43a`
- 로컬 Brier: `0.2479270213638507`
- 로컬 환산 점수: `752.5433411090132`
- 구성: depthwise d6 두 seed와 lossguide 63 두 seed의 평균,
  `scale=1.05`, `mean_shift=linear_extrapolated`

## 해석

이 결과는 큰 구조가 일관되게 좋다는 증거가 아니다. Depth 8과 127·255 leaves는
악화됐고, 중간 크기의 두 구조, seed 평균과 후처리 조합이 개선을 만들었다.
CatBoost blend는 다음 검토 아이디어로만 기록한다. 같은 행과 시간 프로토콜의
정렬된 OOF를 확인하고 가중치와 gate를 사전 고정하기 전에는 실행 후보나 제출
승인으로 표현하지 않는다.

## 탐색 기본값

대회 규정, 시간 누출, 데이터 손상 위험이 없다면 모델 깊이, leaves, seed 수,
feature 수, 실행 시간과 GPU 사용량을 사전에 제한하지 않는다. 비용과 시간은 안내와
실행 순서에만 사용한다. Fold·seed 편차, calibration과 과적합 가능성은 탐색을
막지 않고 결과에 기록해 최종 선택 시 판단한다.

사전에 차단하는 항목은 대회 규정 위반, 시간 누출·평가 행 의존성, 원본 및 기존
산출물 덮어쓰기, 무효 예측, 비밀 노출과 승인되지 않은 push·패키징·제출로
제한한다. 이후 비교·앙상블·재시작에 필요한 설정, 점수, 정렬된 OOF와 선택
checkpoint만 보존한다.

## 완료 기준

- README와 장부에서 확인된 XGBoost Public 최고를 찾을 수 있다.
- 기존 XGBoost 라운드 문서에 구성, 실패한 확대 방향과 다음 판단이 기록된다.
- 모든 Drive 링크와 핵심 수치가 테스트로 고정된다.
- Drive 원본과 기존 저장소는 변경되지 않는다.
