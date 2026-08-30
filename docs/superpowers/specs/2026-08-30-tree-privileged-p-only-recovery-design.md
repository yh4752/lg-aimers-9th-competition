# Tree Privileged P-only 복구 설계

## 목적

첫 실행에서 정확 투구 단위 TrackMan 매칭 coverage는 전체 최대 `0.003939`,
최신 시즌 최대 `0.002896`이었다. 계약 기준 `0.30`과 `0.20`에 크게 못 미치므로
증류 후보 `D15`, `D35`, `PD15`, `PD35`는 종료한다. 성능이 측정되지 않은 계층
프로필 후보 `P`만 E2 기준선과 다시 비교한다.

## 실행 범위

- 시간 fold: `2021→2022`, `2022→2023`, `2023→2024`
- 구조 선별 seed: `3407`
- 기준선: 기존 Tree Expert E2 OOF
- 신규 후보: 계층 프로필과 E2 anchor residual을 결합한 `P`
- 실행 환경: Kaggle Tesla T4 두 장
- 전체 데이터 학습과 Save Version 실행은 사용자가 수행한다.

TrackMan 교사 매칭과 teacher OOF는 만들지 않는다. `P`의 세 fold가 기준선을
통과한 경우에만 기존 계약의 추가 seed `42`, `2026`, full-fit 및 독립성 감사로
진행한다.

## CUDA 복구

첫 실행의 `CUDA error 3: initialization error`는 부모 프로세스가 GPU를 확인한 뒤
기본 `fork`로 CatBoost 작업을 시작한 구조에서 발생했다. 후보 worker는 Python
`spawn` multiprocessing context로 시작한다. 각 worker에는 GPU `0` 또는 `1` 하나만
명시적으로 할당한다.

테스트는 executor가 `spawn` context를 받는지 확인하고, 두 GPU에 한 작업씩 배정되는
기존 동작을 함께 보존한다.

## 상태와 산출물

이전 실패 handoff는 감사 근거로만 보관한다. 코드와 계약이 바뀌므로 새 실행의
재개 입력으로 사용하지 않는다.

새 캠페인은 다음 상태만 허용한다.

- `paused`: 시간 부족으로 안전 중단, 재개 handoff만 생성
- `rejected`: P가 검증 기준 미달, review와 handoff 생성
- `accepted`: 모든 OOF gate와 추론 독립성 감사 통과, delivery 생성
- `failed`: 실행 오류, 제출물 생성 금지

Kaggle 캠페인은 DACON 제출 ZIP을 만들지 않는다. `accepted` delivery가 로컬에서
다시 검증된 뒤에만 별도 제출 패키징 단계로 넘어간다.

## 성공 조건

1. P-only 실행에서 teacher cache와 증류 job이 생성되지 않는다.
2. 두 CatBoost worker가 `spawn`으로 시작되고 GPU 0·1에 분리된다.
3. 세 screen fold 중 하나라도 실패하면 후보 승인과 delivery가 차단된다.
4. 기존 시간 OOF, 행 정렬, 확률 범위 및 평가 행 독립성 gate는 유지된다.
5. 현재 contract hash가 반영된 새 입력 ZIP과 한 셀 Kaggle 코드가 생성된다.
6. 전체 학습 없이 수행하는 자동 검사와 런타임 archive import 검사가 통과한다.
