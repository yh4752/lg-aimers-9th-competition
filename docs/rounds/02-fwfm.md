# FwFM 단독·제한 blend와 계열 종료

## 가설

CatBoost와 다른 field-weighted interaction 모델이 R9이 놓친 범주 상호작용을
보완할 수 있다.

## 검증 프로토콜

R9과 같은 746,504행에서 standalone OOF를 만들고 전체·fold·game type을
감사했다. 이어서 game_type=F에만 적용하는 blend grid와 고정 F80 후보를 확인했다.

## 결과

Standalone Brier는 `0.2501797846421676`으로 R9보다 나빴고 Public은
`88.5362742196`이었다. F 제한 blend는 전체 평균을 낮췄지만 최악 fold delta가
gate를 넘었다. F80도 2022·2024 F fold를 악화시켰다. 마지막 읽기 전용 감사에서
`stable_signal_count=0`이었다.

## 판정

Standalone과 두 제한 blend는 `rejected`; 증거를 갖춘 종료 감사에 따라 FwFM
계열은 닫혔다.

## 배운 점

특정 segment의 큰 평균 개선만 보고 높은 가중치를 선택하면 연도 전이에서 불안정할
수 있다. 전체와 segment 개선은 fold 안정성과 함께 판단해야 한다.

## 다음 결정

FwFM을 다시 패키징하지 않는다. 독립적인 다른 모델 계열은 계속 탐색할 수 있다.

## 근거

- [Standalone rejection](../../reports/rejections/fwfm_standalone_rejection.json)
- [F blend rejection](../../reports/rejections/r9_fwfm_game_type_f_blend_rejection.json)
- [F80 rejection](../../reports/rejections/r9_fwfm_game_type_f_blend_w080_rejection.json)
- [Exit diagnostic](../../reports/diagnostics/fwfm_failure_boundary_exit_audit.json)
