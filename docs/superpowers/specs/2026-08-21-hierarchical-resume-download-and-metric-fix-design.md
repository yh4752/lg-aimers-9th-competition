# Hierarchical TabM 복구 다운로드·지표 오류 수정 설계

## 목표

학습 체크포인트는 매 epoch 안전하게 로컬 ZIP으로 보존하되, 브라우저 다운로드는 첫 유효 체크포인트·20분 주기·fold 완료에만 요청한다. 전체 실행이 끝나면 review/resume/candidate 산출물만 한 번씩 다운로드한다. 기존 코드 해시 `cdd9f63f48c79f6b715c58a2dc455fb177ae12e40335a4b3a64d5f9599273c60`으로 생성된, 두 OOF fold 완료 후 의사결정 전에 멈춘 복구본은 재학습 없이 이어서 평가할 수 있어야 한다.

## 원인과 수정 경계

`run_supervised_campaign`은 새 active checkpoint를 승격할 때마다 검증 ZIP을 만든 뒤 곧바로 사용자 callback을 호출한다. 이 때문에 epoch 수만큼 같은 이름의 ZIP이 내려간다. 저장과 다운로드를 분리하고, 첫 active snapshot 이후에는 `download_interval_seconds`가 지난 경우에만 callback을 호출한다. 안정 상태에서는 실행 중 fold가 새로 완료된 경우만 callback을 호출하고, terminal state는 셀의 최종 다운로드 경로에 맡긴다.

Stage C anchor 예측은 `row_id`, target, probability와 제한된 식별 열만 가진다. 반면 H1 OOF 예측에는 사전 등록된 segment 열이 모두 있다. `align_anchor_and_h1`은 확률만 정렬하고 H1의 segment 열을 반환값으로 옮기지 않아 `_count_state`가 없다고 실패했다. row_id와 target이 정확히 일치한 뒤 H1의 segment와 month를 H1 쪽에서 정렬해 붙인다. anchor에 동명 열이 있으면 값이 H1과 정확히 같은지도 확인한다.

## 기존 복구본 호환

새 코드는 일반적인 과거 복구본을 허용하지 않는다. 위의 정확한 기존 코드 SHA, 현재 계약·데이터·환경·Stage C·anchor 해시, 완료된 두 OOF job, active/final/decision 부재 조건을 모두 만족하는 복구본만 새 코드 identity로 승격한다. 기존 학습 산출물은 바꾸지 않고, 승격 기록에 원본 ZIP SHA와 이전/새 코드 SHA를 남긴다. 변조되거나 다른 단계의 복구본은 계속 거부한다.

## 검증

- 24개의 연속 checkpoint를 관찰해도 첫 checkpoint와 20분 경계에서만 다운로드되는 회귀 테스트
- fold 완료 시 1회, terminal state에서 supervisor callback 0회인 테스트
- anchor에 segment 열이 없어도 H1의 row-aligned segment로 paired metrics가 계산되는 테스트
- row/target/동명 segment 불일치는 거부하는 테스트
- 정확한 legacy resume만 승격되고 SHA·상태·멤버 변조는 거부되는 테스트
- 생성 Colab 셀 결정성, embedded code identity, 기존 hierarchical 전체 회귀 테스트
