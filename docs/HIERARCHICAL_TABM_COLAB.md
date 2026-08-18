# 계층적 문맥 TabM Colab 실행 안내

이 실험은 기존 TabM에 투수·타자·볼카운트·주자·아웃 상황의 과거 성공률을 붙였을
때 시간 전이 검증 점수가 좋아지는지 확인한다. 평가 데이터는 읽지 않으며 제출 ZIP도
만들지 않는다. 최종 후보 파일이 생기더라도 검토용 산출물이므로 바로 제출하면 안 된다.

## 실행 순서

1. Colab에서 새 노트를 만들고 런타임 유형을 **T4 GPU**로 바꾼다.
2. [COLAB_HIERARCHICAL_TABM_CELL.py](../experiments/hierarchical_tabm/COLAB_HIERARCHICAL_TABM_CELL.py)의 내용을 전부 복사해 한 셀에 붙여 넣는다.
3. 셀을 실행하면 업로드 창이 한 번 열린다. 아래 두 파일을 **같은 창에서 동시에** 선택한다.
   - `catboost_tabm_blend_input.zip`
   - `tabm_colab_stage_C_delivery.zip`
4. 이전 실행을 이어갈 때만 가장 최근의 `hierarchical_tabm_resume.zip` 하나를 같이 고른다. 처음 실행이라면 두 파일만 올린다.
5. 실행이 끝나면 다음 파일을 내려받아 Codex에 전달한다.
   - `hierarchical_tabm_review.zip`
   - `hierarchical_tabm_resume.zip`
   - `hierarchical_tabm_candidate_delivery.zip` — 실제로 생성된 경우에만 전달

파일 이름을 바꿀 필요는 없다. 코드는 이름이 아니라 ZIP 내부 구성을 보고 세 입력을
구분한다. Drive 연결, GitHub clone, 평가 데이터 업로드도 필요 없다.

## 시간과 재개

T4 한 장에서 보통 1~3시간을 예상한다. Colab 상태와 실제 epoch 시간에 따라 차이가
날 수 있다. 한 번의 실행은 최대 3시간이며, 남은 시간이 15분보다 적으면 새 GPU job을
시작하지 않는다.

완료된 fold와 학습 중인 최신 checkpoint는 검증된 resume ZIP으로 묶인다. 첫
checkpoint가 생기면 곧바로 한 번 내려받고, 이후에는 약 20분마다 다시 내려받는다.
fold가 끝났을 때도 새 resume이 내려온다. 다운로드가 여러 번 뜨는 것은 정상이다.
재실행할 때는 가장 번호가 크거나 가장 늦게 받은 `hierarchical_tabm_resume.zip` 하나만
고르면 된다. 이전 입력 ZIP은 그대로 다시 올린다.

브라우저가 여러 다운로드를 막으면 다중 다운로드를 허용해야 한다. 런타임이 끊기기
전에 받은 최신 resume은 로컬에 남으므로 처음부터 다시 학습할 필요가 없다.

## 로그 읽는 법

정상 실행에서는 아래 문구가 순서에 맞춰 나타난다.

- `HIER_INPUTS_VERIFIED`: 두 필수 ZIP과 선택한 resume의 내부 구성·해시 검증 완료
- `HIER_CONTEXT_SELECTED k=<K>`: 2021→2022 개발 구간에서 문맥 평활값 선택 완료
- `HIER_JOB_START fold=<fold>`: 한 개 fold 학습 시작
- `HIER_TRAINING_PROGRESS`: 학습 batch 진행 중
- `HIER_JOB_END fold=<fold>`: 해당 fold 결과와 worker 산출물 확정
- `HIER_DECISION candidate=<H1|H2|H3> status=<status>`: 후보별 gate 판정 완료
- `HIER_FULL_TRAIN_START`: 독립 검증을 통과한 후보가 있어 전체 학습 시작
- `HIER_DELIVERY_READY`: 검토 가능한 후보 파일 생성 완료

오류가 나면 `HIER_ERROR stage=... type=... message=...` 한 줄이 출력된다. 이 줄 전체와
가장 최근에 내려받은 resume ZIP을 함께 전달하면 된다. traceback도 남아 있으면 같이
보내는 편이 원인을 찾기 쉽다.

## H1, H2, H3가 뜻하는 것

세 후보는 서로 다른 모델 세 개가 아니다. 같은 TabM의 OOF 확률에서 보정을 어디까지
허용할지 나눈 것이다.

- **H1**: 계층적 문맥 피처를 붙인 TabM 원확률이다.
- **H2**: H1 확률에 전역 절편과 기울기만 다시 맞춘다.
- **H3**: H2 보정에 사전 지정한 경기·카운트·좌우 조합 등의 작은 효과를 더한다.

H2와 H3의 보정값은 2023 OOF의 1~7월에서 후보를 맞추고 8월 이후로 고른다. 2024
결과나 Public 점수를 보고 보정 강도를 바꾸지 않는다.

## 어떤 경우에 후보로 남는가

판정은 Brier가 낮아지는지만 보지 않는다. 두 시간 fold와 데이터가 충분한 segment가
같이 버텨야 한다.

H1은 다음 조건을 모두 만족하면 `final_candidate`가 된다.

- 두 fold를 행 수로 가중한 anchor 대비 Brier 개선이 `0.00010` 이상
- 최신 2023→2024 fold 개선이 `0.00005` 이상
- 오래된 2022→2023 fold의 악화가 `0.00015` 이하
- 5,000행 이상 segment 가운데 최악의 악화가 `0.00075` 이하

평균과 최신 fold가 좋아도 강한 기준을 다 통과하지 못하면 제한적으로
`public_diagnostic_only`가 될 수 있다. 이 경우 오래된 fold 악화는 `0.00025` 이하여야
하며 최종 제출 후보로 취급하지 않는다.

H2는 최신 fold에서 H1보다 `0.00003` 이상, anchor보다 `0.00008` 이상 좋아야 한다.
동시에 H1 대비 최악 segment 악화가 `0.00020` 이하여야 한다. H3도 같은 조건을
통과해야 하며, 최신 fold에서 H2보다 추가로 `0.00002` 이상 좋아야 한다.

어느 후보도 자기 조건을 통과하지 못하면 전체 학습과 candidate delivery 생성을
건너뛴다. 이 결과도 실패가 아니라 “이번 가설을 채택하지 않는다”는 유효한 결론이다.

## 전달 파일의 용도

- `hierarchical_tabm_review.zip`: OOF, calibration, fold·segment 지표와 판정 검토용
- `hierarchical_tabm_resume.zip`: 완료 fold와 최신 checkpoint를 이어가기 위한 파일
- `hierarchical_tabm_candidate_delivery.zip`: gate를 통과한 후보의 모델·피처 상태·보정 상태

`hierarchical_tabm_candidate_delivery.zip`은 제출 파일이 아니다. Codex가 해시, 독립성
보고서, 후보 역할과 acceptance 상태를 다시 확인한 뒤 별도의 제출 패키징 단계를
거쳐야 한다.
