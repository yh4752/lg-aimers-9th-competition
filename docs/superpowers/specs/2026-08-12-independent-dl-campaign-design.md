# 독립 DL 대규모 탐색 설계

## 목적

다음 주력 연구를 트리 예측에 의존하지 않는 독립 DL로 진행한다. 과거에 기각된
`TabM residual`은 R9 logit 보정용의 작은 단일 구성일 뿐이며, 독립 TabM이나 다른
DL 구조의 성능 근거로 사용하지 않는다.

목표는 무료 Colab의 단일 T4에서도 모델 계열과 용량을 보수적으로 줄이지 않고,
중단 후 이어서 실행할 수 있는 대규모 탐색 코드를 준비하는 것이다. Codex는 코드와
작은 확인만 담당하고 공식 데이터와 GPU 실행은 사용자가 담당한다.

## 성능 우선 확인

1. smoke는 import, CUDA와 데이터 흐름 확인에만 사용하고 성능 근거로 사용하지 않는다.
2. 첫 본 실험부터 전체 학습 행, 큰 모델, 긴 학습과 넓은 탐색을 포함한다.
3. 최고 설정이 width, depth, epoch, ensemble size 또는 retrieval 경계에 있으면 다음
   탐색에서 그 경계를 확장한다.
4. OOM, 시간, 비용, 런타임 종료나 단일 설정 실패를 모델 계열 종료 근거로 사용하지
   않는다.

## 선택한 접근

계층형 대규모 탐색을 사용한다. 모든 조합을 처음부터 3-fold로 실행하면 중간에 비교
가능한 결과가 너무 늦게 나오고, 한 모델 계열을 먼저 끝내면 탐색이 그 계열에
편향된다. 따라서 네 모델 계열을 전체 규모 `2023→2024`에서 폭넓게 탐색한 뒤,
생존 후보를 3-fold와 multi-seed로 심화한다.

첫 단계는 작은 표본이나 짧은 proxy가 아니다. 공식 전체 학습 행과 전체 2024
holdout을 사용하는 본 실험이다.

## 모델 계열과 용량

다음 네 계열을 동시에 후보군에 포함하되 T4에서는 후보를 순차 실행한다.

| 계열 | 첫 본 탐색 범위 |
|---|---|
| TabM / TabMmini | `k=16~64`, width `256~1024`, block `3~8` |
| MLP / ResNet | width `512~2048`, block `4~16` |
| FT-Transformer | token dim `128~512`, layer `3~12` |
| TabR | retrieval `32~256`, width `256~1024`, block `3~8` |

각 계열은 embedding 크기, numerical embedding, dropout, weight decay, optimizer,
scheduler와 learning rate도 탐색한다. 후보별 최대 epoch는 `100~400` 범위에서
사전 설정하며 early stopping patience를 짧게 두지 않는다.

최고 후보가 상한에 있거나 마지막 checkpoint까지 개선되면 다음 범위까지 확장한다.

- MLP/ResNet width `3072~4096`
- FT-Transformer layer `16~20`
- TabM `k=96~128`
- TabR retrieval `384~512`

이는 강제 최종 상한이 아니다. T4 OOM은 micro-batch, gradient accumulation,
mixed precision과 activation checkpoint로 대응하고 모델 계열을 기각하지 않는다.

## 입력 표현

모델 구조뿐 아니라 입력 표현도 함께 탐색한다.

### `raw_typed`

- 공식 수치형, 범주형과 ID형 열을 구분한다.
- `pitcher_id`, `batter_id`, 팀과 경기 상황 값은 embedding으로 처리한다.
- `row_id`와 `control_success`는 입력에서 제외한다.

### `engineered`

- XGBoost 고득점 실험에서 확인된 카운트, 손잡이 matchup, 점수 상황, 승리 기대값,
  비율과 시즌 변화 특성을 원본 수치형과 함께 사용한다.

### `entity_context`

- 투수, 타자, 팀, inning, base state와 game type embedding을 사용한다.
- 투수×타자, 투수×상대 손잡이, inning×주자 상황의 수작업 교차와 학습형
  interaction을 비교한다.

### `trackman_augmented`

- fold cutoff를 지킨 Trackman 투수 특징, 구종 분포, 최근 상태와 계층 통계를
  다른 표현에 추가한다.
- 조회되지 않는 선수는 missing embedding과 indicator로 보존한다.

각 모델 계열은 최소 `raw_typed`와 `trackman_augmented`를 포함한다. 네 표현의
조합을 사전에 한두 개로 제한하지 않는다.

## 누출 방지 경계

성능 탐색을 막는 일반 제약은 두지 않지만 다음 위험은 반드시 차단한다.

- 전처리, 정규화, 범주 사전과 집계 상태는 fold의 학습 시즌에서만 fit한다.
- 검증 시즌에는 고정 상태로 transform만 수행한다.
- target 기반 선수·상황 통계는 학습 행만 사용한다.
- Trackman은 fold별 허용 cutoff 이후 기록을 읽지 않는다.
- 검증·테스트 배치의 다른 행, 행 순서와 batch 크기에 의존하지 않는다.
- feature cache는 row ID, cutoff, schema와 생성 코드 해시가 모두 같을 때만 재사용한다.

## 탐색 파동

### 파동 A: 전체 규모 광역 탐색

- `2023→2024` 전체 holdout에서 최소 64개의 본 후보를 순차 실행한다.
- 네 모델 계열, 네 입력 표현과 중형·대형·초대형 용량을 처음부터 포함한다.
- epoch와 후보 완료 시 checkpoint, 설정, 지표와 예측 해시를 Drive에 기록한다.
- 한 seed 또는 한 설정으로 계열을 판정하지 않는다.

### 파동 B: 경계 확장과 국소 심화

모델 크기, 학습 길이, retrieval, regularization 또는 optimizer의 탐색 경계가
최선이면 그 축을 확장한다. 특정 입력 표현에서 한 계열만 개선되면 해당 결합도
추가 탐색한다.

### 파동 C: 시간 전이 OOF와 multi-seed

생존 후보를 다음 세 fold에서 다시 평가한다.

- `2021→2022`
- `2022→2023`
- `2023→2024`

구조별 최소 3개 seed를 실행하고 확률 평균과 logit 평균을 모두 비교한다. 이후 DL
내부 ensemble을 먼저 평가하고, 마지막에 동일 행과 동일 fold로 정렬된 CatBoost와
XGBoost OOF가 있을 때만 ML+DL ensemble을 평가한다. 검증 프로토콜이 다른 기존
OOF는 혼합하거나 하나의 순위로 비교하지 않는다.

## 후보 생존 규칙

고정된 단일 Brier 임계값으로 후보를 대량 제거하지 않는다. 아래 중 하나를
충족하면 심화 후보로 유지한다.

- 정렬된 강한 ML 기준선에 근접하거나 단독 Brier를 개선한다.
- 사전 고정 weight grid로 ML과 혼합했을 때 Brier를 개선한다.
- 특정 fold 또는 segment에서 재현 가능한 개선을 보인다.
- 다른 후보와 오차 상관이 낮아 ensemble 기여가 있다.
- 해당 모델 계열의 상위 후보에 해당한다.

단독 성능, blend 기여, segment 강점과 오차 다양성의 Pareto 후보군을 유지한다.
한 설정의 실패는 그 설정만 종료한다.

## 재개와 실패 처리

- 후보, seed, fold와 epoch 단위로 완료 상태를 원자적으로 기록한다.
- Colab 재연결 시 완료된 작업은 건너뛰고 마지막 유효 checkpoint부터 이어간다.
- OOM은 micro-batch 축소, gradient accumulation 증가와 activation checkpoint로
  재시도한다.
- NaN은 AMP scale, optimizer와 learning rate를 조정해 동일 용량으로 재시도한다.
- 실패와 재시도 이유는 manifest에 남기고 다른 후보 실행은 계속한다.

## 코드 범위

새 저장소에 이번 캠페인에 필요한 코드만 추가한다.

```text
experiments/independent_dl/
├── configs/campaign_v1.json
├── features.py
├── models/
│   ├── tabm.py
│   ├── mlp_resnet.py
│   ├── ft_transformer.py
│   └── tabr.py
├── campaign.py
├── evaluation.py
└── run_campaign.py
```

범용 실험 플랫폼, 웹 대시보드, 자동 문서 생성기와 제출 패키징은 만들지 않는다.
기존 코드에서 필요한 feature 또는 DL adapter만 선별해 옮긴다.

## 사용자 실행 계약

Codex는 다음만 수행한다.

- 실험 코드와 설정 작성
- 문법과 import 확인
- 아주 작은 합성 데이터로 cutoff, 입출력 형태와 checkpoint 재개 핵심 경로 확인
- Colab 오류의 원래 traceback이 보이도록 실행 경계 작성

사용자는 다음을 수행한다.

- Colab 의존성 설치와 T4 확인
- 공식 전체 데이터 feature cache 생성
- 실제 학습, OOF, multi-seed, full inference와 장시간 재개 실행
- 생성된 지표와 산출물 전달

사용자 실행 시 Drive 마운트, 정확한 저장소 커밋, 의존성, T4와 입력 파일 확인,
캠페인 실행·재개 및 전체 오류 출력을 포함한 하나의 완결된 Colab 셀을 제공한다.
사용자 요청 없이 노트북을 수정하거나 push하지 않는다.

## Drive 산출물

```text
outputs/independent_dl_campaign_v1/
├── campaign_manifest.json
├── candidate_results.jsonl
├── feature_cache/
├── checkpoints/{candidate_id}/{seed}/{fold}/
├── predictions/{candidate_id}/{seed}/{fold}.csv
├── metrics/{candidate_id}.json
├── leaderboards/
│   ├── standalone.csv
│   ├── diversity.csv
│   └── blend_contribution.csv
└── campaign_summary.json
```

대용량 산출물은 Drive에만 저장한다. Git에는 코드, 설정과 작은 실행 판정만 남긴다.

## 의도적으로 하지 않는 검증

- Codex가 공식 데이터를 읽어 실제 모델을 학습하지 않는다.
- 동일 계약을 여러 테스트로 중복 검증하지 않는다.
- 대회 성능과 무관한 파일 경합, 가상 공격과 범용 보안 계층을 추가하지 않는다.
- 사용자가 Colab에서 확인할 패키지·GPU·실데이터 문제를 로컬에서 완벽히 재현하지
  않는다.
- 코드 변경마다 관련 없는 저장소 전체 검증을 반복하지 않는다.

대회 규정, 시간 누출, 데이터 손상, 잘못된 행 정렬과 무효 산출물처럼 결과를 믿을 수
없게 만드는 문제만 실행 전에 차단한다.

## 구현 완료 기준

1. 네 모델 계열과 네 입력 표현의 최소 64개 본 후보가 설정으로 표현된다.
2. T4에서 후보를 순차 실행하고 checkpoint부터 재개할 수 있다.
3. OOM 대응이 모델 용량을 자동 축소하지 않고 실행 설정만 조정한다.
4. fold-safe feature 상태와 cache identity가 기록된다.
5. 단독 성능, 다양성과 정렬된 blend 기여도를 비교할 산출물이 남는다.
6. Codex의 로컬 확인은 작은 합성 테스트에 그치며 공식 전체 실행은 사용자에게 남는다.
7. 제출 ZIP이나 자동 제출 코드는 포함하지 않는다.
