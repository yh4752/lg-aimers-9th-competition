# LG Aimers 9th Competition

투구 직전 정보로 `control_success=1` 확률을 예측하는 DACON 대회의 실험 코드와
검증 기록을 관리한다. 내부 선택은 누출 없는 시간 전이 Brier Score를 사용하며,
공식 제출 결과와 로컬 검증 결과를 구분한다.

## 한눈에 보기

| 항목 | 현재 상태 |
|---|---|
| 현재 기준선 | R9 temporal OOF, 3개 시즌 전이 746,504행 |
| 기준선 Brier | `0.24825099524638927` |
| 닫힌 계열 | FwFM standalone 및 F 제한 blend |
| 기각된 구성 | TabM residual, calibration의 고정 세 변형 |
| 열린 계열 | TabM seed ensemble·행 단위 파생변수, segment-aware calibration |
| XGBoost v3 | Public `820.9583317093`; 현재 규칙 재검토 전 패키지 차단 |
| XGBoost rescue | 2024 `0.24826687414041645`, technical verified, Public 미확인 |
| TabM 첫 제출 | Public `872.3920184667`; 규칙 준수 단일 모델 |
| 다음 주력 | 기존 OOF seed 앙상블 감사 후 행 단위 파생변수 검증 |
| 규칙 안전선 | `competition_rules` → `experiment_contract.json` → 전체행 evidence → `submission/package.py` |

점수가 같은 표에 있어도 검증 프로토콜이 다르면 직접 순위를 매기지 않는다. R9은
세 개 시즌 전이, XGBoost v3는 네 개 역사 fold 선택 후 2024 holdout을 사용했다.

## 현재 결론

- R9은 새 후보를 비교하는 검증된 anchor다.
- FwFM은 단독, 제한 blend와 종료 감사까지 끝나 계열을 닫았다.
- TabM residual 한 구성은 기각됐지만 다른 독립 TabM 후보를 막지 않는다.
- Calibration 세 변형은 평균 Brier를 개선했지만 calibration gap과 fold 안정성
  gate를 통과하지 못했다. Family는 열려 있다.
- XGBoost v3의 depth 6·63 leaves 4-member ensemble은 Public
  `820.9583317093`을 기록했다. 더 큰 depth 8·127·255 leaves가 자동으로 더 좋지는
  않았으며, 넓은 탐색 후 중간 용량·seed ensemble·시즌 보정을 함께 선택한 결과다.
  다만 평가 분포 평균 이동 보정이 포함되어 현재 독립 예측 규칙 아래에서는 재사용·
  재패키징하지 않는다. 점수는 역사 기록으로만 보존한다.
- 확정 전처리 `dl_standard + hand_matchup`의 단일 TabM은 Public
  `872.3920184667`을 기록했다. 최대 43 epoch 검증의 최적 시점이 2~3 epoch였으므로
  학습시간 연장보다 seed 앙상블과 행 단위 파생변수를 먼저 검증한다.

## 규칙 안전선

모든 ML·DL 후보는 현재 정책 `dacon-236743-2026-08-15`, 후보별
`experiment_contract.json`, `current_row_only` 소스 gate와 전체행 독립성 감사를
순서대로 통과해야 한다. 실제 제출 ZIP은 `submission/package.py`만 만들 수 있으며,
규칙을 통과하지 못하면 파일 생성 전에 중단한다. 데이터, 모델과 ZIP은 저장소에
올리지 않는다.
팀·계정, 중복 참가, 일일 제출 잔여량, 마감과 업로드 파일 선택은 사용자가 확인한다.

## 문서

- [실험 장부](reports/EXPERIMENT_LEDGER.md): 완료된 모든 실행과 판정
- [실험 라운드](docs/rounds/README.md): 가설, 결과, 배운 점과 다음 결정
- [실험 실행 계약](docs/EXPERIMENT_CONTRACT.md): 역할, 상태와 패키지 gate
- [로드맵](docs/ROADMAP.md): 다음 후보와 코드 이전 순서
- [저장소 이전 설계](docs/superpowers/specs/2026-08-11-competition-repository-migration-design.md)
- [TabM Kaggle 실행 안내](docs/TABM_CHAMPION_KAGGLE.md): A–D 입력, 시간, 로그와 전달 파일
- [TabM 행 단위 파생변수 실행 안내](docs/TABM_ROW_FEATURE_PROXY_RUNBOOK.md): Stage P 입력, Colab 재개와 검토 파일
- [TabM 첫 공식 제출](docs/rounds/07-tabm-first-submission.md): 모델, 점수와 해석
- [TabM 점수 개선 설계](docs/superpowers/specs/2026-08-15-tabm-score-improvement-design.md): 다음 실험 순서와 gate

## 검증 프로토콜

- 모델 및 변환 상태는 과거 학습 구간에서만 fit한다.
- 다음 시즌 검증 행에는 고정된 상태만 적용한다.
- 평가 데이터의 다른 행, 행 순서와 배치 크기를 특징 생성에 사용하지 않는다.
- 평균 Brier뿐 아니라 fold, 사전 지정 segment와 calibration 안정성을 확인한다.
- Public 점수에 맞춰 사후 가중치를 선택하지 않는다.

## 실행 역할

Codex는 코드와 작은 합성·정적 테스트를 작성한다. 공식 데이터 전처리, 전체 OOF,
GPU 학습, Colab 장시간 실행과 실제 제출 평가는 사용자가 수행한다. acceptance와
현재 artifact 해시가 모두 통과하기 전에는 제출 패키지를 만들지 않는다.

## 데이터와 대용량 결과

원본 데이터, 대용량 OOF, 모델과 ZIP은 Git에 넣지 않고
[Google Drive 실행 저장소](https://drive.google.com/drive/folders/1SG8lbCKCsznkaiWpGFYOF0abc3WPA9kL)에
보관한다. Git에는 판정에 필요한 작은 JSON, 논리 실행 ID와 SHA-256만 둔다.

## 팀 참고 자료

R9 이후 CatBoost 계보, R25 TabM 잔차와 R32 분모 보정 연구는 동료의
[LG Aimers 9th 저장소](https://github.com/castle9612/lg_aimers_9th)를 참고한다.
전체 코드를 병합하지 않고 다음 실험에 필요한 기능만 누출·행 독립성 검토 후
출처와 함께 선별 이식한다.

## 다음 후보

현재 첫 TabM 제출까지 완료했다. 다음 실행 대상은 새 학습이 아니라 Stage C의 기존
seed별 OOF 예측을 읽는 앙상블 감사다. 이 감사가 사전 등록 gate를 통과한 경우에만
추가 seed 전체 학습을 수행하고, 실패하면 행 단위 파생변수 검증으로 이동한다.
