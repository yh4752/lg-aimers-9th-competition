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
| 열린 계열 | segment-aware calibration, 독립 TabM 후보, XGBoost |
| XGBoost v3 | 공격적 4-member ensemble, Public `820.9583317093` |
| XGBoost rescue | 2024 `0.24826687414041645`, technical verified, Public 미확인 |
| 다음 주력 | Colab Pro의 독립 DL 프런티어 캠페인 |

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

## 문서

- [실험 장부](reports/EXPERIMENT_LEDGER.md): 완료된 모든 실행과 판정
- [실험 라운드](docs/rounds/README.md): 가설, 결과, 배운 점과 다음 결정
- [실험 실행 계약](docs/EXPERIMENT_CONTRACT.md): 역할, 상태와 패키지 gate
- [로드맵](docs/ROADMAP.md): 다음 후보와 코드 이전 순서
- [저장소 이전 설계](docs/superpowers/specs/2026-08-11-competition-repository-migration-design.md)

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

구체적인 실행 순서는 [ROADMAP](docs/ROADMAP.md)에 있다. 저장소 기반을 확인한 뒤
기존 TabM raw P1·P2를 보존하고, TabM 입력 표현 교차, 대형
ResNet·FT-Transformer, 연구용 TabICLv2와 확장성 수정된 TabR를 실행한다. 본 실행은
[Colab 노트북](notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb)에서 Drive checkpoint로
재개한다. 비용은 실행 순서 안내에만 사용하고 좋은 후보를 제외하는 기준으로 삼지
않는다.
