# TrackMan 교사·계층 프로필 통합 캠페인 설계

## 1. 목적

현재 승인된 최고 기준선은 Tree Expert E2의 `c1_anchor_residual`이다. 세 개의
CatBoost seed를 평균한 이 모델은 시간 OOF에서 weighted Brier를
`0.0008375137` 개선했고 Public `977.3809532715`를 기록했다.

그 이후의 최근 시즌 분리, 단순 계층 보정, R/F 전문가, 실패 유형 전문가,
XGBoost·LightGBM 잔차 보정, 대규모 anchor·residual·hierarchical 조합은 E2를
안정적으로 넘지 못했다. 따라서 이번 캠페인은 같은 모델이나 후처리를 늘리지 않고,
E2가 추론할 수 없었던 학습 전용 정보와 명시적인 축소 통계를 학생 모델에 전달할 수
있는지를 검증한다.

핵심 가설은 두 가지다.

1. 공식 학습 기간의 현재 투구 TrackMan 측정값으로 학습한 교사는 이진 정답보다
   풍부한 제구 난이도 신호를 제공할 수 있다.
2. 투수·타자·상황 상호작용의 과거 성공률을 계층적으로 축소한 프로필은 CatBoost의
   암묵적인 CTR과 다른 오차를 만들 수 있다.

두 신호를 E2의 anchor residual 구조와 결합해, 단일 신규 피처군과 결합 피처군을
같은 시간 OOF에서 비교한다.

### 기존 작업과의 경계

저장소에는 Temporal Portfolio를 위해 만든 현재 투구 TrackMan 일대일 매칭,
투수-held-out 교사 cross-fit, TabM soft-label 학습 기반이 이미 있다. 현재 저장소의
review·ledger에는 이 LUPI 신호를 E2 CatBoost 잔차 모델과 결합해 승인한 증거가 없다.

이번 캠페인은 기존 매칭과 교사 구현을 다시 만들거나 동일한 TabM 실험을 반복하지
않는다. 다음만 새로 검증한다.

- 기존 감사된 교사 OOF를 CatBoost Brier 학생 target으로 전달하는 방식
- S4의 사후 calibration보다 세분화된 계층 프로필을 잔차 모델 입력으로 사용하는 방식
- 위 두 신호와 E2 anchor를 함께 사용했을 때의 시간 OOF 개선

기존 구현과 새 계약이 충돌하면 기존 바이트를 조용히 재사용하지 않고 입력 단계에서
명시적으로 중단한다.

## 2. 실행 범위와 예산

- 실행 환경: Kaggle T4 GPU 2장
- 남은 GPU quota: 24시간
- 최대 벽시계 실행: 10시간 30분
- 종료·압축·재시작 여유: 최소 1시간
- 사용자 역할: 공식 데이터와 통합 입력을 Kaggle Dataset으로 추가하고 Save Version 실행
- Codex 역할: 코드 작성, 정적 검사, 작은 fixture 검사, 결과 ZIP 검증

T4 두 장을 동시에 사용할 때 GPU quota가 벽시계보다 빠르게 소진될 수 있으므로 한
세션을 10시간 30분보다 길게 잡지 않는다. 두 GPU에는 서로 독립적인 학습 작업만
배치한다. CPU 전처리가 병목일 때 불필요하게 GPU를 점유하지 않도록 프로필과
TrackMan 매칭 캐시는 학습 시작 전에 한 번만 만든다.

캠페인은 한 Kaggle Save Version에서 끝내는 것을 기본으로 한다. 시간 제한이나
예외로 중단되면 검증된 handoff 하나를 받아 새 Save Version에서 재개한다.

## 3. 대회 규칙 경계

평가 행 하나의 예측에는 다음 정보만 사용할 수 있다.

- 해당 평가 행의 원본 변수와 행 내부 파생변수
- 공식 학습 데이터만으로 미리 만든 통계와 조회표
- 공식 학습 데이터와 TrackMan history만으로 학습한 모델
- OOF에서 고정한 모델 구조와 혼합 비율

금지 사항은 다음과 같다.

- 평가 데이터의 다른 행을 이용한 정렬, 차분, 누적, rolling 또는 lag
- 평가 데이터 전체나 일부의 평균, 빈도, 순위, 분포 또는 그룹 통계
- 평가 데이터에 등장한 선수·팀·월·경기별 행 수를 이용한 보정
- 평가 행 간 asof 값의 차이를 이용한 정답 또는 최근 기록 복원
- 외부 데이터와 평가 기간 TrackMan 사용

공식 학습 기간 TrackMan의 현재 투구 측정값은 교사 학습에만 사용할 수 있다. 최종
학생과 배포 모델은 해당 측정값을 입력으로 받지 않는다. 동일한 평가 행을 단독,
셔플, 역순, 여러 batch 크기로 추론했을 때 결과가 같아야 한다.

## 4. 고정 기준선과 시간 검증

- 기준선: E2 `c1_anchor_residual`
- 기준선 seed: `42`, `2026`, `3407`
- 시간 fold: `2021→2022`, `2022→2023`, `2023→2024`
- 구조 선별 seed: `3407`
- 추가 확인 seed: `42`, `2026`
- 평가 지표: Brier score, 낮을수록 우수

구조와 혼합 강도는 앞선 두 fold를 중심으로 고르고 `2023→2024`는 최근 시즌 확인에
사용한다. 전체 판정에서는 세 fold를 모두 포함한다. Public 점수로 상수나 확률
분포를 역추정하지 않는다.

## 5. TrackMan 교사

### 5.1 행 매칭 감사

학습 전에 train과 TrackMan history의 투구 매칭을 다시 계산한다. 매칭은 공식
데이터만 사용하며 결과마다 confidence와 충돌 여부를 남긴다.

감사 항목은 다음과 같다.

- 전체 학습 행 중 고신뢰 매칭 비율
- season별, R/F별, 투수별 매칭 비율
- 하나의 학습 행에 여러 TrackMan 행이 대응하는 비율
- 하나의 TrackMan 행이 여러 학습 행에 재사용되는 비율
- 매칭된 행과 미매칭 행의 target 및 주요 상황 분포 차이

고신뢰 일대일 매칭만 교사 학습에 사용한다. 중복이나 모호한 매칭은 임의 선택하지
않고 제외한다. 전체 coverage가 `30%`보다 낮거나 최근 fold coverage가 `20%`보다
낮으면 증류 후보를 만들지 않고 프로필 후보만 계속한다.

### 5.2 교사 피처와 OOF

교사는 학습 행에서 사용할 수 있는 기존 상황 피처와 현재 투구 TrackMan 측정값을
함께 받는다. 구속, 회전, 수직·수평 무브먼트, 릴리스, 익스텐션, 존 도달 속도,
구종 그룹과 데이터에 실제 존재하는 결측·품질 표시를 포함한다. 매칭되지 않은 행을
임의 보간해 교사 학습에 섞지 않는다.

교사 확률은 반드시 기존 LUPI와 같은 pitcher-held-out 5-fold cross-fitting으로 만든다.
어떤 행의 soft label도 그 행이나 같은 투수의 정답을 학습한 교사에서 나오면 안 된다.
각 투수의 fold는 seed와 canonical pitcher id의 SHA-256으로 결정하고, 각 행의 fold id와
예측 교사 fold를 artifact에 기록한다.

교사는 기존 `CatBoostTeacherBackend`와 여덟 개의 고정 측정값(`rel_speed`,
`spin_rate`, `induced_vert_break`, `horz_break`, `extension`, `rel_height`,
`rel_side`, `zone_speed`)을 사용한다. 하나의 고정 구조와 seed `3407`만 사용한다.
이번 캠페인의 목적은 교사 하이퍼파라미터 탐색이 아니라 증류 신호의 유효성
검증이다.

### 5.3 학생 soft target

학생은 평가 시 사용 가능한 E2 피처만 받으며 TrackMan 측정값이나 매칭 여부는 받지
않는다. 고신뢰 매칭 행의 학습 target만 다음처럼 완화한다.

```text
soft_target = (1 - lambda) * control_success + lambda * teacher_oof_probability
```

미매칭 행은 원래 이진 target을 그대로 사용한다. `lambda`는 `0.15`, `0.35` 두 값만
비교한다. 학생은 soft target을 직접 회귀해 Brier 목적과 맞추고 출력은 `[1e-5,
1-1e-5]`로 제한한다.

## 6. 신규 계층 프로필

기존 계층 보정은 제한된 수준의 성공률을 모델 뒤에서 보정했다. 이번 프로필은 더
많은 상호작용의 표본 수와 축소 성공률을 잔차 모델 입력으로 직접 제공한다.

우선 수준은 다음과 같다.

- 투수
- 타자
- 투수 × 카운트
- 투수 × 타자 손
- 투수 × 베이스 상황
- 투수 × 경기 유형
- 타자 × 투수 손
- 타자 × 카운트
- 투수 팀 × 카운트
- 투수 × 타자 직접 매치업

각 수준에는 count, raw rate, parent rate, shrunk rate, known flag,
`shrunk - parent`, logit difference를 만든다. 직접 매치업과 고차원 상호작용은 최소
표본 수를 충족하지 못하면 부모 수준으로 되돌린다.

축소 강도 후보는 별도 모델을 대량 학습하지 않고 저장된 training target 통계에서
계산한다.

- identity: `25`, `75`, `200`
- context interaction: `50`, `150`, `400`
- direct matchup: `100`, `300`, `800`

앞선 두 fold에서 weighted Brier가 가장 좋은 조합을 하나 고정한 뒤 최근 fold에
적용한다. 학습 행의 프로필은 time-safe cross-fitting으로 만들고 검증·평가 행의
프로필은 해당 cutoff까지의 공식 학습 자료로 만든 고정 조회표에서 가져온다.

## 7. 후보군

`E2`는 변경하지 않는 control이다. 구조 선별 seed `3407`에서 아래 여섯 후보를 세
시간 fold에 학습한다.

| 후보 | 신규 프로필 | 증류 | R/F 혼합 |
| --- | --- | --- | --- |
| `P` | 사용 | 없음 | 공통 |
| `D15` | 없음 | 15% | 공통 |
| `D35` | 없음 | 35% | 공통 |
| `PD15` | 사용 | 15% | 공통 |
| `PD35` | 사용 | 35% | 공통 |
| `PD_RF` | 사용 | 선별된 비율 | 구간별 alpha |

모든 후보는 E2 anchor를 고정하고 CatBoost가 남은 확률 잔차를 학습한다. `PD_RF`는
새로운 정보가 실제로 R/F에서 다르게 작동하는지만 확인한다. R/F별 새 전문가를 처음부터
별도로 대량 학습하지 않고, 공통 결합 후보와 E2의 OOF 예측을 구간별 alpha로 섞는다.
alpha 후보는 `0.35`, `0.60`, `0.80`, `1.00`이다.

## 8. 단계별 선택

### 8.1 구조 선별

여섯 후보를 seed `3407`로 비교한다. 앞선 두 fold 평균이 E2보다 나쁜 후보는 추가
seed를 학습하지 않는다. 통과 후보를 다음 순서로 정렬한다.

1. 세 fold weighted Brier gain
2. 2023→2024 gain
3. 최악 fold gain
4. E2 residual과의 상관이 낮은 후보
5. 더 단순한 후보

상위 두 후보만 다중 seed 확인으로 보낸다. 두 후보의 OOF 예측 상관이 `0.998`을
넘으면 더 좋은 하나만 보낸다.

### 8.2 다중 seed 확인

선별된 후보만 seed `42`, `2026`을 추가해 세 seed 평균을 만든다. seed별 gain,
fold별 gain, 주요 구간 회귀와 bootstrap 구간을 계산한다.

### 8.3 제출 후보 판정

다음 조건을 모두 만족한 후보만 `accepted`로 판정한다.

- 세 fold 행 수 가중 Brier gain `>= 0.00005`
- 세 fold 중 최소 두 fold 개선
- `2023→2024` gain `>= -0.00002`
- 주요 사전 정의 구간의 최대 Brier 악화 `<= 0.00050`
- 세 seed 중 최소 두 seed 개선
- 평균 예측이 finite이고 `[0, 1]` 범위 안에 있음
- 행 독립성 감사 최대 차이 `<= 1e-6`

기준을 통과한 후보가 둘이면 두 후보의 OOF 평균도 평가한다. 평균이 둘 중 최선보다
`0.00002` 이상 추가 개선될 때만 ensemble candidate로 인정한다. 검증을 통과하지
않은 후보는 full-fit이나 제출 패키징으로 보내지 않는다.

## 9. 시간 예산과 조기 종료

| 구간 | 최대 벽시계 |
| --- | ---: |
| 입력·매칭·프로필 캐시 | 45분 |
| 교사 inner OOF | 2시간 |
| 6개 구조 선별 | 3시간 30분 |
| 상위 2개 다중 seed | 2시간 30분 |
| full-fit·감사·artifact | 1시간 30분 |
| 압축·종료 여유 | 1시간 15분 |

다음 조건에서는 시간을 아낀다.

- TrackMan coverage gate 실패: 교사와 증류 후보 생략
- 앞선 두 fold가 모두 악화: 해당 후보 추가 seed 생략
- 두 후보 예측이 사실상 동일: 하나만 확인
- 남은 시간이 2시간 미만: 새 학습을 시작하지 않고 검증된 handoff 생성

## 10. 재개와 로그

전처리 캐시, 교사 OOF, 후보별 fold·seed 학습을 각각 원자적 작업으로 관리한다.
완료 파일을 임시 파일과 구분하고 artifact hash가 일치할 때만 `JOB_REUSED`한다.

필수 로그는 다음과 같다.

- `PRIV_CODE_READY`
- `PRIV_INPUTS_VERIFIED`
- `PRIV_GPU_READY`
- `PRIV_MATCH_AUDIT`
- `PRIV_TEACHER_JOB_START`, `PRIV_TEACHER_JOB_END`
- `PRIV_CANDIDATE_JOB_START`, `PRIV_CANDIDATE_JOB_END`
- `PRIV_DECISION`
- `PRIV_HANDOFF_READY`
- `PRIV_CAMPAIGN_SUCCESS`
- `PRIV_ERROR`

학습 중 브라우저 다운로드를 반복하지 않는다. 성공, 안전 중단 또는 오류 시 마지막으로
검증된 handoff ZIP 하나만 다운로드 요청한다. 정상 완료 시 review ZIP을 추가하고,
accepted 후보가 있을 때만 delivery ZIP을 추가한다.

## 11. 산출물과 제출 경계

- `handoff`: 완료 작업, 캐시, OOF, 체크포인트, 선택 상태와 identity
- `review`: 매칭 감사, fold·seed·구간별 Brier, 후보 선택과 기각 근거, 독립성 감사
- `delivery`: accepted 후보의 full-fit 모델, 학습 데이터 조회표와 고정 혼합 설정

Kaggle 캠페인은 DACON 제출 ZIP을 만들지 않는다. 사용자가 delivery를 전달하면
Codex가 acceptance evidence, artifact hash, 모델·조회표 schema와 독립성 감사를 다시
검증한다. 그 검증을 통과한 이후에만 별도 제출 패키징 단계로 넘어간다.

## 12. 구현 전 검증 범위

전체 학습은 사용자가 실행한다. Codex는 구현 과정에서 다음 작은 검사를 수행한다.

- 모호한 TrackMan 매칭 배제와 coverage gate
- 교사 OOF에서 자기 행 정답 누수 차단
- 계층 프로필의 cutoff 및 parent fallback
- soft target과 미매칭 행 처리
- 후보 선별, 다중 seed, ensemble gate
- GPU 두 작업 동시 실행과 상태 파일 잠금
- 종료 직전 handoff 생성 및 재개
- 압축과 디렉터리 형태 입력 탐색
- 평가 행 단독·셔플·역순·batch 독립성
- 현재 E2 제출 경로와 기존 실험 회귀

## 13. 완료 조건

다음 상태가 모두 충족되면 Kaggle 실행 준비가 완료된 것으로 본다.

- 통합 입력 생성 도구와 한 셀 Kaggle 실행 코드가 현재 contract hash로 고정됨
- 작은 자동 검사가 모두 통과함
- T4 두 장에서 독립 작업만 동시에 실행함
- 최대 벽시계와 종료 여유가 코드로 강제됨
- 중단 후 완료 작업이 재사용됨
- 마지막 handoff 하나만 자동 다운로드됨
- accepted 판정 전에는 delivery와 제출 ZIP 생성이 차단됨
