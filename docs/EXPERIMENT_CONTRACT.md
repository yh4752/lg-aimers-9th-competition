# 실험 실행 계약

이 문서는 저장소의 현재 실험 운영 계약이다. 과거 설계와 구현 기록이 이 문서와
충돌하면 완료된 실행 evidence와 이 계약을 우선한다.

## 고정 원칙

- 설정과 선택 기준은 결과를 보기 전에 고정한다.
- 시간 전이 검증은 학습 구간에서 fit한 상태만 다음 시즌에 적용한다.
- 검증·테스트 행의 집계, 정답과 미래 시즌 정보를 학습 특징에 섞지 않는다.
- 테스트 행은 같은 평가 배치의 다른 행, 행 순서와 배치 크기에 독립적이어야 한다.
- Public 결과에 맞춘 사후 미세 조정을 검증된 개선으로 취급하지 않는다.
- 비용은 실행 순서와 자원 안내에만 사용하며 후보 배제 기준으로 사용하지 않는다.

## DACON 규칙 전이 gate

현재 정책은 `competition_rules/policy.json`의
`dacon-236743-2026-08-13`이다. 각 실험은 자기 설정과 후보 범위를 고정한
`experiment_contract.json`을 가져야 하며, 평가 범위는 `current_row_only`다.

1. **실험 시작 gate**: 공식 train·TrackMan만 사용하고, training rows only,
   pre-pitch only, 외부 API 미사용, 사전학습 가중치의 출처·버전·라이선스·해시를
   계약에 고정한다.
2. **사용자 실행 gate**: `competition_rules`의 소스 검사에서 평가 파일 직접 읽기,
   원격 통신, 평가 행 사이의 groupby·rank·rolling·shift와 평가 행의 수나 분포에
   따른 분기를 거부한다.
3. **후보 수용 gate**: 시간 전이 성능, provenance, 전체 검증 행의 repeat·reverse·
   shuffle·rebatch·singleton 일치와 data/code/config/preprocessing/model/adapter/runtime
   해시가 모두 맞아야 `passed`가 된다.
4. **패키징 gate**: 당일 공식 규칙 검토, 완전한 candidate-local acceptance,
   전체행 감사 manifest, 설치·추론·메모리·크기 benchmark와 실물 해시를 다시
   확인한다. 생성기는 `submission/package.py` 하나뿐이다.

허용되는 추론은 각 평가 행 자체의 feature, 학습 시 고정한 통계·segment·calibration,
고정 ensemble과 row-id로 고정한 per-row TTA다. 같은 평가 배치의 다른 행을 이용한
집계·순위·평균 이동 보정, test 파일에 대한 fit, 행 수별 특수 분기는 금지한다.
규칙을 통과하지 못하면 해당 후보의 패키지만 차단하며 독립 후보 연구는 계속한다.

## 성능 우선 사전 확인

- smoke는 import, CUDA, 데이터 흐름과 출력 형태의 기술 확인 전용이며 성능을
  판단하지 않는다.
- 첫 본 실험은 성능을 판단할 수 있는 충분한 모델 규모, 학습 시간과 탐색 폭을
  포함한다.
- 최선 결과가 탐색 경계에 있으면 그 값을 상한으로 확정하지 않고 다음 범위를 확장한다.
- OOM은 batch size, mixed precision, gradient accumulation과 checkpoint 재시작으로
  대응하며 모델 계열 기각 근거가 아니다.
- 비용과 실행 시간은 안내와 순서 정보로만 사용한다.
- 한 seed, 설정, residual 또는 feature view 실패는 해당 설정만 종료한다.

## 후보 상태

```text
planned → code_ready → waiting_for_user_run → passed → package_ready
                                         ↘ rejected
                                         ↘ failed
```

- `planned`: 가설과 검증 기준만 고정된 상태
- `code_ready`: 코드와 작은 테스트가 준비된 상태
- `waiting_for_user_run`: 전체 데이터 또는 GPU 실행을 사용자가 수행해야 하는 상태
- `passed`: 고정 acceptance 기준과 evidence를 통과한 상태
- `rejected`: 실행은 유효하지만 성능 gate를 통과하지 못한 상태
- `failed`: 환경·실행·산출물 오류로 성능을 판정할 수 없는 상태
- `package_ready`: passed evidence와 현재 파일 해시를 다시 대조한 상태

`rejected`와 `failed`는 해당 후보의 패키지만 막으며 독립 후보를 막지 않는다.

## 실행 소유권

Codex는 코드, 리뷰, import·정적 검사, 작은 합성 데이터와 fixture 테스트를 맡는다.
사용자는 공식 전체 데이터 전처리, 시간 전이 OOF, CPU·GPU 학습, 전체 추론,
Colab·Drive 장시간 실행과 실제 제출 평가를 맡는다. 중량 실행의 소유권은 사용자가
해당 실행을 별도로 승인했을 때만 바뀐다.

## Acceptance evidence

`passed` evidence에는 후보·실험·모델 ID, 전체·fold·필수 segment 지표, 모든
수용 gate의 JSON boolean `true`와 아래 현재 산출물의 소문자 SHA-256이 있어야
한다.

- `data_preflight`
- `predictions`
- `training_code`
- `config`

JSON은 중복 키, `NaN`과 `Infinity`를 허용하지 않는다. evidence와 대상 파일은
저장소 또는 승인된 실행 경계 안의 일반 파일이어야 하며 심볼릭 링크와 경로 이탈을
허용하지 않는다.

## 패키징과 제출

패키징 진입점은 후보 상태, acceptance ID·gate와 현재 네 산출물의 실제 SHA-256을
다시 검사한다. 하나라도 다르면 파일을 만들기 전에 중단한다. `passed`와 현재 해시
검사가 모두 통과하기 전에는 제출 패키지를 만들지 않는다. 제출은 자동화하지 않으며
사람이 evidence와 독립 추론 결과를 확인한 뒤 사용자가 수동으로 수행한다.

저장소는 자동 업로드 또는 자동 제출을 제공하지 않는다. 자동화할 수 없는 최종
확인은 팀·계정과 중복 등록 여부, 당일 일일 제출 잔여량, 대회 마감 여부, 업로드
화면에서 고른 ZIP의 파일명·SHA-256이다. 이는 기계 gate의 `passed`로 기록하지
않고 사용자가 제출 직전에 직접 확인한다.

## 기록

완료된 실행만 `reports/EXPERIMENT_LEDGER.md`에 기록한다. 각 기록은 검증 프로토콜,
핵심 지표, 판정, 작은 Git evidence, 대용량 Drive 원본의 논리 위치와 다음 결정을
포함한다. Public 점수는 실제 제출 결과가 있을 때만 별도로 기록한다.
