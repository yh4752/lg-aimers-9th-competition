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

## 기록

완료된 실행만 `reports/EXPERIMENT_LEDGER.md`에 기록한다. 각 기록은 검증 프로토콜,
핵심 지표, 판정, 작은 Git evidence, 대용량 Drive 원본의 논리 위치와 다음 결정을
포함한다. Public 점수는 실제 제출 결과가 있을 때만 별도로 기록한다.
