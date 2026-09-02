---
title: "9시간 학습이 OOM으로 끝난 뒤 바꾼 ML 실험 운영 방식"
date: "2026-09-02"
type: "debugging"
project: "lg-aimers-9th-competition"
tags: ["AI", "Debugging", "Performance", "Testing"]
summary: "Kaggle T4×2에서 33,677.4초 뒤 메모리 부족으로 끝난 E3를 운영 실패로 기록하고, 다음 실험의 시간·메모리 예산 방식을 정리합니다."
featured: false
draft: true
canonicalProjectPath: "docs/blog/2026-09-02-ml-experiment-oom-retrospective.md"
sourceRepository: "https://github.com/yh4752/lg-aimers-9th-competition"
---

> 한 줄 요약: 9시간 넘게 실행한 E3는 성능이 나쁜 후보가 아니라 성능을 판정하기 전에 메모리가 바닥난 실험이었다.

## 문제

Tree Expert E2로 Public `977.3809532715`를 확인한 뒤 더 넓은 전문가 앙상블을 설계했다. E2가 놓친 실패 유형과 시즌 변화를 나눠 학습하면 1,000점에 가까워질 수 있다는 가설이었다.

남은 GPU 시간은 제한돼 있었고 한 번의 Kaggle 실행 안에서 가능한 한 많은 후보를 비교하고 싶었다. 이 판단은 모델 가설에는 공격적이었지만 운영 예산에는 느슨했다. 각 작업의 시간과 메모리를 작은 파일럿으로 재지 않은 채 63개 OOF 작업을 한 캠페인에 넣었다.

## E3 설계

E3는 E2 확률을 anchor로 두고 여러 CatBoost 전문가가 잔차를 나눠 맡는 구조였다.

- 전체 기간과 최근 기간의 성공 패턴
- 정규 시즌 `R`과 별도 경기 유형 `F`
- `middle`, `wild`, `reverse`로 나눈 실패 유형
- 각 행에서 어떤 전문가를 얼마나 적용할지 정하는 gate
- 세 개 seed와 세 시간 전이 OOF

이전 실패 유형 감사에서 `middle`과 `reverse`가 약 7.16~7.44% 겹쳤다. E3는 이 중첩을 라벨 오류로 지우지 않고 두 현상이 같이 나타나는 신호로 남겼다. `wild`는 성공도, middle도, reverse도 아닌 실패로 정의했다.

구조는 넓었지만 비교 순서는 정해 뒀다. Screening에서 약한 후보를 줄이고 confirmation으로 다시 확인한 뒤 extra seed를 추가할 계획이었다. 모든 OOF가 끝나야 최종 decision, 전체 데이터 학습과 행 독립성 감사로 넘어갈 수 있었다.

## 실제 로그

실행 환경은 Kaggle의 Tesla T4 GPU 2장이었다. Version 70은 `33,677.4초`, 약 9시간 21분 동안 실행됐다.

마지막 상태 파일을 열어 보니 진행 상황은 아래와 같았다.

| 단계 | 계획 | 완료 |
| --- | ---: | ---: |
| Screening OOF | 14 | 14 |
| Confirmation OOF | 7 | 7 |
| Extra seed OOF | 42 | 32 |
| 전체 OOF | 63 | 53 |

남은 OOF 작업은 10개였다. 이때 런타임 메모리가 한도를 넘었고 프로세스가 종료됐다. 최종 decision, 전체 학습과 행 독립성 감사는 시작하지 못했다.

## 왜 rejected가 아닌 failed인가

`rejected`는 실행을 끝낸 후보가 사전에 정한 성능 기준을 통과하지 못했다는 뜻이다. `failed`는 코드, 환경이나 자원 문제로 그 판단 자체를 끝내지 못했다는 뜻이다.

E3에는 중간 Brier가 남아 있었다. 하지만 63개 중 53개만 끝난 값으로 후보를 고르면 먼저 완료된 seed와 fold에 유리한 결론을 만들 수 있다. 아직 decision 단계에도 도달하지 않았으므로 좋다거나 나쁘다고 판정할 근거가 없었다.

Registry에는 상태를 `failed`, 실패 분류를 `runtime`, 세부 원인을 `resource_memory`로 기록했다. `submission_eligible=false`, `performance_claim_allowed=false`도 함께 남겼다. 이 값들은 E3 일부 결과를 제출이나 성능 주장에 쓰지 못하게 막는다.

## 복구한 증거

Kaggle 세션은 끝났지만 결과 ZIP과 review, handoff 파일은 회수했다. 원본 실행 코드와 산출물 안의 코드를 대조한 결과 47개 소스 중 47개가 일치했다. 실행 시간, 완료 작업 수와 마지막 단계도 로그와 상태 파일에서 다시 확인했다.

이 검사는 모델 성능을 복구한 것이 아니다. “어떤 코드가 어디까지 실행됐고 왜 멈췄는가”를 복구했다. 그래서 E3의 성능 순위는 비워 두고 운영 실패의 근거만 A등급 evidence로 등록했다.

## 원인

직접 원인은 메모리 부족(OOM, out of memory)이었다. 더 근본적인 문제는 하나의 긴 캠페인에 많은 모델과 OOF 산출물을 묶고 단계가 바뀔 때 메모리가 충분히 회수되는지 미리 재지 않은 데 있었다.

GPU가 2장이라고 시스템 메모리도 두 배가 되는 것은 아니다. 두 작업이 동시에 모델과 데이터를 올리면 GPU 메모리뿐 아니라 CPU RAM과 파일 캐시도 함께 늘어난다. 실행 시간이 길어질수록 여러 단계의 객체가 남아 있는지 확인하기도 어려웠다.

## 선택지

| 선택지 | 장점 | 판단 |
| --- | --- | --- |
| 같은 캠페인을 그대로 재실행 | 추가 코드 수정 없이 남은 작업을 기대할 수 있다. | 같은 메모리 패턴이 반복될 가능성이 커서 선택하지 않는다. |
| 작업 수만 절반으로 줄이기 | 당장 종료 위험을 낮춘다. | 어떤 자원이 병목인지 모르면 임의 축소에 그친다. |
| 파일럿으로 시간과 peak RAM을 측정한 뒤 단계별 예산 설정 | 다음 실행 규모를 계산할 수 있다. | 준비 작업이 늘지만 원인을 제어할 수 있어 채택한다. |
| 모든 후보를 별도 세션으로 분리 | 장애 범위를 줄인다. | 업로드와 복구 비용이 커서 강한 후보의 confirmation 단계에만 제한적으로 쓴다. |

## 다음 운영 계약

다음 장기 실험에는 아래 조건을 먼저 적용한다.

1. 대표 fold와 seed 한 개를 파일럿으로 실행해 작업당 시간과 peak RAM을 잰다.
2. 측정값에 여유 폭을 더해 한 세션에서 끝낼 작업 수를 계산한다.
3. Screening, confirmation과 extra seed 사이에서 모델 풀과 데이터 객체를 명시적으로 해제한다.
4. 중간 resume은 여러 개를 자동 다운로드하지 않고 검증된 최신 파일 하나만 유지한다.
5. 남은 시간으로 끝낼 수 없는 새 작업은 시작하지 않는다.
6. Decision 전에 멈춘 부분 결과는 성능 비교와 제출 패키징에서 제외한다.

한 번에 실험하는 후보 수는 줄어든다. 그래도 9시간 뒤 아무 판정도 남지 않는 것보다는 낫다. 중간에 멈춰도 어느 단계부터 다시 시작할지 알 수 있다.

## 검증한 것과 미검증

검증한 것은 실행 코드 47개의 일치, 실행 시간 `33,677.4초`, OOF 53/63 완료, 마지막 상태 `extra_seeds`, 메모리 부족 종료와 후속 단계 미실행이다.

검증하지 못한 것은 E3의 최종 성능, 최적 gate, 전체 학습 결과와 행 독립성이다. 따라서 E3를 기존 E2보다 좋거나 나쁘다고 표현하지 않는다. 제출 가능한 모델도 만들지 않았다.

## Codex에게 맡긴 것과 내가 검토한 것

Codex는 E3 캠페인 코드, resume과 handoff 구조, 로그 분석과 실패 evidence 작성을 지원했다. 전체 Kaggle 실행과 파일 회수는 내가 직접 진행했다.

나는 남은 GPU 시간 안에서 E3의 범위를 선택했고 종료 로그와 다운로드한 파일을 확인했다. 실행이 끝난 뒤 일부 수치로 후보를 억지로 평가하지 않고 실패로 기록하기로 결정했다. 다음 실험에서는 모델 아이디어뿐 아니라 자원 예산도 승인 기준에 넣어야 한다는 판단 역시 이 경험에서 나왔다.

## 코드에서 다시 볼 지점

| 자료 | 다시 볼 내용 |
| --- | --- |
| [E3 실패 evidence](https://github.com/yh4752/lg-aimers-9th-competition/blob/main/reports/evidence/failure_regime_e3_20260902.json) | 실행 시간, 작업 수, 실패 분류와 금지된 주장 |
| [실험 registry](https://github.com/yh4752/lg-aimers-9th-competition/blob/main/reports/experiment_registry.json) | `failed`, `runtime`, `repeat_policy=blocked` 기록 |
| [실험 장부](https://github.com/yh4752/lg-aimers-9th-competition/blob/main/reports/EXPERIMENT_LEDGER.md) | E3와 이전 실험의 상태 구분 |
| [E3 실행 코드](https://github.com/yh4752/lg-aimers-9th-competition/tree/main/experiments/failure_regime_e3) | 단계별 후보와 복구 구조 |
| [E3 산출물 테스트](https://github.com/yh4752/lg-aimers-9th-competition/blob/main/tests/test_failure_regime_e3_artifacts.py) | Resume, review와 handoff 무결성 검사 |

## 면접에서 설명할 수 있어야 할 질문

1. E3를 성능 기각이 아니라 실행 실패로 분류한 이유는 무엇인가?
2. 53개 OOF 결과만으로 후보를 선택하면 어떤 편향이 생길 수 있는가?
3. GPU 2장을 쓸 때 CPU RAM까지 따로 예산화해야 하는 이유는 무엇인가?
4. 파일럿 한 작업으로 전체 캠페인의 시간과 메모리를 어떻게 추정할 것인가?
5. 최신 resume 하나만 유지하면 복구와 저장 비용이 어떻게 달라지는가?
6. AI가 작성한 장기 실행 코드에서 사용자가 직접 승인해야 할 자원 조건은 무엇인가?
