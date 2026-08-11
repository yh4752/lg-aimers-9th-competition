# R9 검증 기반

## 가설

같은 시즌의 무작위 OOF 대신 과거 시즌에서 fit한 상태만 다음 시즌에 적용하면
간접적인 시간 누출 없이 후보를 비교할 수 있다.

## 검증 프로토콜

2021→2022, 2022→2023, 2023→2024의 세 전이를 독립적으로 재생했다. 검증 행과
테스트 행의 집계치는 학습 특징에 사용하지 않고 행 독립성을 검사했다.

## 결과

총 746,504행의 R9 OOF Brier는 `0.24825099524638927`이다. 예측, preflight,
학습 코드와 manifest 해시가 acceptance evidence에 고정돼 있다.

## 판정

`passed`. 이후 후보가 같은 행과 프로토콜을 사용할 때 공통 anchor로 사용한다.

## 배운 점

평균 Brier만으로는 부족하다. fold와 game type 등 사전에 정한 segment 안정성,
provenance와 행 독립성을 함께 봐야 한다.

## 다음 결정

새 후보는 R9을 교체한다고 가정하지 않고 먼저 독립 OOF와 수용 gate를 통과한다.

## 근거

- [R9 OOF acceptance](../../reports/acceptances/round9_temporal_oof_acceptance.json)
- [Anchor Brier audit](../../reports/acceptances/anchor_brier_audit_acceptance.json)
- 동료 저장소의 [R9 모델 계보](https://github.com/castle9612/lg_aimers_9th/blob/main/README.md)
