# Tree Hierarchical Residual Calibration Design

**Date:** 2026-08-27  
**Status:** Approved design; implementation not started

## 1. Objective

Build one rule-safe submission candidate from the accepted Tree Expert E2 lineage by
adding a second residual correction and a rolling hierarchical calibration. The campaign
must fit within about 24 remaining T4x2 GPU hours, survive Kaggle interruptions, and block
full-fit delivery unless paired temporal OOF evidence passes fixed gates.

This is not a new broad model search. It tests the specific high-scoring pattern
`anchor + residual correction + hierarchical calibration` while preserving the current-row
inference rule.

## 2. Evidence and decisions

### 2.1 Fixed base

The base candidate is the accepted Tree Expert E2 handoff:

- artifact: `tree_expert_e2_handoff (1).zip`;
- SHA-256: `4dd0c9012b9a59e5235b76d6fe1f6ded4df4b404cab0082c0f3084b8c384050f`;
- model: R9 anchor plus three-seed CatBoost residual;
- temporal weighted Brier gain: `0.0008375137`;
- worst fold gain: `0.0003439904`;
- public score: `977.3809532715`;
- row-independence maximum error: `0` in the recorded audit.

E2 is candidate `C0` and remains the fallback if every new candidate fails.

### 2.2 Closed temporal branch

The completed T3 handoff is evidence, not a campaign input:

- artifact SHA-256: `5157f1d0922655f85345c296b6a4ad6b821349a008b17d0fc72687b7018d680b`;
- completed structure jobs: 12;
- selected decay/recent weight: `0.75 / 0.70`;
- weighted gain: `-0.0000505772`;
- 2021→2022 gain: `-0.0001582684`;
- 2023→2024 gain: `-0.0000457619`;
- maximum segment regression: `0.0015859681`;
- decision: rejected.

T3 is not rerun and its recent/multi-season blend is not silently reused.

### 2.3 Independent failure-label audit

The failure-label audit remains open as a CPU diagnostic. It runs independently and does
not block or alter this campaign. A passing `middle`, `reverse`, or `other_failure` audit is
saved for a later preregistered experiment; it is not added adaptively to C1 or C2.

## 3. Candidates

### C0: accepted E2

`p0` is the existing E2 three-seed probability. It is read from verified OOF evidence for
validation and produced by the verified E2 delivery at final inference.

### C1: hierarchical second residual

C1 learns a second correction to E2:

```text
residual_target = control_success - p0
p1 = clip(p0 + residual_model(features), 1e-5, 1 - 1e-5)
```

The residual model is CatBoost regression. It receives the existing E2 current-row feature
frame, `p0`, and rule-safe hierarchical features. It does not receive validation targets,
future rows, or any statistic derived from evaluation rows.

Three shrinkage profiles are screened with seed `3407`:

| profile | identity/player shrinkage | interaction shrinkage | purpose |
|---|---:|---:|---|
| `hc_strong` | strong | very strong | safest sparse-ID correction |
| `hc_balanced` | medium | strong | primary candidate |
| `hc_light` | light | medium | higher-variance boundary |

The JSON contract contains the exact numeric shrinkage constants. No unregistered profile
or interpolation is allowed after seeing validation or leaderboard results.

### C2: rolling hierarchical calibration

C2 applies a small additive logit correction to C1:

```text
logit(p2) = logit(p1) + global_effect
                         + game_type_effect
                         + player_effects
                         + stable_context_effects
```

Every effect is estimated from earlier OOF errors and shrunk toward its parent. Unknown or
sparse values fall back one level at a time and ultimately to zero. The calibrator is not
permitted to inspect a batch of evaluation rows.

## 4. Hierarchy and feature rules

The hierarchy is fixed from broad to narrow:

1. global;
2. `game_type`;
3. `pitcher_id` and `batter_id` reliability;
4. `pitcher_hand × batter_hand`;
5. `balls_before × strikes_before`, `outs_before`, and `base_state`;
6. `pitcher_id × game_type` and `batter_id × pitcher_hand` only above the fixed minimum count.

Target-derived hierarchy tables for a training row use seasons strictly earlier than that
row's season. Validation tables use only the outer fold's training period. Final inference
tables use official training data only. The provided `asof_*` columns remain current-row
features and are never reconstructed from other evaluation rows.

For a group with count `n`, raw estimate `r`, parent estimate `r_parent`, and fixed strength
`k`, the stored value is:

```text
shrunk_rate = r_parent + n / (n + k) * (r - r_parent)
```

The implementation stores counts and parent identity with every table so fallback and
provenance can be audited.

## 5. Leakage-free stacking and calibration

The existing E2 evidence covers the main 2022, 2023, and 2024 validation seasons. One
additional frozen-E2 fold `2020→2021` is created only to provide the earliest OOF error
source.

Rolling use is fixed as follows:

| prediction season | residual/calibration evidence allowed |
|---:|---|
| 2022 | 2021 OOF only |
| 2023 | 2021–2022 OOF |
| 2024 | 2021–2023 OOF |
| final evaluation | 2021–2024 OOF |

The same validation row never supplies its own residual or calibration effect. The final
C1 meta-model is trained on verified 2021–2024 OOF pairs and anchored by the E2 full model
at inference.

## 6. Selection and acceptance

### 6.1 Structure and seed protocol

- profile selection folds: `2021→2022`, `2022→2023`;
- sealed confirmation fold in this campaign: `2023→2024`;
- structure seed: `3407`;
- confirmation seeds: `42`, `2026`;
- main comparison uses paired rows against C0;
- uncertainty uses pitcher-cluster bootstrap, not independent row bootstrap.

### 6.2 C1 gates against C0

- weighted Brier gain `>= 0.00010`;
- 2024 fold gain `> 0`;
- minimum fold gain `>= -0.00005`;
- maximum eligible segment regression `<= 0.00035`;
- 95% pitcher-cluster bootstrap gain lower bound `> 0`;
- at least two of three seeds are non-worse on every fold.

### 6.3 C2 gates

C2 can survive even when C1 alone misses its acceptance gate, because the composite is
judged independently. C2 must satisfy all of the following:

- weighted Brier gain versus C0 `>= 0.00013`;
- incremental gain versus C1 `>= 0.00003`;
- minimum fold gain versus C0 `>= -0.00003`;
- global calibration gap no worse than C0;
- 10-bin expected calibration error no worse than C0;
- maximum eligible segment regression `<= 0.00020`;
- 95% pitcher-cluster bootstrap gain lower bound versus C0 `> 0`.

If both C1 and C2 pass, the lower weighted Brier wins. A Brier difference no larger than
`0.00002` selects simpler C1. Leaderboard scores do not refit calibration constants or
continuous weights.

## 7. Segment diagnostics

Only segments with at least 5,000 validation rows participate in the hard regression gate.
The review still reports smaller groups without using them for selection.

Hard-gated segment families are:

- validation year;
- `game_type`;
- known/unknown pitcher and batter identity;
- pitcher/batter hand matchup;
- count state;
- base occupancy state;
- prediction decile based on C0, with edges fit on training evidence.

## 8. Runtime plan

The campaign uses one generated Kaggle cell and one stage-aware resume format. Each Save
Version selects the next incomplete stage.

| stage | maximum wall time | work | output |
|---|---:|---|---|
| H1 | 6 hours | frozen E2 `2020→2021` source fold, three profiles, structure seed, 2024 confirmation | review + resume |
| H2 | 8 hours | selected profile, seeds 42 and 2026, rolling C1/C2 decisions | acceptance review + resume |
| H3 | 5 hours | accepted candidate only, final meta fit and calibration state | model delivery |

About five hours remain for interrupted-run recovery, inference verification, and a later
submission packaging step.

T4x2 runs two independent jobs at a time. Every completed job is immutable and reused after
resume. A stable snapshot is written at least every 15 minutes, but Kaggle does not trigger
repeated browser downloads.

## 9. Inputs and artifacts

H1 inputs are exactly:

- official `lg-aimers-9th-data` with fixed train/history hashes;
- accepted E2 handoff SHA-256 `4dd0c901...384050f`.

H2 and H3 add exactly one previous `tree_hierarchical_handoff.zip`. Expanded Kaggle dataset
copies and ZIP copies with the same manifest identity are deduplicated. Two distinct resume
identities fail before training.

Each Save Version publishes one `tree_hierarchical_handoff.zip`. It may contain:

- one verified resume;
- one review;
- acceptance evidence when decided;
- model delivery only after acceptance.

The handoff contains aggregate evidence and required OOF predictions but no evaluation data
or submission package.

## 10. Logging, interruption, and errors

Stable operator markers are:

```text
TREE_HC_STAGE_SELECTED stage=<H1|H2|H3>
TREE_HC_JOB_START job=<id> gpu=<id>
TREE_HC_TRAINING_PROGRESS job=<id> ...
TREE_HC_JOB_END job=<id> status=<completed|failed>
TREE_HC_DECISION candidate=<C1|C2> status=<accepted|rejected>
TREE_HC_ARTIFACT_READY path=<path>
TREE_HC_SUCCESS stage=<stage> handoff=<path>
TREE_HC_ERROR stage=<stage> type=<type> message=<message>
```

A binding mismatch, unsafe archive, non-finite prediction, row identity mismatch, deadline,
memory limit, or failed independence audit stops the stage. The error path may publish one
last verified resume but never a model delivery or submission artifact.

## 11. Rule and submission boundary

Inference for a row may use only that row, official training-derived tables, and persisted
models/states. Singleton, shuffled, rebatched, and duplicate-row audits must have maximum
probability error `<= 1e-12`.

Before model delivery:

- inference time must be at most 480 seconds;
- package-ready model state must be at most 2 GB;
- source and artifact hashes must match the accepted evidence;
- no evaluation `groupby`, rolling, lag, rank, mean, frequency, or batch-dependent transform
  is permitted.

This design does not authorize submission-package code. After H3, Codex must verify the
acceptance artifact, model delivery, hashes, inference limits, and independence evidence.
Only then may a separate packaging design and implementation create a submission ZIP.

The rule basis is the DACON evaluation-row independence notice and official Q&A: model
selection is allowed, but predictions may not use another evaluation row or evaluation-wide
statistics.

## 12. Testing and completion criteria

Implementation uses test-first development with synthetic fixtures for:

- strict contract parsing and identity;
- prior-season-only hierarchy construction;
- shrinkage and fallback;
- rolling OOF boundaries;
- paired Brier, calibration, cluster bootstrap, and segment gates;
- independent C1/C2 decisions and tie-break;
- resume deduplication and interrupted-stage reuse;
- deterministic archives and runtime embedding;
- singleton/shuffle/rebatch/duplicate-row invariance;
- prohibition of a submission artifact before accepted evidence.

The implementation is complete when the generated cell is deterministic and under Kaggle's
kernel source limit, focused and existing tree-expert regression tests pass, static boundary
checks find no evaluation aggregation path, and the user receives one exact H1 run operation.
Full-data training remains a user-run operation.
