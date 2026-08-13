# TabM Champion Campaign and DACON Submission Design

**Status:** Conversationally approved; pending written-spec review
**Competition:** DACON 236743, Aimers 9기 투구 제구 성공 확률 예측 AI 온라인 해커톤
**Rules snapshot date:** 2026-08-13 (Asia/Seoul)
**Scope:** Fixed preprocessing, TabM-only model research, candidate verification, and fail-closed submission preparation

## 1. Objective

Build the strongest defensible TabM candidate that can be trained within an
aggregate T4 x2 research budget of ten hours and executed on the official
evaluation server without violating row-independent inference or package
requirements.

The semantic preprocessing contract is frozen:

- preprocessing profile: `dl_standard`
- only added component: `hand_matchup`
- `hand_matchup = pitcher_hand + "_" + batter_hand`
- no Trackman augmentation in this candidate
- no evaluation-derived feature, statistic, calibration, or state

The campaign may vary TabM architecture, numerical embeddings, loss,
optimization, and seed. It may not reopen feature engineering or introduce a
second model family.

## 2. Official constraints

The implementation must enforce the stricter interpretation whenever official
materials could be read in more than one way.

### 2.1 Allowed information

- Official `train.csv` may be used for fitting preprocessing, category maps,
  numerical bin edges, models, early stopping, selection, and fixed ensemble
  decisions.
- Official `test.csv` may be used only as independent input rows to the frozen
  inference function.
- Official `trackman_history.csv` is an allowed competition file but is outside
  this fixed-preprocessing candidate.
- Official `asof_*` input columns are permitted row attributes.

External data, remote APIs, remote inference, pseudo-labeling, test-time
adaptation, and test-time downloads are prohibited. This candidate uses no
pretrained weight.

### 2.2 Row independence

For an evaluation row A, the prediction may depend only on:

1. the input values in row A;
2. row-local deterministic derivations such as `hand_matchup`;
3. immutable state fitted from official training rows; and
4. immutable TabM weights fitted from official training rows.

It must not depend on another evaluation row, evaluation row count, evaluation
order, evaluation distribution, or mutable state left by a previous prediction.

Prohibited evaluation operations include groupby aggregation, frequency,
ranking, quantiles, mean or standard-deviation fitting, rolling, expanding,
lag, cumulative state, target encoding, retrieval from other test rows,
calibration, and global prediction adjustment.

### 2.3 Metric

The leaderboard metric is Brier Skill Score:

`max(0, 100000 * (1 - Brier / reference_Brier))`.

The reference term is fixed for the hidden evaluation labels, so offline model
selection minimizes Brier Score directly. Public uses 100% of evaluation rows;
Private is the Public score at competition close. Leaderboard feedback must not
be used to fit weights, calibration, hyperparameters, or preprocessing.

### 2.4 Submission contract

The archive must have exactly this top-level layout:

```text
submit.zip
├── model/
├── script.py
└── requirements.txt
```

The evaluation server adds `data/` and `output/`, executes
`python script.py`, and requires `output/submission.csv`. The result columns
must be exactly `row_id` and `control_success`, with one finite probability in
`[0, 1]` for each test row and exact test `row_id` coverage.

The submission script uses `data/test.csv` and `data/sample_submission.csv`.
The generic guide and the competition evaluation page both define `data/`; a
single stray `open/` wording in the competition notes is not used as the server
contract.

Official resource limits:

- compressed archive: at most 10 GB
- extracted archive: at most 32 GB
- package installation: at most 10 minutes
- inference: at most 10 minutes
- Python 3.11.15
- Ubuntu 22.04.5 LTS
- NVIDIA L4, 22.4 GiB VRAM
- 6 vCPU
- 28 GB system RAM
- CUDA 12.8
- no internet after package installation
- hidden evaluation size: 245,789 rows

Internal safety targets are stricter:

- inference at most 8 minutes on the T4 dry run
- peak GPU memory at most 20 GiB
- peak system RAM at most 22 GB
- compressed package target below 2 GB
- installation at most 8 minutes in a clean Python 3.11 environment

## 3. Candidate search space

### 3.1 Fixed input representation

Every candidate receives the same train-fitted arrays produced by
`PreprocessingSpec("dl_standard", ("hand_matchup",))`.

For each temporal fold:

- medians, means, and standard deviations are fitted only on the fold's
  training years;
- categorical maps are fitted only on the fold's training years;
- piecewise bin edges are fitted only on the fold's training arrays;
- validation rows use the frozen fold state;
- unseen categories map deterministically to the reserved index;
- row count, order, and `row_id` are preserved.

### 3.2 Capacity levels

| Name | `k` | width | blocks | default dropout | Purpose |
|---|---:|---:|---:|---:|---|
| P2 | 32 | 512 | 4 | 0.10 | Confirmed reference |
| P3-lite | 32 | 768 | 6 | 0.15 | More capacity without doubling `k` |
| P3-full | 64 | 768 | 6 | 0.15 | High-capacity boundary candidate |

P3-full is allowed into final consideration only if its measured inference cost
fits the package budget. Search success alone cannot override the runtime gate.

### 3.3 Numerical embeddings

- `piecewise_linear`
- `periodic`

Train-fitted piecewise edges or periodic configuration are serialized in the
model artifact. Test values never refit embedding state.

### 3.4 Losses

- binary cross entropy on member logits
- Brier loss on member-mean probabilities
- equal-weight BCE plus Brier hybrid

All candidates report Brier regardless of training loss.

### 3.5 Optimization refinement

The broad screen uses learning rate `0.0006` with AdamW, weight decay `0.0001`,
effective batch size 4096, micro-batch size 512, and AMP enabled. After the
winning structural configuration is known, refinement compares learning rates
`0.0003`, `0.0006`, and `0.0009`, and dropout values centered on the winning
default at offsets `-0.05`, `0`, and `+0.05`, clipped to `[0.0, 0.30]`.

The nine refinement combinations first run at proxy fidelity. Only the two best
advance to full primary-fold validation.

### 3.6 Seeds and ensembles

The final structure uses seeds `42`, `2026`, and `3407`.

Eligible final predictors are:

- best single seed;
- equal mean of the best two seeds;
- equal mean of all three seeds; and
- equal mean of the two best distinct structures at seed 42.

No learned stacking weight, leaderboard-derived weight, test-derived weight, or
post-hoc mean alignment is allowed. A final predictor contains at most three
TabM weights.

## 4. Multi-fidelity campaign

Each stage is a separate restartable Kaggle Save Version. A stage has an
absolute wall-time deadline and a ten-minute finalization reserve. The same
generated cell detects the hash-validated resume bundle and advances exactly
one stage.

### 4.1 Version A: broad proxy screen

**Budget:** at most 2 hours wall time on T4 x2.

The screen evaluates 18 combinations:

`3 capacities * 2 numerical embeddings * 3 losses`.

- Training years: through 2023.
- Model-training sample: deterministic 25% hash sample of training `row_id`.
- Preprocessing fit: all allowed training rows through 2023, not the sample.
- Validation: all 2024 rows.
- Maximum: 8 epochs.
- Minimum before stopping: 3 epochs.
- Patience: 3 completed validation epochs.

The deterministic sample is selected only by a published hash of `row_id` and
the campaign seed. It does not use the target or evaluation data.

Four survivors advance:

1. best overall proxy Brier;
2. best P2 candidate;
3. best P3-lite or P3-full candidate not already selected; and
4. best candidate with a different loss or numerical embedding from the best
   overall candidate.

This diversity rule prevents a noisy proxy from eliminating an entire useful
design axis.

### 4.2 Version B: full temporal validation

**Budget:** at most 3 hours wall time on T4 x2.

The four survivors run on all training rows through 2023 and validate on all
2024 rows:

- maximum 30 epochs;
- minimum 3 epochs;
- patience 8;
- epoch-level atomic checkpoint;
- validation predictions saved for every completed candidate.

The best two structures, plus the original P2 reference if it is not already
among them, are checked on training through 2022 and validation on 2023 with the
same stopping rules. At most three structures therefore run on the older fold.

Candidates are compared by fold-relative Brier delta against P2:

`selection_delta = 0.70 * delta_2024 + 0.30 * delta_2023`.

A candidate cannot become champion if it worsens 2024 by more than `0.00005`,
worsens 2023 by more than `0.00010`, or has a positive weighted delta.

### 4.3 Version C: refinement, stability, and ensemble evidence

**Budget:** at most 3 hours wall time on T4 x2.

The nine learning-rate and dropout refinements first use the Version A proxy
protocol. The best two run the full 2023-to-2024 fold. The refined champion is
then trained with seeds 42, 2026, and 3407 on the full primary fold.

If the three-seed mean improves primary-fold Brier, its direction is confirmed
on the 2022-to-2023 fold. The additional older-fold seed runs stop as soon as
the ensemble verdict is determined or the stage deadline is reached.

Version C compares the four eligible predictors listed in section 3.6. Equal
averaging is performed row by row and is therefore row independent.

### 4.4 Version D: final fit and evaluation-server simulation

**Budget:** at most 2 hours wall time on T4 x2.

For each selected final member, the epoch count is the rounded median of its
`best_epoch + 1` values from completed full temporal folds and seeds, clipped to
`[2, 30]`. That rule is fixed before final training.

Final preprocessing fits on all 2019-2024 official training rows. Final TabM
weights train on the same rows for the fixed epoch count without validation or
test access.

Version D performs:

- inference on the official five-row sample;
- a 245,789-row synthetic workload made only by repeating the public sample
  rows, used for timing and memory measurement;
- singleton, full-batch, reversed-order, shuffled-order, and changed-batch-size
  invariance checks;
- state hash checks before and after inference;
- exact output schema and row coverage checks;
- clean package-install simulation; and
- archive size and extracted-size checks.

Version D produces a final review bundle, not `submit.zip`.

## 5. Selection evidence

Every completed candidate records:

- overall Brier;
- best epoch and complete learning curve;
- elapsed training and inference time;
- peak CPU RAM and GPU VRAM;
- predictions aligned by `row_id`;
- `game_type` segment Brier;
- `game_month` segment Brier;
- pitcher-OOV and batter-OOV segment Brier;
- pairwise prediction and squared-error correlation for ensemble candidates;
- fold-relative delta against P2; and
- exact config, code, data-row, preprocessing-state, and prediction hashes.

Segments with fewer than 1,000 rows are reported but do not trigger a hard
gate. A segment with at least 1,000 rows and regression greater than `0.00050`
against P2 blocks automatic promotion and requires explicit review. Smaller
segment regressions are advisory and weighed against both temporal folds.

The previously tested linear OOF calibration is excluded. Fitting it on 2023
OOF changed 2024 Brier from `0.248097409196` to `0.248611328276`, a regression
of `0.000513919080`.

## 6. GPU scheduling and deadlines

Each T4 runs one independent process. The campaign does not use DDP and does
not split one model across two GPUs. Jobs are assigned deterministically by
candidate ID and expected cost.

The scheduler must:

- print a heartbeat at least every five percent of an epoch;
- checkpoint only at completed epoch boundaries;
- preserve the last completed epoch before a deadline;
- mark a deadline-stopped candidate `inconclusive` when selection evidence is
  insufficient;
- never silently convert a deadline stop into `pending` and retry forever;
- resume only when the candidate config, code hash, data hash, and checkpoint
  hash all match; and
- reserve ten minutes to seal artifacts and bundles.

`failed` blocks only the failed candidate. `inconclusive` cannot win but does
not block independent candidates. A complete stage can advance only from
completed evidence.

## 7. Logs and artifacts

Required log milestones:

```text
RULES_GATE_PASSED
STAGE_START stage=...
JOB_START candidate=... gpu=...
TRAINING_PROGRESS candidate=... epoch=... batch=...
EPOCH_CHECKPOINTED candidate=... brier=...
JOB_COMPLETE candidate=... best_brier=...
STAGE_DECISION survivors=...
ROW_INDEPENDENCE_PASSED
BUNDLE_SUCCESS review=... resume=...
```

Errors use:

```text
TABM_CAMPAIGN_ERROR stage=... candidate=... type=... message=...
```

Versions A-C emit:

- `tabm_search_stage_XX_review_bundle.zip`
- `tabm_search_stage_XX_resume_bundle.zip`

The review bundle contains a canonical manifest, candidate metrics, learning
curves, segment metrics, resource use, decisions, rules evidence, validation
predictions, and the full log. The resume bundle contains only hash-validated
state, completed evidence, and checkpoints required to resume or advance.

Version D emits:

- `tabm_hand_matchup_final_review_bundle.zip`

It includes frozen preprocessing state, model configuration, model metadata,
one to three weights, resource benchmarks, row-independence evidence, dependency
install evidence, and SHA-256 for every member.

Bundles are written atomically and use fixed timestamps and sorted members.
Existing immutable bundles are never overwritten.

## 8. Inference and package implementation

Research code may contain fit functions. Production `script.py` may not. It
contains only:

1. local imports;
2. manifest and artifact hash validation;
3. exact `data/test.csv` and `data/sample_submission.csv` loading;
4. frozen row-local transformation;
5. frozen TabM reconstruction and weight loading;
6. batched probability inference in `eval()` and inference mode;
7. row-ID alignment; and
8. atomic `output/submission.csv` publication.

The model directory contains only the reviewed weights, frozen preprocessing
state, frozen model metadata, configuration, and manifest. No training data,
validation predictions, checkpoints with optimizer state, cache, notebook, or
log is included.

`requirements.txt` initially contains only exact TabM-specific dependencies:

- `tabm==0.0.3`
- `rtdl-num-embeddings==0.0.12`
- `rtdl-revisiting-models==0.0.2`

Packages listed as preinstalled by the official server, including PyTorch,
pandas, NumPy, SciPy, and scikit-learn, are omitted. If a clean install test
shows that these exact three lines cannot meet the eight-minute internal limit
without changing preinstalled packages, packaging stops for a reviewed
dependency strategy; it does not silently vendor or change versions.

## 9. Verification gates

### 9.1 Code and rules

- The current DACON policy and official-source review must match.
- The experiment contract allows only this candidate family and fixed feature
  derivation.
- Static checks reject prohibited test-wide operations in inference paths.
- Inference imports no research runner or training entry point.
- No network URL, API client, downloader, or pretrained loader is reachable.

### 9.2 Row independence

For canonical eight-decimal probabilities, all of the following must match by
`row_id`:

- row alone;
- row in the full batch;
- reversed batch;
- deterministic shuffled batch;
- batch sizes 1, 257, 2048, and the production batch size.

The preprocessing and model state digest must be unchanged after every path.

### 9.3 Output

- exact two-column schema;
- exact test-ID set and sample-submission order;
- no null, infinity, or probability outside `[0, 1]`;
- exact 245,789 rows in the full-scale simulation;
- atomic output with no partial file left after failure.

### 9.4 Packaging

- current candidate status is accepted;
- every required evidence gate passes;
- artifact hashes equal the reviewed Version D hashes;
- same-day official policy review passes in Asia/Seoul;
- only `model/`, `script.py`, and `requirements.txt` are at archive root;
- compressed, extracted, install, inference, VRAM, and RAM limits pass.

No code path may create `submit.zip` before all gates pass. A rejected candidate
blocks only its own packaging.

## 10. User workflow

The user performs full-data and GPU execution. Codex writes and reviews code,
runs syntax and dependency-free checks, and supplies one complete Kaggle cell.

For each Save Version, the user returns both the review and resume bundle. Codex
reviews the stage decision before the next stage is run. After Version D, the
user returns only the final review bundle. Codex validates it and may then open
a separate, explicit packaging task.

The Kaggle handoff must state purpose, required input datasets, expected files,
runtime, rerun safety, success text, and exact error text to return.

## 11. Official sources

- Rules: https://dacon.io/competitions/official/236743/overview/rules
- Evaluation and submission specification: https://dacon.io/competitions/official/236743/overview/evaluation
- Data: https://dacon.io/competitions/official/236743/data
- Row-independence notice: https://dacon.io/competitions/official/236743/talkboard/417123?page=1&dtype=recent
- Rules reminder: https://dacon.io/competitions/official/236743/talkboard/417094?page=1&dtype=recent
- Generic code-submission guide: https://cfiles.dacon.co.kr/competitions/236564/guide.html
- Anti-cheating policy: https://dacon.io/notice/notice/13
