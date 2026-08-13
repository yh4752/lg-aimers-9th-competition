# DACON Rules Compliance Guardrail Design

**Date:** 2026-08-13

**Repository:** `lg-aimers-9th-competition`

**Competition:** DACON official competition 236743

**Status:** approved design, implementation not started

## Purpose

Make official-rule compliance a required transition gate for every competition
experiment that can affect data, features, preprocessing, models, calibration,
ensembles, inference, or submission packaging. The system must prevent an
ineligible experiment from advancing and must prevent a submission ZIP from
being created when current compliance evidence is missing, stale, incomplete,
or inconsistent with current artifacts.

The guardrail protects the supported repository workflow. It does not claim to
prevent someone from manually creating or uploading an unrelated archive
outside the repository.

## Official rule sources

The initial policy version is `dacon-236743-2026-08-13` and is grounded in:

- Competition rules: <https://dacon.io/competitions/official/236743/overview/rules>
- Evaluation and code-submission guide:
  <https://dacon.io/competitions/official/236743/overview/evaluation>
- Evaluation-row independence notice:
  <https://dacon.io/competitions/official/236743/talkboard/417123?page=1&dtype=recent>
- Data description: <https://dacon.io/competitions/official/236743/data>
- Competition description:
  <https://dacon.io/competitions/official/236743/overview/description>
- DACON anti-cheating policy: <https://dacon.io/notice/notice/13>

Before any candidate becomes `package_ready`, the sources above and newer
official notices must be reviewed again. A new or changed rule increments the
policy version and invalidates package readiness produced under the previous
version. It does not erase historical experiment results.

Packaging additionally requires a review recorded on the same Korea Standard
Time calendar date as the package build. The review records the URLs, visible
official titles and update times when available, and the digest of the policy
summary derived from them. The package gate never claims that a locally stored
policy proves the absence of a newer official notice.

## Required rule interpretation

### Evaluation-row independence

The prediction for evaluation row A may depend only on:

- the input values in row A;
- features derived from row A alone;
- official training data supplied by the organizer;
- models, statistics, feature states, calibration parameters, and retrieval
  corpora fitted only from permitted official training data.

It must not depend on any other evaluation row, the number or order of
evaluation rows, the evaluation batch size, or the evaluation distribution.

Prohibited examples include:

- evaluation-set means, distributions, frequencies, ranks, or target-mean
  matching;
- evaluation-row groupby, cumulative, rolling, lag, or history features;
- aggregation across evaluation rows sharing a player, team, month, or game;
- evaluation-row attention, nearest-neighbor lookup, or retrieval;
- fitting scalers, encoders, calibration, BatchNorm state, or other state from
  evaluation rows;
- pseudo-labeling, transductive learning, or test-time adaptation;
- retaining an earlier evaluation row in mutable inference state;
- treating an apparently earlier evaluation row as history for a later row.
- hard-coded predictions, row-ID-to-probability lookup tables, or packaged
  prediction vectors that replace model inference for specific evaluation
  rows.

The historical XGBoost v3 `shift_mean` submission used the evaluation
prediction mean and is therefore not reusable. Its public-score record remains
historical evidence, but its record must carry `rules_review_required` and
`package_blocked: true`. Its archive and predictions must not be repackaged or
used as an ensemble component.

### Data, time, models, and APIs

- Only official Phase 2 data may be used as data. `train.csv` and
  `trackman_history.csv` are the permitted training sources.
- Features must be available before the predicted pitch or be fitted from
  permitted historical training data. Future-pitch or post-pitch information
  is prohibited.
- Pretrained weights are permitted only when they are publicly available and
  have a license allowing at least non-commercial use. The source, version,
  license, and weight SHA-256 must be recorded.
- Remote inference, remote model APIs, external feature APIs, and evaluation-
  time downloads are prohibited. All work must be locally reproducible.

### Submission runtime and structure

The generated submission must respect the current official limits:

- Python 3.11.15 on Ubuntu 22.04;
- NVIDIA L4 22.4 GiB, 6 vCPU, and 28 GiB RAM;
- package installation within 10 minutes;
- inference of 245,789 rows within 10 minutes;
- compressed archive at most 10 GB and extracted contents at most 32 GB;
- offline inference after package installation;
- UTF-8 source and comments;
- the exact official submission layout and required `submission.csv` output.

The official `baseline_submit.zip` is the source of truth for input paths and
archive layout when website wording is ambiguous. Compliance receipts are
stored beside the submission archive, never as extra top-level ZIP members.

### Manual competition operations

Team membership, duplicate registration, daily submission quota, deadline,
and the final file selected in the DACON UI are account- and platform-state
rules that repository code cannot prove. The packager never uploads a file or
claims these checks passed. Its handoff shows a mandatory human checklist for
the user to confirm in DACON immediately before upload. The current policy
records a maximum of five team members, no duplicate personal/team
registration, and at most five submissions per day. A failed or unknown manual
item blocks the user's upload decision, but it is not forged into a
machine-generated compliance receipt.

## Scope

The guardrail applies to changes that can affect:

- datasets and feature sources;
- preprocessing and fitted feature state;
- validation and temporal OOF;
- ML and DL models;
- calibration and post-processing;
- ensembles and retrieval;
- full-data training and inference;
- submission runtime and packaging.

It does not gate ordinary documentation edits, result summaries, or other
non-model repository maintenance. Research is not exempt when its output is a
candidate predictor: a method that cannot be deployed under the official rules
must not be implemented as a candidate experiment.

## Architecture

There is one policy package and one submission package:

```text
competition_rules/
├── policy.json       # versioned official sources and fixed policy values
├── contract.py       # experiment design contract validation
├── code_gate.py      # source policy and small row-independence checks
└── evidence_gate.py  # full-run evidence and artifact binding

submission/
├── contract.py       # package candidate and artifact contract
├── adapters.py       # closed registry of reviewed model-family adapters
├── audit.py          # final policy, evidence, hash, format, and runtime gate
├── runtime.py        # reviewed row-separable inference runtime
└── package.py        # sole DACON submission-archive creator
```

This is not a general Python sandbox. It is a fail-closed workflow for trusted
candidate code written in this repository. New model families use reviewed,
registered adapters; a manifest cannot select an arbitrary Python entry point.
At package time, the central packager renders the selected reviewed adapter and
runtime into the single required top-level `script.py`; it does not add another
top-level Python package to the official archive layout.

## Candidate lifecycle and gates

The existing lifecycle remains:

```text
planned → code_ready → waiting_for_user_run → passed → package_ready
                                         ↘ rejected
                                         ↘ failed
```

Rule eligibility is a required gate on each transition rather than a parallel
candidate status. A blocked transition reports `blocked_by_rules` with a
machine-readable reason. It must not be converted into a performance rejection
or an infrastructure failure.

### Design gate: `planned → code_ready`

Every new candidate has an `experiment_contract.json` before implementation.
It records at least:

```json
{
  "rules_version": "dacon-236743-2026-08-13",
  "data_sources": ["official_train", "official_trackman"],
  "fit_scope": "training_rows_only",
  "evaluation_scope": "current_row_only",
  "time_scope": "pre_pitch_only",
  "external_api": false,
  "pretrained_model": null
}
```

When pretrained weights are used, `pretrained_model` contains the immutable
source, version, license, and SHA-256. Unknown sources, ambiguous feature time,
evaluation-fitted state, evaluation-row interactions, external data, and remote
APIs fail the design gate. No candidate code is written until the contract
passes.

One contract may cover an explicitly enumerated campaign grid when every member
has the same data, fitting, time, API, and inference-isolation posture. The
contract then binds the immutable campaign-config hash and candidate IDs. A
candidate that changes any rule-relevant dimension, including a pretrained
checkpoint, retrieval corpus, feature source, or inference strategy, requires
a separate contract or a validated contract extension. This avoids thousands
of duplicate files without allowing an unlisted candidate to inherit approval.

A config-bound campaign may also declare deterministic child derivations such
as a listed seed confirmation or capacity-boundary expansion. The child passes
only when its approved parent, derivation type, derived model/training spec,
and child ID are reproduced by the already hash-bound campaign parser. A free-
form child manifest is never sufficient. Changing data, feature-time scope,
pretrained weights, retrieval, evaluation behavior, or APIs is not a permitted
derivation and requires a new contract.

### Code gate: `code_ready → waiting_for_user_run`

Before the user receives a full-data or GPU run:

- only the central runtime may read the evaluation table;
- registered predictors receive approved artifacts and feature batches, not an
  evaluation path or arbitrary entry point;
- source checks reject evaluation-table aggregation, evaluation-time fitting,
  network/API access, mutable cross-row state, direct evaluation-file reads in
  adapters, and row-specific hard-coded predictions;
- synthetic tests compare one-row, reordered, and differently batched
  inference;
- preprocessing and model state hashes must be unchanged by inference;
- DL must use deterministic evaluation mode; training-mode dropout or changing
  normalization state is rejected.

Failure keeps the candidate out of `waiting_for_user_run` and reports the
specific rule reason. It does not block independent candidates.

Static checks are defense in depth, not proof by themselves. Vectorized row-
local arithmetic, missing-value replacement, and column-wise transforms fitted
from training data are permitted even when they operate on a batch object.
Behavioral invariance and the registered adapter boundary are the primary
evidence. A compliant implementation blocked by a source heuristic must be
rewritten to make its row-local boundary explicit; the heuristic is never
bypassed by an exception flag.

### Evidence gate: `waiting_for_user_run → passed`

The user runs full-data preprocessing, temporal OOF, model training, and the
heavy independence audit. Evidence must bind:

- policy version;
- candidate, experiment, and model IDs;
- data, code, config, preprocessing, and model hashes;
- train-only temporal fitting evidence;
- feature source and availability-time evidence;
- performance metrics and acceptance booleans;
- row-independence audit results.

The row-independence audit uses the official five-row test sample and a sealed
full validation season. It compares:

- the normal production batch path;
- multiple batch sizes;
- reversed row order;
- a deterministic shuffled order;
- all rows predicted as singletons.

Predictions are canonicalized to the exact decimal strings that would be
written to `submission.csv`. The current design uses eight digits after the
decimal point. Every canonical prediction for a `row_id` must match exactly.
The audit is intentionally performed before packaging, where it may run longer
than the evaluator's 10-minute inference limit.

The heavy audit is restartable by immutable chunks because Colab and Kaggle
runtimes can disconnect. Each chunk binds the policy, candidate, validation
data, runtime, adapter, preprocessing, and model hashes plus its exact `row_id`
range and prediction digest. Publication requires every expected chunk exactly
once and a final whole-audit manifest. Missing, duplicated, reordered,
mismatched, or partially written chunks cannot count as evidence.

### Package gate: `passed → package_ready`

Immediately before packaging, the gate verifies:

- the current official rules and notices have been reviewed under the recorded
  policy version on the package-build date in Korea Standard Time;
- all evidence IDs, booleans, paths, and live SHA-256 values match;
- data and model provenance and licenses are complete;
- full row-independence evidence matches the exact runtime, adapter, model, and
  preprocessing artifacts being packaged;
- a clean evaluator-compatible installation and inference smoke test passed;
- a T4 or L4 end-to-end `script.py` benchmark is at most eight minutes,
  including model loading, live artifact checks, canaries, preprocessing, full
  prediction, and atomic CSV publication, leaving safety margin below the
  official ten-minute limit;
- RAM, VRAM, compressed size, extracted size, UTF-8, output schema, and archive
  layout all pass;
- no other repository entry point can create a DACON-format submission archive.

There is no `--force`, warning-only mode, manual exception, or alternate
packager. A failure creates neither a submission archive nor a partial
`submission.csv`.

## Inference runtime

The evaluator cannot run all 245,789 rows as singletons multiple times within
the ten-minute limit. Production inference therefore uses an audited
row-separable batch path:

- preprocessing is fitted only on training data and transforms rows without
  cross-row state;
- CatBoost and XGBoost use ordinary row-separable batch prediction;
- DL uses evaluation mode and may mix features within one row, but may not mix
  information across batch rows;
- frozen training BatchNorm state is allowed only when inference does not
  update it;
- retrieval may search only a hash-bound official-training corpus;
- calibration, segmentation, and ensemble weights are fixed before inference
  and operate on each row independently.

Deterministic per-row test-time augmentation or Monte Carlo prediction is
permitted only when every random seed is a pure function of the immutable
package seed and the current row ID, never row position, batch size, or another
evaluation row. It must pass repeat, reorder, rebatch, and singleton checks.
Ordinary stochastic inference without this property is rejected.

The evaluator runtime reloads the approved model, verifies the model and
preprocessing artifact hashes embedded by the packager, runs deterministic
batch-versus-singleton canaries on a fixed subset of the actual evaluation
rows, and then performs the full batch inference. Canary rows are used only to
test invariance; they do not select, fit, calibrate, or modify final
predictions. Any mismatch or state change stops before the output file is
atomically published. The external package receipt, rather than `script.py`
itself, binds the exact `script.py` and `requirements.txt` hashes; the runtime
does not attempt an impossible self-referential script-hash check.

The runtime must accept arbitrary positive evaluation row counts and may not
branch on the public five-row sample or the expected hidden row count. It
requires non-null unique row IDs and restores the official sample-submission ID
order exactly. Duplicate or missing IDs, unexpected columns, non-finite
probabilities, values outside `[0, 1]`, or an output row/column mismatch fail
before publication.

## Repository enforcement

### Documentation

`AGENTS.md` requires the design contract before candidate implementation,
requires the code gate before a user-run handoff, and forbids any submission
archive outside the central packager. `docs/EXPERIMENT_CONTRACT.md` contains the
official interpretation and lifecycle gates. This design is the detailed
source for their concise rules.

Documentation alone is not considered enforcement.

### Automated tests

Repository tests fail when:

- a new candidate experiment has no valid contract;
- an experiment entry point does not invoke the contract gate;
- a transition advances without the required gate evidence;
- a policy version is stale at package time;
- an adapter reads evaluation files or uses prohibited evaluation operations;
- packaged code contains hard-coded evaluation IDs, probabilities, or a
  precomputed prediction vector;
- a non-central path can create the official DACON archive structure;
- a package gate accepts missing, forged, mismatched, non-finite, duplicated,
  unsafe-path, or stale evidence;
- inference changes model or preprocessing state;
- order, batch size, or singleton execution changes a canonical prediction;
- the known XGBoost v3 mean-shift fixture is accepted;
- a violating candidate blocks an unrelated candidate.

Positive tests cover CatBoost, XGBoost, frozen-evaluation DL, fixed OOF
calibration, per-row ML/DL ensembles, per-row segmentation, and training-only
retrieval. They also cover row-local vectorized preprocessing, deterministic
per-row stochastic inference, missing values, unseen categories, arbitrary row
counts, and restart of an interrupted heavy independence audit.

### Rule changes

`policy.json` records the rule version, official URLs, review date, and a digest
of the reviewed policy summary. Updating the policy invalidates package
readiness under earlier versions. Historical scores, OOF metrics, and experiment
records remain available but cannot authorize a new package.

## Artifacts

Git stores only small contracts, policy files, tests, and evidence summaries.
Large predictions, models, full audit outputs, and submission archives remain
outside Git. A package receipt stored beside the archive records:

- policy version and official review date;
- candidate and model IDs;
- live hashes of all packaged artifacts;
- row-independence report hash;
- environment, install, inference, RAM, VRAM, and size results;
- archive SHA-256;
- `package_ready: true`.

The receipt does not authorize a changed archive. Any changed byte requires a
new gate run and receipt.

## Non-goals

The implementation does not add:

- a general-purpose Python sandbox or taint tracker;
- blanket bans on operations such as every call to `mean`;
- risk scores, warning tiers, or exception approvals;
- automatic leaderboard submission;
- false automation of team, quota, deadline, or DACON UI state;
- multiple packagers;
- unrelated refactoring of research code;
- heavy official-data or GPU runs performed by Codex without explicit approval.

Model capacity, search width, training time, feature count, and GPU use remain
performance decisions. The rules guardrail blocks only official-rule,
reproducibility, artifact-integrity, and evaluator-operability failures.

Expected development exceptions are handled fail-closed rather than waived:

- CUDA or library nondeterminism is resolved by a deterministic runtime or a
  different row-separable implementation before promotion;
- OOM and runtime excess are resolved by batching, mixed precision, loading,
  or model optimization without changing the approved prediction semantics;
- interrupted heavy audits resume only from fully verified immutable chunks;
- a safe model rejected by a static heuristic is refactored to expose the safe
  boundary and is retested;
- unknown licenses, rule interpretations, artifact provenance, or official
  notice freshness remain blocked until authoritative evidence is supplied.

No operational exception can turn missing evidence into a passing gate.

## Implementation boundary

Implementation is limited to this repository. The older `LG_AIMERS_2026`
repository is not another submission authority and is not modified by this
project. Current active experiment families are migrated to the design and code
gates without rewriting their training internals. Historical or completed
experiments remain readable; only current rule-compliant evidence may authorize
future execution promotion or packaging.

## Success criteria

The design is complete when all of the following are mechanically true:

1. New candidate code cannot advance without a valid current-version design
   contract.
2. A heavy user-run handoff cannot be produced until the code gate passes.
3. A candidate cannot become `passed` without complete performance, temporal,
   provenance, and full row-independence evidence.
4. A candidate cannot become `package_ready` when rules, evidence, hashes,
   runtime, memory, size, environment, format, or archive structure fail.
5. Only the central package module can create a DACON-format submission ZIP.
6. The evaluator script produces no `submission.csv` when its live checks fail.
7. Updating the official policy version invalidates old package readiness.
8. One candidate's failure does not stop independent rule-compliant research.
