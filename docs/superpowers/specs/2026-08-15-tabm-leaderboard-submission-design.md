# TabM Leaderboard Submission Design

**Date:** 2026-08-15

**Repository:** `lg-aimers-9th-competition`

**Competition:** DACON official competition 236743

**Status:** approved design, implementation not started

## Purpose

Prepare one rule-compliant leaderboard inference candidate from the reviewed
Version D TabM artifact. The supported workflow must keep the frozen model and
training-fitted preprocessing unchanged, prove that the evaluator runtime is
row-independent and compatible with the official environment, and create the
official ZIP only after every candidate-local gate passes with matching hashes.

This work prepares the Phase 2 leaderboard archive only. Phase 3 training code,
presentation material, and reproduction documentation remain separate later
deliverables. Their provenance must be preserved now, but they are not placed
inside the leaderboard ZIP.

## Decision

Use the official minimal archive shape:

```text
submit.zip
├── model/
│   ├── inference_manifest.json
│   ├── numeric_embedding_0.json
│   ├── preprocessing_state.json
│   └── tabm_member_0_seed_3407.pt
├── script.py
└── requirements.txt
```

The archive contains no parent directory, notebook, training code, result CSV,
review report, receipt, or cached evaluation prediction. `script.py` reads only
`./data/test.csv` and `./data/sample_submission.csv`, performs fixed inference,
and creates only `./output/submission.csv`.

The accepted source artifact is
`tabm_hand_matchup_stage_D_review_delivery (1).zip`. Its inner final-review
bundle is `review_ready`, not `passed`, so it is an immutable candidate input
rather than packaging authorization.

## Current official interpretation

The submission policy must be refreshed from these sources before acceptance:

- Competition rules: <https://dacon.io/competitions/official/236743/overview/rules>
- Evaluation and code-submission guide:
  <https://dacon.io/competitions/official/236743/overview/evaluation>
- Evaluation-row independence notice:
  <https://dacon.io/competitions/official/236743/talkboard/417123?page=1&dtype=recent>
- Competition FAQ and official comment answers:
  <https://dacon.io/competitions/official/236743/talkboard/417082?page=1&dtype=recent>
- Data description: <https://dacon.io/competitions/official/236743/data>
- Competition description:
  <https://dacon.io/competitions/official/236743/overview/description>
- DACON anti-cheating policy: <https://dacon.io/notice/notice/13>

The FAQ clarifies that there is no separate final-submission selection. The
highest-scoring compliant leaderboard submission is used, all submissions must
comply from the moment they are submitted, and the complete submission history
may be reviewed. A low-scoring or superseded submission is therefore not a safe
place to test questionable inference behavior.

The FAQ also confirms that fixed mappings, features, and aggregate statistics
derived only from official training data are allowed, including official
Trackman information within the permitted training period. Leaderboard scores
may guide model, hyperparameter, and ensemble-weight selection. None of these
permissions allows one evaluation row or the evaluation-set distribution to
affect another row's prediction.

Because the current stored policy predates this FAQ clarification, introduce a
new policy version and invalidate old `current_rules` acceptance evidence. The
frozen model files do not change merely because the policy evidence changes.

## Frozen candidate identity

The importer accepts exactly one outer delivery ZIP and verifies, without
trusting filenames alone:

1. outer delivery member set and member hashes;
2. inner review-bundle member set and manifest hashes;
3. `final_review.json` status is `review_ready`, version is `D`, fixed epochs
   are three, seed is `3407`, scheduler is `constant`, and review-only is true;
4. `frozen_inference/inference_manifest.json` declares exactly the three model
   state files and each declared SHA-256 matches its bytes;
5. fit scope is `official_train_2019_2024_only` and the frozen row count is
   1,475,092;
6. no optimizer, gradient scaler, RNG state, cached prediction, or undeclared
   member is imported.

It copies the four frozen inference files into a new candidate directory using
exclusive, atomic publication. It also records the source delivery hash, inner
bundle hash, imported file hashes, and combined model-directory digest. It
never overwrites a candidate directory and never creates a submission ZIP.

## Evaluator runtime

The production adapter ID is fixed to
`tabm_hand_matchup_version_d_seed3407_v1`. A closed registry maps that ID to:

- a local adapter factory used by audit code; and
- one reviewed, self-contained `script.py` renderer.

The generated script must not depend on repository modules that will be absent
from the evaluator. It contains only the minimum frozen preprocessing, TabM
construction, artifact verification, input validation, prediction, and output
publication code required for this candidate. Candidate metadata and expected
hashes are embedded deterministically at render time.

At startup the script:

1. requires CUDA and reports package/runtime versions;
2. resolves only `data/test.csv`, `data/sample_submission.csv`, and `model/`;
3. verifies the four model member names and SHA-256 values;
4. validates the exact test schema required by the frozen preprocessing state;
5. rejects null or duplicate `row_id` values and requires exact ID equality
   with the sample submission;
6. reconstructs only training-fitted `dl_standard + hand_matchup` state;
7. loads the single seed-3407 TabM state dict with `weights_only=True`;
8. predicts in bounded GPU batches without fitting or state mutation;
9. validates a one-dimensional, finite probability vector in `[0, 1]`;
10. writes rows in sample-submission order to a temporary UTF-8 CSV and then
    publishes `output/submission.csv` atomically.

The script fails instead of using default probabilities, placeholder rows,
automatic input discovery, CPU fallback, network access, or a cached
row-ID-to-prediction lookup.

## Evaluation-row independence

Permitted inputs to one prediction are the values in that evaluation row and
constants frozen from official training data. The runtime may vectorize rows
in a GPU batch, but it must not compute evaluation-set means, standard
deviations, frequencies, ranks, groups, rolling state, calibration parameters,
or OOV statistics.

The locally supplied official `test.csv` contains five sample rows. The hidden
245,789-row evaluation frame exists only inside the DACON evaluator and cannot
be audited before submission. Evidence must state this limitation plainly.

The pre-submission audit uses the actual candidate script or byte-identical
runtime logic on all five official sample rows. It compares original order,
reverse order, deterministic shuffle, several batch partitions, and every row
as a singleton. Preprocessing features must be bit-identical by `row_id`,
probabilities must agree within the documented floating-point tolerance, and
the adapter state digest must not change.

Capacity testing uses a deterministic 245,789-row scale fixture made by
repeating the five official sample rows and assigning unique synthetic
`row_id` values. This fixture measures runtime and memory only; it is not
described as hidden-test accuracy or full hidden-test independence evidence.

The runtime also runs deterministic singleton canaries before writing the final
CSV. A canary mismatch blocks output publication.

## Compatibility and capacity validation

The existing Version D review ran under Python 3.12.13, PyTorch 2.11.0,
NumPy 2.0.2, and pandas 2.2.2. It cannot prove official-environment
compatibility. Current Colab and Kaggle GPU images may also use Python 3.12, so
requiring the host kernel itself to match the official environment would make
the validation handoff unusable.

Use two complementary probes. First, create an isolated Python 3.11.15
environment with PyTorch 2.7.1 CPU, pandas 2.0.3, and NumPy 1.26.4. In that
environment, install the two submitted requirements, deserialize the real
checkpoint, and produce the deterministic five-row prediction. This proves
Python and library compatibility without pretending that a Colab or Kaggle
image is the evaluator. Second, run the complete five-row independence audit
and the 245,789-row synthetic capacity fixture on the T4 GPU host, recording
its actual Python, PyTorch, CUDA, pandas, and NumPy versions. Compare the
exact-environment CPU probe and GPU-host five-row probabilities within the
documented floating-point tolerance.

The user runs one complete validation cell on a GPU service. The cell receives
the candidate handoff and official data as uploaded inputs, installs only the
two non-default dependencies, and produces a downloadable review ZIP. It never
creates `submit.zip`.

Validation must record:

- exact-probe and GPU-host Python, OS, CUDA, GPU, PyTorch, pandas, NumPy, TabM,
  and numerical-embedding versions;
- dependency installation command, return code, elapsed seconds, and logs;
- successful checkpoint deserialization and a deterministic five-row probe;
- five-row official sample audit and 245,789-row synthetic capacity time, peak
  RAM, peak allocated VRAM, and output digest;
- exact output schema, row count, row order, and probability checks;
- restartable five-row row-independence audit evidence labeled
  `official_sample_plus_synthetic_scale`;
- candidate, model, adapter, rendered-runtime, preprocessing, config, and data
  hashes.

The acceptance threshold is stricter than the platform limit: submitted
dependency installation must be at most 480 seconds, full inference at most
480 seconds, peak RAM below 22 GiB, peak allocated VRAM below 20 GiB, ZIP
projection below 10 GB, and extracted projection below 32 GB. A T4 runtime
passing the time gate is acceptable conservative performance evidence for the
faster official L4, but the report must label both hardware and host-runtime
mismatches rather than claim exact equivalence. A version mismatch in the
isolated Python 3.11.15/PyTorch 2.7.1 probe, checkpoint-load failure, or
five-row prediction disagreement blocks compatibility acceptance.

The validation operation is rerun-safe: completed audit phases and their
prediction shards are reused only after identity and content-hash verification.
New evidence paths are exclusive, and an interrupted run exports a resume
bundle rather than replacing previous evidence.

## Acceptance and packaging gates

Candidate acceptance is a separate explicit operation after the validation ZIP
is returned and reviewed. It creates `acceptance.json` only when all seven
existing gates pass:

- temporal validation;
- performance;
- provenance;
- row independence;
- evaluator runtime;
- pretrained license;
- current rules.

The acceptance identity binds the refreshed policy version, official data,
candidate source, frozen preprocessing, combined model directory, production
adapter, and exact rendered script. The runtime benchmark and complete audit
manifest must carry the identical identity.

`submission/package.py` remains the sole archive writer. It recomputes live
hashes, requires a same-KST-date official policy review, verifies acceptance,
full audit, compatibility benchmark, minimal requirements, exact archive member
set, and output paths, and renders the script again. It must compare the newly
rendered script hash with the accepted runtime hash before writing any archive
member.

The deterministic archive is published only to a new path. A receipt containing
the archive hash and bound identity is stored beside the ZIP, never inside it.
The packager never uploads to DACON.

## Requirements policy

`requirements.txt` contains only:

```text
tabm==0.0.3
rtdl-num-embeddings==0.0.12
```

It does not reinstall default evaluator packages such as torch, pandas, NumPy,
or CUDA components. The audit rejects extra indexes, URLs, editable installs,
local paths, environment markers, unpinned versions, and extra dependencies.

## Explicit non-goals

- Do not retrain, calibrate, ensemble, or alter the Version D model.
- Do not include Trackman files at inference; permitted training-derived state
  is already frozen in the preprocessing artifact.
- Do not add input-path fallbacks for `open/` or recursive data discovery.
- Do not make CPU inference a hidden fallback.
- Do not generate or upload a leaderboard ZIP before acceptance.
- Do not mix the later Phase 3 reproduction package into this archive.
- Do not modify unrelated Stage C recovery files currently dirty in the worktree.

## Success criteria

The work is complete only when:

1. policy and FAQ interpretations are current and same-day reviewed;
2. the frozen delivery imports with all hashes and lineage verified;
3. production adapter and rendered script pass synthetic fail-closed tests;
4. the user-run Python 3.11/PyTorch 2.7.1 validation report passes all runtime,
   output, independence, and capacity gates;
5. reviewed evidence produces a hash-bound `acceptance.json`;
6. the sole packager creates the exact official layout and immediately
   re-verifies it;
7. an isolated dry run of the packaged `script.py` creates exactly one valid
   `output/submission.csv` from official input paths;
8. no unrelated user changes are altered or committed.
