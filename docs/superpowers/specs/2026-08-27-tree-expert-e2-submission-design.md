# Tree Expert E2 Submission Design

## Goal

Build one DACON-ready submission from the accepted `c1_anchor_residual` E2
CatBoost delivery. The package must use the repository's sole writer,
`submission/package.py`, and must refuse to write anything when the reviewed
artifact, acceptance evidence, runtime, or current policy identity differs.

## Accepted source identity

The only accepted source is `tree_expert_e2_handoff (1).zip` with SHA-256
`4dd0c9012b9a59e5235b76d6fe1f6ded4df4b404cab0082c0f3084b8c384050f`.
Its verified delivery has these fixed properties:

- campaign: `tree_expert_e2_v1`
- candidate: `c1_anchor_residual`
- predictor: `catboost`
- seeds: `42`, `2026`, `3407`
- acceptance: `accepted`, grade `breakthrough`
- weighted validation gain: `0.0008375137395728018`
- worst required-fold gain: `0.0003439904301012764`
- bootstrap interval: `[0.0006692336345317838, 0.0010344285313272623]`
- inference audit: 245,789 rows, all independence deltas zero
- measured inference: 64.939 seconds, 3,308,498,944 peak RSS bytes, no GPU

The three model hashes and the frozen-state member hashes are copied from and
verified against the delivery manifest. The builder must not accept a
look-alike artifact or a later artifact with a different hash.

## Package boundary

The final ZIP contains exactly these top-level paths:

```text
model/
script.py
requirements.txt
```

`model/` contains the three CatBoost models, frozen S1 lookup tables,
`feature_state.json`, and a candidate manifest that binds every member. The
submission does not contain training checkpoints, optimizers, evaluation
predictions, review bundles, logs, or source data.

The existing `submission/package.py` remains the only component that writes
`submit.zip`. E2-specific code verifies and imports the accepted delivery,
creates current acceptance evidence, and renders the reviewed runtime, but it
does not write a submission archive itself.

## Runtime

`script.py` is a plain Python entry point with no repository dependency. It:

1. locates `test.csv` and `sample_submission.csv` under the evaluator's
   `data/` directory, with `open/` accepted as an explicit compatibility path;
2. validates unique, non-null `row_id` values and exact test/sample ID parity;
3. loads the immutable training-derived S1 state and the three CatBoost models;
4. creates only current-row features and current-row lookups into frozen
   training statistics;
5. predicts a residual with each model, adds the frozen anchor, clips each
   probability to `[1e-5, 1-1e-5]`, and averages the three members;
6. restores sample-submission order and writes `output/submission.csv` with
   exactly `row_id,control_success`.

The runtime never fits on evaluation data and never uses evaluation-set
`groupby`, frequency, rank, rolling, lag, cumulative statistics, or a
distribution-dependent branch. It performs no network access and launches no
subprocess. CatBoost runs on CPU with at most six threads because the accepted
245,789-row audit is already comfortably below the evaluator limit.

## Evidence and fail-closed gates

Before the sole packager is called, the E2 builder must verify:

- exact outer handoff SHA-256 and safe ZIP structure;
- handoff status `accepted` and delivery presence;
- delivery manifest identity and all member size/hash declarations;
- acceptance decision, full-fit manifest, and inference-audit hashes;
- exact candidate, predictor, seeds, iterations, and three model hashes;
- `status=passed`, `reason=inference_audit_passed`, 245,789 audited rows, and
  zero maximum independence difference;
- runtime and memory evidence below the current safety limits;
- same-KST-day official-rules review;
- source inspection of the exact rendered `script.py`;
- Python 3.11.15 and submitted CatBoost version;
- prediction parity between the trusted E2 implementation and rendered runtime
  on the official five-row local sample;
- current singleton, reverse, shuffle, and rebatch parity on that sample;
- projected and final package size/layout/member hashes.

Any mismatch stops before `submit.zip` is created. Outputs are written into a
new directory and existing outputs are never overwritten.

## Dependency contract

`requirements.txt` contains only `catboost==1.2.10`. NumPy and pandas use the
official base environment versions and are not redundantly installed.

## Verification

Automated tests cover malformed/tampered handoffs, rejected evidence, stale
hashes, unsafe archive members, exact package layout, runtime source rules,
feature and prediction parity, row-order independence, singleton equality,
output schema/order, and deterministic archive hashes. The final build uses
the official Python 3.11.15 environment and the official five-row local sample;
the already accepted 245,789-row run is reused rather than repeated.

## Human handoff

The builder prints `TREE_E2_SUBMISSION_READY` with the archive path, SHA-256,
size, candidate ID, and source handoff SHA-256. The user still checks the DACON
account/team, remaining daily submissions, deadline, and selected file before
manual upload.
