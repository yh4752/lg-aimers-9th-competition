# Final Kaggle Failure Recovery Design

## Goal

Recover the exact source and tests used by the final `failure_regime_e3` Kaggle run, record what the run actually proved and where it stopped, preserve unrelated local work, and synchronize the source-only history to GitHub before writing the project narrative.

## Evidence boundary

The recovery uses three independent sources:

1. the surviving `codex/failure-regime-e3` worktree, which contains the original package, checked-in Kaggle cell, tests, and implementation plan;
2. `failure_regime_e3_review.zip` and `failure_regime_e3_handoff.zip`, which identify the earlier retryable screening failure;
3. the later `results.zip`, whose embedded runtime and campaign state identify the long Kaggle execution at notebook failure time.

Conversation memory is supporting context only. Artifact manifests, embedded source, state files, metrics, and the Kaggle terminal error are the result authority.

## Recovery structure

- Keep `experiments/failure_regime_e3/` and its tests from the surviving worktree.
- Compare every recovered runtime member with the corresponding source embedded in `results.zip`. A mismatch blocks the recovery commit.
- Keep the deterministic `KAGGLE_CELL.py` from the original worktree because output ZIPs contain the expanded runtime but not the notebook wrapper.
- Restore the original implementation plan and update the existing roadmap and experiment ledger rather than creating a second project history.
- Add one small diagnostic JSON containing artifact hashes, completed-job counts, terminal status, interruption reason, and evidence limitations.
- Extend `docs/EXPERIMENT_JOURNEY.md` with a plain-language final-run section suitable for later portfolio documentation.

## Result classification

The final run is `failed`, not `rejected` and not `accepted`.

- The later campaign state was still `running` in phase `extra_seeds`.
- It had 53 completed metric files: 14 screening, 7 confirmation, and 32 extra-seed jobs.
- Ten extra-seed jobs, recipe decision, full fit, inference audit, and submission packaging were not completed.
- Kaggle terminated the notebook after 33,677.4 seconds because the process attempted to allocate more memory than was available.
- Partial Brier values are diagnostic only. They cannot establish an accepted candidate or predict a leaderboard score.

The earlier review/handoff is recorded separately as a retryable screening attempt with four completed jobs and two `NoneType` conversion failures. It must not be confused with the later long execution.

## Git and artifact policy

Do not commit ZIP files, model binaries, predictions, CatBoost temporary files, snapshots, `__pycache__`, personal paths, or raw data. Commit source, tests, compact JSON evidence, and human-readable documentation only. Record SHA-256 values so the local artifacts can be reidentified.

The pre-existing modified TabM and gated-residual files on `main` are a separate change set. Verify and commit them separately; do not fold them into the E3 recovery commit. Preserve all unrelated user edits.

## Verification and synchronization

1. Run the original focused E3 tests and relevant competition-rule regression tests without full-data training.
2. Compile the recovered Python sources and parse the Kaggle cell.
3. Verify source parity against the embedded runtime in `results.zip`.
4. Verify diagnostic counts and hashes from the actual artifacts.
5. Commit E3 source/tests, then result documentation as separate commits.
6. Merge into local `main`, rerun the focused checks, review the separate existing dirty changes, and push only after the local history is coherent.

No heavy model run, submission package, automatic upload, or leaderboard claim is part of this recovery.
