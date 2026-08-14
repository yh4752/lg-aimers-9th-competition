# Colab TabM Stage C Recovery Design

## Purpose

Resume the incomplete TabM Stage C campaign on one Colab T4 without Google
Drive. Preserve all 20 trusted completed Stage C jobs, rerun only the final
incomplete `s3407 / 2022->2023` job, provide continuous logs, and limit work
lost after a Colab VM reset by exporting verified emergency checkpoints to the
user's computer.

This workflow creates research review and resume artifacts only. It does not
create a competition submission package.

## Fixed Inputs

The first run accepts exactly these uploads:

1. `lg-aimers-9th-data.zip`, containing exactly one copy of:
   - `train.csv`
   - `trackman_history.csv`
   - `test.csv`
   - `sample_submission.csv`
2. `tabm_search_stage_C_resume_bundle.zip` with SHA-256
   `6f7cc5b3c8280551f266767f63e71667db05473846e262371803a1fe1272d806`.

A reset recovery run may additionally accept one or more
`tabm_colab_emergency_epoch_*.zip` files. It selects the highest compatible
epoch. Two snapshots claiming the same epoch with different hashes are an
error.

The local data directory is packaged as one deterministic ZIP before handoff.
The handoff records each source file's SHA-256, uncompressed size, and ZIP
member name. The Colab cell embeds and enforces those values.

## Trust and Reproducibility Boundary

The Stage C resume contains 20 completed jobs and one incomplete job. Completed
job metrics and predictions remain immutable and are reused after manifest
verification.

The incomplete Kaggle checkpoint ended after epoch 2 but does not bind the
exact Kaggle PyTorch and CUDA runtime versions. The Colab workflow therefore
does not continue its optimizer state. It removes only that incomplete job's
training-file bindings from the imported resume and starts that one job at
epoch 0. No completed job is retrained. The expected duplicated work is three
epochs, approximately four to six T4 minutes.

Metrics from that incomplete Kaggle attempt remain in the immutable source
archive for audit, but are not eligible for the Stage C decision. Only the
clean Colab rerun's verified result may complete the pending job and update the
stage decision.

Once training begins in Colab, emergency snapshots bind all continuation state
to:

- base Stage C resume archive SHA-256 and manifest SHA-256;
- candidate ID and completed epoch;
- campaign config SHA-256;
- embedded runtime SHA-256 and training-source SHA-256;
- preprocessing cache SHA-256;
- Python, PyTorch, CUDA runtime, NumPy, pandas, TabM, and
  `rtdl-num-embeddings` versions;
- GPU model and checkpoint member SHA-256 values.

A later Colab runtime must match the bound environment and artifact identities
before an emergency checkpoint is restored. A mismatch stops before training;
it never silently falls back to an incompatible checkpoint.

## One-Cell Architecture

The repository provides a generated `COLAB_STAGE_C_RECOVERY_CELL.py`. The user
copies the entire file into one Colab cell. The cell contains a deterministic,
compressed runtime payload and performs these phases:

1. **Local-state discovery**
   - Reuse verified files already under the fixed `/content` work root.
   - Prompt for uploads only when required inputs are absent.
2. **Input ingestion**
   - Save uploaded bytes under fixed names and immediately release the upload
     mapping to avoid retaining a second large in-memory copy.
   - Verify archive and member SHA-256 values before extraction.
3. **Secure data extraction**
   - Reject absolute paths, `..`, symlinks, duplicate names, unexpected files,
     excessive member counts, and excessive total uncompressed bytes.
   - Publish the extracted data directory atomically only after all four files
     pass their hashes.
4. **Resume preparation**
   - Verify the official Stage C resume and its prior/config bindings.
   - On the first Colab run, produce a sanitized resume that removes only the
     incomplete Kaggle training checkpoint.
   - On reset recovery, merge the newest compatible emergency snapshot into
     the sanitized base resume and regenerate a valid resume manifest.
5. **Runtime preparation**
   - Install only the declared campaign dependencies.
   - Require exactly one visible CUDA T4 for this handoff.
   - Record all environment versions before training.
6. **Campaign subprocess**
   - Launch the normal Stage C runner with an explicit visible GPU count of one.
   - Stream every stdout/stderr line to both the cell and an append-only log.
7. **Checkpoint monitor**
   - Watch the target job's atomically written `checkpoint_meta.json`.
   - Export only a complete, hash-stable epoch checkpoint.
8. **Final delivery**
   - Verify generated Stage C review/resume bundles.
   - Package both bundles, the log, and a delivery manifest into one ZIP.
   - Request a browser download and preserve the individual artifacts under
     `/content` for manual fallback.

The embedded runtime and input files persist across a cell interruption in the
same VM. A rerun skips upload, extraction, and dependency installation when
their receipts and live hashes still match.

## Single-GPU Scheduling

The campaign runner currently hard-codes two workers. It will receive an
explicit `gpu_count` parameter with a default of two so existing Kaggle behavior
does not change. The Colab cell passes one. The subprocess scheduler may launch
only GPU index 0, and tests reject attempts to schedule GPU index 1.

Each campaign job was already trained on one GPU, so changing the number of
concurrent workers does not change a job's model architecture, batch size,
seed, or optimization contract. This handoff has only one pending job.

## Emergency Snapshot Contract

The monitor creates a snapshot after a newly completed epoch when at least 20
minutes have passed since the last exported snapshot. It also attempts a final
snapshot on a handled interrupt or campaign error.

Each snapshot has a unique epoch-bearing filename and contains only:

```text
emergency_manifest.json
training/<candidate_id>/checkpoint.pt
training/<candidate_id>/checkpoint_meta.json
training/<candidate_id>/best_checkpoint.pt
logs/colab_stage_C.log
```

The snapshot is written to a temporary path, verified, and atomically renamed.
Existing snapshots are never overwritten or deleted while the cell runs,
because a browser download may still be reading them.

`google.colab.files.download()` is only a download request; Python cannot prove
that the browser saved the file. Therefore each export prints:

```text
EMERGENCY_SNAPSHOT_READY epoch=<n> path=<path> sha256=<sha256>
EMERGENCY_DOWNLOAD_REQUESTED path=<path>
```

The cell also renders a manual download link and tells the user to confirm the
file appears in the local Downloads directory. The recovery guarantee extends
only through the newest snapshot actually present on the user's computer.

## Process and Interrupt Safety

The cell owns one campaign subprocess and records its PID, process start
identity, and run UUID in a lock receipt.

- A second invocation refuses to start while the recorded matching process is
  alive.
- A stale lock with no matching process is archived and replaced.
- A handled notebook interrupt stops new scheduling, asks the child to finish
  its current complete checkpoint boundary, and waits for at most 180 seconds.
  If a new stable checkpoint appears, it exports that checkpoint; otherwise it
  exports the previous stable checkpoint when one exists. It then terminates
  the process group and verifies that the recorded process identity is gone.
- A forced VM deletion or `SIGKILL` cannot run cleanup; recovery then uses the
  newest snapshot already downloaded to the user's computer.

The cell never kills an unrelated process based only on a reused PID.

## Logs and Outputs

All phases emit structured, flushed log lines. Important success lines are:

```text
COLAB_INPUTS_VERIFIED ...
COLAB_GPU_READY device_count=1 name=Tesla T4
COLAB_RESUME_READY source=stage_C_base|emergency epoch=<n-or-zero>
STAGE_SELECTED version=C
JOB_START job=c_final__a__p2__piecewise_linear__bce__plateau__s42__s3407__tr2022__va2023 gpu=0
...
COLAB_DELIVERY_READY path=<path> sha256=<sha256>
COLAB_DOWNLOAD_REQUESTED path=<path>
```

Errors end with:

```text
COLAB_STAGE_C_ERROR stage=<stage> type=<type> message=<message>
```

Normal completion creates:

```text
tabm_colab_stage_C_delivery.zip
├── tabm_search_stage_C_review_bundle.zip
├── tabm_search_stage_C_resume_bundle.zip
├── colab_stage_C.log
└── delivery_manifest.json
```

The delivery manifest binds the component names, sizes, SHA-256 values,
runtime identity, source resume identity, run UUID, and final Stage C state.
The wrapper is for handoff and review. A later campaign run consumes the inner
resume ZIP, not the wrapper itself.

## Failure Behavior

- Missing or duplicate official files: stop before package installation or
  training.
- Wrong data/resume hash: stop without extraction or mutation.
- Unsafe ZIP member: stop before publishing extracted files.
- No T4 or more than one visible GPU for this handoff: stop before training.
- Incompatible emergency environment or base resume: stop before merge.
- Partially written checkpoint: retry later; never export it.
- Browser download failure: keep the verified file and manual link in
  `/content`; do not claim local persistence.
- Campaign failure after a complete Colab epoch: export the latest emergency
  snapshot, then report the original failure.
- VM deletion: no cleanup claim; recover only from a previously downloaded
  snapshot.

## Verification Plan

All verification before handoff is local and synthetic or artifact-based; no
full-data training is run by Codex.

1. Deterministically build the data ZIP twice and compare bytes and SHA-256.
2. Verify exact data member names, sizes, hashes, and secure extraction guards.
3. Verify the supplied Stage C resume and assert exactly 20 completed plus one
   incomplete target job.
4. Verify sanitization removes only the target training files and preserves all
   completed prediction bytes and metrics.
5. Test one-GPU scheduling and retain existing two-GPU Kaggle tests.
6. Test snapshot creation only from a stable complete epoch.
7. Test snapshot tampering, wrong base hash, wrong candidate, older snapshot,
   same-epoch conflict, and environment mismatch rejection.
8. Test merge produces a resume accepted by the existing verifier and resumes
   at the next epoch in a fixture run.
9. Test same-VM rerun reuse, live-lock refusal, stale-lock recovery, and handled
   interrupt cleanup with fixture subprocesses.
10. Test the generated cell compiles, contains no GitHub or Drive dependency,
    requires one T4, and remains comfortably below notebook cell limits.
11. Test the final delivery ZIP and every nested bundle/hash.
12. Run the complete repository test suite and `git diff --check`.

## User Run Contract

The user performs the full Colab upload and GPU run. Expected first-run time is
approximately 20 to 70 minutes for the one remaining job, plus upload,
extraction, cache materialization, and final packaging. A snapshot restart
continues from the next completed Colab epoch. The user returns the final
delivery ZIP, or, after a reset, retains the base Stage C resume and newest
downloaded emergency snapshot for the next attempt.
