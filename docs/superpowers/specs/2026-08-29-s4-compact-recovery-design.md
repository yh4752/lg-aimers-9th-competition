# S4 Compact Recovery Design

## Purpose

Resume the interrupted S4 anchor–residual–hierarchical campaign without rerunning its expensive anchor and structure-seed residual training, while preventing another Kaggle working-disk exhaustion.

The trusted source is:

- artifact: `anchor_residual_hierarchical_handoff.zip`
- SHA-256: `c8becb4037e511ff5d28d0fd146e1a9ec73923125cb4fb9c4467ba3cab6f8e9d`
- artifact status: `review_ready`
- saved phase: `full_chains`
- saved progress: every anchor and structure residual job, plus full-chain jobs `00` through `13`
- predecessor runtime identity: `5e73e15269723679d421f88ddbb04141941b5760a9d7257c550b42183acd35c8`

The live Kaggle log progressed further, but those confirmation jobs were not included in the last valid handoff. Recovery therefore restarts from saved full-chain job `14`, not from the later unsaved log position.

## Non-goals

- Do not produce a DACON submission package.
- Do not weaken temporal validation, candidate acceptance gates, or evaluation-row independence.
- Do not import models or predictions from any non-official source.
- Do not rerun completed anchor or structure-seed residual jobs.
- Do not modify unrelated TabM recovery work in the existing checkout.

## Chosen Architecture

Use a distinct, compact recovery-input artifact rather than uploading the 3.8 GB handoff unchanged or disguising a rewritten artifact as its original identity.

### Local compaction

A local CLI reads the source handoff sequentially and performs four checks before creating output:

1. The outer SHA-256 equals the exact trusted source hash.
2. The outer handoff manifest and every outer member size and SHA-256 are valid.
3. The nested resume manifest, bindings, state, and every retained member are valid.
4. The source contract, official train/history, S4 input, E2 handoff, and predecessor code identities match the recovery contract.

The output kind is `tree_s4_recovery_input_v1`. It contains a compact current-runtime-bound `resume.zip`, the immutable source handoff manifest, and a recovery manifest recording both source and destination bindings. Rebinding is permitted only from the exact predecessor runtime identity named above to the runtime identity embedded in the generated Kaggle cell.

For the saved `full_chains` phase, retain:

- `state/`
- `diagnostics/`
- `decisions/`
- `anchors/`
- `residual_predictions/`
- `full_chains/`
- `s4_campaign.log`

Drop:

- all `jobs/` OOF model files and job-local copies
- `anchor_basis/`, because the anchor phase and every structure residual are complete
- `verified_e2/` and `verified_e2_input.zip`, because they are deterministically reconstructed from the sealed S4 input
- prior `bundles/` and temporary files

The CLI must never overwrite an existing output. It reports source verification, retained/dropped byte counts, output SHA-256, and the destination runtime identity.

### Kaggle recovery

The existing notebook remains in use. Its single cell is replaced with the new generated cell. Kaggle inputs are:

1. official `lg-aimers-9th-data`
2. existing `tree_s4_input`
3. new compact S4 recovery input

The cell verifies all three identities, requires Tesla T4 ×2, restores the compact resume, and starts at the first pending job. For the supplied recovery state this is `full_chains__14`, followed by confirmation jobs.

Kaggle may recursively expand every nested ZIP. Input discovery and verification must accept both the ZIP and recursively expanded directory forms without relaxing manifest or hash checks.

## Runtime Storage Policy

OOF models are not submission models and are not needed after their prediction files and result metadata are committed. New anchor/residual/confirmation jobs therefore do not retain trained model binaries after successful prediction persistence.

After full-chain selection is frozen, the runtime removes large intermediates that no later phase can read:

- anchor candidates not referenced by selected chains
- residual predictions for unselected chains
- full-chain prediction tables for unselected chains
- anchor-basis predictions
- completed job model directories
- reconstructed E2 working copies when they can be regenerated safely

The resume writer independently excludes those same disposable categories as a defense in depth. The review bundle contains state, decisions, diagnostics, selected configuration, logs, and final confirmation evidence, but not raw anchor/residual/full-chain tables for rejected candidates.

Artifact writers stream file content instead of loading multi-gigabyte nested ZIP members into memory. Successfully nested review/resume files are deleted after the outer handoff is committed. Before each snapshot, the runner logs current free bytes and an estimated peak requirement. A periodic snapshot is skipped, without failing training, when free space is below the safe requirement; the last valid handoff is preserved.

## State and Failure Handling

- Existing completed-job identities remain authoritative; a job is skipped only when it is in the verified restored state and its required retained output exists.
- Missing output for a supposedly completed job is a hard recovery error, not an implicit retrain.
- Source or destination binding differences stop before GPU work.
- Disk-space checks occur before snapshot construction.
- A job exception records state first and then attempts one compact emergency handoff.
- The final output remains `anchor_residual_hierarchical_handoff.zip` with `submission_package=false`.
- An accepted decision produces review evidence only. Submission packaging remains blocked until the independent acceptance and artifact gates are checked afterward.

## Verification

Synthetic tests cover:

- exact source-hash and predecessor-binding enforcement
- nested resume member tampering
- phase-aware retained and dropped member sets
- safe current-runtime rebinding
- recursively expanded Kaggle recovery input
- restoring a `full_chains` state with jobs `00`–`13` completed and starting only job `14`
- selected-chain pruning without removing later-phase dependencies
- streaming artifact creation and cleanup of nested temporary bundles
- low-space snapshot skip while preserving the last valid handoff
- final handoff containing no submission entry point

Repository verification includes the complete Tree Expert suite, competition-rule suite, isolated runtime import, generated-cell equality, source compilation, and the Kaggle one-megabyte cell limit.

The user performs the only full artifact compaction and T4 ×2 recovery runs. Codex performs static, synthetic, and fixture-based validation only.
