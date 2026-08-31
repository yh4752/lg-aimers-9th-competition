# Gated Residual G0 Submission Design

## Goal

Build one new DACON-ready submission from the accepted gated-residual candidate
`G0_7628291922`. This is an independent successor to the previously submitted
977-point archive, so the previous file remains reproducible and unchanged.

The submission packages the accepted E2 anchor with a regular-season direct
residual correction. It does not train, tune, select candidates, or use the
evaluation distribution while packaging or predicting.

## Accepted evidence and immutable identity

The builder accepts only these three reviewed artifacts:

- delivery SHA-256:
  `dcb4a602318cf787a247653349de0daf777351cdf296b1a78c10380c5b577568`
- review SHA-256:
  `e7a651c3c8b96aed13ed9270ce7f78c2f719c42baa3e7ee42ef09edef9a7193a`
- handoff SHA-256:
  `e14a5558810cd8e5be9ecb76e0cf95abf0abb3d255dcac56bce89e7e852a055c`

Their verified shared identity is:

- candidate: `G0_7628291922`
- status: `accepted`
- archetype: `G0`
- residual weight: `alpha=0.3`
- code SHA-256:
  `469e739ffa87e94516f2a088340f26e05427751ce85324d0009dd8aa0fded187`
- contract SHA-256:
  `ae2513c7d5a3a5e000271f2b37fffe1253b1ecbc27ee254c589e9c4a50e5c25e`
- training input manifest SHA-256:
  `4a47cf21f6c23634f92b66ab9934c3b059de04997aca0a0e6351d346214b924b`
- official train SHA-256:
  `d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff`
- campaign history SHA-256:
  `f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9`

The accepted evidence reports a positive weighted temporal-OOF gain of
`0.00006637240692574338`, a positive latest-fold gain of
`0.000020338502248306083`, a positive bootstrap lower bound of
`0.00003823806557728248`, and no failed decision gates. All three seeds are
non-worse overall and on the latest fold.

The inference audit passed on five official test rows. Singleton, reverse,
shuffle, rebatch, companion-row, and same-feature-row differences are exactly
zero. These results authorize packaging this candidate only; they are not a
claim about its public-leaderboard score.

## Package boundary

The final ZIP contains exactly:

```text
model/
script.py
requirements.txt
```

`model/` contains the accepted E2 anchor payload, six CatBoost models
(`D0` and `D5` for seeds `42`, `2026`, and `3407`), and the frozen feature,
count, calibration, configuration, and iteration state needed for inference.
Every file is bound by name, size, and SHA-256 in a candidate manifest.

The review bundle, handoff bundle, logs, OOF predictions, training data,
training code, and optimizer state are excluded. The delivery ZIP itself is
not submitted.

`submission/package.py` remains the sole writer of `submit.zip`. Gated-residual
code may verify and import the accepted delivery, build current evidence, and
render `script.py`, but it must not create a submission archive directly.

## Prediction flow

The standalone `script.py` has no repository dependency. It:

1. reads `data/test.csv` and `data/sample_submission.csv`, accepting the
   documented `open/` compatibility directory when present;
2. requires unique, non-null `row_id` values and exact test/sample ID parity;
3. loads only frozen training-derived state and reviewed model files;
4. computes the E2 anchor and the six direct-model predictions from the current
   row and frozen lookup tables;
5. applies the accepted `G0` rule: regular-season rows receive the direct
   residual correction with `alpha=0.3`, while other rows retain the anchor;
6. clips finite probabilities to the reviewed bounds, restores sample order,
   and writes `output/submission.csv` with exactly
   `row_id,control_success`.

The runtime must not fit on evaluation data or use evaluation-set aggregates,
frequency, ranks, quantiles, rolling or lag values, cumulative statistics,
cross-row calibration, or distribution-dependent branches. It performs no
network access and starts no subprocess.

## Builder and fail-closed gates

A dedicated builder imports the three artifacts into a new output directory
after safe ZIP validation. Before the sole packager is called, it verifies:

- exact outer artifact hashes and safe, unique archive paths;
- identical artifact bindings and complete member size/hash declarations;
- accepted candidate ID, archetype, alpha, decision, and empty failed gates;
- exact code, contract, input, train, and history identities listed above;
- the six model files, E2 payload, frozen state, requirements, and iteration
  metadata against the delivery manifest;
- `status=passed` for the inference audit and zero independence deltas;
- current rules review, policy identity, and source inspection of the exact
  rendered runtime;
- Python `3.11.15`, CatBoost `1.2.10`, pandas `2.0.3`, and NumPy `1.26.4` for
  the final local build environment;
- exact prediction parity between the reviewed implementation and standalone
  runtime on the official five-row sample;
- singleton, reverse, shuffle, and rebatch invariance on that sample;
- projected and final package layout, member hashes, compressed size, and
  extracted size.

Any mismatch stops before `submit.zip` exists. The builder never overwrites an
existing output directory and removes a partially published archive if final
verification fails.

## Dependencies and resource limits

`requirements.txt` contains only the packages that the generated runtime must
install explicitly, with reviewed exact versions. NumPy and pandas remain on
the official base versions unless the final isolated import test proves they
must be listed.

The builder enforces the current repository policy:

- installation at most 600 seconds;
- inference safety target at most 480 seconds and hard limit at most 600
  seconds for 245,789 rows;
- compressed package below 10 GB;
- extracted package below 32 GB;
- CPU inference within the official six-core and 28 GiB RAM environment.

The six reviewed CatBoost models and E2 payload are large but remain below the
declared archive limits. A measured local sample is used to reject pathological
startup or per-row behavior; full evaluation inference is left to DACON.

## Verification strategy

Implementation follows test-driven development. Automated tests cover:

- accepted delivery import and exact metadata extraction;
- tampered, stale, rejected, missing, duplicate, and unsafe artifacts;
- exact model member set and deterministic candidate manifest;
- rendered runtime import in an isolated directory;
- trusted-versus-standalone prediction equality on official sample rows;
- row-order, singleton, and rebatch independence;
- exact output columns, order, finite probabilities, and decimal formatting;
- source-rule inspection and absence of prohibited runtime operations;
- central-packager enforcement, exact archive layout, deterministic hashes,
  size limits, receipt identity, and cleanup after failure.

The final build runs only the five-row official local dry run and static or
isolated checks. It does not repeat GPU training or full-data inference.

## Outputs and human handoff

The builder writes a new timestamped directory under
`artifacts/gated_residual_g0_submission_<timestamp>/` containing:

- `submit.zip` — the only file uploaded to DACON;
- `submission_receipt.json` — archive hash, size, candidate, policy, and
  evidence identity;
- local evidence and verification reports used by the central packager.

Success is printed as `GATED_RESIDUAL_SUBMISSION_READY` with the archive path,
SHA-256, byte size, candidate ID, and three source artifact hashes. The user
still checks the selected DACON account/team, remaining daily submission count,
deadline, and exact file before manual upload.

## Non-goals

- No new training, hyperparameter search, leaderboard-driven calibration, or
  model blending occurs during packaging.
- No rejected candidate is silently substituted.
- No prior submission is overwritten or deleted.
- No Git push or DACON upload is performed automatically.
