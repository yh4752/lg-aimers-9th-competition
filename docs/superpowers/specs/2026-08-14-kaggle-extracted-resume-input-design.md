# Kaggle Extracted Resume Input Design

## Goal

Make `experiments/tabm_campaign/KAGGLE_CELL.py` accept the form Kaggle normally
creates when a resume ZIP is uploaded as a Dataset: an extracted directory
containing `manifest.json`, `stage_state.json`, and optional `training/`
artifacts. Preserve support for an intact resume ZIP from Notebook Output.

## Input discovery

The generated cell searches `/kaggle/input` for both:

1. intact files matching `tabm_search_stage_*_resume_bundle.zip`; and
2. extracted directories whose `manifest.json` declares
   `artifact_kind=resume`, `review_only=true`, and version A, B, or C.

Unrelated manifests and official-data files are ignored. Discovery must not
infer a resume from directory names alone.

## Verification and normalization

Every candidate is verified against its own manifest before use:

- the manifest member names must exactly match the files below the extracted
  root, excluding `manifest.json`;
- member paths must be relative, non-traversing regular files;
- every member SHA-256 must match the manifest;
- the campaign configuration and prior-manifest bindings remain checked by the
  existing resume verifier.

An extracted candidate is rebuilt under `/kaggle/working` as a temporary ZIP
and then passed through the existing `verify_resume_bundle` implementation.
This keeps one trusted validation path for both input forms.

## Ambiguity handling

Exactly zero or one logical resume is allowed. If an intact ZIP and an
extracted directory represent the same manifest, they are treated as duplicate
representations of one resume. Different valid manifests, invalid candidates,
or conflicting input versions stop before GPU work with an explicit error.

## Logs

The cell emits one unambiguous line before GPU startup:

```text
RESUME_FOUND source=extracted path=/kaggle/input/... rebuilt=/kaggle/working/...
```

or:

```text
RESUME_FOUND source=zip path=/kaggle/input/...
```

With no resume it emits `RESUME_FOUND source=none path=None`. The next stage
selection remains the authoritative confirmation that an A resume advances to
B, a B resume to C, and a C resume to D.

## Tests

Tests construct a real manifest-bound extracted resume fixture, verify that it
is discovered and rebuilt, and run the existing resume verifier against the
result. Additional tests cover intact ZIP compatibility, duplicate logical
representations, hash mismatch rejection, and multiple different resumes.
The generated cell remains deterministic, self-contained, and below Kaggle's
one-megabyte source limit.
