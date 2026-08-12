# Independent DL Kaggle Notebook Design

## Goal

Provide one Kaggle-ready notebook that resumes the existing independent DL
campaign using private Kaggle Dataset inputs. The user uploads official data and
the current campaign checkpoint directory; Codex does not run the full campaign.

## Scope

- Add `notebooks/KAGGLE_INDEPENDENT_DL_CAMPAIGN.ipynb`.
- Keep the existing Colab notebook and training implementation unchanged.
- Do not create submission CSV or ZIP files.
- Do not push without a separate user request.

## Kaggle inputs

The notebook discovers files below `/kaggle/input` instead of depending on exact
Dataset slugs.

Required official data:

- exactly one `train.csv`
- exactly one `trackman_history.csv`

Optional resume input:

- one directory named `independent_dl_campaign_v1` containing the existing
  campaign artifacts and checkpoints

Ambiguous duplicate matches are errors. Missing official data is an error. A
missing checkpoint directory starts a new campaign only after printing that fact
clearly.

## Runtime flow

The notebook contains one short Korean Markdown instruction cell and one complete
code cell.

The code cell:

1. confirms `/kaggle/input`, `/kaggle/working`, and CUDA availability;
2. locates required data without modifying the read-only input tree;
3. clones the public repository and checks out a sealed code commit;
4. installs only the campaign requirements into a local runtime directory;
5. copies the optional checkpoint tree to
   `/kaggle/working/independent_dl_campaign_v1` without overwriting an existing
   working copy;
6. runs the existing campaign command against the discovered data directory and
   writable output directory;
7. prints the manifest, summary, completion count, and the exact directory the
   user must preserve with Kaggle `Save Version`.

The GPU check accepts any CUDA GPU supplied by Kaggle and does not require a T4.

## Resume and persistence

Kaggle Dataset inputs are treated as immutable. During a session, all new
checkpoints go to `/kaggle/working/independent_dl_campaign_v1`. On a later
session, the user attaches the prior saved output or a refreshed private
checkpoint Dataset. The notebook copies that snapshot back into the writable
working directory and the existing campaign resume logic validates and skips
completed artifacts.

The notebook does not automatically upload or publish a Dataset. This avoids
embedding Kaggle credentials and keeps external writes under explicit user
control.

## Error handling

Errors identify the missing or ambiguous input, repository commit mismatch,
package failure, unavailable CUDA device, checkpoint copy conflict, or campaign
failure. The user should return the complete traceback. Partial outputs remain in
`/kaggle/working` for inspection and saving.

## Verification

Static tests verify:

- notebook format and clean outputs;
- exactly one Markdown cell and one code cell;
- Kaggle paths are used and Colab/Drive APIs are absent;
- the public repository and sealed commit are present;
- input discovery rejects ambiguity;
- checkpoint copy is non-destructive;
- the existing campaign entry point is invoked;
- no submission packaging command appears.

Only small fixture-based tests run locally. The user performs the Kaggle GPU run.
