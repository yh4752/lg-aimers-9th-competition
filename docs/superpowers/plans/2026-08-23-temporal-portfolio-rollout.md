# Temporal Portfolio Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved temporal-expert competition campaign as five independently testable, sequential subprojects without creating a submission package.

**Architecture:** A new `experiments.temporal_portfolio` package owns experiment identities, feature bundles, weighted training, OOF decisions, artifacts, and platform launchers. It reuses the proven independent-DL trainer, preprocessing primitives, pitcher TrackMan lookup, and two-GPU scheduler through narrow adapters rather than modifying existing accepted candidates.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy, scikit-learn, PyTorch 2.7, TabM 0.0.3, CatBoost 1.2.10, pytest, Kaggle T4x2, Colab T4.

---

## Rollout order

1. [Foundation and artifact contracts](2026-08-23-temporal-portfolio-01-foundation.md)
2. [Temporal and TrackMan features](2026-08-23-temporal-portfolio-02-features.md)
3. [Weighted TabM, CatBoost, and LUPI training](2026-08-23-temporal-portfolio-03-training.md)
4. [OOF metrics, decisions, and staged campaign](2026-08-23-temporal-portfolio-04-campaign.md)
5. [Kaggle, Colab, final delivery, and runbooks](2026-08-23-temporal-portfolio-05-platform.md)

Each plan must pass its listed tests before the next starts. Existing submission,
TabM Stage C, CatBoost deployment, hierarchical, and realignment packages remain
unchanged. Full-data and GPU commands are handed to the user only after all local
fixture gates pass.

## Global verification after all five plans

- [ ] Run the complete focused regression suite.

```bash
pytest -q \
  tests/test_temporal_portfolio_contracts.py \
  tests/test_temporal_portfolio_identity.py \
  tests/test_temporal_portfolio_state.py \
  tests/test_temporal_portfolio_artifacts.py \
  tests/test_temporal_portfolio_inputs.py \
  tests/test_temporal_portfolio_features.py \
  tests/test_temporal_portfolio_trackman.py \
  tests/test_temporal_portfolio_lupi.py \
  tests/test_temporal_portfolio_training.py \
  tests/test_temporal_portfolio_metrics.py \
  tests/test_temporal_portfolio_ensembles.py \
  tests/test_temporal_portfolio_decisions.py \
  tests/test_temporal_portfolio_runner.py \
  tests/test_temporal_portfolio_platform.py \
  tests/test_temporal_portfolio_final.py
```

Expected: all tests pass; no test opens a GPU, installs packages, reads the official
full dataset, or creates `submit.zip`.

- [ ] Run existing boundary regressions.

```bash
pytest -q \
  tests/test_independent_dl_feature_sources.py \
  tests/test_independent_dl_features.py \
  tests/test_independent_dl_training.py \
  tests/test_budgeted_preprocessing_scheduler.py \
  tests/test_tabm_campaign_artifacts.py \
  tests/test_submission_package.py
```

Expected: all existing tests pass and rejected legacy candidates remain rejected.

- [ ] Confirm no submission-writing symbol exists in the new package.

```bash
rg -n "submit\.zip|build_submission|write_submission" experiments/temporal_portfolio tools/prepare_temporal_portfolio_input.py
```

Expected: no match except explicit fail-closed error text in `final_delivery.py`.
