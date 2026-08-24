from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Callable

import numpy as np
import pandas as pd

from experiments.catboost_deployment.state import deserialize_feature_state
from experiments.catboost_preprocessing.features import transform_catboost_features

from .artifacts import (
    Bindings,
    CampaignFiles,
    TrustedFile,
    restore_resume,
    write_delivery_bundle,
    write_resume_bundle,
    write_review_bundle,
)
from .contracts import contract_sha256, load_contract
from .inputs import VerifiedRealignInput
from .metrics import (
    CATBOOST_COLUMNS,
    TABM_COLUMNS,
    RealignDecision,
    decision_to_payload,
    evaluate_prefixes,
)
from .state import CampaignState, deserialize_state, serialize_state
from .tabm_fold import TabMFoldResult, run_tabm_f1
from .training import CatBoostJobResult, run_catboost_f1, run_full_fit


class RealignRunnerError(RuntimeError):
    """Raised when the guarded campaign cannot advance safely."""


@dataclass(frozen=True)
class RealignRun:
    status: str
    selected_tree_count: int | None
    review_bundle: Path
    resume_bundle: Path
    delivery_bundle: Path | None


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _code_sha256() -> str:
    from .runtime_inventory import code_identity_sha256

    return code_identity_sha256(Path(__file__).resolve().parents[2])


def _bindings(verified: VerifiedRealignInput) -> Bindings:
    train = verified.data_dir / "train.csv"
    history = verified.data_dir / "trackman_history.csv"
    if not train.is_file() or train.is_symlink() or not history.is_file() or history.is_symlink():
        raise RealignRunnerError("verified training sources are missing")
    return Bindings(
        contract_sha256=contract_sha256(),
        code_sha256=_code_sha256(),
        input_manifest_sha256=verified.manifest_sha256,
        train_sha256=_file_sha(train),
        history_sha256=_file_sha(history),
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_state(root: Path, state: CampaignState) -> None:
    path = root / "state/stage_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(path, serialize_state(state))


def _read_state(root: Path) -> CampaignState:
    return deserialize_state((root / "state/stage_state.json").read_bytes())


def _trusted(path: Path) -> TrustedFile:
    return TrustedFile(path, _file_sha(path))


def _all_resume_members(root: Path) -> dict[str, TrustedFile]:
    output: dict[str, TrustedFile] = {}
    for top in ("state", "jobs", "decision", "logs"):
        base = root / top
        if not base.is_dir() or base.is_symlink():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.is_symlink():
                output[path.relative_to(root).as_posix()] = _trusted(path)
    return output


def _review_members(root: Path) -> dict[str, TrustedFile]:
    output: dict[str, TrustedFile] = {}
    for name in ("state/stage_state.json", "logs/campaign.log"):
        path = root / name
        if path.is_file():
            output[name] = _trusted(path)
    decision_root = root / "decision"
    if decision_root.is_dir():
        for path in sorted(decision_root.glob("*.json")):
            output[path.relative_to(root).as_posix()] = _trusted(path)
    for job_id in ("tabm_f1_2022", "catboost_f1_2022"):
        directory = root / "jobs" / job_id
        if directory.is_dir():
            for path in sorted(directory.glob("*.csv")):
                output[path.relative_to(root).as_posix()] = _trusted(path)
    return output


def _write_decision_files(root: Path, decision: RealignDecision) -> str:
    directory = root / "decision"
    directory.mkdir(parents=True, exist_ok=True)
    payload = decision_to_payload(decision)
    encoded = _canonical_json(payload)
    decision_path = directory / "alignment_decision.json"
    if decision_path.exists() or decision_path.is_symlink():
        if decision_path.is_symlink() or decision_path.read_bytes() != encoded:
            raise RealignRunnerError("recomputed alignment decision differs")
    else:
        _atomic_bytes(decision_path, encoded)
    _atomic_bytes(
        directory / "fold_metrics.json",
        _canonical_json(
            {
                str(item.tree_count): dict(item.fold_gain)
                for item in decision.candidates
            }
        ),
    )
    _atomic_bytes(
        directory / "bootstrap_metrics.json",
        _canonical_json(
            {
                str(item.tree_count): {
                    "status": item.bootstrap_status,
                    "lower": item.latest_bootstrap_lower,
                }
                for item in decision.candidates
            }
        ),
    )
    _atomic_bytes(
        directory / "segment_metrics.json",
        _canonical_json(
            {
                str(item.tree_count): {
                    "eligible_count": item.eligible_segment_count,
                    "maximum_regression": item.maximum_segment_regression,
                }
                for item in decision.candidates
            }
        ),
    )
    return sha256(encoded).hexdigest()


def _campaign_files(root: Path, state: CampaignState) -> CampaignFiles:
    resume = _all_resume_members(root)
    review = _review_members(root)
    delivery: dict[str, TrustedFile] = {}
    if state.status == "completed":
        full = root / "jobs/catboost_full_2024"
        model = full / "model.cbm"
        preprocessing = full / "preprocessing_state.json"
        inference = root / "frozen_catboost/inference_manifest.json"
        inference.parent.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(
            inference,
            _canonical_json(
                {
                    "schema_version": 1,
                    "selected_tree_count": state.selected_tree_count,
                    "model_sha256": _file_sha(model),
                    "preprocessing_sha256": _file_sha(preprocessing),
                }
            ),
        )
        policy = root / "policy/policy.json"
        policy.parent.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(
            policy,
            _canonical_json(
                {
                    "campaign_id": "catboost_50_50_realign_v2",
                    "rule_safe": True,
                    "submission_package": False,
                    "tabm_weight": "0.50",
                    "test_independent": True,
                }
            ),
        )
        delivery_sources = {
            "decision/alignment_decision.json": root / "decision/alignment_decision.json",
            "decision/fold_metrics.json": root / "decision/fold_metrics.json",
            "decision/bootstrap_metrics.json": root / "decision/bootstrap_metrics.json",
            "decision/segment_metrics.json": root / "decision/segment_metrics.json",
            "frozen_catboost/model.cbm": model,
            "frozen_catboost/preprocessing_state.json": preprocessing,
            "frozen_catboost/inference_manifest.json": inference,
            "logs/campaign.log": root / "logs/campaign.log",
            "policy/policy.json": policy,
        }
        delivery = {name: _trusted(path) for name, path in delivery_sources.items()}
    return CampaignFiles(state.status, resume, review, delivery)


def _publish(
    root: Path,
    state: CampaignState,
    phase: str,
    callback: Callable[[Path, str], None] | None,
) -> tuple[Path, Path, Path | None]:
    bundles = root / "bundles"
    bundles.mkdir(exist_ok=True)
    files = _campaign_files(root, state)
    terminal = phase in {"completed", "blocked"}
    resume_name = (
        "catboost_50_50_realign_resume.zip"
        if terminal
        else f"catboost_50_50_realign_resume_{phase}.zip"
    )
    review_name = (
        "catboost_50_50_realign_review.zip"
        if terminal
        else f"catboost_50_50_realign_review_{phase}.zip"
    )
    resume = write_resume_bundle(
        files, bundles / resume_name, state.bindings
    )
    review = write_review_bundle(
        files, bundles / review_name, state.bindings
    )
    delivery = None
    if state.status == "completed":
        delivery = write_delivery_bundle(
            files, bundles / "catboost_50_50_realign_delivery.zip", state.bindings
        )
    if callback is not None:
        callback(resume, phase)
    return review, resume, delivery


def _load_catboost_model(path: Path):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def build_existing_fold_predictions(
    verified: VerifiedRealignInput,
    model_loader: Callable[[Path], object] = _load_catboost_model,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    full = pd.read_csv(verified.data_dir / "train.csv")
    seasons = pd.to_numeric(full["season"], errors="raise").astype("int64")
    tabm: dict[str, pd.DataFrame] = {}
    catboost: dict[str, pd.DataFrame] = {}
    for fold, valid_year in (("2022->2023", 2023), ("2023->2024", 2024)):
        tabm_source = pd.read_csv(verified.tabm_predictions[fold])
        if not set(TABM_COLUMNS).issubset(tabm_source.columns):
            raise RealignRunnerError(f"existing TabM schema differs: {fold}")
        tabm[fold] = tabm_source.loc[:, TABM_COLUMNS].copy()
        state = deserialize_feature_state(verified.catboost_states[fold].read_bytes())
        valid = full.loc[seasons.eq(valid_year)].reset_index(drop=True)
        features = transform_catboost_features(valid, state)
        model = model_loader(verified.catboost_models[fold])
        payload: dict[str, object] = {
            "row_id": valid["row_id"].astype(str),
            "target": pd.to_numeric(valid["control_success"], errors="raise").astype("int8"),
        }
        contract = load_contract()
        for prefix in contract.tree_prefixes:
            raw = np.asarray(model.predict(features, ntree_end=prefix), dtype="float64").reshape(-1)
            if len(raw) != len(valid) or not np.isfinite(raw).all():
                raise RealignRunnerError(f"existing CatBoost prediction differs: {fold}")
            payload[f"p_{prefix}"] = np.clip(raw, 0.0, 1.0)
        payload.update(
            {
                "game_type": valid["game_type"].astype(str),
                "game_month": valid["game_month"],
                "pitcher_id_known": np.where(
                    features["pitcher_id"].astype(str).isin(state.category_values["pitcher_id"]),
                    "known",
                    "oov",
                ),
                "batter_id_known": np.where(
                    features["batter_id"].astype(str).isin(state.category_values["batter_id"]),
                    "known",
                    "oov",
                ),
            }
        )
        generated = pd.DataFrame(payload).loc[:, CATBOOST_COLUMNS]
        prior = pd.read_csv(verified.catboost_predictions[fold])
        prior_required = {
            "row_id",
            "target",
            "p_4",
            "p_32",
            "game_type",
            "game_month",
            "pitcher_id_known",
            "batter_id_known",
        }
        if not prior_required.issubset(prior.columns) or len(prior) != len(generated):
            raise RealignRunnerError(f"existing CatBoost evidence schema differs: {fold}")
        for column in (
            "row_id",
            "target",
            "game_type",
            "game_month",
            "pitcher_id_known",
            "batter_id_known",
        ):
            left = prior[column].astype("string").reset_index(drop=True)
            right = generated[column].astype("string").reset_index(drop=True)
            if not bool((left.eq(right) | (left.isna() & right.isna())).all()):
                raise RealignRunnerError(
                    f"existing CatBoost evidence alignment differs: {fold} {column}"
                )
        for prefix in (4, 32):
            if f"p_{prefix}" not in prior or not np.allclose(
                generated[f"p_{prefix}"].to_numpy("float64"),
                pd.to_numeric(prior[f"p_{prefix}"], errors="raise").to_numpy("float64"),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RealignRunnerError(f"existing CatBoost parity differs: {fold} p_{prefix}")
        catboost[fold] = generated
    return tabm, catboost


def run_campaign(
    *,
    verified: VerifiedRealignInput,
    output_dir: Path,
    resume_bundle: Path | None,
    absolute_deadline: float,
    on_phase_resume: Callable[[Path, str], None] | None = None,
    tabm_runtime: Callable[..., TabMFoldResult] = run_tabm_f1,
    catboost_runtime: Callable[..., CatBoostJobResult] = run_catboost_f1,
    full_runtime: Callable[..., CatBoostJobResult] = run_full_fit,
    existing_fold_runtime: Callable[[VerifiedRealignInput], tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]] = build_existing_fold_predictions,
    evaluation_runtime: Callable[..., RealignDecision] = evaluate_prefixes,
) -> RealignRun:
    if not np.isfinite(absolute_deadline):
        raise RealignRunnerError("absolute deadline must be finite")
    bindings = _bindings(verified)
    root = Path(output_dir)
    if resume_bundle is None:
        if root.exists() or root.is_symlink():
            raise RealignRunnerError("fresh output directory already exists")
        root.mkdir(parents=True)
        state = CampaignState("fresh", (), None, None, None, bindings)
        _write_state(root, state)
    else:
        restored = restore_resume(Path(resume_bundle), root, bindings)
        state = _read_state(restored.root)
    logs = root / "logs/campaign.log"
    logs.parent.mkdir(parents=True, exist_ok=True)
    terminal_state = state if state.status in {"completed", "deployment_blocked"} else None

    completed = list(state.completed_job_ids)
    if "tabm_f1_2022" not in completed:
        state = CampaignState("f1_active", tuple(completed), "tabm_f1_2022", None, None, bindings)
        _write_state(root, state)
        result = tabm_runtime(
            data_dir=verified.data_dir,
            output_dir=root / "jobs/tabm_f1_2022",
            cache_root=root / "cache/tabm_f1_2022",
            absolute_deadline=absolute_deadline,
        )
        if result.status != "completed":
            logs.write_text("TabM F1 incomplete\n", encoding="utf-8")
            review, resume, _ = _publish(root, state, "tabm_incomplete", on_phase_resume)
            return RealignRun("budget_inconclusive", None, review, resume, None)
        completed.append("tabm_f1_2022")
    if "catboost_f1_2022" not in completed:
        state = CampaignState(
            "f1_active", tuple(completed), "catboost_f1_2022", None, None, bindings
        )
        _write_state(root, state)
        result = catboost_runtime(
            data_dir=verified.data_dir,
            output_dir=root / "jobs/catboost_f1_2022",
            absolute_deadline=absolute_deadline,
            input_manifest_sha256=verified.manifest_sha256,
            code_sha256=bindings.code_sha256,
        )
        if result.status != "completed":
            logs.write_text("CatBoost F1 incomplete\n", encoding="utf-8")
            review, resume, _ = _publish(root, state, "catboost_incomplete", on_phase_resume)
            return RealignRun("budget_inconclusive", None, review, resume, None)
        completed.append("catboost_f1_2022")
    if terminal_state is None:
        state = CampaignState("f1_complete", tuple(completed), None, None, None, bindings)
        _write_state(root, state)
        logs.write_text("F1 complete\n", encoding="utf-8")
        _publish(root, state, "f1_complete", on_phase_resume)

    tabm_existing, catboost_existing = existing_fold_runtime(verified)
    tabm_by_fold = {
        "2021->2022": pd.read_csv(root / "jobs/tabm_f1_2022/tabm_f1_predictions.csv"),
        **tabm_existing,
    }
    catboost_by_fold = {
        "2021->2022": pd.read_csv(root / "jobs/catboost_f1_2022/predictions.csv"),
        **catboost_existing,
    }
    decision = evaluation_runtime(tabm_by_fold, catboost_by_fold, load_contract())
    decision_sha = _write_decision_files(root, decision)
    if terminal_state is not None:
        if decision_sha != terminal_state.decision_sha256:
            raise RealignRunnerError("recomputed decision SHA-256 differs")
        if terminal_state.status == "deployment_blocked":
            if decision.status != "rejected" or decision.selected_tree_count is not None:
                raise RealignRunnerError("blocked resume decision differs")
            review, resume, _ = _publish(
                root, terminal_state, "blocked", on_phase_resume
            )
            return RealignRun("deployment_blocked", None, review, resume, None)
        if (
            decision.status != "promoted"
            or decision.selected_tree_count != terminal_state.selected_tree_count
        ):
            raise RealignRunnerError("completed resume decision differs")
        review, resume, delivery = _publish(
            root, terminal_state, "completed", on_phase_resume
        )
        return RealignRun(
            "completed",
            terminal_state.selected_tree_count,
            review,
            resume,
            delivery,
        )
    if decision.status != "promoted":
        state = CampaignState(
            "deployment_blocked", tuple(completed), None, decision_sha, None, bindings
        )
        _write_state(root, state)
        logs.write_text("Deployment blocked by preregistered gates\n", encoding="utf-8")
        review, resume, _ = _publish(root, state, "blocked", on_phase_resume)
        return RealignRun("deployment_blocked", None, review, resume, None)

    selected = decision.selected_tree_count
    state = CampaignState(
        "full_fit_active", tuple(completed), "catboost_full_2024", decision_sha, selected, bindings
    )
    _write_state(root, state)
    full = full_runtime(
        data_dir=verified.data_dir,
        output_dir=root / "jobs/catboost_full_2024",
        decision=decision,
        absolute_deadline=absolute_deadline,
        input_manifest_sha256=verified.manifest_sha256,
        code_sha256=bindings.code_sha256,
    )
    if full.status != "completed":
        logs.write_text("Full fit incomplete\n", encoding="utf-8")
        review, resume, _ = _publish(root, state, "full_incomplete", on_phase_resume)
        return RealignRun("budget_inconclusive", selected, review, resume, None)
    state = CampaignState(
        "completed", (*completed, "catboost_full_2024"), None, decision_sha, selected, bindings
    )
    _write_state(root, state)
    logs.write_text("Campaign completed\n", encoding="utf-8")
    review, resume, delivery = _publish(root, state, "completed", on_phase_resume)
    return RealignRun("completed", selected, review, resume, delivery)
