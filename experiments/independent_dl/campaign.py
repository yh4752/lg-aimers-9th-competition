"""Atomic, restartable state machine for the independent DL campaign."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import traceback
from types import MappingProxyType
from typing import Mapping, Protocol

from .contracts import CampaignSpec, CandidateSpec


class CampaignStateError(ValueError):
    """Raised when an existing campaign cannot be resumed unambiguously."""


class CandidateExecutionError(RuntimeError):
    """A candidate-local failure that must not stop unrelated candidates."""


@dataclass(frozen=True)
class CandidateRunResult:
    metrics_path: Path
    predictions_path: Path
    best_brier: float


@dataclass(frozen=True)
class CampaignSummary:
    campaign_id: str
    completed: tuple[str, ...]
    failed: tuple[str, ...]
    pending: tuple[str, ...]
    registered: tuple[CandidateSpec, ...]
    output_root: Path


class CampaignRuntime(Protocol):
    def run_candidate(
        self, candidate: CandidateSpec, output_dir: Path
    ) -> CandidateRunResult: ...


class OfficialCampaignRuntime:
    """Official-data runtime used only by the user-owned campaign command."""

    def __init__(
        self,
        data_dir: str | Path,
        *,
        cache_root: str | Path,
        fit_function: object | None = None,
        adapter_factory: object | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.cache_root = Path(cache_root).resolve()
        self._fit_function = fit_function
        self._adapter_factory = adapter_factory
        self._train = None
        self._history = None

    def _load_data(self) -> tuple[object, object]:
        if self._train is None:
            import pandas as pd

            train_path = self.data_dir / "train.csv"
            history_path = self.data_dir / "trackman_history.csv"
            if not train_path.is_file() or not history_path.is_file():
                raise CandidateExecutionError(
                    "train.csv and trackman_history.csv are required"
                )
            self._train = pd.read_csv(train_path)
            self._history = pd.read_csv(history_path)
        return self._train, self._history

    @staticmethod
    def _default_adapter(family: str) -> object:
        from .models import (
            FTTransformerAdapter,
            MLPResNetAdapter,
            TabMAdapter,
            TabRAdapter,
        )

        adapters = {
            "tabm": TabMAdapter,
            "mlp_resnet": MLPResNetAdapter,
            "ft_transformer": FTTransformerAdapter,
            "tabr": TabRAdapter,
        }
        try:
            return adapters[family]()
        except KeyError as error:
            raise CandidateExecutionError(f"unknown model family: {family}") from error

    def run_candidate(
        self, candidate: CandidateSpec, output_dir: Path
    ) -> CandidateRunResult:
        try:
            import numpy as np
            import pandas as pd

            from .features import materialize_fold_cache
            from .training import TrainRequest, fit_candidate

            if candidate.train_end_year is None or candidate.valid_year is None:
                raise ValueError("candidate fold years are missing")
            train_frame, history = self._load_data()
            seasons = pd.to_numeric(train_frame["season"], errors="raise")
            fit_rows = train_frame.loc[seasons.le(candidate.train_end_year)].reset_index(
                drop=True
            )
            valid_rows = train_frame.loc[seasons.eq(candidate.valid_year)].reset_index(
                drop=True
            )
            if fit_rows.empty or valid_rows.empty:
                raise ValueError("candidate fold has no train or validation rows")
            cache = materialize_fold_cache(
                self.cache_root,
                fit_rows,
                valid_rows,
                history,
                candidate.feature_view,
                candidate.train_end_year,
                candidate.valid_year,
            )
            request = TrainRequest(
                candidate_id=candidate.candidate_id,
                family=candidate.family,
                seed=candidate.seed,
                epochs=candidate.epochs,
                model_config=candidate.model,
                training_config=candidate.training,
                train=cache.train,
                valid=cache.valid,
            )
            adapter_factory = self._adapter_factory or self._default_adapter
            fit_function = self._fit_function or fit_candidate
            adapter = adapter_factory(candidate.family)
            result = fit_function(request, adapter, output_dir)
            probability = np.asarray(result.predictions, dtype="float64")
            target = np.asarray(cache.valid.y, dtype="float64")
            if probability.shape != target.shape:
                raise ValueError("candidate predictions are not aligned to validation")
            brier = float(np.mean(np.square(probability - target)))
            output_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = output_dir / "predictions.csv"
            metrics_path = output_dir / "metrics.json"
            prediction_frame = pd.DataFrame(
                {
                    "row_id": cache.valid.row_id.astype(str),
                    "fold": f"valid_{candidate.valid_year}",
                    "season": cache.valid.season.astype("int64"),
                    "game_type": cache.valid.game_type.astype(str),
                    "target": target.astype("int64"),
                    "probability": probability,
                    "candidate_id": candidate.candidate_id,
                }
            )
            prediction_temporary = predictions_path.with_suffix(".csv.tmp")
            prediction_frame.to_csv(prediction_temporary, index=False)
            os.replace(prediction_temporary, predictions_path)
            _atomic_json(
                metrics_path,
                {
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "feature_view": candidate.feature_view,
                    "stage": candidate.stage,
                    "train_end_year": candidate.train_end_year,
                    "valid_year": candidate.valid_year,
                    "seed": candidate.seed,
                    "n": len(prediction_frame),
                    "brier": brier,
                    "best_epoch": int(result.best_epoch),
                    "checkpoint": str(Path(result.checkpoint).resolve()),
                },
            )
            return CandidateRunResult(
                metrics_path=metrics_path,
                predictions_path=predictions_path,
                best_brier=brier,
            )
        except CandidateExecutionError:
            raise
        except Exception as error:
            raise CandidateExecutionError(
                f"{candidate.candidate_id}: {type(error).__name__}: {error}\n"
                f"{traceback.format_exc()}"
            ) from error


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_payload(candidate: CandidateSpec) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "feature_view": candidate.feature_view,
        "seed": candidate.seed,
        "epochs": candidate.epochs,
        "model": dict(candidate.model),
        "training": dict(candidate.training),
        "train_end_year": candidate.train_end_year,
        "valid_year": candidate.valid_year,
        "stage": candidate.stage,
        "parent_candidate_id": candidate.parent_candidate_id,
        "parent_model": (
            None if candidate.parent_model is None else dict(candidate.parent_model)
        ),
    }


def _candidate_from_payload(payload: Mapping[str, object]) -> CandidateSpec:
    parent_model = payload.get("parent_model")
    return CandidateSpec(
        candidate_id=str(payload["candidate_id"]),
        family=str(payload["family"]),
        feature_view=str(payload["feature_view"]),
        seed=int(payload["seed"]),
        epochs=int(payload["epochs"]),
        model=MappingProxyType(dict(payload["model"])),
        training=MappingProxyType(dict(payload["training"])),
        train_end_year=(
            None if payload.get("train_end_year") is None else int(payload["train_end_year"])
        ),
        valid_year=(
            None if payload.get("valid_year") is None else int(payload["valid_year"])
        ),
        stage=str(payload.get("stage", "exploration")),
        parent_candidate_id=(
            None
            if payload.get("parent_candidate_id") is None
            else str(payload["parent_candidate_id"])
        ),
        parent_model=(
            None
            if parent_model is None
            else MappingProxyType(dict(parent_model))
        ),
    )


def _config_sha256(candidate: CandidateSpec) -> str:
    encoded = json.dumps(
        _candidate_payload(candidate), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _new_entry(candidate: CandidateSpec) -> dict[str, object]:
    return {
        "candidate": _candidate_payload(candidate),
        "config_sha256": _config_sha256(candidate),
        "state": "pending",
        "attempts": 0,
        "checkpoint": None,
        "metrics_path": None,
        "predictions_path": None,
        "metrics_sha256": None,
        "predictions_sha256": None,
        "best_brier": None,
        "failure_reason": None,
        "updated_at": _utc_now(),
    }


def _new_manifest(campaign: CampaignSpec) -> dict[str, object]:
    return {
        "schema_version": 1,
        "campaign_id": campaign.campaign_id,
        "protocol": campaign.protocol,
        "boundary_expansion_registered": False,
        "confirmation_registered": False,
        "candidates": {
            candidate.candidate_id: _new_entry(candidate)
            for candidate in campaign.candidates
        },
        "updated_at": _utc_now(),
    }


def _load_manifest(path: Path, campaign: CampaignSpec) -> dict[str, object]:
    if not path.exists():
        return _new_manifest(campaign)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignStateError(f"cannot read campaign manifest: {error}") from error
    if not isinstance(payload, dict) or payload.get("campaign_id") != campaign.campaign_id:
        raise CampaignStateError("campaign manifest identity does not match")
    entries = payload.get("candidates")
    if not isinstance(entries, dict):
        raise CampaignStateError("campaign candidate registry is invalid")
    for candidate in campaign.candidates:
        entry = entries.get(candidate.candidate_id)
        if entry is None:
            entries[candidate.candidate_id] = _new_entry(candidate)
        elif not isinstance(entry, dict) or entry.get("config_sha256") != _config_sha256(candidate):
            raise CampaignStateError(
                f"candidate config changed: {candidate.candidate_id}"
            )
    return payload


def _artifact_path(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        return None
    return path


def _completion_is_valid(root: Path, entry: Mapping[str, object]) -> bool:
    if entry.get("state") != "completed":
        return False
    for prefix in ("metrics", "predictions"):
        path = _artifact_path(root, entry.get(f"{prefix}_path"))
        expected = entry.get(f"{prefix}_sha256")
        if path is None or not path.is_file() or not isinstance(expected, str):
            return False
        if _file_sha256(path) != expected:
            return False
    return True


def _register_candidate(manifest: dict[str, object], candidate: CandidateSpec) -> None:
    entries = manifest["candidates"]
    if candidate.candidate_id not in entries:
        entries[candidate.candidate_id] = _new_entry(candidate)


def _registered_candidates(manifest: Mapping[str, object]) -> tuple[CandidateSpec, ...]:
    return tuple(
        _candidate_from_payload(entry["candidate"])
        for entry in manifest["candidates"].values()
    )


def _terminal(entry: Mapping[str, object]) -> bool:
    return entry.get("state") in {"completed", "failed"}


def _register_boundary_expansions(
    campaign: CampaignSpec, manifest: dict[str, object]
) -> None:
    entries = manifest["candidates"]
    initial = [
        candidate
        for candidate in _registered_candidates(manifest)
        if candidate.stage == "exploration"
        and entries[candidate.candidate_id]["state"] == "completed"
    ]
    for family, axes in campaign.boundary_expansion.items():
        family_candidates = [item for item in initial if item.family == family]
        if not family_candidates:
            continue
        winner = min(
            family_candidates,
            key=lambda item: (
                float(entries[item.candidate_id]["best_brier"]), item.candidate_id
            ),
        )
        for axis, expansion_values in axes.items():
            if axis not in winner.model:
                continue
            boundary = max(
                int(item.model[axis])
                for item in family_candidates
                if axis in item.model
            )
            if int(winner.model[axis]) != boundary:
                continue
            for value in expansion_values:
                if value <= boundary:
                    continue
                model = dict(winner.model)
                model[axis] = value
                expanded = replace(
                    winner,
                    candidate_id=f"{winner.candidate_id}__expand_{axis}_{value}",
                    model=MappingProxyType(model),
                    stage="boundary_expansion",
                    parent_candidate_id=winner.candidate_id,
                    parent_model=MappingProxyType(dict(winner.model)),
                )
                _register_candidate(manifest, expanded)
    manifest["boundary_expansion_registered"] = True


def _register_confirmations(campaign: CampaignSpec, manifest: dict[str, object]) -> None:
    entries = manifest["candidates"]
    exploration = [
        candidate
        for candidate in _registered_candidates(manifest)
        if candidate.stage in {"exploration", "boundary_expansion"}
        and entries[candidate.candidate_id]["state"] == "completed"
    ]
    top_k = int(campaign.survivor_policy["top_k_per_family"])
    proximity = float(campaign.survivor_policy["proximity_delta"])
    survivors: dict[str, CandidateSpec] = {}
    if exploration:
        best_global = min(
            float(entries[item.candidate_id]["best_brier"]) for item in exploration
        )
        for item in exploration:
            if float(entries[item.candidate_id]["best_brier"]) <= best_global + proximity:
                survivors[item.candidate_id] = item
        families = sorted({item.family for item in exploration})
        for family in families:
            ranked = sorted(
                (item for item in exploration if item.family == family),
                key=lambda item: (
                    float(entries[item.candidate_id]["best_brier"]), item.candidate_id
                ),
            )
            for item in ranked[:top_k]:
                survivors[item.candidate_id] = item

    for survivor in survivors.values():
        seeds = (survivor.seed,) + tuple(
            seed for seed in campaign.confirmation_seeds if seed != survivor.seed
        )
        for train_end_year, valid_year in campaign.oof_folds:
            for seed in seeds:
                if (
                    survivor.train_end_year == train_end_year
                    and survivor.valid_year == valid_year
                    and survivor.seed == seed
                ):
                    continue
                confirmed = replace(
                    survivor,
                    candidate_id=(
                        f"{survivor.candidate_id}__confirm_"
                        f"f{train_end_year}_{valid_year}__s{seed}"
                    ),
                    seed=seed,
                    train_end_year=train_end_year,
                    valid_year=valid_year,
                    stage="confirmation",
                    parent_candidate_id=survivor.candidate_id,
                    parent_model=MappingProxyType(dict(survivor.model)),
                )
                _register_candidate(manifest, confirmed)
    manifest["confirmation_registered"] = True


def _save_manifest(path: Path, manifest: dict[str, object]) -> None:
    manifest["updated_at"] = _utc_now()
    _atomic_json(path, manifest)


def run_campaign(
    campaign: CampaignSpec,
    output_root: str | Path,
    runtime: CampaignRuntime,
) -> CampaignSummary:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "campaign_manifest.json"
    manifest = _load_manifest(manifest_path, campaign)
    entries = manifest["candidates"]
    for entry in entries.values():
        if entry["state"] == "running" or (
            entry["state"] == "completed" and not _completion_is_valid(root, entry)
        ):
            entry["state"] = "pending"
            entry["failure_reason"] = None
    _save_manifest(manifest_path, manifest)

    while True:
        for candidate in _registered_candidates(manifest):
            entry = entries[candidate.candidate_id]
            if entry["state"] != "pending":
                continue
            entry["state"] = "running"
            entry["attempts"] = int(entry["attempts"]) + 1
            entry["updated_at"] = _utc_now()
            _save_manifest(manifest_path, manifest)
            candidate_dir = root / "candidates" / candidate.candidate_id
            print(
                f"[independent-dl] 시작: {candidate.candidate_id} "
                f"({candidate.stage}, fold={candidate.train_end_year}->{candidate.valid_year})",
                flush=True,
            )
            try:
                result = runtime.run_candidate(candidate, candidate_dir)
            except CandidateExecutionError as error:
                entry["state"] = "failed"
                entry["failure_reason"] = str(error)
                entry["updated_at"] = _utc_now()
                _save_manifest(manifest_path, manifest)
                print(f"[independent-dl] 후보 실패:\n{error}", flush=True)
                continue
            except BaseException:
                _save_manifest(manifest_path, manifest)
                raise
            metrics = Path(result.metrics_path).resolve()
            predictions = Path(result.predictions_path).resolve()
            if (
                not metrics.is_file()
                or not predictions.is_file()
                or candidate_dir.resolve() not in metrics.parents
                or candidate_dir.resolve() not in predictions.parents
            ):
                raise CampaignStateError(
                    f"candidate outputs are incomplete: {candidate.candidate_id}"
                )
            entry.update(
                {
                    "state": "completed",
                    "metrics_path": str(metrics.relative_to(root)),
                    "predictions_path": str(predictions.relative_to(root)),
                    "metrics_sha256": _file_sha256(metrics),
                    "predictions_sha256": _file_sha256(predictions),
                    "best_brier": float(result.best_brier),
                    "failure_reason": None,
                    "updated_at": _utc_now(),
                }
            )
            _save_manifest(manifest_path, manifest)
            print(
                f"[independent-dl] 완료: {candidate.candidate_id} "
                f"brier={float(result.best_brier):.12f}",
                flush=True,
            )

        registered = _registered_candidates(manifest)
        if not manifest["boundary_expansion_registered"]:
            initial_entries = [
                entries[item.candidate_id]
                for item in registered
                if item.stage == "exploration"
            ]
            if all(_terminal(entry) for entry in initial_entries):
                _register_boundary_expansions(campaign, manifest)
                _save_manifest(manifest_path, manifest)
                continue
        if (
            manifest["boundary_expansion_registered"]
            and not manifest["confirmation_registered"]
        ):
            exploration_entries = [
                entries[item.candidate_id]
                for item in registered
                if item.stage != "confirmation"
            ]
            if all(_terminal(entry) for entry in exploration_entries):
                _register_confirmations(campaign, manifest)
                _save_manifest(manifest_path, manifest)
                continue
        if not any(entry["state"] == "pending" for entry in entries.values()):
            break

    registered = _registered_candidates(manifest)
    completed = tuple(
        item.candidate_id
        for item in registered
        if entries[item.candidate_id]["state"] == "completed"
    )
    failed = tuple(
        item.candidate_id
        for item in registered
        if entries[item.candidate_id]["state"] == "failed"
    )
    pending = tuple(
        item.candidate_id
        for item in registered
        if entries[item.candidate_id]["state"] in {"pending", "running"}
    )
    summary = CampaignSummary(
        campaign_id=campaign.campaign_id,
        completed=completed,
        failed=failed,
        pending=pending,
        registered=registered,
        output_root=root,
    )
    _atomic_json(
        root / "campaign_summary.json",
        {
            "campaign_id": summary.campaign_id,
            "completed": list(summary.completed),
            "failed": list(summary.failed),
            "pending": list(summary.pending),
            "registered_count": len(summary.registered),
            "output_root": str(summary.output_root),
        },
    )
    return summary
