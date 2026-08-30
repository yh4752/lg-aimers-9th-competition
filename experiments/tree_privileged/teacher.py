"""Adapter that creates leakage-safe, pitcher-held-out TrackMan teacher OOF."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from experiments.temporal_portfolio.lupi_teacher import (
    CatBoostTeacherBackend,
    TEACHER_FEATURES,
    TeacherBackend,
    build_teacher_vector,
    crossfit_teacher,
    teacher_fold,
)

from .contracts import load_contract
from .matching import MATCH_COLUMNS, match_training_pitches


class PrivilegedTeacherError(ValueError):
    pass


@dataclass(frozen=True)
class TeacherEvidence:
    probability: np.ndarray
    accepted_mask: np.ndarray
    coverage: float
    latest_coverage: float
    coverage_by_segment: Mapping[str, float]
    status: str
    distillation_allowed: bool
    split_evidence: tuple[tuple[tuple[object, ...], tuple[object, ...]], ...]
    match_sha256: str
    mapping_sha256: str
    teacher_oof_sha256: str | None


def save_teacher_evidence(evidence: TeacherEvidence, directory: Path) -> Path:
    if type(evidence) is not TeacherEvidence:
        raise PrivilegedTeacherError("teacher evidence type differs")
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise PrivilegedTeacherError("teacher cache is not empty")
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "probability.npy", np.asarray(evidence.probability, dtype="float32"), allow_pickle=False)
    np.save(root / "accepted_mask.npy", np.asarray(evidence.accepted_mask, dtype="bool"), allow_pickle=False)
    metadata = {
        "coverage": evidence.coverage, "latest_coverage": evidence.latest_coverage,
        "coverage_by_segment": dict(evidence.coverage_by_segment), "status": evidence.status,
        "distillation_allowed": evidence.distillation_allowed, "match_sha256": evidence.match_sha256,
        "mapping_sha256": evidence.mapping_sha256, "teacher_oof_sha256": evidence.teacher_oof_sha256,
        "probability_sha256": sha256((root / "probability.npy").read_bytes()).hexdigest(),
        "accepted_mask_sha256": sha256((root / "accepted_mask.npy").read_bytes()).hexdigest(),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    return root


def load_teacher_evidence(directory: Path) -> TeacherEvidence:
    root = Path(directory)
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        probability_path, mask_path = root / "probability.npy", root / "accepted_mask.npy"
        if (sha256(probability_path.read_bytes()).hexdigest() != metadata["probability_sha256"]
                or sha256(mask_path.read_bytes()).hexdigest() != metadata["accepted_mask_sha256"]):
            raise PrivilegedTeacherError("teacher cache member differs")
        probability = np.load(probability_path, allow_pickle=False)
        accepted = np.load(mask_path, allow_pickle=False)
    except PrivilegedTeacherError:
        raise
    except Exception as error:
        raise PrivilegedTeacherError(f"teacher cache cannot be loaded: {error}") from error
    return TeacherEvidence(
        _readonly(probability, "float32"), _readonly(accepted, "bool"), float(metadata["coverage"]),
        float(metadata["latest_coverage"]), MappingProxyType(dict(metadata["coverage_by_segment"])),
        str(metadata["status"]), bool(metadata["distillation_allowed"]), (),
        str(metadata["match_sha256"]), str(metadata["mapping_sha256"]), metadata["teacher_oof_sha256"],
    )


def _frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n", na_rep="<NA>").encode("utf-8")
    return sha256(payload).hexdigest()


def _vector_sha256(row_ids: pd.Series, values: np.ndarray) -> str:
    digest = sha256()
    for row_id in row_ids:
        encoded = f"{type(row_id).__name__}:{row_id}".encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big")); digest.update(encoded)
    digest.update(np.asarray(values, dtype="float32").tobytes())
    return digest.hexdigest()


def _readonly(values: np.ndarray, dtype: str) -> np.ndarray:
    output = np.asarray(values, dtype=dtype).copy()
    output.setflags(write=False)
    return output


def build_teacher_oof(
    train: pd.DataFrame,
    history: pd.DataFrame,
    *,
    cutoff_year: int,
    backend: TeacherBackend | None = None,
    matches: pd.DataFrame | None = None,
) -> TeacherEvidence:
    """Build teacher values for labeled training rows; unmatched rows remain NaN."""

    required_train = {"row_id", "season", "game_type", "pitcher_id", "control_success"}
    if type(train) is not pd.DataFrame or not required_train.issubset(train.columns):
        raise PrivilegedTeacherError("teacher train schema differs")
    if train.empty or train["row_id"].isna().any() or not train["row_id"].is_unique:
        raise PrivilegedTeacherError("teacher train row IDs differ")
    if type(history) is not pd.DataFrame or "trackman_id" not in history.columns:
        raise PrivilegedTeacherError("teacher history schema differs")
    match_frame = match_training_pitches(train, history, cutoff_year=cutoff_year) if matches is None else matches.copy(deep=True)
    if tuple(match_frame.columns) != MATCH_COLUMNS or match_frame["row_id"].tolist() != train["row_id"].tolist():
        raise PrivilegedTeacherError("teacher match evidence differs")
    accepted_mask = match_frame["lupi_match_accepted"].eq(1).to_numpy(dtype=bool)
    coverage = float(accepted_mask.mean())
    latest_mask = train["season"].eq(cutoff_year).to_numpy(dtype=bool)
    latest_coverage = float(accepted_mask[latest_mask].mean()) if latest_mask.any() else 0.0
    coverage_by_segment: dict[str, float] = {}
    for season in sorted(set(train["season"].tolist())):
        mask = train["season"].eq(season).to_numpy(dtype=bool)
        coverage_by_segment[f"season:{season}"] = float(accepted_mask[mask].mean())
    for game_type in ("R", "F"):
        mask = train["game_type"].eq(game_type).to_numpy(dtype=bool)
        if mask.any():
            coverage_by_segment[f"game_type:{game_type}"] = float(accepted_mask[mask].mean())
    contract = load_contract()
    allowed = (
        coverage >= float(contract.teacher.minimum_total_coverage)
        and latest_coverage >= float(contract.teacher.minimum_latest_coverage)
    )
    probability = np.full(len(train), np.nan, dtype="float32")
    match_sha = _frame_sha256(match_frame)
    mapping_sha = sha256(
        match_frame.loc[accepted_mask, ["row_id", "trackman_id"]]
        .to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    if not allowed:
        return TeacherEvidence(
            probability=_readonly(probability, "float32"), accepted_mask=_readonly(accepted_mask, "bool"),
            coverage=coverage, latest_coverage=latest_coverage,
            coverage_by_segment=MappingProxyType(coverage_by_segment), status="insufficient_coverage",
            distillation_allowed=False, split_evidence=(), match_sha256=match_sha,
            mapping_sha256=mapping_sha, teacher_oof_sha256=None,
        )

    if history["trackman_id"].duplicated().any():
        raise PrivilegedTeacherError("TrackMan history IDs must be unique")
    history_indexed = history.set_index("trackman_id")
    accepted = match_frame.loc[accepted_mask, ["row_id", "trackman_id", "lupi_match_accepted"]].copy()
    try:
        joined = accepted.join(history_indexed.loc[:, TEACHER_FEATURES], on="trackman_id", validate="one_to_one")
    except (KeyError, ValueError) as error:
        raise PrivilegedTeacherError("teacher TrackMan features differ") from error
    train_indexed = train.set_index("row_id")
    joined["pitcher_id"] = joined["row_id"].map(train_indexed["pitcher_id"])
    joined["control_success"] = joined["row_id"].map(train_indexed["control_success"])
    teacher_rows = joined.loc[:, ["row_id", "pitcher_id", "control_success", "lupi_match_accepted", *TEACHER_FEATURES]]
    actual_backend = backend or CatBoostTeacherBackend(device="GPU")
    teacher_oof = crossfit_teacher(teacher_rows, seed=3407, folds=contract.teacher.folds, backend=actual_backend)
    probability = build_teacher_vector(train["row_id"].tolist(), teacher_oof)
    split_evidence = []
    folds = teacher_oof.fold_by_row_id
    used = sorted(set(folds.values()))
    for fold in used:
        valid_ids = {row_id for row_id, assigned in folds.items() if assigned == fold}
        valid_pitchers = tuple(sorted(set(teacher_rows.loc[teacher_rows["row_id"].isin(valid_ids), "pitcher_id"]), key=str))
        train_pitchers = tuple(sorted(set(teacher_rows.loc[~teacher_rows["row_id"].isin(valid_ids), "pitcher_id"]), key=str))
        if set(train_pitchers) & set(valid_pitchers):
            raise PrivilegedTeacherError("teacher pitcher groups overlap")
        split_evidence.append((train_pitchers, valid_pitchers))
    return TeacherEvidence(
        probability=_readonly(probability, "float32"), accepted_mask=_readonly(accepted_mask, "bool"),
        coverage=coverage, latest_coverage=latest_coverage,
        coverage_by_segment=MappingProxyType(coverage_by_segment), status="ready",
        distillation_allowed=True, split_evidence=tuple(split_evidence), match_sha256=match_sha,
        mapping_sha256=mapping_sha, teacher_oof_sha256=_vector_sha256(train["row_id"], probability),
    )
