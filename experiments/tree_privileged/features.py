"""Candidate feature composition with strict training/inference boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from experiments.temporal_portfolio.lupi_teacher import TEACHER_FEATURES
from experiments.tree_expert.features import (
    TreeFeatureBatch, TreeFeatureState, fit_tree_features, transform_tree_features,
)

from .matching import MATCH_COLUMNS
from .profiles import (
    ProfileState, ProfileStrengths, build_training_profiles, fit_profiles, transform_profiles,
)
from .teacher import TeacherEvidence, build_teacher_oof


class CandidateFeatureError(ValueError):
    pass


class CandidateFeatureSkip(CandidateFeatureError):
    pass


_CANDIDATES = {"P": (True, 0.0), "D15": (False, 0.15), "D35": (False, 0.35),
               "PD15": (True, 0.15), "PD35": (True, 0.35)}
_FORBIDDEN_INFERENCE = {"control_success", "teacher_probability", *TEACHER_FEATURES, *MATCH_COLUMNS[1:]}


@dataclass(frozen=True)
class CandidateFeatureState:
    candidate_id: str
    tree_state: TreeFeatureState
    profile_state: ProfileState | None
    profile_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    selected_strengths: ProfileStrengths | None
    teacher_evidence_hashes: Mapping[str, str]


@dataclass(frozen=True)
class CandidateFeatureBatch:
    frame: pd.DataFrame
    anchor: np.ndarray
    row_id: np.ndarray
    target: np.ndarray | None
    soft_target: np.ndarray | None
    teacher_probability: np.ndarray | None


def _readonly(values: np.ndarray, dtype: str) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy(); result.setflags(write=False); return result


def _append_profiles(base: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    if len(base) != len(profiles) or set(base.columns) & set(profiles.columns):
        raise CandidateFeatureError("profile feature alignment or names differ")
    result = base.reset_index(drop=True).copy(deep=True)
    for name in profiles:
        result[name] = pd.to_numeric(profiles[name], errors="raise").to_numpy(dtype="float32")
    return result


def fit_candidate_features(
    train: pd.DataFrame,
    history: pd.DataFrame,
    *,
    valid_year: int,
    candidate_id: str,
    strengths: ProfileStrengths | tuple[int, int, int] = (75, 150, 300),
    base_builder: Callable[..., tuple[TreeFeatureState, TreeFeatureBatch]] = fit_tree_features,
    teacher_builder: Callable[..., TeacherEvidence] = build_teacher_oof,
    teacher_evidence: TeacherEvidence | None = None,
) -> tuple[CandidateFeatureState, CandidateFeatureBatch]:
    if candidate_id not in _CANDIDATES:
        raise CandidateFeatureError("candidate ID differs")
    use_profiles, teacher_lambda = _CANDIDATES[candidate_id]
    tree_state, base = base_builder(train, None, valid_year=valid_year, use_trackman=False)
    if base.target is None:
        raise CandidateFeatureError("training target is absent")
    frame = base.frame.copy(deep=True)
    profile_state = None
    profile_columns: tuple[str, ...] = ()
    selected = ProfileStrengths(*strengths) if type(strengths) is tuple else strengths
    if use_profiles:
        training_profiles = build_training_profiles(train, valid_year=valid_year, strengths=selected)
        profile_state = fit_profiles(train, cutoff_year=valid_year - 1, strengths=selected)
        profile_columns = tuple(training_profiles.columns)
        frame = _append_profiles(frame, training_profiles)

    teacher_probability = None
    hashes: dict[str, str] = {}
    hard_target = np.asarray(base.target, dtype="float64")
    soft_target = hard_target.copy()
    if teacher_lambda:
        evidence = teacher_evidence or teacher_builder(train, history, cutoff_year=valid_year - 1)
        hashes = {
            "match_sha256": evidence.match_sha256,
            "mapping_sha256": evidence.mapping_sha256,
            "teacher_oof_sha256": evidence.teacher_oof_sha256 or "0" * 64,
        }
        if not evidence.distillation_allowed:
            raise CandidateFeatureSkip("teacher_coverage_gate")
        teacher_probability = np.asarray(evidence.probability, dtype="float64")
        finite = np.isfinite(teacher_probability)
        soft_target[finite] = ((1.0 - teacher_lambda) * hard_target[finite]
                               + teacher_lambda * teacher_probability[finite])
    state = CandidateFeatureState(
        candidate_id=candidate_id, tree_state=tree_state, profile_state=profile_state,
        profile_columns=profile_columns, feature_columns=tuple(frame.columns),
        categorical_columns=tree_state.categorical_columns,
        selected_strengths=selected if use_profiles else None,
        teacher_evidence_hashes=MappingProxyType(hashes),
    )
    return state, CandidateFeatureBatch(
        frame=frame, anchor=base.anchor, row_id=base.row_id, target=base.target,
        soft_target=_readonly(soft_target, "float64"),
        teacher_probability=None if teacher_probability is None else _readonly(teacher_probability, "float64"),
    )


def transform_candidate_features(rows: pd.DataFrame, state: CandidateFeatureState) -> CandidateFeatureBatch:
    if type(rows) is not pd.DataFrame or type(state) is not CandidateFeatureState:
        raise CandidateFeatureError("inference rows or state differ")
    forbidden = sorted(_FORBIDDEN_INFERENCE & set(rows.columns))
    if forbidden:
        raise CandidateFeatureError(f"inference contains forbidden columns: {forbidden}")
    base = transform_tree_features(rows, state.tree_state)
    frame = base.frame.copy(deep=True)
    if state.profile_state is not None:
        profiles = transform_profiles(rows, state.profile_state)
        if tuple(profiles.columns) != state.profile_columns:
            raise CandidateFeatureError("profile inference schema differs")
        frame = _append_profiles(frame, profiles)
    if tuple(frame.columns) != state.feature_columns:
        raise CandidateFeatureError("candidate inference feature schema differs")
    return CandidateFeatureBatch(frame, base.anchor, base.row_id, None, None, None)
