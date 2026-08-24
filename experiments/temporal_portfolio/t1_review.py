"""Evaluate a complete T1 review and prepare the minimal T2-A evidence input."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import io
import json
from numbers import Integral
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Mapping
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

import numpy as np
import pandas as pd

from .contracts import build_stage_jobs, load_contract
from .decisions import CandidateMetrics, decide_candidate
from .metrics import (
    T1Recipe,
    align_oof,
    apply_anchor,
    blend_logit,
    blend_probability,
    brier,
    build_t1_recipes,
)
from .t1_artifacts import _read_bundle
from .uncertainty import SEGMENT_COLUMNS, segment_regressions


class T1ReviewError(ValueError):
    pass


@dataclass(frozen=True)
class T1Decision:
    review_sha256: str
    data_rows_sha256: str
    recipe_id: str
    decay: Decimal
    recent_weight: Decimal
    mode: str
    beta: Decimal
    status: str
    tier: str
    weighted_gain: float
    latest_gain: float
    fold_gains: Mapping[int, float]
    bootstrap_lower: float
    max_segment_regression: float
    sha256: str


@dataclass(frozen=True)
class VerifiedT2AInput:
    path: Path
    archive_sha256: str
    decision_sha256: str
    data_rows_sha256: str
    review_sha256: str


_PREDICTION_COLUMNS = (
    "row_id",
    "target",
    "probability",
    "pitcher_id",
    "batter_id",
    "game_type",
    "pitcher_id_known",
    "batter_id_known",
    "trackman_available",
    "hand_matchup",
    "history_count_bucket",
    "runner_state",
    "leverage_bucket",
)
_T2A_MEMBERS = {
    "manifest.json",
    "decision.json",
    "t1_anchor_2024.csv",
    "t1_multi_2024.csv",
}
_MODEL_RELATIVE_SEGMENTS = ("pitcher_id_known", "batter_id_known")
_STABLE_ALIGNMENT_COLUMNS = (
    "pitcher_id",
    "batter_id",
    *(column for column in SEGMENT_COLUMNS if column not in _MODEL_RELATIVE_SEGMENTS),
)
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def evaluate_t1_review(path: str | Path) -> T1Decision:
    source = Path(path)
    manifest, members = _read_bundle(
        source, expected_kind="temporal_t1_review_v1"
    )
    specs = build_stage_jobs(load_contract(), "T1")
    expected_jobs = {spec.job_id for spec in specs}
    if (
        set(manifest.get("completed", ())) != expected_jobs
        or manifest.get("pending") != []
        or manifest.get("failed") != []
    ):
        raise T1ReviewError("T1 review is not a complete 15-job result")
    identities = manifest.get("training_identities")
    if type(identities) is not dict or set(identities) != expected_jobs:
        raise T1ReviewError("T1 review training identities differ")

    frames: dict[str, pd.DataFrame] = {}
    rates: dict[str, float] = {}
    data_hashes: set[str] = set()
    for spec in specs:
        prefix = f"jobs/{spec.job_id}/"
        try:
            frame = pd.read_csv(io.BytesIO(members[prefix + "predictions.csv"]))
            metrics = json.loads(members[prefix + "metrics.json"])
            checkpoint = json.loads(members[prefix + "checkpoint_meta.json"])
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise T1ReviewError(f"T1 job evidence is unreadable: {spec.job_id}") from error
        if tuple(frame.columns) != _PREDICTION_COLUMNS:
            raise T1ReviewError(f"T1 prediction schema differs: {spec.job_id}")
        identity = checkpoint.get("training_identity_sha256")
        binding = checkpoint.get("checkpoint_binding")
        if identity != identities[spec.job_id] or type(binding) is not dict:
            raise T1ReviewError(f"T1 job binding differs: {spec.job_id}")
        data_rows = binding.get("data_rows_sha256")
        if not _is_sha256(data_rows):
            raise T1ReviewError(f"T1 data binding is invalid: {spec.job_id}")
        data_hashes.add(data_rows)
        if checkpoint.get("train_request_sha256") != metrics.get(
            "train_request_sha256"
        ):
            raise T1ReviewError(f"T1 request binding differs: {spec.job_id}")
        rate = metrics.get("weighted_train_target_rate")
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not np.isfinite(float(rate))
            or not 0 <= float(rate) <= 1
        ):
            raise T1ReviewError(f"T1 train-only anchor differs: {spec.job_id}")
        frames[spec.job_id] = frame
        rates[spec.job_id] = float(rate)
    if len(data_hashes) != 1:
        raise T1ReviewError("T1 jobs are bound to different official data")

    fold_pairs = _fold_pairs(specs, frames, rates)
    ranked = sorted(
        (
            (_point_metrics(recipe, fold_pairs), recipe)
            for recipe in build_t1_recipes(load_contract())
        ),
        key=lambda item: (
            -item[0][0],
            -item[0][1],
            item[1].recipe_id,
        ),
    )
    selected: tuple[T1Recipe, CandidateMetrics, dict[int, float]] | None = None
    for (weighted_gain, latest_gain, fold_gains), recipe in ranked:
        oof = _candidate_oof(recipe, fold_pairs)
        bootstrap_lower = _bootstrap_lower(oof, repeats=1000, seed=3407)
        segments = segment_regressions(oof, minimum_rows=5_000)
        eligible = [item for item in segments if item.eligible]
        max_regression = max((max(0.0, -item.brier_gain) for item in eligible), default=0.0)
        regressions = tuple(Decimal(str(max(0.0, -fold_gains[year]))) for year in sorted(fold_gains))
        metrics = CandidateMetrics(
            candidate_id=recipe.recipe_id,
            family="temporal",
            temporal_gain=Decimal(str(weighted_gain)),
            weighted_gain=Decimal(str(weighted_gain)),
            latest_gain=Decimal(str(latest_gain)),
            bootstrap_lower=Decimal(str(bootstrap_lower)),
            max_segment_regression=Decimal(str(max_regression)),
            fold_regressions=regressions,
            improved_fold_count=sum(value > 0 for value in fold_gains.values()),
            worst_fold_regression=max(regressions),
            latest_regression=Decimal(str(max(0.0, -latest_gain))),
        )
        if decide_candidate(metrics).status == "champion":
            selected = recipe, metrics, fold_gains
            break
    if selected is None:
        raise T1ReviewError("T1 produced no champion recipe")
    recipe, metrics, fold_gains = selected
    decision_status = decide_candidate(metrics)
    payload = {
        "schema_version": 1,
        "review_sha256": _file_sha256(source),
        "data_rows_sha256": next(iter(data_hashes)),
        "recipe_id": recipe.recipe_id,
        "decay": str(recipe.decay),
        "recent_weight": str(recipe.recent_weight),
        "mode": recipe.mode,
        "beta": str(recipe.beta),
        "status": decision_status.status,
        "tier": decision_status.tier,
        "weighted_gain": float(metrics.weighted_gain),
        "latest_gain": float(metrics.latest_gain),
        "fold_gains": {str(year): value for year, value in sorted(fold_gains.items())},
        "bootstrap_lower": float(metrics.bootstrap_lower),
        "max_segment_regression": float(metrics.max_segment_regression),
    }
    digest = sha256(_json_bytes(payload)).hexdigest()
    return T1Decision(
        payload["review_sha256"],
        payload["data_rows_sha256"],
        recipe.recipe_id,
        recipe.decay,
        recipe.recent_weight,
        recipe.mode,
        recipe.beta,
        decision_status.status,
        decision_status.tier,
        float(metrics.weighted_gain),
        float(metrics.latest_gain),
        MappingProxyType(dict(sorted(fold_gains.items()))),
        float(metrics.bootstrap_lower),
        float(metrics.max_segment_regression),
        digest,
    )


def build_t2a_input(
    review: str | Path, decision: T1Decision, output: str | Path
) -> Path:
    verified_decision = evaluate_t1_review(review)
    if verified_decision.sha256 != decision.sha256:
        raise T1ReviewError("T1 decision differs from verified review")
    manifest, members = _read_bundle(
        Path(review), expected_kind="temporal_t1_review_v1"
    )
    del manifest
    year = 2024
    recent_id = f"t1__recent__va{year}__s3407"
    decay_id = str(decision.decay).replace(".", "p")
    multi_id = f"t1__multi_d{decay_id}__va{year}__s3407"
    recent = pd.read_csv(io.BytesIO(members[f"jobs/{recent_id}/predictions.csv"]))
    multi = pd.read_csv(io.BytesIO(members[f"jobs/{multi_id}/predictions.csv"]))
    aligned = _align_pair(recent, multi, year)
    probability = _blend(
        aligned["recent"].to_numpy(),
        aligned["multi"].to_numpy(),
        decision,
        anchor_rate=None,
    )
    base_columns = (
        "row_id",
        "valid_year",
        "target",
        "pitcher_id",
        "batter_id",
        *SEGMENT_COLUMNS,
    )
    anchor = aligned.loc[:, base_columns].copy(deep=True)
    anchor.insert(3, "probability", probability)
    multi_frame = aligned.loc[:, base_columns].copy(deep=True)
    multi_frame.insert(3, "probability", aligned["multi"].to_numpy())
    payload = _decision_payload(decision)
    archive_members = {
        "decision.json": _json_bytes(payload),
        "t1_anchor_2024.csv": anchor.to_csv(index=False).encode("utf-8"),
        "t1_multi_2024.csv": multi_frame.to_csv(index=False).encode("utf-8"),
    }
    records = {
        name: {"size_bytes": len(data), "sha256": sha256(data).hexdigest()}
        for name, data in sorted(archive_members.items())
    }
    manifest_payload = {
        "schema_version": 1,
        "artifact_kind": "temporal_t2a_input_v1",
        "review_sha256": decision.review_sha256,
        "data_rows_sha256": decision.data_rows_sha256,
        "decision_sha256": decision.sha256,
        "members": records,
    }
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with ZipFile(temporary, "w") as archive:
        archive.writestr(_zip_info("manifest.json"), _json_bytes(manifest_payload))
        for name, data in sorted(archive_members.items()):
            archive.writestr(_zip_info(name), data)
    os.replace(temporary, destination)
    verify_t2a_input(destination)
    return destination


def verify_t2a_input(path: str | Path) -> VerifiedT2AInput:
    source = Path(path)
    if not source.is_file() or source.is_symlink() or source.stat().st_size > 512_000_000:
        raise T1ReviewError("T2-A input file is invalid")
    try:
        with ZipFile(source) as archive:
            if set(archive.namelist()) != _T2A_MEMBERS or len(archive.infolist()) != 4:
                raise T1ReviewError("T2-A input members differ")
            manifest_raw = archive.read("manifest.json")
            manifest = json.loads(manifest_raw)
            if manifest_raw != _json_bytes(manifest) or manifest.get(
                "artifact_kind"
            ) != "temporal_t2a_input_v1":
                raise T1ReviewError("T2-A input manifest differs")
            records = manifest.get("members")
            if type(records) is not dict or set(records) != _T2A_MEMBERS - {"manifest.json"}:
                raise T1ReviewError("T2-A input member manifest differs")
            for name, record in records.items():
                data = archive.read(name)
                if (
                    type(record) is not dict
                    or set(record) != {"size_bytes", "sha256"}
                    or record["size_bytes"] != len(data)
                    or record["sha256"] != sha256(data).hexdigest()
                ):
                    raise T1ReviewError(f"T2-A input member differs: {name}")
            decision_raw = archive.read("decision.json")
            decision_payload = json.loads(decision_raw)
            decision_sha = sha256(decision_raw).hexdigest()
            if decision_sha != manifest.get("decision_sha256"):
                raise T1ReviewError("T2-A decision binding differs")
            if (
                decision_payload.get("status") != "champion"
                or decision_payload.get("decay") != "0.55"
                or decision_payload.get("recent_weight") != "0.50"
                or decision_payload.get("mode") != "logit"
                or decision_payload.get("beta") != "0"
                or decision_payload.get("data_rows_sha256")
                != manifest.get("data_rows_sha256")
                or decision_payload.get("review_sha256")
                != manifest.get("review_sha256")
            ):
                raise T1ReviewError("T2-A input lacks the fixed champion T1 decision")
            expected_columns = (
                "row_id",
                "valid_year",
                "target",
                "probability",
                "pitcher_id",
                "batter_id",
                *SEGMENT_COLUMNS,
            )
            frames = []
            for name in ("t1_anchor_2024.csv", "t1_multi_2024.csv"):
                frame = pd.read_csv(archive.open(name))
                probability = frame.get("probability")
                if (
                    frame.empty
                    or tuple(frame.columns) != expected_columns
                    or frame["valid_year"].ne(2024).any()
                    or frame["row_id"].duplicated().any()
                    or probability is None
                    or not np.isfinite(probability.to_numpy(dtype="float64")).all()
                    or not probability.between(0, 1).all()
                ):
                    raise T1ReviewError(f"T2-A OOF evidence differs: {name}")
                frames.append(frame)
            reference, candidate = frames
            comparison_columns = tuple(
                column for column in expected_columns if column != "probability"
            )
            if not reference.loc[:, comparison_columns].equals(
                candidate.loc[:, comparison_columns]
            ):
                raise T1ReviewError("T2-A paired OOF evidence differs")
            return VerifiedT2AInput(
                source.resolve(),
                _file_sha256(source),
                decision_sha,
                str(manifest["data_rows_sha256"]),
                str(manifest["review_sha256"]),
            )
    except T1ReviewError:
        raise
    except (OSError, BadZipFile, KeyError, json.JSONDecodeError) as error:
        raise T1ReviewError("cannot verify T2-A input") from error


def _fold_pairs(specs, frames, rates):
    pairs = {}
    for year in (2022, 2023, 2024):
        recent_id = f"t1__recent__va{year}__s3407"
        recent = frames[recent_id]
        for decay in load_contract().decays:
            decay_id = str(decay).replace(".", "p")
            multi_id = f"t1__multi_d{decay_id}__va{year}__s3407"
            pairs[(year, decay)] = (
                _align_pair(recent, frames[multi_id], year),
                rates[recent_id],
                rates[multi_id],
            )
    return pairs


def _align_pair(recent: pd.DataFrame, multi: pd.DataFrame, year: int) -> pd.DataFrame:
    left = recent.assign(valid_year=year)
    right = multi.assign(valid_year=year)
    aligned = align_oof(
        (
            (
                "recent",
                left.drop(columns=list(_MODEL_RELATIVE_SEGMENTS)),
            ),
            (
                "multi",
                right.drop(columns=list(_MODEL_RELATIVE_SEGMENTS)),
            ),
        ),
        segment_columns=_STABLE_ALIGNMENT_COLUMNS,
    )
    positions = {
        _row_id_token(value): position
        for position, value in enumerate(left["row_id"])
    }
    ordered_positions = [positions[_row_id_token(value)] for value in aligned["row_id"]]
    for column in _MODEL_RELATIVE_SEGMENTS:
        aligned[column] = left.iloc[ordered_positions][column].to_numpy(copy=True)
    return aligned.loc[
        :,
        (
            "row_id",
            "valid_year",
            "target",
            "pitcher_id",
            "batter_id",
            *SEGMENT_COLUMNS,
            "recent",
            "multi",
        ),
    ]


def _row_id_token(value: object) -> tuple[str, object]:
    if isinstance(value, bool):
        raise T1ReviewError("T1 row_id is invalid")
    if isinstance(value, Integral):
        return "int", int(value)
    if type(value) is str and value:
        return "str", value
    raise T1ReviewError("T1 row_id is invalid")


def _point_metrics(recipe: T1Recipe, pairs):
    gains = {}
    rows = {}
    for year in (2022, 2023, 2024):
        frame, recent_rate, multi_rate = pairs[(year, recipe.decay)]
        anchor = float(recipe.recent_weight) * recent_rate + (
            1.0 - float(recipe.recent_weight)
        ) * multi_rate
        candidate = _blend(
            frame["recent"].to_numpy(),
            frame["multi"].to_numpy(),
            recipe,
            anchor_rate=anchor,
        )
        gains[year] = brier(frame["target"], frame["recent"]) - brier(
            frame["target"], candidate
        )
        rows[year] = len(frame)
    weighted = sum(gains[year] * rows[year] for year in gains) / sum(rows.values())
    return weighted, gains[2024], gains


def _candidate_oof(recipe: T1Recipe, pairs) -> pd.DataFrame:
    output = []
    for year in (2022, 2023, 2024):
        frame, recent_rate, multi_rate = pairs[(year, recipe.decay)]
        anchor = float(recipe.recent_weight) * recent_rate + (
            1.0 - float(recipe.recent_weight)
        ) * multi_rate
        candidate = _blend(
            frame["recent"].to_numpy(),
            frame["multi"].to_numpy(),
            recipe,
            anchor_rate=anchor,
        )
        part = frame.loc[
            :,
            (
                "row_id",
                "valid_year",
                "pitcher_id",
                "target",
                *SEGMENT_COLUMNS,
            ),
        ].copy(deep=True)
        part["baseline"] = frame["recent"].to_numpy()
        part["candidate"] = candidate
        output.append(part)
    return pd.concat(output, ignore_index=True)


def _blend(left, right, recipe, *, anchor_rate):
    if recipe.mode == "logit":
        probability = blend_logit(left, right, recipe.recent_weight)
    else:
        probability = blend_probability(left, right, recipe.recent_weight)
    if recipe.beta and anchor_rate is not None:
        probability = apply_anchor(
            probability, anchor_rate=anchor_rate, beta=recipe.beta
        )
    return probability


def _bootstrap_lower(frame: pd.DataFrame, *, repeats: int, seed: int) -> float:
    delta = np.square(frame["baseline"].to_numpy() - frame["target"].to_numpy()) - np.square(
        frame["candidate"].to_numpy() - frame["target"].to_numpy()
    )
    work = pd.DataFrame({"pitcher_id": frame["pitcher_id"].astype(str), "delta": delta})
    grouped = work.groupby("pitcher_id", sort=True)["delta"].agg(["sum", "size"])
    sums = grouped["sum"].to_numpy(dtype="float64")
    sizes = grouped["size"].to_numpy(dtype="int64")
    rng = np.random.default_rng(seed)
    gains = np.empty(repeats, dtype="float64")
    for index in range(repeats):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        gains[index] = sums[selected].sum() / sizes[selected].sum()
    return float(np.quantile(gains, 0.025))


def _decision_payload(decision: T1Decision) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "review_sha256": decision.review_sha256,
        "data_rows_sha256": decision.data_rows_sha256,
        "recipe_id": decision.recipe_id,
        "decay": str(decision.decay),
        "recent_weight": str(decision.recent_weight),
        "mode": decision.mode,
        "beta": str(decision.beta),
        "status": decision.status,
        "tier": decision.tier,
        "weighted_gain": decision.weighted_gain,
        "latest_gain": decision.latest_gain,
        "fold_gains": {str(year): value for year, value in decision.fold_gains.items()},
        "bootstrap_lower": decision.bootstrap_lower,
        "max_segment_regression": decision.max_segment_regression,
    }
    if sha256(_json_bytes(payload)).hexdigest() != decision.sha256:
        raise T1ReviewError("T1 decision object integrity differs")
    return payload


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, _ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info
