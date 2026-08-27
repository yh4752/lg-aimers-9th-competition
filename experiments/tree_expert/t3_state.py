from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from experiments.independent_dl.feature_sources.seasonal import SeasonalSnapshot
from experiments.temporal_portfolio.seasonal_features import S1State

from .features import TreeFeatureState


class T3StateError(ValueError):
    pass


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")


def _dtypes(frame: pd.DataFrame) -> dict[str, str]:
    return {str(column): str(dtype) for column, dtype in frame.dtypes.items()}


def _cast(frame: pd.DataFrame, dtypes: dict[str, str]) -> pd.DataFrame:
    if set(frame.columns) != set(dtypes):
        raise T3StateError("frozen-state columns differ")
    output = frame.copy(deep=True)
    for column in output:
        output[column] = output[column].astype(object if dtypes[column] == "object" else dtypes[column])
    return output


def export_frozen_tree_state(
    state: TreeFeatureState,
    destination: Path,
    *,
    candidate_id: str = "c1_anchor_residual",
) -> Path:
    if type(state) is not TreeFeatureState or candidate_id != "c1_anchor_residual" or state.trackman_state is not None:
        raise T3StateError("T3 frozen-state identity differs")
    root = Path(destination)
    if root.exists() and (root.is_symlink() or any(root.iterdir())):
        raise T3StateError("frozen-state destination is not empty")
    root.mkdir(parents=True, exist_ok=True)
    snapshot = state.s1_state.snapshot
    payloads = {
        "s1_pitcher.csv": _frame_bytes(snapshot.pitcher),
        "s1_batter.csv": _frame_bytes(snapshot.batter),
    }
    manifest = {
        "schema_version": 1, "artifact_kind": "tree_expert_t3_frozen_state_v1",
        "candidate_id": candidate_id, "valid_year": state.valid_year,
        "prior_rate": state.prior_rate,
        "categorical_columns": list(state.categorical_columns),
        "feature_columns": list(state.feature_columns),
        "source_hashes": dict(state.source_hashes), "anchor_formula": "e1_anchor_v1",
        "s1": {
            "cutoff_year": snapshot.cutoff_year,
            "pitcher_dtypes": _dtypes(snapshot.pitcher),
            "batter_dtypes": _dtypes(snapshot.batter),
        },
        "members": {
            name: {"sha256": sha256(payload).hexdigest(), "size": len(payload)}
            for name, payload in sorted(payloads.items())
        },
    }
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    (root / "feature_state.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    return root


def load_frozen_tree_state(root: Path) -> TreeFeatureState:
    source = Path(root)
    try:
        manifest = json.loads((source / "feature_state.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise T3StateError("frozen-state manifest cannot be loaded") from error
    if (
        type(manifest) is not dict
        or manifest.get("artifact_kind") != "tree_expert_t3_frozen_state_v1"
        or manifest.get("candidate_id") != "c1_anchor_residual"
        or manifest.get("anchor_formula") != "e1_anchor_v1"
    ):
        raise T3StateError("frozen-state identity differs")
    payloads: dict[str, bytes] = {}
    for name in ("s1_pitcher.csv", "s1_batter.csv"):
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise T3StateError(f"frozen-state member is missing: {name}")
        payload = path.read_bytes()
        metadata = manifest.get("members", {}).get(name, {})
        if metadata.get("size") != len(payload) or metadata.get("sha256") != sha256(payload).hexdigest():
            raise T3StateError(f"frozen-state SHA-256 differs: {name}")
        payloads[name] = payload
    pitcher = _cast(pd.read_csv(source / "s1_pitcher.csv"), manifest["s1"]["pitcher_dtypes"])
    batter = _cast(pd.read_csv(source / "s1_batter.csv"), manifest["s1"]["batter_dtypes"])
    snapshot = SeasonalSnapshot(
        cutoff_year=int(manifest["s1"]["cutoff_year"]), pitcher=pitcher, batter=batter,
    )
    s1 = S1State._from_snapshot(
        valid_year=int(manifest["valid_year"]),
        prior_rate=float(manifest["prior_rate"]), snapshot=snapshot,
    )
    return TreeFeatureState(
        valid_year=int(manifest["valid_year"]),
        prior_rate=float(manifest["prior_rate"]),
        categorical_columns=tuple(manifest["categorical_columns"]),
        feature_columns=tuple(manifest["feature_columns"]),
        s1_state=s1, trackman_state=None,
        source_hashes=MappingProxyType(dict(manifest["source_hashes"])),
    )
