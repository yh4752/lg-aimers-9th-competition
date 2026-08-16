from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.tabm_campaign.ensemble_audit import (
    EnsembleAuditError,
    audit_prediction_frames,
    load_ensemble_contract,
)


CONTRACT_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "tabm_campaign"
    / "score_improvement_contract.json"
)


def _contract_payload() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _write_contract(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    return path


def test_contract_seals_folds_seeds_ensembles_and_gates() -> None:
    contract = load_ensemble_contract()

    assert contract.folds == ("2022->2023", "2023->2024")
    assert contract.seeds == (42, 2026, 3407)
    assert tuple(contract.ensembles) == ("mean_42_3407", "mean_all")
    assert dict(contract.ensembles["mean_42_3407"]) == {42: 0.5, 3407: 0.5}
    assert dict(contract.ensembles["mean_all"]) == {
        42: pytest.approx(1.0 / 3.0),
        2026: pytest.approx(1.0 / 3.0),
        3407: pytest.approx(1.0 / 3.0),
    }
    assert contract.min_weighted_gain == pytest.approx(0.00003)
    assert contract.max_fold_degrade == pytest.approx(0.00003)

    with pytest.raises(TypeError):
        contract.ensembles["mean_all"][42] = 1.0  # type: ignore[index]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update({"extra": True}), "unknown keys"),
        (lambda payload: payload.update({"seeds": [42, 2026, 9999]}), "seeds"),
        (lambda payload: payload.update({"seeds": [42, 42, 3407]}), "seeds"),
        (
            lambda payload: payload["ensembles"].update(  # type: ignore[union-attr]
                {"mean_all": {"42": 0.5, "2026": 0.5, "3407": 0.5}}
            ),
            "sum to 1",
        ),
        (
            lambda payload: payload["ensembles"].update(  # type: ignore[union-attr]
                {"mean_all": {"42": 0.5, "2026": -0.1, "3407": 0.6}}
            ),
            "positive",
        ),
        (
            lambda payload: payload["ensembles"].update(  # type: ignore[union-attr]
                {"mean_all": {"42": 0.4, "2026": 0.3, "9999": 0.3}}
            ),
            "unknown seed",
        ),
        (lambda payload: payload.update({"min_weighted_gain": float("nan")}), "finite"),
        (lambda payload: payload.update({"max_fold_degrade": -0.1}), "non-negative"),
    ],
)
def test_contract_rejects_invalid_values(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    payload = _contract_payload()
    mutation(payload)  # type: ignore[operator]

    with pytest.raises(EnsembleAuditError, match=message):
        load_ensemble_contract(_write_contract(tmp_path, payload))


def test_contract_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(
        '{"schema_version":1,"schema_version":1,"folds":[],"seeds":[],"ensembles":{},'
        '"min_weighted_gain":0.00003,"max_fold_degrade":0.00003}',
        encoding="utf-8",
    )

    with pytest.raises(EnsembleAuditError, match="duplicate JSON key"):
        load_ensemble_contract(path)


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [
                "TRAIN_2023_1",
                "TRAIN_2023_2",
                "TRAIN_2023_3",
                "TRAIN_2023_4",
                "TRAIN_2024_1",
                "TRAIN_2024_2",
                "TRAIN_2024_3",
                "TRAIN_2024_4",
            ],
            "season": [2023] * 4 + [2024] * 4,
            "control_success": [0, 1, 0, 1] * 2,
        }
    )


def _prediction_frames() -> dict[tuple[str, int], pd.DataFrame]:
    probabilities = {
        42: [0.1, 0.7, 0.3, 0.9],
        2026: [0.2, 0.75, 0.25, 0.8],
        3407: [0.31, 0.9, 0.1, 0.7],
    }
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for fold, year in (("2022->2023", 2023), ("2023->2024", 2024)):
        for seed, values in probabilities.items():
            frame = pd.DataFrame(
                {
                    "row_id": [f"TRAIN_{year}_{index}" for index in range(1, 5)],
                    "target": [0, 1, 0, 1],
                    "probability": values,
                    "ignored_column": ["x"] * 4,
                }
            )
            frames[(fold, seed)] = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return frames


def test_audit_aligns_rows_and_promotes_the_best_sealed_ensemble() -> None:
    result = audit_prediction_frames(load_ensemble_contract(), _prediction_frames(), _truth())

    assert result.decision == "promoted"
    assert result.baseline_candidate_id == "seed_42"
    assert result.selected_candidate_id == "mean_42_3407"
    assert result.row_counts == {"2022->2023": 4, "2023->2024": 4}
    assert result.candidates["seed_42"].fold_brier == {
        "2022->2023": pytest.approx(0.05),
        "2023->2024": pytest.approx(0.05),
    }
    promoted = result.candidates["mean_42_3407"]
    assert promoted.weighted_brier == pytest.approx(0.04050625)
    assert promoted.weighted_gain == pytest.approx(0.00949375)
    assert promoted.worst_fold_degrade == pytest.approx(-0.00949375)
    assert promoted.gate_passed is True


def test_audit_keeps_the_best_single_when_no_ensemble_clears_the_gate() -> None:
    frames = _prediction_frames()
    for fold in ("2022->2023", "2023->2024"):
        reference = frames[(fold, 42)].set_index("row_id")["probability"]
        for seed in (42, 2026, 3407):
            frame = frames[(fold, seed)]
            frame["probability"] = frame["row_id"].map(reference)

    result = audit_prediction_frames(load_ensemble_contract(), frames, _truth())

    assert result.decision == "keep_single"
    assert result.selected_candidate_id == result.baseline_candidate_id
    assert not any(
        result.candidates[candidate_id].gate_passed
        for candidate_id in ("mean_all", "mean_42_3407")
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda frames: frames[("2022->2023", 42)].drop(index=0, inplace=True),
            "row_id set differs",
        ),
        (
            lambda frames: frames[("2022->2023", 42)].__setitem__(
                "row_id", ["duplicate"] * 4
            ),
            "duplicate row_id",
        ),
        (
            lambda frames: frames[("2022->2023", 42)].__setitem__(
                "target", [1, 1, 0, 1]
            ),
            "target differs",
        ),
        (
            lambda frames: frames[("2022->2023", 42)].__setitem__(
                "probability", [0.2, np.nan, 0.3, 0.4]
            ),
            "probability",
        ),
        (
            lambda frames: frames[("2022->2023", 42)].__setitem__(
                "probability", [0.2, 1.01, 0.3, 0.4]
            ),
            "probability",
        ),
    ],
)
def test_audit_rejects_untrustworthy_prediction_frames(
    mutation: object,
    message: str,
) -> None:
    frames = _prediction_frames()
    mutation(frames)  # type: ignore[operator]

    with pytest.raises(EnsembleAuditError, match=message):
        audit_prediction_frames(load_ensemble_contract(), frames, _truth())


def test_audit_rejects_duplicate_or_incomplete_truth() -> None:
    duplicate = pd.concat([_truth(), _truth().iloc[[0]]], ignore_index=True)
    with pytest.raises(EnsembleAuditError, match="truth has duplicate row_id"):
        audit_prediction_frames(load_ensemble_contract(), _prediction_frames(), duplicate)

    incomplete = _truth().iloc[:-1].copy()
    with pytest.raises(EnsembleAuditError, match="row_id set differs"):
        audit_prediction_frames(load_ensemble_contract(), _prediction_frames(), incomplete)
