from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.tabm_campaign.ensemble_audit import (
    EnsembleAuditError,
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
