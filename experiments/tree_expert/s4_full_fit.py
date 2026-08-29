from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .s4_artifacts import S4Bindings
from .s4_contracts import load_s4_contract
from .s4_decisions import S4Decision


class S4FullFitError(ValueError):
    pass


@dataclass(frozen=True)
class S4FullFitToken:
    schema_version: int
    status: str
    candidate_id: str
    full_fit_iterations: int
    fold_iterations: Mapping[tuple[int, int], int]
    bindings: S4Bindings


def median_plus_one_iterations(
    fold_iterations: Mapping[tuple[int, int], int], *, maximum: int
) -> int:
    folds = load_s4_contract().folds
    if set(fold_iterations) != set(folds) or type(maximum) is not int or maximum <= 0:
        raise S4FullFitError("fold iteration evidence differs")
    values = []
    for fold in folds:
        value = fold_iterations[fold]
        if type(value) is not int or type(value) is bool or value < 0:
            raise S4FullFitError("fold iteration value differs")
        values.append(value)
    median = sorted(values)[len(values) // 2]
    return min(maximum, median + 1)


def issue_full_fit_token(
    decision: S4Decision,
    bindings: S4Bindings,
    *,
    fold_iterations: Mapping[tuple[int, int], int],
    maximum_iterations: int | None = None,
) -> S4FullFitToken:
    if type(decision) is not S4Decision or decision.status != "accepted" or decision.failed_gates:
        raise S4FullFitError("accepted decision required")
    contract = load_s4_contract()
    maximum = int(contract.parameters["catboost"]["iterations"]) if maximum_iterations is None else maximum_iterations
    iterations = median_plus_one_iterations(fold_iterations, maximum=maximum)
    return S4FullFitToken(
        schema_version=1,
        status="accepted",
        candidate_id=decision.candidate_id,
        full_fit_iterations=iterations,
        fold_iterations=MappingProxyType(dict(fold_iterations)),
        bindings=bindings,
    )


def full_fit_token_payload(token: S4FullFitToken) -> dict[str, object]:
    return {
        "schema_version": token.schema_version,
        "status": token.status,
        "candidate_id": token.candidate_id,
        "full_fit_iterations": token.full_fit_iterations,
        "fold_iterations": {
            f"{fold[0]}->{fold[1]}": value for fold, value in token.fold_iterations.items()
        },
        "bindings": asdict(token.bindings),
    }


def save_full_fit_token(token: S4FullFitToken, path: Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(full_fit_token_payload(token), sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output
