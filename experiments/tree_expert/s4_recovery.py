from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class S4RecoveryError(ValueError):
    pass


_CONTRACT_KEYS = {
    "schema_version",
    "artifact_kind",
    "source_handoff_sha256",
    "predecessor_code_sha256",
    "s4_contract_sha256",
    "source_bindings",
}
_BINDING_KEYS = {
    "contract_sha256",
    "code_sha256",
    "input_manifest_sha256",
    "train_sha256",
    "history_sha256",
    "e2_handoff_sha256",
}


@dataclass(frozen=True)
class S4RecoveryContract:
    schema_version: int
    artifact_kind: str
    source_handoff_sha256: str
    predecessor_code_sha256: str
    s4_contract_sha256: str
    source_bindings: Mapping[str, str]


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def parse_recovery_contract(payload: object) -> S4RecoveryContract:
    if type(payload) is not dict or set(payload) != _CONTRACT_KEYS:
        raise S4RecoveryError("recovery contract keys differ")
    bindings = payload["source_bindings"]
    if type(bindings) is not dict or set(bindings) != _BINDING_KEYS:
        raise S4RecoveryError("recovery contract binding keys differ")
    hashes = (
        payload["source_handoff_sha256"],
        payload["predecessor_code_sha256"],
        payload["s4_contract_sha256"],
        *bindings.values(),
    )
    if any(not _is_sha256(value) for value in hashes):
        raise S4RecoveryError("recovery contract SHA-256 differs")
    if (
        payload["schema_version"] != 1
        or payload["artifact_kind"] != "tree_s4_recovery_contract_v1"
        or payload["predecessor_code_sha256"] != bindings["code_sha256"]
        or payload["s4_contract_sha256"] != bindings["contract_sha256"]
    ):
        raise S4RecoveryError("recovery contract identity differs")
    return S4RecoveryContract(
        schema_version=1,
        artifact_kind="tree_s4_recovery_contract_v1",
        source_handoff_sha256=str(payload["source_handoff_sha256"]),
        predecessor_code_sha256=str(payload["predecessor_code_sha256"]),
        s4_contract_sha256=str(payload["s4_contract_sha256"]),
        source_bindings=MappingProxyType(dict(bindings)),
    )


def load_recovery_contract(path: Path | None = None) -> S4RecoveryContract:
    source = Path(__file__).with_name("s4_recovery_contract.json") if path is None else Path(path)
    if source.is_symlink() or not source.is_file():
        raise S4RecoveryError("recovery contract source differs")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S4RecoveryError("recovery contract is invalid") from error
    return parse_recovery_contract(payload)
