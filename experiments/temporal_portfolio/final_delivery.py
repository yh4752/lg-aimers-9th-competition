"""Review artifact for a completed fit. Submission creation stays disabled here."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

from .artifacts import canonical_json
from .state import Bindings


class FinalDeliveryError(ValueError):
    pass


class SubmissionNotAuthorized(FinalDeliveryError):
    pass


@dataclass(frozen=True)
class AcceptedFullFit:
    bindings: Bindings
    decision_sha256: str
    confirmation_sha256: str
    frozen_members: Mapping[str, Path]
    acceptance: Mapping[str, object]
    row_independence: Mapping[str, object]
    runtime: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.bindings) is not Bindings:
            raise FinalDeliveryError("full-fit bindings are invalid")
        if not _is_sha(self.decision_sha256) or not _is_sha(self.confirmation_sha256):
            raise FinalDeliveryError("full-fit evidence hashes are invalid")
        if self.acceptance.get("status") != "accepted":
            raise FinalDeliveryError("full-fit acceptance is missing")
        if self.row_independence.get("accepted") is not True:
            raise FinalDeliveryError("row-independence acceptance is missing")
        members: dict[str, Path] = {}
        for name, raw_path in self.frozen_members.items():
            _member_name(name)
            path = Path(raw_path).resolve()
            if not path.is_file() or path.is_symlink():
                raise FinalDeliveryError("frozen member is not a regular file")
            members[name] = path
        if not members:
            raise FinalDeliveryError("frozen members are empty")
        object.__setattr__(self, "frozen_members", MappingProxyType(members))
        object.__setattr__(self, "acceptance", MappingProxyType(dict(self.acceptance)))
        object.__setattr__(
            self, "row_independence", MappingProxyType(dict(self.row_independence))
        )
        object.__setattr__(self, "runtime", MappingProxyType(dict(self.runtime)))


@dataclass(frozen=True)
class VerifiedTrainingDelivery:
    path: Path
    sha256: str
    members: tuple[str, ...]
    policy: Mapping[str, object]


def write_training_delivery(root: str | Path, full_fit: AcceptedFullFit) -> Path:
    if type(full_fit) is not AcceptedFullFit:
        raise FinalDeliveryError("accepted full-fit result is required")
    output_root = Path(root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    members = {
        f"frozen/{name}": path.read_bytes()
        for name, path in full_fit.frozen_members.items()
    }
    frozen_hashes = {
        name: sha256(value).hexdigest() for name, value in sorted(members.items())
    }
    members.update(
        {
            "frozen/manifest.json": canonical_json(
                {
                    "decision_sha256": full_fit.decision_sha256,
                    "confirmation_sha256": full_fit.confirmation_sha256,
                    "members": frozen_hashes,
                }
            ),
            "policy/policy.json": canonical_json(
                {
                    "campaign_id": "temporal_portfolio_v1",
                    "submission_package": False,
                    "evaluation_row_independent": True,
                    "external_data": False,
                }
            ),
            "review/acceptance.json": canonical_json(dict(full_fit.acceptance)),
            "review/row_independence.json": canonical_json(
                dict(full_fit.row_independence)
            ),
            "review/runtime.json": canonical_json(dict(full_fit.runtime)),
        }
    )
    manifest = {
        "artifact_kind": "temporal_training_delivery_v1",
        "bindings": {
            "campaign_id": full_fit.bindings.campaign_id,
            "contract_sha256": full_fit.bindings.contract_sha256,
            "input_manifest_sha256": full_fit.bindings.input_manifest_sha256,
        },
        "members": {
            name: sha256(value).hexdigest() for name, value in sorted(members.items())
        },
    }
    members["manifest.json"] = canonical_json(manifest)
    path = output_root / "temporal_portfolio_training_delivery.zip"
    temporary = path.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(members.items()):
            info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value)
    os.replace(temporary, path)
    verify_training_delivery(path)
    return path


def verify_training_delivery(path: str | Path) -> VerifiedTrainingDelivery:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise FinalDeliveryError("training delivery is missing")
    try:
        with ZipFile(source) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or "manifest.json" not in names:
                raise FinalDeliveryError("training delivery members are invalid")
            if any(info.is_dir() or info.file_size > 512 * 1024 * 1024 for info in infos):
                raise FinalDeliveryError("training delivery member is invalid")
            for name in names:
                _member_name(name)
            values = {info.filename: archive.read(info) for info in infos}
    except (BadZipFile, OSError) as error:
        raise FinalDeliveryError("training delivery cannot be read") from error
    manifest = _json(values.pop("manifest.json"), "delivery manifest")
    if manifest.get("artifact_kind") != "temporal_training_delivery_v1":
        raise FinalDeliveryError("training delivery kind differs")
    declared = manifest.get("members")
    observed = {name: sha256(value).hexdigest() for name, value in sorted(values.items())}
    if declared != observed:
        raise FinalDeliveryError("training delivery member hashes differ")
    required = {
        "frozen/manifest.json",
        "policy/policy.json",
        "review/acceptance.json",
        "review/row_independence.json",
        "review/runtime.json",
    }
    if not required.issubset(values) or any(
        PurePosixPath(name).name == "submit.zip" for name in values
    ):
        raise FinalDeliveryError("training delivery policy members differ")
    policy = _json(values["policy/policy.json"], "delivery policy")
    if policy != {
        "campaign_id": "temporal_portfolio_v1",
        "submission_package": False,
        "evaluation_row_independent": True,
        "external_data": False,
    }:
        raise FinalDeliveryError("training delivery policy differs")
    if _json(values["review/acceptance.json"], "acceptance").get("status") != "accepted":
        raise FinalDeliveryError("training acceptance differs")
    if _json(values["review/row_independence.json"], "row independence").get("accepted") is not True:
        raise FinalDeliveryError("row-independence evidence differs")
    return VerifiedTrainingDelivery(
        source,
        sha256(source.read_bytes()).hexdigest(),
        tuple(sorted((*values, "manifest.json"))),
        MappingProxyType(policy),
    )


def assert_submission_packaging_authorized(_value: object) -> None:
    raise SubmissionNotAuthorized(
        "submission packaging requires a separate reviewed step"
    )


def _json(value: bytes, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalDeliveryError(f"{label} is invalid") from error
    if type(parsed) is not dict or canonical_json(parsed) != value:
        raise FinalDeliveryError(f"{label} is not canonical")
    return parsed


def _member_name(name: object) -> None:
    if type(name) is not str or not name or "\\" in name:
        raise FinalDeliveryError("delivery member name is invalid")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise FinalDeliveryError("delivery member path is unsafe")


def _is_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
