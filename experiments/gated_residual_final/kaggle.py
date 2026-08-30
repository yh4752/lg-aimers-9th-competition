from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from .inputs import canonical_json


class FinalKaggleError(ValueError):
    pass


@dataclass(frozen=True)
class DiscoveredInputs:
    official_data: Path
    final_input: Path
    previous_handoff: Path | None


def _descriptor(path: Path) -> tuple[str, str] | None:
    try:
        if path.is_dir():
            payload = (path / "manifest.json").read_bytes()
        else:
            with ZipFile(path) as archive:
                payload = archive.read("manifest.json")
        manifest = json.loads(payload)
    except (OSError, KeyError, BadZipFile, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if type(manifest) is not dict or type(manifest.get("artifact_kind")) is not str:
        return None
    return manifest["artifact_kind"], sha256(canonical_json(manifest)).hexdigest()


def discover_inputs(root: Path) -> DiscoveredInputs:
    official: set[Path] = set()
    final_inputs: dict[str, Path] = {}
    handoffs: dict[str, Path] = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            continue
        if path.is_dir() and (path / "train.csv").is_file() and (path / "trackman_history.csv").is_file():
            official.add(path)
        if not ((path.is_file() and path.suffix.lower() == ".zip") or (path.is_dir() and (path / "manifest.json").is_file())):
            continue
        descriptor = _descriptor(path)
        if descriptor is None:
            continue
        kind, identity = descriptor
        if kind == "gated_residual_final_input_v1":
            final_inputs.setdefault(identity, path)
        elif kind == "gated_residual_final_handoff_v1":
            handoffs.setdefault(identity, path)
    if len(official) != 1:
        raise FinalKaggleError(f"official data count must be one; found={len(official)}")
    if len(final_inputs) != 1:
        raise FinalKaggleError(f"final input count must be one; found={len(final_inputs)}")
    if len(handoffs) > 1:
        raise FinalKaggleError(f"handoff count must be zero or one; found={len(handoffs)}")
    return DiscoveredInputs(
        official_data=next(iter(official)),
        final_input=next(iter(final_inputs.values())),
        previous_handoff=next(iter(handoffs.values())) if handoffs else None,
    )
