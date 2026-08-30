from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import time
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from .contracts import load_contract
from .inputs import canonical_json
from .inputs import verify_and_extract_final_input


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


def verify_t4x2(torch_module) -> tuple[str, str]:
    names = tuple(
        torch_module.cuda.get_device_name(index)
        for index in range(torch_module.cuda.device_count())
    )
    if len(names) != 2 or any("T4" not in name for name in names):
        raise FinalKaggleError("two Tesla T4 GPUs are required")
    return names


def _smoke_gpu() -> None:
    from catboost import CatBoostClassifier

    frame = pd.DataFrame({"x": np.arange(200), "category": ["a", "b"] * 100})
    target = np.asarray([0, 1] * 100)

    def fit(device: int) -> None:
        CatBoostClassifier(
            iterations=20, depth=4, task_type="GPU", devices=str(device), verbose=False,
        ).fit(frame, target, cat_features=[1])

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(fit, (0, 1)))


def run_kaggle_campaign(input_root: Path, work_root: Path):
    import torch

    from .runner import run_campaign
    from .runtime import ProductionFinalRuntime

    found = discover_inputs(Path(input_root))
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=False)
    verified = verify_and_extract_final_input(found.final_input, root / "verified_input")
    print(
        f"FINAL_CANDIDATE_INPUTS_VERIFIED input_sha256={verified.manifest_sha256}",
        flush=True,
    )
    names = verify_t4x2(torch)
    print(f"FINAL_CANDIDATE_GPU_READY count=2 names={' | '.join(names)}", flush=True)
    _smoke_gpu()
    print("FINAL_CANDIDATE_SMOKE_SUCCESS", flush=True)
    runtime = ProductionFinalRuntime(
        official_root=found.official_data,
        verified_input=verified,
        work_root=root / "runtime",
    )
    result = run_campaign(
        runtime=runtime,
        output_dir=root / "campaign",
        previous_handoff=found.previous_handoff,
        absolute_deadline=time.time() + load_contract().maximum_runtime_seconds,
    )
    print(f"FINAL_CANDIDATE_CAMPAIGN_STATUS status={result.status}", flush=True)
    print(f"FINAL_CANDIDATE_REVIEW_READY path={result.review.resolve()}", flush=True)
    print(f"FINAL_CANDIDATE_HANDOFF_READY path={result.handoff.resolve()}", flush=True)
    if result.delivery is not None:
        print(f"FINAL_CANDIDATE_DELIVERY_READY path={result.delivery.resolve()}", flush=True)
    return result
