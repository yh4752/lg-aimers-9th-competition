from __future__ import annotations

import argparse
import base64
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import gzip
import io
import json
from pathlib import Path
import tarfile
import threading
import time
from types import MappingProxyType
from typing import Mapping
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from .artifacts import DirectExpertBindings
from .contracts import ExpertJob, load_contract
from .features import fit_direct_features, transform_direct_features
from .inputs import (
    EXPECTED_E2_SUBMISSION_SHA256,
    canonical_json,
    file_sha256,
    verify_and_extract_input,
)
from .runtime_inventory import code_identity_sha256, runtime_members
from .stage_a import run_stage_a
from .stage_b import run_stage_b
from .training import FoldData, run_fold_job


class DirectExpertKaggleError(ValueError):
    pass


@dataclass(frozen=True)
class DiscoveredInputs:
    official_data: Path
    campaign_input: Path
    stage_a_handoff: Path | None = None
    previous_stage_b_handoff: Path | None = None


def _artifact_descriptor(path: Path) -> tuple[str, str] | None:
    try:
        if path.is_dir():
            manifest = (path / "manifest.json").read_bytes()
        else:
            with ZipFile(path) as archive:
                manifest = archive.read("manifest.json")
        payload = json.loads(manifest)
    except (OSError, KeyError, BadZipFile, json.JSONDecodeError):
        return None
    if type(payload) is not dict or type(payload.get("artifact_kind")) is not str:
        return None
    return payload["artifact_kind"], sha256(canonical_json(payload)).hexdigest()


def discover_inputs(root: Path, *, stage: str) -> DiscoveredInputs:
    source = Path(root)
    official = []
    campaigns: dict[str, Path] = {}
    stage_a: dict[str, Path] = {}
    stage_b: dict[str, Path] = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_dir() and (path / "train.csv").is_file() and (path / "trackman_history.csv").is_file():
            official.append(path)
        if (path.is_file() and path.suffix.lower() == ".zip") or (path.is_dir() and (path / "manifest.json").is_file()):
            descriptor = _artifact_descriptor(path)
            if descriptor is None:
                continue
            kind, identity = descriptor
            if kind == "direct_expert_input_v1": campaigns.setdefault(identity, path)
            elif kind == "direct_expert_stage_a_handoff_v1": stage_a.setdefault(identity, path)
            elif kind == "direct_expert_handoff_v1": stage_b.setdefault(identity, path)
    if len(official) != 1:
        raise DirectExpertKaggleError(f"official data count must be one; found={len(official)}")
    if len(campaigns) != 1:
        raise DirectExpertKaggleError(f"campaign input count must be one; found={len(campaigns)}")
    if stage == "A":
        if len(stage_a) > 1:
            raise DirectExpertKaggleError(f"Stage A handoff count must be zero or one; found={len(stage_a)}")
        return DiscoveredInputs(official[0], next(iter(campaigns.values())), next(iter(stage_a.values())) if stage_a else None)
    if stage != "B" or len(stage_a) != 1 or len(stage_b) > 1:
        raise DirectExpertKaggleError("Stage B handoff counts differ")
    return DiscoveredInputs(
        official[0],
        next(iter(campaigns.values())),
        next(iter(stage_a.values())),
        next(iter(stage_b.values())) if stage_b else None,
    )


def verify_t4x2(torch_module) -> tuple[str, str]:
    names = tuple(torch_module.cuda.get_device_name(index) for index in range(torch_module.cuda.device_count()))
    if len(names) != 2 or any("T4" not in name for name in names):
        raise DirectExpertKaggleError("two Tesla T4 GPUs are required")
    return names


class ProductionStageARuntime:
    def __init__(self, official_root: Path, campaign_input: Path, work_root: Path):
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        train_path = Path(official_root) / "train.csv"
        history_path = Path(official_root) / "trackman_history.csv"
        train_sha = file_sha256(train_path)
        history_sha = file_sha256(history_path)
        if train_sha != "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff" or history_sha != "f7818f9ee0ccefe7c2cf69fa99efe6e5cb882d8b886dd96d2394bcf3b53f33a9":
            raise DirectExpertKaggleError("official data SHA-256 differs")
        self.verified = verify_and_extract_input(campaign_input, self.work_root / "verified_input")
        if self.verified.train_sha256 != train_sha or self.verified.history_sha256 != history_sha:
            raise DirectExpertKaggleError("campaign input data binding differs")
        self.train = pd.read_csv(train_path)
        self.history = pd.read_csv(history_path)
        contract_path = Path(__file__).with_name("contract.json")
        repository_root = Path(__file__).resolve().parents[2]
        self.bindings = DirectExpertBindings(
            file_sha256(contract_path),
            code_identity_sha256(repository_root),
            self.verified.manifest_sha256,
            train_sha,
            history_sha,
            EXPECTED_E2_SUBMISSION_SHA256,
        )
        self._folds: OrderedDict[int, FoldData] = OrderedDict()
        self._fold_lock = threading.Lock()

    def now(self) -> float:
        return time.time()

    def _fold(self, valid_year: int) -> FoldData:
        with self._fold_lock:
            if valid_year not in self._folds:
                prefix = self.train.loc[pd.to_numeric(self.train["season"]).lt(valid_year)].copy()
                valid_source = self.train.loc[pd.to_numeric(self.train["season"]).eq(valid_year)].copy()
                state, train_batch = fit_direct_features(prefix, self.history, valid_year=valid_year)
                target = pd.to_numeric(valid_source.pop("control_success"), errors="raise").to_numpy(dtype="int8")
                valid_batch = transform_direct_features(valid_source, state)
                metadata = valid_source.loc[:, ["row_id", "game_type", "pitcher_id"]].copy()
                self._folds[valid_year] = FoldData(
                    train_batch,
                    valid_batch,
                    target,
                    metadata,
                    state.categorical_columns,
                    MappingProxyType({
                        "contract": self.bindings.contract_sha256,
                        "code": self.bindings.code_sha256,
                        "train": self.bindings.train_sha256,
                        "history": self.bindings.history_sha256,
                        "input": self.bindings.input_manifest_sha256,
                    }),
                )
                while len(self._folds) > 2:
                    self._folds.popitem(last=False)
            result = self._folds[valid_year]
            self._folds.move_to_end(valid_year)
            return result

    def clear_fold_cache(self) -> None:
        with self._fold_lock:
            self._folds.clear()

    def run_job(self, job: ExpertJob, gpu: int, output: Path):
        print(f"DIRECT_EXPERT_JOB_START candidate={job.job_id} fold={job.fold[0]}_{job.fold[1]} seed={job.seed} gpu={gpu}", flush=True)
        result = run_fold_job(job, self._fold(job.fold[1]), output, gpu=gpu)
        print(f"DIRECT_EXPERT_JOB_END candidate={job.job_id} status={result.status}", flush=True)
        return result

    def e2_oof(self, year: int) -> pd.DataFrame:
        return pd.read_csv(self.verified.e2_oof_paths[year])


def _smoke_gpu(torch_module) -> None:
    from catboost import CatBoostClassifier
    x = pd.DataFrame({"x": np.arange(200), "category": ["a", "b"] * 100})
    y = np.asarray([0, 1] * 100)
    def fit(device: int) -> None:
        CatBoostClassifier(iterations=20, depth=4, task_type="GPU", devices=str(device), verbose=False).fit(x, y, cat_features=[1])
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(fit, (0, 1)))


def run_kaggle_stage_a(input_root: Path, work_root: Path) -> Path:
    import torch
    found = discover_inputs(input_root, stage="A")
    print("DIRECT_EXPERT_INPUTS_VERIFIED", flush=True)
    names = verify_t4x2(torch)
    print(f"DIRECT_EXPERT_GPU_READY count=2 names={' | '.join(names)}", flush=True)
    _smoke_gpu(torch)
    print("DIRECT_EXPERT_SMOKE_SUCCESS", flush=True)
    runtime = ProductionStageARuntime(found.official_data, found.campaign_input, Path(work_root) / "runtime")
    deadline = time.time() + int(load_contract().runtime["stage_a_wall_seconds"])
    result = run_stage_a(
        runtime,
        Path(work_root) / "campaign",
        absolute_deadline=deadline,
        previous_handoff=found.stage_a_handoff,
    )
    print(f"DIRECT_EXPERT_CAMPAIGN_STATUS stage=A status={result.status}", flush=True)
    print(f"DIRECT_EXPERT_HANDOFF_READY path={result.handoff.resolve()}", flush=True)
    return result.handoff


def run_kaggle_stage_b(input_root: Path, work_root: Path):
    import torch
    from .stage_b_runtime import ProductionStageBRuntime

    found = discover_inputs(input_root, stage="B")
    print("DIRECT_EXPERT_INPUTS_VERIFIED", flush=True)
    names = verify_t4x2(torch)
    print(f"DIRECT_EXPERT_GPU_READY count=2 names={' | '.join(names)}", flush=True)
    _smoke_gpu(torch)
    print("DIRECT_EXPERT_SMOKE_SUCCESS", flush=True)
    runtime = ProductionStageBRuntime(
        found.official_data,
        found.campaign_input,
        found.stage_a_handoff,
        Path(work_root) / "runtime",
    )
    deadline = time.time() + int(load_contract().runtime["stage_b_wall_seconds"])
    result = run_stage_b(
        runtime,
        found.stage_a_handoff,
        Path(work_root) / "campaign",
        absolute_deadline=deadline,
        previous_handoff=found.previous_stage_b_handoff,
    )
    print(f"DIRECT_EXPERT_CAMPAIGN_STATUS stage=B status={result.status}", flush=True)
    print(f"DIRECT_EXPERT_REVIEW_READY path={result.review.resolve()}", flush=True)
    print(f"DIRECT_EXPERT_HANDOFF_READY path={result.handoff.resolve()}", flush=True)
    if result.delivery is not None:
        print(f"DIRECT_EXPERT_DELIVERY_READY path={result.delivery.resolve()}", flush=True)
    return result


def _runtime_archive(root: Path) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name in runtime_members(root):
                payload = (root / name).read_bytes()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mtime = 0
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def build_kaggle_cell(stage: str, output: Path, *, root: Path | None = None) -> Path:
    if stage not in {"A", "B"}:
        raise DirectExpertKaggleError("cell stage differs")
    repository = Path(root) if root else Path(__file__).resolve().parents[2]
    payload = _runtime_archive(repository)
    encoded = base64.b64encode(payload).decode("ascii")
    if stage == "A":
        run_line = "run_kaggle_stage_a(Path('/kaggle/input'), Path('/kaggle/working/direct_expert'))"
    else:
        run_line = "run_kaggle_stage_b(Path('/kaggle/input'), Path('/kaggle/working/direct_expert'))"
    terminal_markers = (
        "# terminal markers: DIRECT_EXPERT_HANDOFF_READY\n"
        if stage == "A"
        else "# terminal markers: DIRECT_EXPERT_REVIEW_READY DIRECT_EXPERT_HANDOFF_READY DIRECT_EXPERT_DELIVERY_READY\n"
    )
    source = f'''from __future__ import annotations
import base64, importlib.metadata, io, subprocess, sys, tarfile
from hashlib import sha256
from pathlib import Path

PAYLOAD = base64.b64decode("{encoded}")
print(f"DIRECT_EXPERT_CODE_READY sha256={{sha256(PAYLOAD).hexdigest()}} size_bytes={{len(PAYLOAD)}}", flush=True)
try:
    version = importlib.metadata.version("catboost")
except importlib.metadata.PackageNotFoundError:
    version = None
if version != "1.2.10":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "catboost==1.2.10"], check=True)
runtime_root = Path("/kaggle/working/direct_expert_runtime")
runtime_root.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(PAYLOAD), mode="r:gz") as archive:
    for member in archive.getmembers():
        target = (runtime_root / member.name).resolve()
        if not target.is_relative_to(runtime_root.resolve()) or not member.isfile():
            raise RuntimeError("unsafe embedded runtime member")
    archive.extractall(runtime_root)
sys.path.insert(0, str(runtime_root))
from experiments.direct_expert.kaggle import run_kaggle_stage_a, run_kaggle_stage_b
{terminal_markers}
{run_line}
'''.encode()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("A", "B"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build_kaggle_cell(args.stage, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
