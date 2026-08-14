from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from typing import Callable, Mapping

from .final_review import review_frozen_artifact
from .final_training import (
    fit_final_member,
    load_epoch_checkpoint,
    resolve_final_candidate,
)
from .version_d import (
    RecoverySelection,
    canonical_json,
    extract_review_inputs,
    extract_training_input,
    file_sha256,
    load_version_d_contract,
    restore_emergency_snapshot,
    restore_frozen_snapshot,
    select_recovery_snapshot,
    verify_stage_c_delivery,
    write_emergency_snapshot,
    write_frozen_snapshot,
    write_review_delivery,
)


class ColabVersionDError(RuntimeError):
    pass


@dataclass(frozen=True)
class VersionDContext:
    contract: Mapping[str, object]
    contract_sha256: str
    stage_c: Mapping[str, object]
    checkpoint_identity: Mapping[str, str]
    candidate: Mapping[str, object]


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def verify_inputs(
    data_archive: Path,
    stage_c_delivery: Path,
    runtime_sha256: str,
    runtime_versions: Mapping[str, str],
) -> VersionDContext:
    if not _valid_sha256(runtime_sha256):
        raise ColabVersionDError("embedded runtime SHA-256 differs")
    contract_path = Path(__file__).with_name("version_d_contract.json")
    contract = load_version_d_contract(contract_path)
    data_contract = contract["data_archive"]
    if file_sha256(data_archive) != data_contract["sha256"]:
        raise ColabVersionDError("official data archive SHA-256 differs")
    stage_c = verify_stage_c_delivery(stage_c_delivery, contract)
    contract_sha256 = file_sha256(contract_path)
    environment_sha256 = sha256(canonical_json(dict(runtime_versions))).hexdigest()
    checkpoint_identity = {
        "contract_sha256": contract_sha256,
        "data_archive_sha256": str(data_contract["sha256"]),
        "train_sha256": str(data_contract["members"]["train.csv"]["sha256"]),
        "runtime_sha256": runtime_sha256,
        "environment_sha256": environment_sha256,
        "training_source_sha256": file_sha256(
            Path(__file__).with_name("final_training.py")
        ),
    }
    candidate = resolve_final_candidate(
        contract["selected_candidate"], contract["final_fit"]
    )
    print(
        "VERSION_D_INPUTS_VERIFIED "
        f"data_sha256={data_contract['sha256']} "
        f"stage_c_sha256={stage_c['delivery_sha256']} "
        f"contract_sha256={contract_sha256}",
        flush=True,
    )

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ColabVersionDError("Version D requires exactly one CUDA GPU")
    device_name = torch.cuda.get_device_name(0)
    if "T4" not in device_name.upper():
        raise ColabVersionDError(f"Version D requires a T4 GPU; found={device_name}")
    print(f"VERSION_D_GPU_READY device_count=1 name={device_name}", flush=True)
    return VersionDContext(
        contract,
        contract_sha256,
        stage_c,
        checkpoint_identity,
        candidate,
    )


def restore_recovery(
    context: VersionDContext,
    recovery_archive: Path | None,
    work_root: Path,
) -> RecoverySelection:
    snapshot_root = work_root / "snapshots"
    artifact_root = work_root / "frozen_inference"
    checkpoint_path = work_root / "training/final_epoch_checkpoint.pt"
    frozen_paths = list(snapshot_root.glob("tabm_version_D_frozen_model.zip"))
    emergency_paths = list(
        snapshot_root.glob("tabm_version_D_emergency_epoch_*.zip")
    )
    if artifact_root.joinpath("inference_manifest.json").is_file() and not frozen_paths:
        frozen_paths.append(
            write_frozen_snapshot(
                snapshot_root,
                artifact_root,
                context.checkpoint_identity,
            )
        )
    if recovery_archive is not None:
        if recovery_archive.name == "tabm_version_D_frozen_model.zip":
            frozen_paths.append(recovery_archive)
        elif recovery_archive.name.startswith("tabm_version_D_emergency_epoch_"):
            emergency_paths.append(recovery_archive)
        else:
            raise ColabVersionDError("recovery archive filename differs")
    selection = select_recovery_snapshot(
        emergency_paths,
        frozen_paths,
        context.checkpoint_identity,
    )
    if selection.mode == "frozen":
        assert selection.path is not None
        restore_frozen_snapshot(
            selection.path,
            artifact_root,
            context.checkpoint_identity,
        )
    elif selection.mode == "emergency":
        assert selection.path is not None
        restore_emergency_snapshot(
            selection.path,
            checkpoint_path,
            artifact_root,
            context.checkpoint_identity,
        )
    elif checkpoint_path.is_file():
        checkpoint = load_epoch_checkpoint(
            checkpoint_path,
            context.checkpoint_identity,
        )
        selection = RecoverySelection(
            "emergency",
            None,
            int(checkpoint["completed_epochs"]),
        )
    print(
        f"VERSION_D_RECOVERY_READY mode={selection.mode} epoch={selection.epoch}",
        flush=True,
    )
    return selection


def fit_final(
    context: VersionDContext,
    data_archive: Path,
    work_root: Path,
    absolute_deadline: float,
    on_download: Callable[[Path, str], None],
) -> tuple[dict[str, object], Path]:
    train_root = extract_training_input(
        data_archive,
        work_root / "train_only",
        context.contract,
    )
    artifact_root = work_root / "frozen_inference"
    checkpoint_path = work_root / "training/final_epoch_checkpoint.pt"

    def on_epoch(checkpoint: Path, epoch: int) -> None:
        snapshot = write_emergency_snapshot(
            work_root / "snapshots",
            checkpoint,
            artifact_root / "preprocessing_state.json",
            artifact_root / "numeric_embedding_0.json",
            epoch,
            context.checkpoint_identity,
        )
        print(
            f"VERSION_D_EMERGENCY_SNAPSHOT_READY epoch={epoch} path={snapshot}",
            flush=True,
        )
        on_download(snapshot, "emergency")

    fit_report = fit_final_member(
        data_dir=train_root,
        artifact_root=artifact_root,
        candidate=context.candidate,
        seed=3407,
        epochs=3,
        absolute_deadline=absolute_deadline,
        checkpoint_path=checkpoint_path,
        checkpoint_identity=context.checkpoint_identity,
        on_epoch_checkpoint=on_epoch,
    )
    frozen = write_frozen_snapshot(
        work_root / "snapshots",
        artifact_root,
        context.checkpoint_identity,
    )
    return fit_report, frozen


def _frozen_fit_report(artifact_root: Path) -> dict[str, object]:
    manifest_path = artifact_root / "inference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "status": "completed",
        "rows": int(manifest["row_count"]),
        "epochs": 3,
        "seeds": [3407],
        "scheduler": "constant",
        "resumed_from_epoch": 3,
        "recovery": "frozen",
        "manifest_sha256": file_sha256(manifest_path),
    }


def review_final(
    context: VersionDContext,
    data_archive: Path,
    work_root: Path,
    fit_report: Mapping[str, object],
    runtime_versions: Mapping[str, str],
) -> Path:
    review_data = extract_review_inputs(
        data_archive,
        work_root / "review_inputs",
        context.contract,
    )
    result = review_frozen_artifact(
        artifact_root=work_root / "frozen_inference",
        review_data_dir=review_data,
        output_dir=work_root / "review",
        prior_manifest_sha256=str(context.stage_c["prior_manifest_sha256"]),
        version_d_contract_sha256=context.contract_sha256,
        fit_report=fit_report,
        runtime_versions=runtime_versions,
    )
    return result.bundles.review


def publish_delivery(
    context: VersionDContext,
    review_bundle: Path,
    frozen_snapshot: Path,
    log_path: Path,
    work_root: Path,
    runtime_versions: Mapping[str, str],
) -> Path:
    return write_review_delivery(
        output_dir=work_root / "delivery",
        review_bundle=review_bundle,
        log_path=log_path,
        contract_sha256=context.contract_sha256,
        data_archive_sha256=str(context.contract["data_archive"]["sha256"]),
        stage_c_delivery_sha256=str(context.stage_c["delivery_sha256"]),
        prior_manifest_sha256=str(context.stage_c["prior_manifest_sha256"]),
        frozen_sha256=file_sha256(frozen_snapshot),
        runtime_versions=runtime_versions,
    )


def run_version_d(
    *,
    data_archive: Path,
    stage_c_delivery: Path,
    work_root: Path,
    recovery_archive: Path | None,
    absolute_deadline: float,
    runtime_sha256: str,
    runtime_versions: Mapping[str, str],
    log_path: Path,
    on_download: Callable[[Path, str], None],
) -> Path:
    if time.time() >= absolute_deadline:
        raise ColabVersionDError("Version D deadline was reached before setup")
    work_root.mkdir(parents=True, exist_ok=True)
    context = verify_inputs(
        data_archive,
        stage_c_delivery,
        runtime_sha256,
        runtime_versions,
    )
    selection = restore_recovery(context, recovery_archive, work_root)
    artifact_root = work_root / "frozen_inference"
    if selection.mode == "frozen":
        assert selection.path is not None
        frozen_snapshot = selection.path
        fit_report = _frozen_fit_report(artifact_root)
    else:
        fit_report, frozen_snapshot = fit_final(
            context,
            data_archive,
            work_root,
            absolute_deadline,
            on_download,
        )
    print(f"VERSION_D_FROZEN_MODEL_READY path={frozen_snapshot}", flush=True)
    on_download(frozen_snapshot, "frozen")
    review_bundle = review_final(
        context,
        data_archive,
        work_root,
        fit_report,
        runtime_versions,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    delivery = publish_delivery(
        context,
        review_bundle,
        frozen_snapshot,
        log_path,
        work_root,
        runtime_versions,
    )
    print(f"VERSION_D_DELIVERY_READY path={delivery}", flush=True)
    on_download(delivery, "delivery")
    print(f"VERSION_D_DOWNLOAD_REQUESTED path={delivery}", flush=True)
    return delivery
