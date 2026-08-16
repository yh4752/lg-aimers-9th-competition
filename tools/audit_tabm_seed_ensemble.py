"""Audit sealed TabM seed ensembles from existing temporal predictions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from experiments.tabm_campaign.artifacts import verify_review_bundle
from experiments.tabm_campaign.ensemble_audit import (
    EnsembleAuditError,
    EnsembleAuditResult,
    audit_prediction_frames,
    load_ensemble_contract,
    prediction_member,
)


CONTRACT_PATH = (
    PROJECT_ROOT / "experiments/tabm_campaign/score_improvement_contract.json"
)
OUTPUT_NAME = "tabm_seed_ensemble_audit_review.zip"
_REVIEW_NAMES = {"audit.log", "ensemble_audit.json", "manifest.json"}
_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class AuditReview:
    path: Path
    sha256: str
    decision: str
    reused: bool


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _unique_object(items: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in items:
        if key in output:
            raise EnsembleAuditError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> object:
    raise EnsembleAuditError(f"JSON number must be finite: {value}")


def _load_json(value: bytes, label: str) -> dict[str, object]:
    try:
        result = json.loads(
            value.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except EnsembleAuditError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EnsembleAuditError(f"invalid JSON: {label}") from error
    if not isinstance(result, dict):
        raise EnsembleAuditError(f"JSON must be an object: {label}")
    return result


def _source_inputs(
    review_path: Path,
) -> tuple[dict[tuple[str, int], pd.DataFrame], dict[str, str], str]:
    contract = load_ensemble_contract(CONTRACT_PATH)
    verified = verify_review_bundle(review_path)
    if verified.version != "C":
        raise EnsembleAuditError("source review must be Version C")
    if verified.campaign_config_sha256 != contract.stage_c_campaign_config_sha256:
        raise EnsembleAuditError("source review campaign config SHA-256 differs")

    expected = {
        prediction_member(fold, seed): (fold, seed)
        for fold in contract.folds
        for seed in contract.seeds
    }
    required = {*expected, "metrics/job_results.json"}
    if not required.issubset(verified.member_sha256):
        raise EnsembleAuditError("source review is missing sealed prediction evidence")

    try:
        with ZipFile(verified.path) as archive:
            metrics_value = archive.read("metrics/job_results.json")
            prediction_values = {name: archive.read(name) for name in expected}
    except (BadZipFile, KeyError, OSError) as error:
        raise EnsembleAuditError("cannot read source review evidence") from error

    try:
        metrics = json.loads(
            metrics_value.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except EnsembleAuditError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EnsembleAuditError("invalid source job metrics") from error
    if not isinstance(metrics, list):
        raise EnsembleAuditError("source job metrics must be a list")
    result_by_id: dict[str, dict[str, object]] = {}
    for item in metrics:
        if not isinstance(item, dict) or not isinstance(item.get("candidate_id"), str):
            raise EnsembleAuditError("source job metric identity is invalid")
        candidate_id = str(item["candidate_id"])
        if candidate_id in result_by_id:
            raise EnsembleAuditError("source job metric identity is duplicated")
        result_by_id[candidate_id] = item
    for name in expected:
        candidate_id = Path(name).stem
        result = result_by_id.get(candidate_id)
        if result is None or result.get("status") != "completed":
            raise EnsembleAuditError(
                f"sealed Stage C job must be completed: {candidate_id}"
            )

    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for name, identity in expected.items():
        try:
            frame = pd.read_csv(
                io.BytesIO(prediction_values[name]),
                usecols=["row_id", "target", "probability"],
                dtype={"row_id": "string"},
            )
        except Exception as error:
            raise EnsembleAuditError(f"cannot read prediction CSV: {name}") from error
        frame["row_id"] = frame["row_id"].astype(object)
        frames[identity] = frame
    prediction_hashes = {
        name: verified.member_sha256[name] for name in sorted(expected)
    }
    return frames, prediction_hashes, verified.manifest_sha256


def _load_truth(path: Path) -> pd.DataFrame:
    if path.is_symlink() or not path.is_file():
        raise EnsembleAuditError("train CSV is missing or unsafe")
    try:
        frame = pd.read_csv(
            path,
            usecols=["row_id", "season", "control_success"],
            dtype={"row_id": "string"},
        )
    except Exception as error:
        raise EnsembleAuditError("cannot read train CSV") from error
    frame["row_id"] = frame["row_id"].astype(object)
    return frame


def _result_payload(result: EnsembleAuditResult) -> dict[str, object]:
    return {
        "schema_version": 1,
        "decision": result.decision,
        "baseline_candidate_id": result.baseline_candidate_id,
        "selected_candidate_id": result.selected_candidate_id,
        "row_counts": dict(result.row_counts),
        "candidates": {
            candidate_id: {
                "weights": {str(seed): weight for seed, weight in audit.weights.items()},
                "fold_brier": dict(audit.fold_brier),
                "weighted_brier": audit.weighted_brier,
                "weighted_gain": audit.weighted_gain,
                "worst_fold_degrade": audit.worst_fold_degrade,
                "gate_passed": audit.gate_passed,
            }
            for candidate_id, audit in sorted(result.candidates.items())
        },
    }


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, members[name])
    return buffer.getvalue()


def _publish_without_overwrite(path: Path, value: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise EnsembleAuditError("output directory is unsafe")
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != value:
            raise EnsembleAuditError("existing audit review differs")
        return True

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != value:
                raise EnsembleAuditError("existing audit review differs")
            return True
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return False


def run_audit(stage_c_review: str | Path, train_csv: str | Path, output_dir: str | Path) -> AuditReview:
    review_path = Path(stage_c_review).expanduser().resolve(strict=True)
    train_path = Path(train_csv).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().resolve(strict=False)
    if review_path.is_symlink() or not review_path.is_file():
        raise EnsembleAuditError("source review is missing or unsafe")

    contract = load_ensemble_contract(CONTRACT_PATH)
    frames, prediction_hashes, source_manifest_sha = _source_inputs(review_path)
    truth = _load_truth(train_path)
    result = audit_prediction_frames(contract, frames, truth)

    result_bytes = _canonical_json(_result_payload(result))
    log_bytes = (
        f"decision={result.decision} baseline={result.baseline_candidate_id} "
        f"selected={result.selected_candidate_id} rows={sum(result.row_counts.values())}\n"
    ).encode("utf-8")
    evidence_members = {
        "audit.log": log_bytes,
        "ensemble_audit.json": result_bytes,
    }
    manifest = {
        "schema_version": 1,
        "artifact_kind": "tabm_seed_ensemble_audit_review",
        "review_only": True,
        "source_review_sha256": _sha256_file(review_path),
        "source_review_manifest_sha256": source_manifest_sha,
        "stage_c_campaign_config_sha256": contract.stage_c_campaign_config_sha256,
        "train_csv_sha256": _sha256_file(train_path),
        "contract_sha256": _sha256_file(CONTRACT_PATH),
        "prediction_members": prediction_hashes,
        "members": {
            name: _sha256_bytes(value) for name, value in sorted(evidence_members.items())
        },
    }
    archive_bytes = _zip_bytes(
        {**evidence_members, "manifest.json": _canonical_json(manifest)}
    )
    path = output / OUTPUT_NAME
    reused = _publish_without_overwrite(path, archive_bytes)
    return AuditReview(path, _sha256_bytes(archive_bytes), result.decision, reused)


def verify_audit_review(path: str | Path) -> AuditReview:
    review_path = Path(path).expanduser().resolve(strict=True)
    if review_path.is_symlink() or not review_path.is_file():
        raise EnsembleAuditError("audit review is missing or unsafe")
    try:
        with ZipFile(review_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != _REVIEW_NAMES:
                raise EnsembleAuditError("audit review member set differs")
            if archive.testzip() is not None:
                raise EnsembleAuditError("audit review has a corrupt member")
            manifest_bytes = archive.read("manifest.json")
            audit_bytes = archive.read("ensemble_audit.json")
            log_bytes = archive.read("audit.log")
    except EnsembleAuditError:
        raise
    except (BadZipFile, KeyError, OSError) as error:
        raise EnsembleAuditError("cannot read audit review") from error

    manifest = _load_json(manifest_bytes, "manifest.json")
    audit = _load_json(audit_bytes, "ensemble_audit.json")
    expected_manifest_keys = {
        "schema_version",
        "artifact_kind",
        "review_only",
        "source_review_sha256",
        "source_review_manifest_sha256",
        "stage_c_campaign_config_sha256",
        "train_csv_sha256",
        "contract_sha256",
        "prediction_members",
        "members",
    }
    if set(manifest) != expected_manifest_keys:
        raise EnsembleAuditError("audit review manifest keys differ")
    if (
        manifest["schema_version"] != 1
        or manifest["artifact_kind"] != "tabm_seed_ensemble_audit_review"
        or manifest["review_only"] is not True
    ):
        raise EnsembleAuditError("audit review manifest identity differs")
    for key in (
        "source_review_sha256",
        "source_review_manifest_sha256",
        "stage_c_campaign_config_sha256",
        "train_csv_sha256",
        "contract_sha256",
    ):
        if not _valid_sha256(manifest[key]):
            raise EnsembleAuditError(f"audit review {key} is invalid")
    members = manifest["members"]
    if not isinstance(members, dict) or set(members) != {"audit.log", "ensemble_audit.json"}:
        raise EnsembleAuditError("audit review member manifest differs")
    actual = {
        "audit.log": _sha256_bytes(log_bytes),
        "ensemble_audit.json": _sha256_bytes(audit_bytes),
    }
    if members != actual:
        raise EnsembleAuditError("audit review member SHA-256 differs")
    prediction_members = manifest["prediction_members"]
    contract = load_ensemble_contract(CONTRACT_PATH)
    expected_predictions = {
        prediction_member(fold, seed)
        for fold in contract.folds
        for seed in contract.seeds
    }
    if (
        not isinstance(prediction_members, dict)
        or set(prediction_members) != expected_predictions
        or not all(_valid_sha256(value) for value in prediction_members.values())
    ):
        raise EnsembleAuditError("audit review prediction binding differs")
    if _canonical_json(manifest) != manifest_bytes or _canonical_json(audit) != audit_bytes:
        raise EnsembleAuditError("audit review JSON is not canonical")
    decision = audit.get("decision")
    if decision not in {"promoted", "keep_single"}:
        raise EnsembleAuditError("audit review decision is invalid")
    return AuditReview(
        review_path,
        _sha256_file(review_path),
        str(decision),
        True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-c-review")
    parser.add_argument("--train-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--verify-review")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    verify_mode = args.verify_review is not None
    audit_values = (args.stage_c_review, args.train_csv, args.output_dir)
    if verify_mode and any(value is not None for value in audit_values):
        print(
            "TABM_ENSEMBLE_AUDIT_ERROR stage=arguments "
            "type=EnsembleAuditError message=verify_and_audit_arguments_are_mutually_exclusive"
        )
        return 1
    if not verify_mode and any(value is None for value in audit_values):
        print(
            "TABM_ENSEMBLE_AUDIT_ERROR stage=arguments "
            "type=EnsembleAuditError message=audit_arguments_are_required"
        )
        return 1

    stage = "verify" if verify_mode else "input"
    try:
        if verify_mode:
            result = verify_audit_review(args.verify_review)
            print(
                "TABM_ENSEMBLE_REVIEW_VERIFIED "
                f"decision={result.decision} review={result.path} sha256={result.sha256}"
            )
            return 0
        result = run_audit(args.stage_c_review, args.train_csv, args.output_dir)
        print(
            "TABM_ENSEMBLE_AUDIT_SUCCESS "
            f"decision={result.decision} review={result.path} sha256={result.sha256} "
            f"reused={str(result.reused).lower()}"
        )
        return 0
    except Exception as error:
        message = "_".join(str(error).split())
        print(
            f"TABM_ENSEMBLE_AUDIT_ERROR stage={stage} "
            f"type={type(error).__name__} message={message}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
