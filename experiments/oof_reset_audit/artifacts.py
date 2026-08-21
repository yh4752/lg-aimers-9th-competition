from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Callable, Mapping
from zipfile import ZipFile

import pandas as pd

from experiments.catboost_deployment.artifacts import verify_deployment_review
from experiments.catboost_tabm_blend.colab import verify_delivery as verify_blend_delivery
from experiments.hierarchical_tabm.artifacts import verify_review_bundle as verify_hierarchical_review
from experiments.hierarchical_tabm.calibration import apply_calibration, calibration_state_from_payload
from experiments.tabm_campaign.colab_recovery import verify_delivery_bundle as verify_stage_c_delivery
from experiments.tabm_campaign.row_feature_colab import verify_delivery as verify_row_feature_delivery

from .metrics import validate_prediction_frame
from .types import ArtifactRecord, ArtifactRole, PredictionSet, TrustClass


class AuditArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedArtifacts:
    predictions: tuple[PredictionSet, ...]
    inventory: tuple[ArtifactRecord, ...]


_MAX_MEMBERS = 2_000
_MAX_MEMBER_SIZE = 2 * 1024**3
_MAX_TOTAL_SIZE = 4 * 1024**3
_FOLD = re.compile(r"__tr(2022|2023)__va(2023|2024)")
_STAGE_C = re.compile(r"__s(?:42|2026|3407)__s(42|2026|3407)__tr(2022|2023)__va(2023|2024)\.csv$")


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _names(archive: ZipFile) -> set[str]:
    infos = archive.infolist()
    if not infos or len(infos) > _MAX_MEMBERS:
        raise AuditArtifactError("ZIP member count is invalid")
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise AuditArtifactError("ZIP contains duplicate members")
    total = 0
    for info in infos:
        pure = PurePosixPath(info.filename)
        if (
            not info.filename
            or "\\" in info.filename
            or pure.is_absolute()
            or ".." in pure.parts
            or stat.S_ISLNK(info.external_attr >> 16)
            or info.file_size > _MAX_MEMBER_SIZE
        ):
            raise AuditArtifactError(f"unsafe ZIP member: {info.filename}")
        total += info.file_size
        if total > _MAX_TOTAL_SIZE:
            raise AuditArtifactError("ZIP expanded size is invalid")
        if info.file_size > 16 * 1024**2 and info.compress_size and info.file_size / info.compress_size > 1_000:
            raise AuditArtifactError("ZIP compression ratio is invalid")
    return set(names)


def _json_member(archive: ZipFile, name: str) -> Mapping[str, object]:
    try:
        info = archive.getinfo(name)
        if info.file_size > 4 * 1024**2:
            raise AuditArtifactError(f"JSON member is too large: {name}")
        value = json.loads(archive.read(info))
    except AuditArtifactError:
        raise
    except Exception as error:
        raise AuditArtifactError(f"invalid JSON member: {name}") from error
    if not isinstance(value, Mapping):
        raise AuditArtifactError(f"JSON member must be an object: {name}")
    return value


def _bindings(manifest: Mapping[str, object]) -> dict[str, str]:
    value = manifest.get("bindings")
    if not isinstance(value, Mapping):
        raise AuditArtifactError("artifact bindings are missing")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise AuditArtifactError("artifact bindings are invalid")
        result[key] = item
    return result


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for alias, canonical in (("pitcher_id_known", "pitcher_known"), ("batter_id_known", "batter_known")):
        if alias not in result:
            continue
        if canonical in result and not result[canonical].equals(result[alias]):
            raise AuditArtifactError(f"segment aliases differ: {canonical}")
        if canonical not in result:
            result[canonical] = result[alias]
        result = result.drop(columns=[alias])
    try:
        return validate_prediction_frame(result)
    except ValueError as error:
        raise AuditArtifactError(str(error)) from error


def _prediction(
    raw: bytes, *, path: Path, artifact_sha: str, member: str, model_id: str,
    fold: str, trust: TrustClass = TrustClass.RULE_SAFE,
) -> PredictionSet:
    try:
        frame = pd.read_csv(BytesIO(raw))
    except Exception as error:
        raise AuditArtifactError(f"prediction CSV is invalid: {member}") from error
    return PredictionSet(path, artifact_sha, member, sha256(raw).hexdigest(), model_id, fold, trust, _normalise(frame))


def _nested(data: bytes) -> tuple[ZipFile, BytesIO]:
    stream = BytesIO(data)
    archive = ZipFile(stream)
    _names(archive)
    return archive, stream


def _stage_c(path: Path, digest: str, outer: ZipFile) -> list[PredictionSet]:
    verify_stage_c_delivery(path)
    review, stream = _nested(outer.read("tabm_search_stage_C_review_bundle.zip"))
    try:
        rows = []
        for name in sorted(_names(review)):
            match = _STAGE_C.search(name)
            if not match or not name.startswith("predictions/"):
                continue
            seed, train, valid = match.groups()
            rows.append(_prediction(review.read(name), path=path, artifact_sha=digest, member=name,
                                    model_id=f"tabm_stage_c_seed_{seed}", fold=f"{train}->{valid}"))
        return rows
    finally:
        review.close(); stream.close()


def _row_feature(path: Path, digest: str, outer: ZipFile, manifest: Mapping[str, object]) -> list[PredictionSet]:
    bindings = _bindings(manifest)
    verify_row_feature_delivery(path, **bindings)
    review, stream = _nested(outer.read("tabm_row_feature_stage_P_review_bundle.zip"))
    try:
        payload = json.loads(review.read("metrics/job_results.json"))
        jobs = payload.get("results") if isinstance(payload, Mapping) else payload
        if not isinstance(jobs, list):
            raise AuditArtifactError("row feature job results are invalid")
        rows = []
        for job in jobs:
            if not isinstance(job, Mapping) or job.get("status") != "completed":
                continue
            candidate = job.get("candidate_id")
            if not isinstance(candidate, str):
                raise AuditArtifactError("row feature candidate is invalid")
            fold = job.get("fold")
            if not isinstance(fold, str):
                match = _FOLD.search(candidate)
                if not match:
                    raise AuditArtifactError("row feature fold is missing")
                fold = f"{match.group(1)}->{match.group(2)}"
            member = job.get("predictions", f"predictions/{candidate}.csv")
            if not isinstance(member, str):
                raise AuditArtifactError("row feature prediction member is invalid")
            model_id = _FOLD.sub("", candidate)
            rows.append(_prediction(review.read(member), path=path, artifact_sha=digest,
                                    member=member, model_id=model_id, fold=fold))
        return rows
    finally:
        review.close(); stream.close()


def _deployment(path: Path, digest: str, outer: ZipFile, manifest: Mapping[str, object]) -> list[PredictionSet]:
    bindings = _bindings(manifest)
    verify_deployment_review(path, expected_bindings=bindings)
    rows = []
    for name in sorted(_names(outer)):
        match = re.fullmatch(r"predictions/align_(2022|2023)_(2023|2024)\.csv", name)
        if not match:
            continue
        raw = outer.read(name)
        frame = pd.read_csv(BytesIO(raw))
        for column in sorted((c for c in frame if re.fullmatch(r"p_\d+", c)), key=lambda c: int(c[2:])):
            derived = frame.drop(columns=[c for c in frame if c.startswith("p_")]).copy()
            derived["probability"] = frame[column]
            value = derived.to_csv(index=False).encode()
            rows.append(_prediction(value, path=path, artifact_sha=digest, member=f"{name}#{column}",
                                    model_id=f"catboost_prefix_{column[2:]}", fold=f"{match.group(1)}->{match.group(2)}"))
    return rows


def _blend(path: Path, digest: str, outer: ZipFile, manifest: Mapping[str, object]) -> list[PredictionSet]:
    bindings = _bindings(manifest)
    verify_blend_delivery(path, expected_bindings=bindings)
    review, stream = _nested(outer.read("catboost_tabm_blend_review.zip"))
    try:
        rows = []
        for name in sorted(_names(review)):
            match = re.fullmatch(
                r"predictions/catboost__hand_matchup__tr(2022|2023)__va(2023|2024)__s42\.csv",
                name,
            )
            if match:
                rows.append(_prediction(
                    review.read(name), path=path, artifact_sha=digest, member=name,
                    model_id="catboost_hand_matchup_seed_42",
                    fold=f"{match.group(1)}->{match.group(2)}",
                ))
        return rows
    finally:
        review.close(); stream.close()


def _hierarchical(path: Path, digest: str, outer: ZipFile, manifest: Mapping[str, object]) -> list[PredictionSet]:
    bindings = _bindings(manifest)
    verify_hierarchical_review(path, expected_bindings=bindings)
    states = {kind: calibration_state_from_payload(json.loads(outer.read(f"calibration/{kind}.json"))) for kind in ("H2", "H3")}
    rows = []
    for name in sorted(_names(outer)):
        match = re.fullmatch(r"jobs/h1__tr(2022|2023)__va(2023|2024)__s3407/predictions\.csv", name)
        if not match:
            continue
        raw = outer.read(name)
        base = _normalise(pd.read_csv(BytesIO(raw)))
        fold = f"{match.group(1)}->{match.group(2)}"
        rows.append(_prediction(raw, path=path, artifact_sha=digest, member=name, model_id="H1", fold=fold))
        for kind, state in states.items():
            derived = base.copy()
            derived["probability"] = apply_calibration(base["probability"].to_numpy(), base, state)
            value = derived.to_csv(index=False).encode()
            rows.append(_prediction(value, path=path, artifact_sha=digest, member=f"{name}#{kind}", model_id=kind, fold=fold))
    return rows


def _quarantine(path: Path, digest: str, outer: ZipFile, manifest: Mapping[str, object]) -> list[PredictionSet]:
    predictions = manifest.get("predictions")
    if (set(manifest) != {"schema_version", "artifact_kind", "trust", "submission_package", "predictions"}
            or manifest.get("schema_version") != 1 or manifest.get("trust") != TrustClass.QUARANTINED_DIAGNOSTIC.value
            or manifest.get("submission_package") is not False or not isinstance(predictions, Mapping)):
        raise AuditArtifactError("quarantined manifest is invalid")
    if _names(outer) != {"manifest.json", *predictions}:
        raise AuditArtifactError("quarantined member set differs")
    rows = []
    for member, evidence in predictions.items():
        if not isinstance(member, str) or not isinstance(evidence, Mapping):
            raise AuditArtifactError("quarantined prediction evidence is invalid")
        raw = outer.read(member)
        if evidence.get("size") != len(raw) or evidence.get("sha256") != sha256(raw).hexdigest():
            raise AuditArtifactError("quarantined prediction hash differs")
        model_id, fold = evidence.get("model_id"), evidence.get("fold")
        if not isinstance(model_id, str) or not isinstance(fold, str):
            raise AuditArtifactError("quarantined prediction identity is invalid")
        rows.append(_prediction(raw, path=path, artifact_sha=digest, member=member, model_id=model_id,
                                fold=fold, trust=TrustClass.QUARANTINED_DIAGNOSTIC))
    return rows


_KINDS: dict[str, tuple[ArtifactRole, Callable[..., list[PredictionSet]]]] = {
    "tabm_colab_stage_C_delivery": (ArtifactRole.STAGE_C_TABM, _stage_c),
    "tabm_row_feature_stage_P_delivery": (ArtifactRole.ROW_FEATURE, _row_feature),
    "catboost_tabm_blend_delivery": (ArtifactRole.CATBOOST_BLEND, _blend),
    "catboost_deployment_review": (ArtifactRole.CATBOOST_DEPLOYMENT, _deployment),
    "hierarchical_tabm_review_v1": (ArtifactRole.HIERARCHICAL, _hierarchical),
    "oof_reset_quarantined_v1": (ArtifactRole.QUARANTINED_XGBOOST, _quarantine),
}


def load_artifacts(paths: list[Path], *, expected_roles: tuple[ArtifactRole, ...]) -> LoadedArtifacts:
    predictions: list[PredictionSet] = []
    inventory: list[ArtifactRecord] = []
    observed: set[ArtifactRole] = set()
    for supplied in paths:
        path = Path(supplied)
        if not path.is_file() or path.is_symlink():
            raise AuditArtifactError(f"artifact must be a regular file: {path}")
        before = _digest(path)
        try:
            with ZipFile(path) as outer:
                names = _names(outer)
                control = "delivery_manifest.json" if "delivery_manifest.json" in names else "manifest.json" if "manifest.json" in names else None
                manifest = _json_member(outer, control) if control else {}
                kind = manifest.get("artifact_kind")
                adapter = _KINDS.get(kind) if isinstance(kind, str) else None
                if adapter is None:
                    inventory.append(ArtifactRecord(path, before, ArtifactRole.UNKNOWN, str(kind or "unknown"), "missing_evidence", "unsupported artifact"))
                    continue
                role, decoder = adapter
                try:
                    decoded = decoder(path, before, outer, manifest) if decoder in {_row_feature, _blend, _deployment, _hierarchical, _quarantine} else decoder(path, before, outer)
                except AuditArtifactError:
                    raise
                except Exception as error:
                    raise AuditArtifactError(f"artifact verification failed: {path.name}") from error
                predictions.extend(decoded)
                observed.add(role)
                inventory.append(ArtifactRecord(path, before, role, str(kind), "verified", None))
        except AuditArtifactError:
            raise
        except Exception as error:
            raise AuditArtifactError(f"artifact ZIP is invalid: {path.name}") from error
        if _digest(path) != before:
            raise AuditArtifactError(f"artifact changed while reading: {path.name}")
    unique: dict[tuple[str, str], PredictionSet] = {}
    for row in predictions:
        key = (row.model_id, row.fold)
        previous = unique.get(key)
        if previous is not None and previous.prediction_sha256 != row.prediction_sha256:
            raise AuditArtifactError(f"conflicting prediction evidence: {row.model_id} {row.fold}")
        unique.setdefault(key, row)
    for role in expected_roles:
        if role not in observed:
            inventory.append(ArtifactRecord(Path("__MISSING__") / role.value, "0" * 64, role, "missing", "missing_evidence", None))
    return LoadedArtifacts(tuple(unique.values()), tuple(inventory))
