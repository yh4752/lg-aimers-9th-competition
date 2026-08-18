from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import multiprocessing
from pathlib import Path
import platform
import resource
import shutil
import tempfile
from types import MappingProxyType
from typing import Callable, Mapping
from zipfile import ZipFile

import numpy as np
import pandas as pd

from experiments.independent_dl.models.common import ModelMetadata, import_runtime_module
from experiments.independent_dl.models.tabm import TabMAdapter

from .artifacts import verify_candidate_delivery
from .calibration import (
    CalibrationState,
    apply_calibration,
    calibration_state_from_payload,
    calibration_state_payload,
)
from .contracts import HierarchicalContract, load_contract
from .feature_adapter import (
    HierarchicalFeatureState,
    feature_state_from_payload,
    feature_state_payload,
    transform_with_state,
)


class HierarchicalInferenceError(ValueError):
    """Raised when a frozen candidate cannot be evaluated independently."""


@dataclass(frozen=True)
class IndependenceReport:
    row_count: int
    features_exact: bool
    maximum_probability_delta: float
    state_before: str
    state_after: str
    batch_sizes: tuple[int, ...]


@dataclass(frozen=True)
class InferenceResourceReport:
    row_count: int
    python_version: str
    elapsed_seconds: float
    peak_gpu_bytes: int
    peak_rss_bytes: int
    artifact_bytes: int
    passed: bool


@dataclass(frozen=True)
class CandidateValidation:
    accepted_roles: Mapping[str, str]
    report_paths: Mapping[str, Path]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _calibration_segments(frame: pd.DataFrame) -> pd.DataFrame:
    required = (
        "row_id", "game_type", "balls_before", "strikes_before",
        "pitcher_hand", "batter_hand", "base_state", "outs_before",
    )
    missing = [column for column in required if column not in frame]
    if missing:
        raise HierarchicalInferenceError(
            f"missing calibration source columns: {', '.join(missing)}"
        )

    def bounded(column: str, maximum: int) -> pd.Series:
        values = pd.to_numeric(frame[column], errors="coerce")
        if (
            values.isna().any()
            or not np.equal(values.to_numpy(), np.floor(values.to_numpy())).all()
            or ((values < 0) | (values > maximum)).any()
        ):
            raise HierarchicalInferenceError(f"invalid calibration source: {column}")
        return values.astype("int64").astype(str)

    category = lambda column: frame[column].astype("string").fillna("__MISSING__").astype(str)
    output = pd.DataFrame(index=frame.index)
    output["row_id"] = frame["row_id"].astype(str)
    output["game_type"] = category("game_type")
    output["count_state"] = bounded("balls_before", 3) + "_" + bounded("strikes_before", 2)
    output["hand_matchup"] = category("pitcher_hand") + "_" + category("batter_hand")
    output["base_out_state"] = category("base_state") + "_" + bounded("outs_before", 2)
    return output


class FrozenHierarchicalPredictor:
    def __init__(
        self,
        *,
        feature_state: HierarchicalFeatureState,
        candidate_id: str,
        raw_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
        model_digest: str,
        calibration_state: CalibrationState | None,
        artifact_bytes: int,
        live_model_digest: Callable[[], str] | None = None,
        reload_spec: tuple[object, ...] | None = None,
    ) -> None:
        if candidate_id not in {"H1", "H2", "H3"}:
            raise HierarchicalInferenceError("candidate_id differs")
        if (candidate_id == "H1") != (calibration_state is None):
            raise HierarchicalInferenceError("candidate calibration differs")
        if calibration_state is not None and calibration_state.kind != candidate_id:
            raise HierarchicalInferenceError("candidate calibration identity differs")
        if (
            not isinstance(model_digest, str)
            or len(model_digest) != 64
            or any(character not in "0123456789abcdef" for character in model_digest)
        ):
            raise HierarchicalInferenceError("model digest is invalid")
        if isinstance(artifact_bytes, bool) or not isinstance(artifact_bytes, int) or artifact_bytes < 0:
            raise HierarchicalInferenceError("artifact size is invalid")
        self._feature_state = feature_state
        self._candidate_id = candidate_id
        self._raw_predict = raw_predict
        self._model_digest = model_digest
        self._calibration_state = calibration_state
        self._artifact_bytes = artifact_bytes
        self._live_model_digest = live_model_digest
        self._reload_spec = reload_spec

    @property
    def artifact_bytes(self) -> int:
        return self._artifact_bytes

    def state_digest(self) -> str:
        live = self._model_digest if self._live_model_digest is None else self._live_model_digest()
        if live != self._model_digest:
            return sha256(f"MUTATED:{live}".encode("utf-8")).hexdigest()
        payload = {
            "candidate_id": self._candidate_id,
            "model_sha256": self._model_digest,
            "feature_state": feature_state_payload(self._feature_state),
            "calibration_state": (
                None
                if self._calibration_state is None
                else calibration_state_payload(self._calibration_state)
            ),
        }
        return sha256(_canonical_json(payload)).hexdigest()

    def encode(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise HierarchicalInferenceError("prediction frame must be non-empty")
        source = frame.drop(columns=["control_success"], errors="ignore").copy()
        batch = transform_with_state(source, self._feature_state)
        return np.asarray(batch.x_num).copy(), np.asarray(batch.x_cat).copy()

    def predict_batch(self, frame: pd.DataFrame, *, batch_size: int = 2048) -> np.ndarray:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise HierarchicalInferenceError("batch_size must be positive")
        x_num, x_cat = self.encode(frame)
        chunks: list[np.ndarray] = []
        for start in range(0, len(frame), batch_size):
            values = np.asarray(
                self._raw_predict(x_num[start : start + batch_size], x_cat[start : start + batch_size]),
                dtype="float64",
            )
            expected = min(batch_size, len(frame) - start)
            if values.shape != (expected,) or not np.isfinite(values).all():
                raise HierarchicalInferenceError("model probabilities are invalid")
            chunks.append(values)
        probability = np.concatenate(chunks)
        if ((probability < 0) | (probability > 1)).any():
            raise HierarchicalInferenceError("model probabilities are outside [0, 1]")
        if self._calibration_state is not None:
            probability = apply_calibration(
                probability, _calibration_segments(frame), self._calibration_state
            )
        return np.asarray(probability, dtype="float64")


def _model_state_digest(model: object) -> str:
    digest = sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _preliminary_bindings(delivery: Path) -> Mapping[str, str]:
    try:
        with ZipFile(delivery) as archive:
            info = archive.getinfo("manifest.json")
            if info.file_size > 2 * 1024 * 1024:
                raise HierarchicalInferenceError("delivery manifest exceeds limit")
            payload = json.loads(archive.read(info))
    except HierarchicalInferenceError:
        raise
    except Exception as error:
        raise HierarchicalInferenceError("delivery manifest is unreadable") from error
    bindings = payload.get("bindings") if type(payload) is dict else None
    if not isinstance(bindings, dict):
        raise HierarchicalInferenceError("delivery bindings are missing")
    return MappingProxyType(dict(bindings))


def _copy_member(archive: ZipFile, name: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(name) as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)


def _load_predictor_files(
    *,
    checkpoint_path: Path,
    feature_state_path: Path,
    calibration_path: Path | None,
    candidate_id: str,
    device: str,
    artifact_bytes: int,
    reload_spec: tuple[object, ...],
) -> FrozenHierarchicalPredictor:
    feature_payload = json.loads(feature_state_path.read_text(encoding="utf-8"))
    feature_state = feature_state_from_payload(feature_payload)
    calibration_state = None
    if calibration_path is not None:
        calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        calibration_state = calibration_state_from_payload(calibration_payload)
    torch = import_runtime_module("torch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if type(checkpoint) is not dict or set(checkpoint) != {"model", "epoch"}:
        raise HierarchicalInferenceError("final checkpoint schema differs")
    if (
        isinstance(checkpoint["epoch"], bool)
        or not isinstance(checkpoint["epoch"], int)
        or checkpoint["epoch"] < 0
    ):
        raise HierarchicalInferenceError("final checkpoint epoch is invalid")
    contract = load_contract()
    metadata = ModelMetadata(
        n_num_features=len(feature_state.numeric_columns),
        categorical_cardinalities=tuple(
            len(feature_state.category_maps[column]) + 1
            for column in feature_state.categorical_columns
        ),
        train_x_num=None,
        piecewise_bin_edges=tuple(
            np.asarray(edges, dtype="float32")
            for edges in feature_state.piecewise_bin_edges
        ),
    )
    adapter = TabMAdapter("bce")
    model = adapter.build(
        {
            "architecture": "tabm",
            "k": contract.model.k,
            "width": contract.model.width,
            "blocks": contract.model.blocks,
            "dropout": contract.model.dropout,
            "num_embedding": contract.model.num_embedding,
        },
        metadata,
        device,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    model_digest = _model_state_digest(model)

    def raw_predict(x_num: np.ndarray, x_cat: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            num = torch.as_tensor(x_num, dtype=torch.float32, device=device)
            cat = torch.as_tensor(x_cat, dtype=torch.long, device=device)
            return adapter.probabilities(model, num, cat).detach().cpu().numpy()

    return FrozenHierarchicalPredictor(
        feature_state=feature_state,
        candidate_id=candidate_id,
        raw_predict=raw_predict,
        model_digest=model_digest,
        calibration_state=calibration_state,
        artifact_bytes=artifact_bytes,
        live_model_digest=lambda: _model_state_digest(model),
        reload_spec=reload_spec,
    )


def load_candidate_predictor(
    delivery: Path,
    *,
    candidate_id: str,
    device: str,
    expected_bindings: Mapping[str, str] | None = None,
) -> FrozenHierarchicalPredictor:
    source = Path(delivery)
    if source.is_symlink() or not source.is_file():
        raise HierarchicalInferenceError("candidate delivery is not a regular file")
    bindings = (
        MappingProxyType(dict(expected_bindings))
        if expected_bindings is not None
        else _preliminary_bindings(source)
    )
    manifest = verify_candidate_delivery(source, expected_bindings=bindings)
    contract = load_contract()
    if (
        bindings.get("train_sha256") != contract.official_train_sha256
        or bindings.get("history_sha256") != contract.official_history_sha256
        or bindings.get("stage_c_delivery_sha256") != contract.source_stage_c_delivery_sha256
    ):
        raise HierarchicalInferenceError("delivery is not bound to the sealed campaign inputs")
    candidate_ids = manifest["delivery_candidate_ids"]
    if candidate_id not in candidate_ids:
        raise HierarchicalInferenceError("candidate is absent from delivery")
    calibration_name = None if candidate_id == "H1" else f"state/calibration_{candidate_id}.json"
    required = {"model/final_checkpoint.pt", "state/feature_state.json"}
    if calibration_name is not None:
        required.add(calibration_name)
    if not required.issubset(manifest["members"]):
        raise HierarchicalInferenceError("candidate delivery state is incomplete")

    temporary = Path(tempfile.mkdtemp(prefix="hierarchical-tabm-inference-"))
    try:
        with ZipFile(source) as archive:
            model_path = temporary / "final_checkpoint.pt"
            feature_path = temporary / "feature_state.json"
            _copy_member(archive, "model/final_checkpoint.pt", model_path)
            _copy_member(archive, "state/feature_state.json", feature_path)
            calibration_path = None
            if calibration_name is not None:
                calibration_path = temporary / f"calibration_{candidate_id}.json"
                _copy_member(archive, calibration_name, calibration_path)
        return _load_predictor_files(
            checkpoint_path=model_path,
            feature_state_path=feature_path,
            calibration_path=calibration_path,
            candidate_id=candidate_id,
            artifact_bytes=source.stat().st_size,
            device=device,
            reload_spec=(
                "delivery", str(source.resolve()), candidate_id, device, dict(bindings)
            ),
        )
    except HierarchicalInferenceError:
        raise
    except Exception as error:
        raise HierarchicalInferenceError(f"cannot load frozen candidate: {error}") from error
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def audit_frozen_predictor(
    predictor: FrozenHierarchicalPredictor,
    frame: pd.DataFrame,
    *,
    batch_sizes: tuple[int, ...] = (1, 257, 2048),
    tolerance: float = 1e-6,
) -> IndependenceReport:
    if (
        not batch_sizes
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in batch_sizes)
        or len(batch_sizes) != len(set(batch_sizes))
        or not math.isfinite(float(tolerance))
        or tolerance < 0
    ):
        raise HierarchicalInferenceError("independence audit settings are invalid")
    if "row_id" not in frame or frame["row_id"].isna().any() or frame["row_id"].astype(str).duplicated().any():
        raise HierarchicalInferenceError("audit row_id is invalid")
    before = predictor.state_digest()
    baseline_num, baseline_cat = predictor.encode(frame)
    baseline = predictor.predict_batch(frame, batch_size=batch_sizes[0])
    maximum_delta = 0.0
    features_exact = True
    for batch_size in batch_sizes[1:]:
        current = predictor.predict_batch(frame, batch_size=batch_size)
        maximum_delta = max(maximum_delta, float(np.max(np.abs(current - baseline))))
    positions = np.arange(len(frame))[::-1]
    if len(frame) > 1:
        positions = np.random.default_rng(3407).permutation(len(frame))
    shuffled = frame.iloc[positions].copy()
    shuffled_num, shuffled_cat = predictor.encode(shuffled)
    inverse = np.argsort(positions)
    features_exact = features_exact and np.array_equal(shuffled_num[inverse], baseline_num)
    features_exact = features_exact and np.array_equal(shuffled_cat[inverse], baseline_cat)
    shuffled_probability = predictor.predict_batch(shuffled, batch_size=batch_sizes[-1])[inverse]
    maximum_delta = max(maximum_delta, float(np.max(np.abs(shuffled_probability - baseline))))
    for position in range(len(frame)):
        single_num, single_cat = predictor.encode(frame.iloc[[position]])
        features_exact = features_exact and np.array_equal(single_num[0], baseline_num[position])
        features_exact = features_exact and np.array_equal(single_cat[0], baseline_cat[position])
        single = predictor.predict_batch(frame.iloc[[position]], batch_size=1)[0]
        maximum_delta = max(maximum_delta, abs(float(single - baseline[position])))
    after = predictor.state_digest()
    if not features_exact or maximum_delta > tolerance or before != after:
        raise HierarchicalInferenceError("frozen predictor independence audit failed")
    return IndependenceReport(
        len(frame), features_exact, maximum_delta, before, after, batch_sizes
    )


def audit_inference_limits(
    predictor: FrozenHierarchicalPredictor,
    frame: pd.DataFrame,
    contract: HierarchicalContract,
    *,
    resource_probe: Callable[[], tuple[float, int, int]] | None = None,
) -> InferenceResourceReport:
    if resource_probe is None:
        if predictor._reload_spec is None:
            raise HierarchicalInferenceError(
                "bounded resource audit requires a verified delivery reload specification"
            )
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_resource_audit_child,
            args=(child, predictor._reload_spec, frame),
        )
        process.start()
        child.close()
        timeout = float(contract.inference_limits["inference_seconds"]) + 60.0
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join(5.0)
            if process.is_alive():
                process.kill()
                process.join(5.0)
            parent.close()
            raise HierarchicalInferenceError("bounded inference audit timed out")
        if process.exitcode != 0 or not parent.poll():
            parent.close()
            raise HierarchicalInferenceError("bounded inference audit failed")
        accepted, payload = parent.recv()
        parent.close()
        if accepted is not True:
            raise HierarchicalInferenceError(f"bounded inference audit failed: {payload}")
        elapsed, gpu, rss = payload
    else:
        observed = resource_probe()
        if (
            not isinstance(observed, tuple)
            or len(observed) != 3
            or isinstance(observed[0], bool)
            or not isinstance(observed[0], (int, float))
            or any(isinstance(value, bool) or not isinstance(value, int) for value in observed[1:])
        ):
            raise HierarchicalInferenceError("resource probe result is invalid")
        elapsed, gpu, rss = float(observed[0]), int(observed[1]), int(observed[2])
    if not math.isfinite(elapsed) or elapsed < 0 or gpu < 0 or rss < 0:
        raise HierarchicalInferenceError("resource measurements are invalid")
    version = platform.python_version()
    limits = contract.inference_limits
    passed = (
        version == limits["python_version"]
        and elapsed <= int(limits["inference_seconds"])
        and gpu <= int(limits["gpu_bytes"])
        and rss <= int(limits["rss_bytes"])
        and predictor.artifact_bytes <= int(limits["artifact_bytes"])
    )
    return InferenceResourceReport(
        len(frame), version, elapsed, gpu, rss, predictor.artifact_bytes, passed
    )


def _resource_audit_child(
    connection,
    reload_spec: tuple[object, ...],
    frame: pd.DataFrame,
) -> None:
    try:
        import time

        if reload_spec[0] == "delivery" and len(reload_spec) == 5:
            _, delivery, candidate_id, device, bindings = reload_spec
            predictor = load_candidate_predictor(
                Path(str(delivery)),
                candidate_id=str(candidate_id),
                device=str(device),
                expected_bindings=dict(bindings),
            )
        elif reload_spec[0] == "files" and len(reload_spec) == 7:
            (
                _, checkpoint, feature_state, calibration, candidate_id, device,
                artifact_bytes,
            ) = reload_spec
            predictor = _load_predictor_files(
                checkpoint_path=Path(str(checkpoint)),
                feature_state_path=Path(str(feature_state)),
                calibration_path=None if calibration is None else Path(str(calibration)),
                candidate_id=str(candidate_id),
                device=str(device),
                artifact_bytes=int(artifact_bytes),
                reload_spec=reload_spec,
            )
        else:
            raise HierarchicalInferenceError("resource reload specification differs")
        torch = import_runtime_module("torch")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        predictor.predict_batch(frame)
        elapsed = time.monotonic() - started
        gpu = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rss = peak if platform.system() == "Darwin" else peak * 1024
        connection.send((True, (elapsed, gpu, rss)))
    except BaseException as error:
        connection.send((False, f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def validate_candidate_artifacts(
    *,
    bindings: Mapping[str, str],
    delivery_roles: Mapping[str, str],
    final_checkpoint_path: Path,
    feature_state_path: Path,
    calibration_paths: Mapping[str, Path],
    audit_frame: pd.DataFrame,
    output_dir: Path,
    device: str,
    contract: HierarchicalContract | None = None,
    resource_probe: Callable[[], tuple[float, int, int]] | None = None,
) -> CandidateValidation:
    sealed = load_contract() if contract is None else contract
    roles = dict(delivery_roles)
    if not roles:
        return CandidateValidation(MappingProxyType({}), MappingProxyType({}))
    if len(audit_frame) < 245_789:
        raise HierarchicalInferenceError("scale audit requires 245789 train-derived rows")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = Path(final_checkpoint_path)
    feature_path = Path(feature_state_path)
    sources = [model_path, feature_path, *(Path(path) for path in calibration_paths.values())]
    if any(path.is_symlink() or not path.is_file() for path in sources):
        raise HierarchicalInferenceError("candidate validation source is invalid")
    artifact_bytes = sum(path.stat().st_size for path in sources)
    scale_frame = audit_frame.iloc[:245_789].drop(
        columns=["control_success"], errors="ignore"
    ).copy()
    independence_frame = scale_frame.iloc[: min(257, len(scale_frame))].copy()
    accepted: dict[str, str] = {}
    reports: dict[str, Path] = {}
    for candidate_id, role in roles.items():
        report_path = output / "reports" / f"{candidate_id}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        member_paths = {
            "model/final_checkpoint.pt": model_path,
            "state/feature_state.json": feature_path,
        }
        if candidate_id != "H1" and candidate_id in calibration_paths:
            member_paths[f"state/calibration_{candidate_id}.json"] = Path(
                calibration_paths[candidate_id]
            )
        validated_members = {
            name: _file_sha256(path) for name, path in member_paths.items()
        }
        try:
            calibration_path = (
                None if candidate_id == "H1" else Path(calibration_paths[candidate_id])
            )
            reload_spec = (
                "files", str(model_path.resolve()), str(feature_path.resolve()),
                None if calibration_path is None else str(calibration_path.resolve()),
                candidate_id, device, artifact_bytes,
            )
            predictor = _load_predictor_files(
                checkpoint_path=model_path,
                feature_state_path=feature_path,
                calibration_path=calibration_path,
                candidate_id=candidate_id,
                device=device,
                artifact_bytes=artifact_bytes,
                reload_spec=reload_spec,
            )
            independence = audit_frozen_predictor(
                predictor, independence_frame, batch_sizes=(1, 257, 2048)
            )
            resources = audit_inference_limits(
                predictor,
                scale_frame,
                sealed,
                resource_probe=resource_probe,
            )
            if any(
                _file_sha256(path) != validated_members[name]
                for name, path in member_paths.items()
            ):
                raise HierarchicalInferenceError(
                    "candidate source changed during validation"
                )
            passed = resources.passed
            payload = {
                "schema_version": 1,
                "candidate_id": candidate_id,
                "delivery_role": role,
                "passed": passed,
                "independence": asdict(independence),
                "resources": asdict(resources),
                "validated_members": validated_members,
                "failure": None if passed else "sealed_inference_limit_exceeded",
            }
            if passed:
                accepted[candidate_id] = role
        except Exception as error:
            payload = {
                "schema_version": 1,
                "candidate_id": candidate_id,
                "delivery_role": role,
                "passed": False,
                "independence": None,
                "resources": None,
                "validated_members": validated_members,
                "failure": f"{type(error).__name__}: {error}",
            }
        report_path.write_bytes(_canonical_json(payload))
        reports[candidate_id] = report_path
    return CandidateValidation(
        MappingProxyType(accepted), MappingProxyType(reports)
    )
