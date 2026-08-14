"""Standalone evaluator for the frozen Version D TabM model."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


EMBEDDED_METADATA = None
MISSING_CATEGORY = "__MISSING__"
MODEL_NAMES = {
    "inference_manifest.json",
    "numeric_embedding_0.json",
    "preprocessing_state.json",
    "tabm_member_0_seed_3407.pt",
}


class EvaluatorError(ValueError):
    """Raised before an invalid prediction file can be published."""


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    def unique(items: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in items:
            if key in output:
                raise EvaluatorError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    def finite(value: str) -> object:
        raise EvaluatorError(f"non-finite JSON value: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=finite,
        )
    except EvaluatorError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluatorError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise EvaluatorError(f"JSON must be an object: {path}")
    return value


def _category(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_CATEGORY)


def _number(series: pd.Series, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if (series.notna() & values.isna()).any():
        raise EvaluatorError(f"non-numeric value in {column}")
    return values.astype("float64")


def _model_digest(model_dir: Path) -> str:
    digest = sha256()
    for name in sorted(MODEL_NAMES):
        value = (model_dir / name).read_bytes()
        digest.update(name.encode("utf-8") + b"\0" + sha256(value).digest())
    return digest.hexdigest()


def _verify_model_directory(model_dir: Path) -> dict[str, object]:
    if not isinstance(EMBEDDED_METADATA, dict):
        raise EvaluatorError("embedded candidate metadata is missing")
    if not model_dir.is_dir() or model_dir.is_symlink():
        raise EvaluatorError("model directory is missing or unsafe")
    paths = list(model_dir.iterdir())
    if {path.name for path in paths} != MODEL_NAMES or any(
        path.is_symlink() or not path.is_file() for path in paths
    ):
        raise EvaluatorError("model directory member set differs")
    expected = EMBEDDED_METADATA.get("members")
    if not isinstance(expected, dict) or set(expected) != MODEL_NAMES:
        raise EvaluatorError("embedded model member manifest differs")
    for name in MODEL_NAMES:
        if _sha256_file(model_dir / name) != expected[name]:
            raise EvaluatorError(f"model member SHA-256 differs: {name}")
    if _model_digest(model_dir) != EMBEDDED_METADATA.get("model_sha256"):
        raise EvaluatorError("combined model SHA-256 differs")
    manifest = _load_json(model_dir / "inference_manifest.json")
    files = manifest.get("files")
    expected_files = MODEL_NAMES - {"inference_manifest.json"}
    if not isinstance(files, dict) or set(files) != expected_files:
        raise EvaluatorError("inference manifest file set differs")
    for name in expected_files:
        if files[name] != expected[name]:
            raise EvaluatorError(f"inference manifest SHA-256 differs: {name}")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("epochs") != 3
        or manifest.get("seeds") != [3407]
        or manifest.get("scheduler") != "constant"
        or manifest.get("fit_scope") != "official_train_2019_2024_only"
        or manifest.get("row_count") != 1_475_092
        or manifest.get("preprocessing_state") != "preprocessing_state.json"
    ):
        raise EvaluatorError("inference manifest identity differs")
    return manifest


def _verify_preprocessing_state(state: dict[str, object]) -> None:
    preprocessing = state.get("preprocessing")
    if state.get("view") != "raw_typed" or state.get("trackman") is not None:
        raise EvaluatorError("frozen feature view differs")
    if not isinstance(preprocessing, dict) or preprocessing.get("spec") != {
        "profile": "dl_standard",
        "components": ["hand_matchup"],
    }:
        raise EvaluatorError("frozen preprocessing spec differs")
    if preprocessing.get("yeo_johnson_lambda") != {}:
        raise EvaluatorError("unexpected nonlinear preprocessing state")
    if preprocessing.get("entity_frequency") != {}:
        raise EvaluatorError("unexpected entity-frequency state")
    required_lists = (
        "source_columns",
        "output_columns",
        "numeric_columns",
        "categorical_columns",
    )
    if any(not isinstance(preprocessing.get(name), list) for name in required_lists):
        raise EvaluatorError("preprocessing column state differs")
    if state.get("numeric_columns") != preprocessing.get("numeric_columns"):
        raise EvaluatorError("numeric feature order differs")
    if state.get("categorical_columns") != preprocessing.get("categorical_columns"):
        raise EvaluatorError("categorical feature order differs")
    if "hand_matchup" not in preprocessing["categorical_columns"]:
        raise EvaluatorError("hand_matchup feature is missing")


def _transform(frame: pd.DataFrame, state: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    _verify_preprocessing_state(state)
    preprocessing = state["preprocessing"]
    source = frame.drop(columns=[name for name in ("row_id", "control_success") if name in frame]).copy()
    duplicate = "asof_pitcher_pitchmix_n"
    if duplicate in source:
        if "asof_pitcher_n" not in source:
            raise EvaluatorError("duplicate count source is incomplete")
        left = _number(source["asof_pitcher_n"], "asof_pitcher_n")
        right = _number(source[duplicate], duplicate)
        observed = left.notna() & right.notna()
        if not left.isna().equals(right.isna()) or not np.array_equal(
            left.loc[observed].to_numpy(), right.loc[observed].to_numpy()
        ):
            raise EvaluatorError("asof_pitcher_pitchmix_n differs from asof_pitcher_n")
        source = source.drop(columns=[duplicate])
    if source.columns.tolist() != preprocessing["source_columns"]:
        missing = sorted(set(preprocessing["source_columns"]) - set(source.columns))
        extra = sorted(set(source.columns) - set(preprocessing["source_columns"]))
        raise EvaluatorError(f"preprocessing source schema differs: missing={missing}, extra={extra}")
    prepared = source.copy()
    prepared["hand_matchup"] = (
        _category(prepared["pitcher_hand"]) + "_" + _category(prepared["batter_hand"])
    )
    if prepared.columns.tolist() != preprocessing["output_columns"]:
        raise EvaluatorError("preprocessing output schema differs")

    numeric_columns = preprocessing["numeric_columns"]
    medians = preprocessing.get("numeric_median")
    means = preprocessing.get("numeric_mean")
    scales = preprocessing.get("numeric_std")
    if not all(isinstance(item, dict) for item in (medians, means, scales)):
        raise EvaluatorError("numeric preprocessing state differs")
    if any(set(item) != set(numeric_columns) for item in (medians, means, scales)):
        raise EvaluatorError("numeric preprocessing keys differ")
    for column in numeric_columns:
        values = _number(prepared[column], column).replace([np.inf, -np.inf], np.nan)
        imputed = values.fillna(float(medians[column])).to_numpy(dtype="float64")
        scale = float(scales[column])
        if not math.isfinite(scale) or scale <= 0:
            raise EvaluatorError(f"invalid numeric scale: {column}")
        transformed = (imputed - float(means[column])) / scale
        if not np.isfinite(transformed).all():
            raise EvaluatorError(f"non-finite transformed values: {column}")
        prepared[column] = transformed

    x_num = prepared.loc[:, state["numeric_columns"]].to_numpy(
        dtype="float32", copy=True
    )
    maps = state.get("category_maps")
    if not isinstance(maps, dict) or set(maps) != set(state["categorical_columns"]):
        raise EvaluatorError("category map state differs")
    encoded = []
    for column in state["categorical_columns"]:
        mapping = maps[column]
        if not isinstance(mapping, dict):
            raise EvaluatorError(f"invalid category map: {column}")
        encoded.append(
            _category(prepared[column]).map(mapping).fillna(0).to_numpy(dtype="int64")
        )
    x_cat = np.column_stack(encoded).astype("int64", copy=True)
    return x_num, x_cat


class FrozenTabMPredictor:
    def __init__(self, model_dir: Path, manifest: dict[str, object], *, device: str) -> None:
        import rtdl_num_embeddings
        import tabm
        import torch

        self._torch = torch
        self._device = device
        self._model_dir = model_dir
        self._state = _load_json(model_dir / "preprocessing_state.json")
        _verify_preprocessing_state(self._state)
        numeric = _load_json(model_dir / "numeric_embedding_0.json")
        if numeric.get("mode") != "piecewise_linear" or numeric.get("embedding_dim") != 32:
            raise EvaluatorError("numeric embedding state differs")
        edges = numeric.get("piecewise_bin_edges")
        if not isinstance(edges, list) or len(edges) != len(self._state["numeric_columns"]):
            raise EvaluatorError("piecewise bin-edge state differs")
        bins = [torch.as_tensor(edge, dtype=torch.float32) for edge in edges]
        embedding = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            bins, 32, activation=False, version="B"
        )
        members = manifest.get("members")
        if not isinstance(members, list) or len(members) != 1:
            raise EvaluatorError("one frozen model member is required")
        member = members[0]
        config = member.get("model_config") if isinstance(member, dict) else None
        expected_config = {
            "architecture": "tabm",
            "blocks": 4,
            "dropout": 0.1,
            "k": 32,
            "num_embedding": "piecewise_linear",
            "width": 512,
        }
        if (
            not isinstance(member, dict)
            or member.get("seed") != 3407
            or member.get("weights") != "tabm_member_0_seed_3407.pt"
            or member.get("numeric_state") != "numeric_embedding_0.json"
            or config != expected_config
        ):
            raise EvaluatorError("frozen model member differs")
        cardinalities = [
            len(self._state["category_maps"][name]) + 1
            for name in self._state["categorical_columns"]
        ]
        backbone = tabm.TabM.make(
            n_num_features=len(self._state["numeric_columns"]),
            cat_cardinalities=cardinalities,
            d_out=1,
            num_embeddings=embedding,
            n_blocks=4,
            d_block=512,
            dropout=0.1,
            k=32,
            arch_type="tabm",
        )

        class ModelWrapper(torch.nn.Module):
            def __init__(self, model: object) -> None:
                super().__init__()
                self.model = model

            def forward(self, x_num: object, x_cat: object) -> object:
                return self.model(x_num, x_cat)

        model = ModelWrapper(backbone).to(device)
        weights = torch.load(
            model_dir / "tabm_member_0_seed_3407.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(weights, strict=True)
        model.eval()
        self._model = model
        self._digest = _model_digest(model_dir)

    def state_digest(self) -> str:
        return self._digest

    def encode(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return _transform(frame.reset_index(drop=True), self._state)

    def predict_batch(self, frame: pd.DataFrame, *, batch_size: int = 2048) -> np.ndarray:
        if type(batch_size) is not int or batch_size < 1:
            raise EvaluatorError("batch_size must be positive")
        x_num, x_cat = self.encode(frame)
        outputs = []
        with self._torch.inference_mode():
            for start in range(0, len(frame), batch_size):
                stop = min(start + batch_size, len(frame))
                num = self._torch.as_tensor(
                    x_num[start:stop], dtype=self._torch.float32, device=self._device
                )
                cat = self._torch.as_tensor(
                    x_cat[start:stop], dtype=self._torch.long, device=self._device
                )
                values = self._model(num, cat).squeeze(-1).sigmoid().mean(dim=1)
                outputs.append(values.to(dtype=self._torch.float32, device="cpu").numpy())
        result = np.concatenate(outputs).astype("float64", copy=False)
        if result.shape != (len(frame),) or not np.isfinite(result).all():
            raise EvaluatorError("predictions must be finite and row-aligned")
        if ((result < 0) | (result > 1)).any():
            raise EvaluatorError("predictions must be inside [0, 1]")
        return result


def load_frozen_predictor(model_dir: Path, *, device: str = "cuda") -> FrozenTabMPredictor:
    if device not in {"cuda", "cpu"}:
        raise EvaluatorError("device must be cuda or cpu")
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise EvaluatorError("CUDA is required for submitted inference")
    manifest = _verify_model_directory(model_dir)
    return FrozenTabMPredictor(model_dir, manifest, device=device)


def validate_inputs(
    test: pd.DataFrame, sample: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(test, pd.DataFrame) or not isinstance(sample, pd.DataFrame):
        raise EvaluatorError("test and sample submission must be DataFrames")
    if test.empty or "row_id" not in test:
        raise EvaluatorError("test rows must be non-empty with row_id")
    if sample.columns.tolist() != ["row_id", "control_success"]:
        raise EvaluatorError("sample submission columns differ")
    test_copy = test.copy(deep=True)
    sample_copy = sample.copy(deep=True)
    for label, frame in (("test", test_copy), ("sample", sample_copy)):
        ids = frame["row_id"].astype("string")
        if ids.isna().any() or ids.astype(str).duplicated().any():
            raise EvaluatorError(f"{label} row_id must be non-null and unique")
        frame["row_id"] = ids.astype(str)
    if len(test_copy) != len(sample_copy) or set(test_copy["row_id"]) != set(sample_copy["row_id"]):
        raise EvaluatorError("test row IDs must exactly match sample submission")
    return test_copy, sample_copy


def _canonical_probability(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise EvaluatorError("probability must be finite")
    if number < 0 or number > 1:
        raise EvaluatorError("probability must be inside [0, 1]")
    return f"{number:.8f}"


def _predict_checked(predictor: object, frame: pd.DataFrame) -> list[str]:
    before = predictor.state_digest()
    values = np.asarray(predictor.predict_batch(frame.copy(deep=True)), dtype="float64")
    if predictor.state_digest() != before:
        raise EvaluatorError("predictor state changed during inference")
    if values.shape != (len(frame),):
        raise EvaluatorError("predictions must be one-dimensional and row-aligned")
    return [_canonical_probability(value) for value in values]


def write_submission(
    sample: pd.DataFrame, by_id: Mapping[str, str], output: Path
) -> None:
    if output.exists():
        raise EvaluatorError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.partial")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["row_id", "control_success"])
            writer.writerows((row_id, by_id[row_id]) for row_id in sample["row_id"])
            handle.flush()
            os.fsync(handle.fileno())
        with output.open("xb") as target, temporary.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
    except Exception:
        if output.exists():
            output.unlink()
        raise
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    if not isinstance(EMBEDDED_METADATA, dict):
        raise EvaluatorError("script is not bound to a candidate")
    data_dir = Path("data")
    model_dir = Path("model")
    output = Path("output/submission.csv")
    test = pd.read_csv(data_dir / "test.csv", dtype={"row_id": "string"})
    sample = pd.read_csv(
        data_dir / "sample_submission.csv", dtype={"row_id": "string"}
    )
    test, sample = validate_inputs(test, sample)
    predictor = load_frozen_predictor(model_dir)
    predictions = _predict_checked(predictor, test)
    by_id = dict(zip(test["row_id"], predictions, strict=True))
    ordered = sorted(
        range(len(test)),
        key=lambda index: sha256(
            (CANDIDATE_SEED + "\0" + test.iloc[index]["row_id"]).encode("utf-8")
        ).hexdigest(),
    )[: min(8, len(test))]
    for index in ordered:
        single = test.iloc[[index]].reset_index(drop=True)
        single_value = float(_predict_checked(predictor, single)[0])
        batch_value = float(by_id[single.iloc[0]["row_id"]])
        if abs(single_value - batch_value) > 1e-6:
            raise EvaluatorError("row-independence canary mismatch")
    write_submission(sample, by_id, output)
    print(
        f"SUBMISSION_SUCCESS rows={len(sample)} output={output} "
        f"candidate={EMBEDDED_METADATA['candidate_id']}"
    )
    return 0


CANDIDATE_SEED = "dacon-236743-tabm-version-d-v1"


if __name__ == "__main__":
    raise SystemExit(main())
