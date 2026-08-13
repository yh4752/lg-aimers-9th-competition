"""Defense-in-depth source and behavioral gates for inference code."""

from __future__ import annotations

import ast
from hashlib import sha256
from math import isfinite
from numbers import Real
from pathlib import Path
import stat
from typing import Callable, Sequence


class RulesCodeGateError(ValueError):
    """Raised when inference code cannot prove row-local behavior."""


_BLOCKED_IMPORT_ROOTS = {
    "anthropic",
    "boto3",
    "http",
    "openai",
    "requests",
    "socket",
    "urllib",
}
_BLOCKED_INFERENCE_METHODS = {
    "cumcount",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "ewm",
    "expanding",
    "groupby",
    "lag",
    "rank",
    "rolling",
    "shift",
}
_INFERENCE_PREFIXES = ("predict", "transform", "forward", "inference", "infer_batch")


def _safe_sources(
    paths: Sequence[str | Path], project_root: str | Path
) -> list[Path]:
    root = Path(project_root).expanduser().resolve(strict=True)
    output: list[Path] = []
    for raw in paths:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            relative = candidate.relative_to(root)
        except ValueError as error:
            raise RulesCodeGateError(f"source is outside project root: {candidate}") from error
        current = root
        try:
            for part in relative.parts:
                current = current / part
                if stat.S_ISLNK(current.lstat().st_mode):
                    raise RulesCodeGateError(f"symlink source is not allowed: {current}")
        except RulesCodeGateError:
            raise
        except (OSError, ValueError) as error:
            raise RulesCodeGateError(f"cannot inspect source: {candidate}") from error
        if current.is_dir():
            nested = sorted(path for path in current.rglob("*.py") if path.is_file())
            if not nested:
                raise RulesCodeGateError(f"source directory has no Python files: {current}")
            output.extend(_safe_sources(nested, root))
        elif current.is_file() and current.suffix == ".py":
            output.append(current)
        else:
            raise RulesCodeGateError(f"source must be a Python file or directory: {current}")
    unique = sorted(set(output), key=lambda path: path.relative_to(root).as_posix())
    if not unique:
        raise RulesCodeGateError("at least one inference source is required")
    return unique


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _numeric_dict(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Dict)
        and bool(node.keys)
        and all(isinstance(key, ast.Constant) and isinstance(key.value, str) for key in node.keys)
        and all(
            isinstance(value, ast.Constant)
            and isinstance(value.value, (int, float))
            and not isinstance(value.value, bool)
            for value in node.values
        )
    )


class _SourceVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []
        self._inference_depth = 0

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".", 1)[0] in _BLOCKED_IMPORT_ROOTS:
                self.errors.append(f"blocked import: {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.split(".", 1)[0] in _BLOCKED_IMPORT_ROOTS:
            self.errors.append(f"blocked import: {node.module}")

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        name = node.name.lower().lstrip("_")
        is_inference = any(
            name == prefix or name.startswith(prefix + "_")
            for prefix in _INFERENCE_PREFIXES
        )
        self._inference_depth += int(is_inference)
        self.generic_visit(node)
        self._inference_depth -= int(is_inference)

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Assign(self, node: ast.Assign) -> None:
        names = [target.id.lower() for target in node.targets if isinstance(target, ast.Name)]
        if any("prediction" in name or "probability_lookup" in name for name in names):
            if _numeric_dict(node.value):
                self.errors.append("hard-coded row prediction lookup")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._inference_depth:
            if isinstance(node.func, ast.Attribute):
                method = node.func.attr.lower()
                if method in _BLOCKED_INFERENCE_METHODS:
                    self.errors.append(f"blocked evaluation-row operation: {method}")
                if method in {"read_csv", "read_parquet", "read_feather"} and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        if Path(first.value).name in {"test.csv", "sample_submission.csv"}:
                            self.errors.append("adapter reads an evaluation file")
            if isinstance(node.func, ast.Name) and node.func.id == "open" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    if Path(first.value).name in {"test.csv", "sample_submission.csv"}:
                        self.errors.append("adapter opens an evaluation file")
            for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
                if isinstance(argument, ast.Constant) and argument.value in {5, 245789}:
                    if isinstance(node.func, ast.Name) and node.func.id in {"range", "full", "zeros", "ones"}:
                        self.errors.append("evaluation-size-specific inference")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if self._inference_depth:
            values = [node.left, *node.comparators]
            if any(
                isinstance(value, ast.Constant) and value.value in {5, 245789}
                for value in values
            ) and any(
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "len"
                for value in values
            ):
                self.errors.append("evaluation-size-specific inference")
        self.generic_visit(node)


def inspect_inference_source(
    paths: Sequence[str | Path], *, project_root: str | Path
) -> dict[str, object]:
    """Scan registered inference source without treating it as proof by itself."""

    root = Path(project_root).expanduser().resolve(strict=True)
    sources = _safe_sources(paths, root)
    digest = sha256()
    for path in sources:
        try:
            data = path.read_bytes()
            tree = ast.parse(data.decode("utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as error:
            raise RulesCodeGateError(f"cannot parse inference source: {path}") from error
        visitor = _SourceVisitor()
        visitor.visit(tree)
        if visitor.errors:
            detail = "; ".join(sorted(set(visitor.errors)))
            raise RulesCodeGateError(f"{path.relative_to(root)}: {detail}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return {
        "status": "passed",
        "file_count": len(sources),
        "source_sha256": digest.hexdigest(),
    }


def canonical_probability(value: float, *, decimal_places: int = 8) -> str:
    """Return one bounded finite probability using fixed decimal precision."""

    if type(decimal_places) is not int or decimal_places < 1 or decimal_places > 15:
        raise RulesCodeGateError("decimal_places is invalid")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RulesCodeGateError("probability must be a real number")
    number = float(value)
    if not isfinite(number) or number < 0.0 or number > 1.0:
        raise RulesCodeGateError("probability must be finite and inside [0, 1]")
    return f"{number:.{decimal_places}f}"


def assert_row_independent(
    frame: "object",
    *,
    load_predictor: Callable[[], Callable[["object"], object]],
    batch_sizes: Sequence[int] = (1, 2, 3, 17),
    decimal_places: int = 8,
    state_digest: Callable[[object], str] | None = None,
) -> dict[str, object]:
    """Compare repeat, reverse, fixed shuffle, rebatch, and singleton outputs."""

    import numpy as np
    import pandas as pd

    if not isinstance(frame, pd.DataFrame) or frame.empty or "row_id" not in frame:
        raise RulesCodeGateError("row independence requires a non-empty frame with row_id")
    row_ids = frame["row_id"].astype("string")
    if row_ids.isna().any():
        raise RulesCodeGateError("row_id must be non-null")
    normalized = row_ids.astype(str)
    if normalized.duplicated().any():
        raise RulesCodeGateError("row_id must be unique after string normalization")
    source = frame.copy(deep=True)
    source["row_id"] = normalized

    def predict(value: pd.DataFrame) -> list[str]:
        predictor = load_predictor()
        digest_function = state_digest
        if digest_function is None and callable(getattr(predictor, "state_digest", None)):
            digest_function = lambda current: str(current.state_digest())
        before = None if digest_function is None else digest_function(predictor)
        output = np.asarray(predictor(value.copy(deep=True)), dtype="float64")
        after = None if digest_function is None else digest_function(predictor)
        if before != after:
            raise RulesCodeGateError("predictor state changed during inference")
        if output.shape != (len(value),):
            raise RulesCodeGateError("prediction output must be one-dimensional and row-aligned")
        return [canonical_probability(item, decimal_places=decimal_places) for item in output]

    def mapped(value: pd.DataFrame, predictions: list[str]) -> dict[str, str]:
        return dict(zip(value["row_id"].astype(str), predictions, strict=True))

    baseline_predictions = predict(source)
    baseline = mapped(source, baseline_predictions)

    variants: list[tuple[str, dict[str, str]]] = []
    variants.append(("repeat", mapped(source, predict(source))))
    reversed_frame = source.iloc[::-1].reset_index(drop=True)
    variants.append(("reverse", mapped(reversed_frame, predict(reversed_frame))))
    order = sorted(
        range(len(source)),
        key=lambda index: sha256(source.iloc[index]["row_id"].encode("utf-8")).hexdigest(),
    )
    shuffled = source.iloc[order].reset_index(drop=True)
    variants.append(("shuffle", mapped(shuffled, predict(shuffled))))

    for size in batch_sizes:
        if type(size) is not int or size < 1:
            raise RulesCodeGateError("batch_sizes must contain positive integers")
        combined: dict[str, str] = {}
        for start in range(0, len(source), size):
            part = source.iloc[start : start + size].reset_index(drop=True)
            combined.update(mapped(part, predict(part)))
        variants.append((f"batch_{size}", combined))

    singleton: dict[str, str] = {}
    for index in range(len(source)):
        part = source.iloc[[index]].reset_index(drop=True)
        singleton.update(mapped(part, predict(part)))
    variants.append(("singleton", singleton))

    for label, candidate in variants:
        if candidate != baseline:
            raise RulesCodeGateError(f"row independence mismatch: {label}")

    result_digest = sha256()
    for row_id in sorted(baseline):
        result_digest.update(row_id.encode("utf-8"))
        result_digest.update(b"\0")
        result_digest.update(baseline[row_id].encode("ascii"))
        result_digest.update(b"\n")
    return {
        "status": "passed",
        "row_count": len(source),
        "decimal_places": decimal_places,
        "variant_count": len(variants) + 1,
        "prediction_sha256": result_digest.hexdigest(),
    }
