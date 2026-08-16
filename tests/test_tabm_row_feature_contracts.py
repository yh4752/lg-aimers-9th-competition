from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from experiments.independent_dl.row_features import ROW_FEATURE_BUNDLES
from experiments.tabm_campaign.row_feature_contracts import (
    DEFAULT_ROW_FEATURE_PROXY_CONTRACT,
    RowFeatureContractError,
    load_row_feature_proxy_contract,
    row_feature_contract_sha256,
)


CONFIG = Path("experiments/tabm_campaign/configs/row_feature_proxy_v1.json")
CHAMPION_CONFIG = Path("experiments/tabm_campaign/configs/champion_v1.json")
EXPECTED_DEFAULT_SHA256 = "529f201dc360be0a1bab88cea989a8aba1d4f4e9d5355614ccc72daa5c501d36"


def _payload() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _write_payload(tmp_path: Path, payload: Any, *, allow_nan: bool = True) -> Path:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload, allow_nan=allow_nan), encoding="utf-8")
    return path


def _cursor(payload: Any, path: tuple[str, ...]) -> dict[str, Any]:
    current = payload
    for key in path:
        current = current[key]
    assert isinstance(current, dict)
    return current


def _replace(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    parent = _cursor(payload, path[:-1])
    parent[path[-1]] = value


def test_default_contract_loads_exact_immutable_values() -> None:
    contract = load_row_feature_proxy_contract()

    assert DEFAULT_ROW_FEATURE_PROXY_CONTRACT == CONFIG.resolve()
    assert contract.schema_version == 1
    assert contract.stage == "P"
    assert contract.review_only is True
    assert contract.official_train_sha256 == (
        "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"
    )
    assert contract.baseline.capacity == "p2"
    assert contract.baseline.k == 32
    assert contract.baseline.width == 512
    assert contract.baseline.blocks == 4
    assert contract.baseline.dropout == 0.1
    assert contract.baseline.num_embedding == "piecewise_linear"
    assert contract.baseline.loss == "bce"
    assert contract.baseline.scheduler == "plateau"
    assert contract.baseline.learning_rate == 0.0006
    assert contract.baseline.weight_decay == 0.0001
    assert contract.baseline.effective_batch_size == 4096
    assert contract.baseline.micro_batch_size == 512
    assert contract.fold.train_end_year == 2023
    assert contract.fold.valid_year == 2024
    assert contract.sample.mode == "proxy"
    assert contract.sample.max_rows == 400_000
    assert (contract.training.max_epochs, contract.training.min_epochs) == (8, 3)
    assert contract.training.patience == 3
    assert (contract.budget.wall_seconds, contract.budget.new_job_guard_seconds) == (
        10_800,
        900,
    )
    assert contract.proxy_gate.mean_delta_max == -0.00003
    assert contract.proxy_gate.worst_seed_delta_max == 0.00005

    with pytest.raises(FrozenInstanceError):
        contract.stage = "Q"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        contract.baseline.k = 64  # type: ignore[misc]


def test_contract_exposes_only_immutable_ordered_sequences() -> None:
    contract = load_row_feature_proxy_contract()

    assert contract.seeds == (42, 3407)
    assert isinstance(contract.seeds, tuple)
    assert contract.feature_bundles == ROW_FEATURE_BUNDLES
    assert contract.feature_bundles == (
        "count_context",
        "pressure_context",
        "hand_state_interactions",
        "pitcher_batter_gap",
        "recent_trend",
        "pitchmix_shape",
    )
    assert isinstance(contract.feature_bundles, tuple)


def test_default_contract_raw_sha256_is_pinned() -> None:
    assert row_feature_contract_sha256() == EXPECTED_DEFAULT_SHA256


def test_baseline_matches_sealed_champion_candidate() -> None:
    contract = load_row_feature_proxy_contract()
    champion = json.loads(CHAMPION_CONFIG.read_text(encoding="utf-8"))
    candidate = next(
        item
        for item in champion["candidates"]
        if item["candidate_id"] == "a__p2__piecewise_linear__bce__plateau__s42"
    )

    assert (
        contract.baseline.capacity,
        contract.baseline.k,
        contract.baseline.width,
        contract.baseline.blocks,
        contract.baseline.dropout,
        contract.baseline.num_embedding,
        contract.baseline.loss,
        contract.baseline.scheduler,
        contract.baseline.learning_rate,
    ) == (
        candidate["capacity"],
        candidate["k"],
        candidate["width"],
        candidate["blocks"],
        candidate["dropout"],
        candidate["num_embedding"],
        candidate["loss"],
        candidate["scheduler"],
        candidate["learning_rate"],
    )
    assert (
        contract.baseline.weight_decay,
        contract.baseline.effective_batch_size,
        contract.baseline.micro_batch_size,
    ) == (
        champion["optimization"]["weight_decay"],
        champion["optimization"]["effective_batch_size"],
        champion["optimization"]["micro_batch_size"],
    )


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ((), "schema_version"),
        (("baseline",), "capacity"),
        (("fold",), "train_end_year"),
        (("sample",), "mode"),
        (("training",), "max_epochs"),
        (("budget",), "wall_seconds"),
        (("proxy_gate",), "mean_delta_max"),
    ],
)
@pytest.mark.parametrize("change", ["missing", "extra"])
def test_contract_rejects_missing_or_extra_fields_in_every_section(
    tmp_path: Path,
    section: tuple[str, ...],
    field: str,
    change: str,
) -> None:
    payload = _payload()
    target = _cursor(payload, section)
    if change == "missing":
        del target[field]
    else:
        target["surprise"] = "not sealed"

    with pytest.raises(RowFeatureContractError, match="keys"):
        load_row_feature_proxy_contract(_write_payload(tmp_path, payload))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), True),
        (("stage",), 80),
        (("review_only",), 1),
        (("official_train_sha256",), 123),
        (("baseline",), []),
        (("baseline", "k"), True),
        (("baseline", "width"), "512"),
        (("baseline", "dropout"), True),
        (("baseline", "learning_rate"), "0.0006"),
        (("seeds",), "42,3407"),
        (("seeds",), [42, True]),
        (("feature_bundles",), "count_context"),
        (("fold", "valid_year"), 2024.0),
        (("sample", "max_rows"), False),
        (("training", "patience"), "3"),
        (("budget", "wall_seconds"), True),
        (("proxy_gate", "mean_delta_max"), False),
    ],
)
def test_contract_rejects_wrong_types(
    tmp_path: Path, path: tuple[str, ...], value: Any
) -> None:
    payload = _payload()
    _replace(payload, path, value)

    with pytest.raises(RowFeatureContractError):
        load_row_feature_proxy_contract(_write_payload(tmp_path, payload))


@pytest.mark.parametrize(
    "path",
    [
        ("baseline", "dropout"),
        ("baseline", "learning_rate"),
        ("baseline", "weight_decay"),
        ("proxy_gate", "mean_delta_max"),
        ("proxy_gate", "worst_seed_delta_max"),
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_contract_rejects_nonfinite_numbers(
    tmp_path: Path, path: tuple[str, ...], value: float
) -> None:
    payload = _payload()
    _replace(payload, path, value)

    with pytest.raises(RowFeatureContractError, match="non-finite"):
        load_row_feature_proxy_contract(_write_payload(tmp_path, payload))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), 2),
        (("stage",), "A"),
        (("review_only",), False),
        (("official_train_sha256",), "0" * 64),
        (("official_train_sha256",), "D" * 64),
        (("baseline", "capacity"), "p3_lite"),
        (("baseline", "k"), 64),
        (("baseline", "width"), 768),
        (("baseline", "blocks"), 6),
        (("baseline", "dropout"), 0.15),
        (("baseline", "num_embedding"), "periodic"),
        (("baseline", "loss"), "brier"),
        (("baseline", "scheduler"), "one_cycle"),
        (("baseline", "learning_rate"), 0.0009),
        (("baseline", "weight_decay"), 0.0002),
        (("baseline", "effective_batch_size"), 2048),
        (("baseline", "micro_batch_size"), 256),
        (("fold", "train_end_year"), 2022),
        (("fold", "valid_year"), 2025),
        (("sample", "mode"), "full"),
        (("sample", "max_rows"), 399_999),
        (("training", "max_epochs"), 9),
        (("training", "min_epochs"), 2),
        (("training", "patience"), 4),
        (("budget", "wall_seconds"), 10_801),
        (("budget", "new_job_guard_seconds"), 899),
        (("proxy_gate", "mean_delta_max"), -0.00002),
        (("proxy_gate", "worst_seed_delta_max"), 0.00006),
    ],
)
def test_contract_rejects_every_changed_sealed_scalar(
    tmp_path: Path, path: tuple[str, ...], value: Any
) -> None:
    payload = _payload()
    _replace(payload, path, value)

    with pytest.raises(RowFeatureContractError):
        load_row_feature_proxy_contract(_write_payload(tmp_path, payload))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("seeds",), [3407, 42]),
        (("seeds",), [42, 42]),
        (("feature_bundles",), list(reversed(ROW_FEATURE_BUNDLES))),
        (("feature_bundles",), [*ROW_FEATURE_BUNDLES[:-1], ROW_FEATURE_BUNDLES[0]]),
        (("feature_bundles",), [*ROW_FEATURE_BUNDLES[:-1], "unknown_bundle"]),
    ],
)
def test_contract_rejects_changed_ordered_sequences(
    tmp_path: Path, path: tuple[str, ...], value: Any
) -> None:
    payload = _payload()
    _replace(payload, path, value)

    with pytest.raises(RowFeatureContractError):
        load_row_feature_proxy_contract(_write_payload(tmp_path, payload))


@pytest.mark.parametrize(
    ("mutations", "message"),
    [
        ({("fold", "train_end_year"): 2024}, "yearly transition"),
        (
            {
                ("training", "max_epochs"): 2,
                ("training", "min_epochs"): 3,
            },
            "min_epochs",
        ),
        (
            {
                ("budget", "wall_seconds"): 900,
                ("budget", "new_job_guard_seconds"): 900,
            },
            "new_job_guard_seconds",
        ),
        (
            {
                ("baseline", "effective_batch_size"): 4097,
                ("baseline", "micro_batch_size"): 512,
            },
            "divisible",
        ),
        ({("sample", "max_rows"): 0}, "positive"),
        ({("baseline", "dropout"): 1.0}, "dropout"),
    ],
)
def test_contract_rejects_invalid_ranges_and_relationships(
    tmp_path: Path,
    mutations: dict[tuple[str, ...], Any],
    message: str,
) -> None:
    payload = _payload()
    for path, value in mutations.items():
        _replace(payload, path, value)

    with pytest.raises(RowFeatureContractError, match=message):
        load_row_feature_proxy_contract(_write_payload(tmp_path, payload))


def test_duplicate_keys_are_rejected_at_top_and_nested_levels(tmp_path: Path) -> None:
    raw = CONFIG.read_text(encoding="utf-8")
    top = tmp_path / "top.json"
    top.write_text(raw.replace('"schema_version": 1,', '"schema_version": 1,\n  "schema_version": 1,'), encoding="utf-8")
    nested = tmp_path / "nested.json"
    nested.write_text(raw.replace('"capacity": "p2",', '"capacity": "p2", "capacity": "p2",'), encoding="utf-8")

    with pytest.raises(RowFeatureContractError, match="duplicate JSON key"):
        load_row_feature_proxy_contract(top)
    with pytest.raises(RowFeatureContractError, match="duplicate JSON key"):
        load_row_feature_proxy_contract(nested)


def test_contract_rejects_missing_nonregular_and_symlink_paths(tmp_path: Path) -> None:
    with pytest.raises(RowFeatureContractError, match="missing"):
        load_row_feature_proxy_contract(tmp_path / "missing.json")
    with pytest.raises(RowFeatureContractError, match="regular"):
        load_row_feature_proxy_contract(tmp_path)

    target = tmp_path / "target.json"
    target.write_bytes(CONFIG.read_bytes())
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(RowFeatureContractError, match="symlink"):
        load_row_feature_proxy_contract(link)


def test_contract_rejects_symlinked_ancestor_directory(tmp_path: Path) -> None:
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    contract = real_directory / "contract.json"
    contract.write_bytes(CONFIG.read_bytes())
    linked_directory = tmp_path / "linked"
    linked_directory.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(RowFeatureContractError, match="symlink"):
        load_row_feature_proxy_contract(linked_directory / "contract.json")


def test_contract_rejects_leaf_swapped_to_symlink_before_secure_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "contract.json"
    target.write_bytes(CONFIG.read_bytes())
    original = tmp_path / "original.json"
    real_open = os.open
    swapped = False

    def swap_then_open(path: str, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal swapped
        if not swapped and path == target.name and kwargs.get("dir_fd") is not None:
            target.rename(original)
            target.symlink_to(original)
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap_then_open)

    with pytest.raises(RowFeatureContractError, match="symlink"):
        load_row_feature_proxy_contract(target)
    assert swapped is True


def test_contract_secure_reader_accepts_regular_relative_and_absolute_paths(
    tmp_path: Path,
) -> None:
    contract = tmp_path / "contract.json"
    contract.write_bytes(CONFIG.read_bytes())
    relative = Path(os.path.relpath(contract, Path.cwd()))

    assert load_row_feature_proxy_contract(relative).stage == "P"
    assert load_row_feature_proxy_contract(contract.absolute()).stage == "P"


def test_contract_secure_reader_closes_every_opened_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = tmp_path / "contract.json"
    contract.write_bytes(CONFIG.read_bytes())
    real_open = os.open
    real_close = os.close
    opened: set[int] = set()
    observed_open_count = 0

    def tracking_open(path: str, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal observed_open_count
        descriptor = real_open(path, flags, *args, **kwargs)
        opened.add(descriptor)
        observed_open_count += 1
        return descriptor

    def tracking_close(descriptor: int) -> None:
        try:
            real_close(descriptor)
        finally:
            opened.discard(descriptor)

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(os, "close", tracking_close)

    assert load_row_feature_proxy_contract(contract).stage == "P"
    assert observed_open_count >= 2
    assert opened == set()


@pytest.mark.parametrize(
    "capability", ["O_NOFOLLOW", "O_DIRECTORY", "O_NONBLOCK"]
)
def test_contract_rejects_unavailable_secure_traversal_capability(
    monkeypatch: pytest.MonkeyPatch, capability: str
) -> None:
    monkeypatch.delattr(os, capability)

    with pytest.raises(RowFeatureContractError, match="secure path traversal"):
        load_row_feature_proxy_contract(CONFIG)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is unavailable")
def test_contract_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "contract.json"
    os.mkfifo(fifo)
    script = """
import sys
from experiments.tabm_campaign.row_feature_contracts import (
    RowFeatureContractError,
    load_row_feature_proxy_contract,
)

try:
    load_row_feature_proxy_contract(sys.argv[1])
except RowFeatureContractError:
    raise SystemExit(0)
raise SystemExit(2)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, os.fspath(fifo)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        pytest.fail("contract loader blocked while opening a FIFO")

    assert process.returncode == 0, (stdout, stderr)


def test_contract_rejects_malformed_json_and_utf8(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"schema_version":', encoding="utf-8")
    invalid_utf8 = tmp_path / "invalid-utf8.json"
    invalid_utf8.write_bytes(b"{\xff}")

    with pytest.raises(RowFeatureContractError, match="JSON"):
        load_row_feature_proxy_contract(malformed)
    with pytest.raises(RowFeatureContractError, match="UTF-8"):
        load_row_feature_proxy_contract(invalid_utf8)


def test_hash_function_validates_before_returning_digest(tmp_path: Path) -> None:
    payload = _payload()
    payload["stage"] = "A"
    invalid = _write_payload(tmp_path, payload)

    with pytest.raises(RowFeatureContractError):
        row_feature_contract_sha256(invalid)
