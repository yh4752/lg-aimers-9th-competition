from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from competition_rules.code_gate import (
    RulesCodeGateError,
    assert_row_independent,
    canonical_probability,
    inspect_inference_source,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"row_id": ["a", "b", "c", "d"], "x": [0.1, np.nan, 0.7, 0.3]}
    )


def test_mean_shift_is_rejected() -> None:
    def predict(value: pd.DataFrame) -> np.ndarray:
        raw = value["x"].fillna(0.0).to_numpy(dtype="float64")
        return raw - raw.mean() + 0.5

    with pytest.raises(RulesCodeGateError, match="row independence"):
        assert_row_independent(_frame(), load_predictor=lambda: predict)


def test_row_local_vectorized_transform_passes() -> None:
    def predict(value: pd.DataFrame) -> np.ndarray:
        raw = value["x"].fillna(-1.0).to_numpy(dtype="float64")
        return 1.0 / (1.0 + np.exp(-raw))

    report = assert_row_independent(_frame(), load_predictor=lambda: predict)

    assert report["status"] == "passed"
    assert report["decimal_places"] == 8
    assert report["row_count"] == 4


def test_deterministic_row_seed_passes() -> None:
    def predict(value: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [
                (int.from_bytes(str(row_id).encode(), "little") % 1000) / 1000
                for row_id in value["row_id"]
            ]
        )

    assert assert_row_independent(
        _frame(), load_predictor=lambda: predict
    )["status"] == "passed"


def test_cross_row_rank_and_mutable_state_are_rejected() -> None:
    def rank_predict(value: pd.DataFrame) -> np.ndarray:
        return value["x"].fillna(0.0).rank(pct=True).to_numpy()

    with pytest.raises(RulesCodeGateError, match="row independence"):
        assert_row_independent(_frame(), load_predictor=lambda: rank_predict)

    class Stateful:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, value: pd.DataFrame) -> np.ndarray:
            self.calls += 1
            return np.full(len(value), 0.5)

    with pytest.raises(RulesCodeGateError, match="state changed"):
        assert_row_independent(
            _frame(),
            load_predictor=Stateful,
            state_digest=lambda predictor: str(predictor.calls),
        )


@pytest.mark.parametrize("value", [True, float("nan"), -0.1, 1.1, "0.5"])
def test_canonical_probability_rejects_invalid_values(value: object) -> None:
    with pytest.raises(RulesCodeGateError):
        canonical_probability(value)  # type: ignore[arg-type]


def test_canonical_probability_uses_fixed_precision() -> None:
    assert canonical_probability(0.5) == "0.50000000"


@pytest.mark.parametrize(
    "source",
    [
        "import requests\ndef predict(frame): return requests.get('https://x')\n",
        "import socket\ndef predict(frame): return socket.socket()\n",
        "import pandas as pd\ndef predict(frame): return pd.read_csv('test.csv')\n",
        "def predict(frame): return frame.groupby('team').size()\n",
        "PREDICTION_LOOKUP = {'known-row': 0.9}\ndef predict(frame): return [PREDICTION_LOOKUP[x] for x in frame.row_id]\n",
        "def predict(frame):\n    if len(frame) == 245789: return [0.5] * len(frame)\n    return [0.4] * len(frame)\n",
    ],
)
def test_source_gate_rejects_prohibited_inference_code(
    tmp_path: Path, source: str
) -> None:
    path = tmp_path / "predictor.py"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(RulesCodeGateError):
        inspect_inference_source([path], project_root=tmp_path)


def test_source_gate_accepts_row_local_vectorized_code(tmp_path: Path) -> None:
    path = tmp_path / "predictor.py"
    path.write_text(
        "import numpy as np\n"
        "def predict(frame):\n"
        "    x = frame['x'].fillna(0.0).to_numpy()\n"
        "    return 1.0 / (1.0 + np.exp(-x))\n",
        encoding="utf-8",
    )

    report = inspect_inference_source([path], project_root=tmp_path)

    assert report["status"] == "passed"
    assert report["file_count"] == 1
    assert len(report["source_sha256"]) == 64


def test_source_gate_does_not_treat_training_mapping_as_inference(tmp_path: Path) -> None:
    path = tmp_path / "training_helpers.py"
    path.write_text(
        "def _infer_team_mapping(training_rows):\n"
        "    return training_rows.groupby('team').size()\n",
        encoding="utf-8",
    )

    assert inspect_inference_source([path], project_root=tmp_path)["status"] == "passed"


def test_source_gate_rejects_symlinked_source(tmp_path: Path) -> None:
    target = tmp_path / "target.py"
    target.write_text("def predict(frame): return frame['x']\n", encoding="utf-8")
    link = tmp_path / "predict.py"
    link.symlink_to(target)

    with pytest.raises(RulesCodeGateError, match="symlink"):
        inspect_inference_source([link], project_root=tmp_path)
