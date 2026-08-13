from __future__ import annotations

import pandas as pd
from pathlib import Path
import sys

from experiments.tabm_campaign.dependency_probe import run_python_probe
from experiments.tabm_campaign.final_review import fixed_epoch_count, synthetic_scale_frame


def test_fixed_epoch_rule_is_median_plus_one_clipped() -> None:
    assert fixed_epoch_count([0, 4, 9]) == 5
    assert fixed_epoch_count([100, 200]) == 40
    assert fixed_epoch_count([0]) == 2


def test_scale_frame_reuses_only_features_and_assigns_unique_ids() -> None:
    sample = pd.DataFrame({"row_id": ["a", "b"], "x": [1.0, 2.0]})
    scale = synthetic_scale_frame(sample, row_count=7)
    assert scale["row_id"].tolist() == [f"synthetic_{i:06d}" for i in range(7)]
    assert scale["row_id"].is_unique
    assert scale["x"].tolist() == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0]


def test_clean_python_probe_records_output_hash(tmp_path: Path) -> None:
    result = run_python_probe(
        Path(sys.executable),
        "print('five-row-probe-ok')",
        timeout_seconds=10,
    )
    assert result["status"] == "passed"
    assert result["return_code"] == 0
    assert len(result["stdout_sha256"]) == 64
