from pathlib import Path
import os
import subprocess
import sys
import tarfile
import io

from experiments.tree_expert.kaggle import (
    _runtime_archive,
    classify_e1_input,
    runtime_member_names,
)


def test_runtime_inventory_contains_only_declared_tree_expert_and_reused_modules() -> None:
    names = runtime_member_names()
    assert "experiments/tree_expert/runner.py" in names
    assert "experiments/temporal_portfolio/seasonal_features.py" in names
    assert all("submission" not in name for name in names)


def test_kaggle_input_discovery_accepts_expanded_dataset(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"artifact_kind":"tree_expert_e1_input_v1"}', encoding="utf-8"
    )
    (tmp_path / "baseline_predictions.csv").write_text("row_id\nr0\n", encoding="utf-8")

    assert classify_e1_input(tmp_path) == "tree_expert_e1_input_v1"


def test_embedded_runtime_imports_without_repository_on_pythonpath(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    extracted = tmp_path / "runtime"
    extracted.mkdir()
    with tarfile.open(fileobj=io.BytesIO(_runtime_archive(repository)), mode="r:gz") as archive:
        archive.extractall(extracted, filter="data")

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(extracted)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiments.tree_expert import runner; print(runner.__file__)",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(extracted) in completed.stdout
