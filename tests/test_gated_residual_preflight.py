from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from experiments.gated_residual_final.inputs import prepare_final_input
from tools.preflight_gated_residual_final import run_preflight


def test_preflight_replays_zip_expanded_and_isolated_import(tmp_path: Path) -> None:
    sources = []
    for name in ("e2.zip", "a.zip", "b.zip"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        sources.append(path)
    hashes = {
        key: sha256(path.read_bytes()).hexdigest()
        for key, path in zip(("e2_input", "stage_a", "stage_b"), sources)
    }
    prepared = prepare_final_input(
        e2_input=sources[0], stage_a=sources[1], stage_b=sources[2],
        output=tmp_path / "input.zip", expected_hashes=hashes,
    )

    report = run_preflight(
        input_path=prepared,
        cell_path=Path("experiments/gated_residual_final/KAGGLE_CELL.py"),
        work_root=tmp_path / "preflight",
        enforce_contract_hashes=False,
    )

    assert report == {
        "layouts": ("zip", "expanded"),
        "imports": "isolated",
        "submission_created": False,
    }
