from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from submission.tabm_candidate import CANDIDATE_ID, ImportedTabMCandidate
from tools.prepare_tabm_submission_validation_handoff import prepare_handoff


ROOT = Path(__file__).resolve().parents[1]


def _fake_import(delivery: Path, destination: Path) -> ImportedTabMCandidate:
    del delivery
    model = destination / "model"
    model.mkdir(parents=True)
    values = {
        "inference_manifest.json": b"{}\n",
        "numeric_embedding_0.json": b"{}\n",
        "preprocessing_state.json": b"{}\n",
        "tabm_member_0_seed_3407.pt": b"weights",
    }
    member_hashes = {}
    digest = sha256()
    for name, value in sorted(values.items()):
        (model / name).write_bytes(value)
        member_hashes[name] = sha256(value).hexdigest()
        digest.update(name.encode() + b"\0" + sha256(value).digest())
    (destination / "candidate_manifest.json").write_text("{}\n", encoding="utf-8")
    return ImportedTabMCandidate(
        candidate_id=CANDIDATE_ID,
        root=destination,
        model_dir=model,
        delivery_sha256="1" * 64,
        review_bundle_sha256="2" * 64,
        model_sha256=digest.hexdigest(),
        member_sha256=member_hashes,
    )


def test_handoff_contains_validation_runtime_not_submission(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "tools.prepare_tabm_submission_validation_handoff.import_review_delivery",
        _fake_import,
    )
    delivery = tmp_path / "delivery.zip"
    delivery.write_bytes(b"fixture")

    result = prepare_handoff(delivery, tmp_path / "handoff")

    manifest = json.loads((result.root / "handoff_manifest.json").read_text())
    assert manifest["artifact_kind"] == "tabm_submission_validation_handoff_v1"
    assert manifest["candidate_id"] == CANDIDATE_ID
    assert (result.root / "candidate/model/inference_manifest.json").is_file()
    assert (result.root / "runtime/submission/tabm_validation.py").is_file()
    assert (result.root / "runtime/competition_rules/policy.json").is_file()
    assert (result.root / "requirements.txt").read_text() == (
        "tabm==0.0.3\nrtdl-num-embeddings==0.0.12\n"
    )
    assert not list(result.root.rglob("submit.zip"))
    assert not list(result.root.rglob("submission.csv"))


def test_prepare_tool_imports_from_outside_repository(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools/prepare_tabm_submission_validation_handoff.py"), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--delivery" in completed.stdout


def test_kaggle_cell_is_one_offline_repository_independent_program() -> None:
    path = ROOT / "experiments/tabm_campaign/KAGGLE_SUBMISSION_VALIDATION_CELL.py"
    source = path.read_text(encoding="utf-8")

    compile(source, str(path), "exec")
    assert "github.com" not in source
    assert "drive.mount" not in source
    assert "VALIDATION_HANDOFF_FOUND" in source
    assert "EXACT_ENV_READY" in source
    assert "VALIDATION_SUCCESS" in source
    assert "VALIDATION_ERROR" in source
    assert "submit.zip" not in source
