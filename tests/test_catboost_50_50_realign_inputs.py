from __future__ import annotations

from decimal import Decimal
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from experiments.catboost_50_50_realign.contracts import load_contract
from experiments.catboost_50_50_realign.inputs import (
    RealignInputError,
    RealignSourcePaths,
    TrustedSource,
    VerifiedSources,
    prepare_input_archive,
    verify_and_extract_input,
    verify_source_artifacts,
)


def _source_paths(tmp_path: Path) -> RealignSourcePaths:
    paths = [tmp_path / f"source_{index}.zip" for index in range(5)]
    for path in paths:
        path.write_bytes(path.name.encode("ascii"))
    return RealignSourcePaths(*paths)


def _verified_sources(tmp_path: Path) -> VerifiedSources:
    contents = {
        "data/train.csv": b"row_id,season,control_success\na,2021,1\n",
        "data/trackman_history.csv": b"pitcher_id\np1\n",
        "tabm/2022_2023.csv": b"row_id,target,probability\na,1,0.6\n",
        "tabm/2023_2024.csv": b"row_id,target,probability\nb,0,0.4\n",
        "catboost/2022_2023/model.cbm": b"model-old",
        "catboost/2022_2023/preprocessing_state.json": b"{}",
        "catboost/2022_2023/predictions.csv": b"row_id,target,p_4,p_32\na,1,0.5,0.6\n",
        "catboost/2023_2024/model.cbm": b"model-new",
        "catboost/2023_2024/preprocessing_state.json": b"{}",
        "catboost/2023_2024/predictions.csv": b"row_id,target,p_4,p_32\nb,0,0.5,0.4\n",
        "audit/next_experiment.json": b'{"supported_directions":["DIVERSE_BLEND"]}',
        "audit/artifact_inventory.json": b"[]",
    }
    members: dict[str, TrustedSource] = {}
    from hashlib import sha256

    for name, payload in contents.items():
        path = tmp_path / "members" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        members[name] = TrustedSource(path, sha256(payload).hexdigest(), len(payload))
    return VerifiedSources(
        members=members,
        source_sha256=dict(load_contract().source_sha256),
        fold_keys=("2021->2022", "2022->2023", "2023->2024"),
        audit_weight=Decimal("0.50"),
    )


def test_prepare_and_verify_single_handoff_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.catboost_50_50_realign.inputs as module

    verified_sources = _verified_sources(tmp_path)
    monkeypatch.setattr(
        module,
        "verify_source_artifacts",
        lambda sources, contract: verified_sources,
    )
    sources = _source_paths(tmp_path)
    first = prepare_input_archive(sources, tmp_path / "first.zip", load_contract())
    second = prepare_input_archive(sources, tmp_path / "second.zip", load_contract())

    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == second.path.read_bytes()
    restored = verify_and_extract_input(
        first.path, tmp_path / "restored", load_contract()
    )
    assert restored.fold_keys == ("2021->2022", "2022->2023", "2023->2024")
    assert restored.audit_weight == Decimal("0.50")
    assert restored.data_dir.joinpath("train.csv").is_file()
    assert set(restored.tabm_predictions) == {"2022->2023", "2023->2024"}
    assert set(restored.catboost_models) == {"2022->2023", "2023->2024"}


@pytest.mark.parametrize("unsafe_name", ("../escape", "/absolute", "a\\b"))
def test_verify_rejects_unsafe_members(tmp_path: Path, unsafe_name: str) -> None:
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as output:
        output.writestr(unsafe_name, b"bad")

    with pytest.raises(RealignInputError, match="unsafe ZIP member"):
        verify_and_extract_input(archive, tmp_path / "out", load_contract())


def test_verify_rejects_duplicate_members(tmp_path: Path) -> None:
    archive = tmp_path / "duplicate.zip"
    info = ZipInfo("manifest.json")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with ZipFile(archive, "w") as output:
            output.writestr(info, b"{}")
            output.writestr(info, b"{}")

    with pytest.raises(RealignInputError, match="duplicate ZIP member"):
        verify_and_extract_input(archive, tmp_path / "out", load_contract())


def test_source_verification_rejects_missing_regular_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.zip"
    sources = RealignSourcePaths(missing, missing, missing, missing, missing)

    with pytest.raises(RealignInputError, match="regular file"):
        verify_source_artifacts(sources, load_contract())


def test_source_verification_combines_all_verified_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.catboost_50_50_realign.inputs as module

    sources = _source_paths(tmp_path)
    expected = _verified_sources(tmp_path / "expected")
    by_group = {
        "training": {
            key: value for key, value in expected.members.items() if key.startswith("data/")
        },
        "stage_c": {
            key: value for key, value in expected.members.items() if key.startswith("tabm/")
        },
        "deployment": {
            key: value
            for key, value in expected.members.items()
            if key.startswith("catboost/")
        },
        "audit": {
            key: value for key, value in expected.members.items() if key.startswith("audit/")
        },
    }
    source_by_path = {
        sources.training_input: load_contract().source_sha256["training_input"],
        sources.stage_c_delivery: load_contract().source_sha256["stage_c_delivery"],
        sources.deployment_resume: load_contract().source_sha256["deployment_resume"],
        sources.deployment_review: load_contract().source_sha256["deployment_review"],
        sources.oof_audit: load_contract().source_sha256["oof_audit"],
    }
    monkeypatch.setattr(module, "file_sha256", lambda path: source_by_path[Path(path)])
    monkeypatch.setattr(module, "_verify_training_source", lambda path, root: by_group["training"])
    monkeypatch.setattr(module, "_verify_stage_c_source", lambda path, root: by_group["stage_c"])
    monkeypatch.setattr(
        module,
        "_verify_deployment_sources",
        lambda resume, review, root: by_group["deployment"],
    )
    monkeypatch.setattr(
        module,
        "_verify_audit_source",
        lambda path, root: (by_group["audit"], Decimal("0.50")),
    )

    verified = verify_source_artifacts(sources, load_contract())

    assert set(verified.members) == set(expected.members)
    assert verified.source_sha256 == load_contract().source_sha256
    assert verified.audit_weight == Decimal("0.50")


def _audit_zip(path: Path, *, weight: float = 0.5) -> None:
    decision = {
        "schema_version": 1,
        "supported_directions": ["DIVERSE_BLEND"],
        "evidence": {
            "diverse_blend": {
                "supported": True,
                "stable_candidates": ["catboost_hand_matchup_seed_42"],
                "stable_blends": [
                    {
                        "candidate_model_id": "catboost_hand_matchup_seed_42",
                        "candidate_weight": weight,
                        "eligible_segment_count": 26,
                        "fold_gains": {
                            "2022->2023": 0.00017567321480116852,
                            "2023->2024": 0.0001844426727390278,
                        },
                        "latest_interval_lower": 9.215809950985597e-05,
                        "maximum_eligible_segment_regression": 0.0004936941230183067,
                    }
                ],
            }
        },
    }
    inventory = [
        {
            "artifact_kind": "tabm_colab_stage_C_delivery",
            "role": "stage_c_tabm",
            "sha256": load_contract().source_sha256["stage_c_delivery"],
            "status": "verified",
        },
        {
            "artifact_kind": "catboost_deployment_review",
            "role": "catboost_deployment",
            "sha256": load_contract().source_sha256["deployment_review"],
            "status": "verified",
        },
    ]
    names = {
        "artifact_inventory.json": json.dumps(inventory).encode(),
        "audit_summary.md": b"summary",
        "calibration_deciles.csv": b"x\n",
        "correlation_matrix.csv": b"x\n",
        "model_comparison.csv": b"x\n",
        "next_experiment.json": json.dumps(decision).encode(),
        "paired_comparison.csv": b"x\n",
        "segment_diagnostics.csv": b"x\n",
    }
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in names.items():
            archive.writestr(name, payload)


def test_audit_source_requires_the_sealed_stable_half_weight(tmp_path: Path) -> None:
    from experiments.catboost_50_50_realign.inputs import _verify_audit_source

    valid = tmp_path / "valid.zip"
    _audit_zip(valid)
    members, weight = _verify_audit_source(valid, tmp_path / "valid_out")
    assert set(members) == {
        "audit/next_experiment.json",
        "audit/artifact_inventory.json",
    }
    assert weight == Decimal("0.50")

    changed = tmp_path / "changed.zip"
    _audit_zip(changed, weight=0.25)
    with pytest.raises(RealignInputError, match="stable 0.50 blend"):
        _verify_audit_source(changed, tmp_path / "changed_out")


def test_training_source_delegates_to_the_sealed_blend_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.catboost_50_50_realign.inputs as module

    source = tmp_path / "training.zip"
    source.write_bytes(b"source")

    def fake_verify(path: Path, destination: Path, contract: object) -> object:
        assert path == source
        destination.mkdir()
        (destination / "train.csv").write_bytes(b"train")
        (destination / "trackman_history.csv").write_bytes(b"history")
        return SimpleNamespace(data_dir=destination)

    monkeypatch.setattr(module, "verify_and_extract_training_input", fake_verify, raising=False)
    monkeypatch.setattr(module, "load_blend_contract", lambda: object(), raising=False)

    members = module._verify_training_source(source, tmp_path / "out")

    assert set(members) == {"data/train.csv", "data/trackman_history.csv"}
    assert members["data/train.csv"].size == 5


def test_stage_c_source_selects_only_the_two_seed_3407_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.catboost_50_50_realign.inputs as module

    source = tmp_path / "stage_c.zip"
    source.write_bytes(b"source")

    def fake_verify(path: Path, destination: Path, contract: object) -> object:
        assert path == source
        destination.mkdir()
        older = destination / "older.csv"
        latest = destination / "latest.csv"
        older.write_bytes(b"older")
        latest.write_bytes(b"latest")
        return SimpleNamespace(
            prediction_paths={"2022->2023": older, "2023->2024": latest}
        )

    monkeypatch.setattr(module, "verify_and_extract_stage_c", fake_verify, raising=False)
    monkeypatch.setattr(module, "load_blend_contract", lambda: object(), raising=False)

    members = module._verify_stage_c_source(source, tmp_path / "out")

    assert set(members) == {"tabm/2022_2023.csv", "tabm/2023_2024.csv"}
    assert members["tabm/2023_2024.csv"].size == 6


def _deployment_zip(
    path: Path, *, kind: str, bindings: dict[str, str], members: dict[str, bytes]
) -> None:
    manifest = {
        "artifact_kind": kind,
        "bindings": bindings,
        "members": {
            name: {"size": len(payload), "sha256": sha256(payload).hexdigest()}
            for name, payload in members.items()
        },
    }
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, payload in members.items():
            archive.writestr(name, payload)


def test_deployment_sources_bind_review_predictions_to_resume_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.catboost_50_50_realign.inputs as module

    bindings = {f"binding_{index}": f"{index:x}" * 64 for index in range(1, 12)}
    resume = tmp_path / "resume.zip"
    review = tmp_path / "review.zip"
    _deployment_zip(
        resume,
        kind="catboost_deployment_resume",
        bindings=bindings,
        members={
            "jobs/align_2022_2023/model.cbm": b"m1",
            "jobs/align_2022_2023/preprocessing_state.json": b"s1",
            "jobs/align_2023_2024/model.cbm": b"m2",
            "jobs/align_2023_2024/preprocessing_state.json": b"s2",
        },
    )
    _deployment_zip(
        review,
        kind="catboost_deployment_review",
        bindings=bindings,
        members={
            "predictions/align_2022_2023.csv": b"p1",
            "predictions/align_2023_2024.csv": b"p2",
        },
    )
    monkeypatch.setattr(module, "verify_deployment_resume", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(module, "verify_deployment_review", lambda *args, **kwargs: None, raising=False)

    members = module._verify_deployment_sources(resume, review, tmp_path / "out")

    assert set(members) == {
        "catboost/2022_2023/model.cbm",
        "catboost/2022_2023/preprocessing_state.json",
        "catboost/2022_2023/predictions.csv",
        "catboost/2023_2024/model.cbm",
        "catboost/2023_2024/preprocessing_state.json",
        "catboost/2023_2024/predictions.csv",
    }

    changed = tmp_path / "changed_review.zip"
    _deployment_zip(
        changed,
        kind="catboost_deployment_review",
        bindings={**bindings, "binding_1": "f" * 64},
        members={
            "predictions/align_2022_2023.csv": b"p1",
            "predictions/align_2023_2024.csv": b"p2",
        },
    )
    with pytest.raises(RealignInputError, match="bindings differ"):
        module._verify_deployment_sources(resume, changed, tmp_path / "changed_out")


def test_prepare_cli_imports_the_repository_when_run_outside_it(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "tools/prepare_catboost_50_50_realign_input.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--training-input" in completed.stdout
    assert "--oof-audit" in completed.stdout


def test_failed_extraction_does_not_publish_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.catboost_50_50_realign.inputs as module

    verified_sources = _verified_sources(tmp_path)
    monkeypatch.setattr(
        module,
        "verify_source_artifacts",
        lambda sources, contract: verified_sources,
    )
    valid = prepare_input_archive(
        _source_paths(tmp_path), tmp_path / "valid.zip", load_contract()
    ).path
    tampered = tmp_path / "tampered.zip"
    with ZipFile(valid) as source, ZipFile(tampered, "w", compression=ZIP_DEFLATED) as output:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "data/train.csv":
                payload += b"tampered"
            output.writestr(info, payload)

    destination = tmp_path / "published"
    with pytest.raises(RealignInputError, match="member content differs"):
        verify_and_extract_input(tampered, destination, load_contract())
    assert not destination.exists()


def test_prepare_rejects_member_with_symlink_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.catboost_50_50_realign.inputs as module

    verified = _verified_sources(tmp_path / "real")
    real_member_root = tmp_path / "real/members"
    alias = tmp_path / "alias"
    alias.symlink_to(real_member_root, target_is_directory=True)
    redirected = {
        name: replace(member, path=alias / Path(name))
        for name, member in verified.members.items()
    }
    monkeypatch.setattr(
        module,
        "verify_source_artifacts",
        lambda sources, contract: replace(verified, members=redirected),
    )

    with pytest.raises(RealignInputError, match="symlink ancestor"):
        prepare_input_archive(
            _source_paths(tmp_path), tmp_path / "output.zip", load_contract()
        )


def test_prepare_failure_removes_unpublished_temporary_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.catboost_50_50_realign.inputs as module

    verified = _verified_sources(tmp_path)
    members = dict(verified.members)
    first_name = sorted(members)[0]
    members[first_name] = replace(members[first_name], sha256="0" * 64)
    monkeypatch.setattr(
        module,
        "verify_source_artifacts",
        lambda sources, contract: replace(verified, members=members),
    )
    output = tmp_path / "output.zip"

    with pytest.raises(RealignInputError, match="changed while reading"):
        prepare_input_archive(_source_paths(tmp_path), output, load_contract())

    assert not output.exists()
    assert not output.with_suffix(".zip.tmp").exists()


def test_prepare_does_not_replace_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.catboost_50_50_realign.inputs as module

    verified = _verified_sources(tmp_path)
    monkeypatch.setattr(
        module,
        "verify_source_artifacts",
        lambda sources, contract: verified,
    )
    output = tmp_path / "output.zip"
    output.write_bytes(b"keep-me")

    with pytest.raises(RealignInputError, match="already exists"):
        prepare_input_archive(_source_paths(tmp_path), output, load_contract())

    assert output.read_bytes() == b"keep-me"


def test_source_artifact_with_symlink_ancestor_is_rejected(
    tmp_path: Path,
) -> None:
    from experiments.catboost_50_50_realign.inputs import verify_source_artifacts

    real = tmp_path / "real"
    real.mkdir()
    source = real / "source.zip"
    source.write_bytes(b"placeholder")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    paths = RealignSourcePaths(
        training_input=alias / "source.zip",
        stage_c_delivery=source,
        deployment_resume=source,
        deployment_review=source,
        oof_audit=source,
    )

    with pytest.raises(RealignInputError, match="symlink ancestor"):
        verify_source_artifacts(paths, load_contract())
