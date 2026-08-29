from hashlib import sha256
import io
import json
from pathlib import Path
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from experiments.tree_expert.s4_inputs import (
    S4InputError,
    discover_s4_input_candidates,
    prepare_s4_input,
    verify_and_extract_s4_input,
)


def _info(name: str) -> ZipInfo:
    info = ZipInfo(name, (2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _zip(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(_info(name), payload)
    return stream.getvalue()


def _manifest(kind: str, members: dict[str, bytes], **extra: object) -> bytes:
    return json.dumps({
        "schema_version": 1,
        "artifact_kind": kind,
        **extra,
        "members": {
            name: {"size": len(payload), "sha256": sha256(payload).hexdigest()}
            for name, payload in sorted(members.items())
        },
    }, sort_keys=True, separators=(",", ":")).encode()


def _e2_handoff(path: Path) -> Path:
    delivery_members = {"model.cbm": b"model"}
    delivery = _zip({
        **delivery_members,
        "manifest.json": _manifest(
            "tree_expert_e2_model_delivery_v1", delivery_members,
            campaign_id="tree_expert_e2_v1", review_only=False,
            submission_package=False, candidate_id="c1_anchor_residual",
            predictor="catboost", seeds=[42, 2026, 3407],
            iterations={"42": 1, "2026": 1, "3407": 1}, decision_sha256="1" * 64,
        ),
    })
    members = {
        "tree_expert_e2.log": b"ok\n",
        "tree_expert_e2_review.zip": b"review",
        "tree_expert_e2_resume.zip": b"resume",
        "tree_expert_e2_model_delivery.zip": delivery,
    }
    manifest = _manifest(
        "tree_expert_e2_handoff_v1", members, campaign_id="tree_expert_e2_v1",
        review_only=False, submission_package=False, status="accepted", delivery=True,
    )
    path.write_bytes(_zip({**members, "handoff_manifest.json": manifest}))
    return path


def _expand_nested_zips(root: Path) -> None:
    while True:
        expanded = False
        for archive_path in sorted(root.rglob("*.zip")):
            try:
                with ZipFile(archive_path) as archive:
                    members = {name: archive.read(name) for name in archive.namelist()}
            except BadZipFile:
                continue
            destination = archive_path.with_suffix("")
            destination.mkdir(parents=True)
            for name, payload in members.items():
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            archive_path.unlink()
            expanded = True
            break
        if not expanded:
            return


def test_prepare_and_verify_s4_input_round_trip(tmp_path: Path) -> None:
    handoff = _e2_handoff(tmp_path / "e2.zip")
    expected = sha256(handoff.read_bytes()).hexdigest()
    archive = prepare_s4_input(handoff, tmp_path / "s4.zip", expected_e2_sha256=expected)
    verified = verify_and_extract_s4_input(archive, tmp_path / "verified", expected_e2_sha256=expected)
    assert verified.artifact_kind == "tree_s4_input_v1"
    assert verified.e2_handoff_sha256 == expected
    assert verified.e2_handoff.is_file()


def test_recursively_expanded_kaggle_input_is_reconstructed(tmp_path: Path) -> None:
    handoff = _e2_handoff(tmp_path / "e2.zip")
    expected = sha256(handoff.read_bytes()).hexdigest()
    archive = prepare_s4_input(handoff, tmp_path / "s4.zip", expected_e2_sha256=expected)
    expanded = tmp_path / "kaggle_dataset"
    with ZipFile(archive) as source:
        source.extractall(expanded)
    _expand_nested_zips(expanded)

    verified = verify_and_extract_s4_input(
        expanded, tmp_path / "verified_expanded", expected_e2_sha256=expected
    )

    assert verified.e2_handoff_sha256 == expected


def test_zip_and_expanded_copy_are_deduplicated(tmp_path: Path) -> None:
    handoff = _e2_handoff(tmp_path / "e2.zip")
    expected = sha256(handoff.read_bytes()).hexdigest()
    archive = prepare_s4_input(handoff, tmp_path / "s4.zip", expected_e2_sha256=expected)
    expanded = tmp_path / "expanded"
    with ZipFile(archive) as source:
        source.extractall(expanded)
    assert len(discover_s4_input_candidates((archive, expanded))) == 1


def test_distinct_inputs_are_rejected(tmp_path: Path) -> None:
    first_handoff = _e2_handoff(tmp_path / "e2a.zip")
    first_hash = sha256(first_handoff.read_bytes()).hexdigest()
    first = prepare_s4_input(first_handoff, tmp_path / "first.zip", expected_e2_sha256=first_hash)
    changed = tmp_path / "changed"
    with ZipFile(first) as source:
        source.extractall(changed)
    manifest = json.loads((changed / "manifest.json").read_text())
    manifest["e2_handoff_sha256"] = "0" * 64
    (changed / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    with pytest.raises(S4InputError, match="distinct S4 input identities"):
        discover_s4_input_candidates((first, changed))


def test_tampered_e2_handoff_member_is_rejected(tmp_path: Path) -> None:
    handoff = _e2_handoff(tmp_path / "e2.zip")
    with ZipFile(handoff) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["tree_expert_e2.log"] = b"changed\n"
    handoff.write_bytes(_zip(members))
    expected = sha256(handoff.read_bytes()).hexdigest()
    with pytest.raises(S4InputError, match="E2 handoff member differs"):
        prepare_s4_input(handoff, tmp_path / "s4.zip", expected_e2_sha256=expected)
