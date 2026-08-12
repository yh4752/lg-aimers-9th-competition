from __future__ import annotations

import base64
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import tarfile


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/KAGGLE_INDEPENDENT_DL_CAMPAIGN.ipynb"
FAMILIES = {
    "TabM 후보 1개": "tabm",
    "MLP/ResNet 후보 1개": "mlp_resnet",
    "FT-Transformer 후보 1개": "ft_transformer",
    "TabR 후보 1개": "tabr",
    "TabICLv2 후보 1개": "tabicl_v2",
}


def _load() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _all_text() -> str:
    return "\n".join(_source(cell) for cell in _load()["cells"])


def test_kaggle_notebook_has_ordered_split_flow() -> None:
    notebook = _load()
    cells = notebook["cells"]
    markdown = [_source(cell) for cell in cells if cell["cell_type"] == "markdown"]
    headings = ["공통 준비", "현재 상태 확인", *FAMILIES, "handoff ZIP", "결과 요약"]
    positions = [
        next(index for index, text in enumerate(markdown) if heading in text)
        for heading in headings
    ]
    assert positions == sorted(positions)
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_kaggle_setup_is_github_free_and_finds_official_dataset() -> None:
    text = _all_text()
    for required in (
        "/kaggle/input",
        "/kaggle/working",
        "train.csv",
        "test.csv",
        "sample_submission.csv",
        "trackman_history.csv",
        "len(data_directories) != 1",
        "independent_dl_campaign_v1",
        "len(checkpoint_sources) > 1",
        "dirs_exist_ok=False",
        "Internet",
        "environment.json",
        "torch.cuda.device_count()",
        "vram_gib",
    ):
        assert required in text
    for forbidden in (
        "github.com",
        "GITHUB_TOKEN",
        "kaggle_secrets",
        "git clone",
        "google.colab",
        "/content/drive",
        "submission.zip",
        "submit.zip",
    ):
        assert forbidden not in text


def test_each_family_cell_runs_one_candidate_with_complete_guidance() -> None:
    cells = _load()["cells"]
    text = _all_text()
    assert '"--family"' in text
    assert '"--max-candidates"' in text
    assert '"1"' in text
    assert 'entry.get("state") != "completed"' in text
    assert "failure_reason" in text
    for heading, family in FAMILIES.items():
        index = next(
            index
            for index, cell in enumerate(cells)
            if cell["cell_type"] == "markdown" and heading in _source(cell)
        )
        explanation = _source(cells[index])
        code = _source(cells[index + 1])
        assert f'run_one_family("{family}")' in code
        for phrase in (
            "후보 하나",
            "예상 시간",
            "checkpoint",
            "재실행",
            "정상 완료",
            "traceback",
            "제출",
        ):
            assert phrase in explanation
    assert "P3" in text and "P4" in text
    assert "boundary expansion" in text and "confirmation" in text
    assert "research_only" in text


def test_handoff_cell_requires_explicit_candidate_and_prints_download_path() -> None:
    text = _all_text()
    assert 'HANDOFF_CANDIDATE_ID = ""' in text
    assert '"handoff"' in text
    assert "codex_handoffs" in text
    assert "_handoff.zip" in text
    assert "HANDOFF_READY" in text
    assert "size_bytes" in text
    assert "sha256" in text
    assert "checkpoint" in text
    assert "feature cache" in text
    assert "제출 파일" in text


def test_embedded_runtime_is_hash_bound_and_contains_current_handoff_code() -> None:
    text = _all_text()
    encoded_match = re.search(r'EMBEDDED_RUNTIME_B64 = "([A-Za-z0-9+/=]+)"', text)
    digest_match = re.search(r'EMBEDDED_RUNTIME_SHA256 = "([0-9a-f]{64})"', text)
    assert encoded_match and digest_match
    archive_bytes = base64.b64decode(encoded_match.group(1), validate=True)
    assert sha256(archive_bytes).hexdigest() == digest_match.group(1)

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        names = {member.name for member in members}
    assert all(not member.issym() and not member.islnk() for member in members)
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in names)
    for required in (
        "experiments/independent_dl/handoff.py",
        "experiments/independent_dl/run_campaign.py",
        "experiments/independent_dl/campaign.py",
        "experiments/independent_dl/features.py",
        "experiments/independent_dl/configs/campaign_v1.json",
        "experiments/independent_dl/requirements-colab.txt",
    ):
        assert required in names


def test_renderer_output_matches_tracked_notebook() -> None:
    renderer = (ROOT / "tools/render_independent_dl_kaggle_notebook.py").read_text(
        encoding="utf-8"
    )
    assert "gzip.compress" in renderer
    assert "mtime=0" in renderer
    assert "git" in renderer and "archive" in renderer
