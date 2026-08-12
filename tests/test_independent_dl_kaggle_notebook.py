from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/KAGGLE_INDEPENDENT_DL_CAMPAIGN.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_kaggle_notebook_is_a_clean_two_cell_handoff():
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert [cell["cell_type"] for cell in notebook["cells"]] == ["markdown", "code"]
    assert "Kaggle" in "".join(notebook["cells"][0]["source"])

    code_cell = notebook["cells"][1]
    assert code_cell["execution_count"] is None
    assert code_cell["outputs"] == []
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["kaggle"]["title"] == "Independent DL Campaign"


def test_kaggle_notebook_uses_kaggle_paths_and_existing_campaign():
    code = "".join(_load_notebook()["cells"][1]["source"])

    required = (
        "/kaggle/input",
        "/kaggle/working",
        "c36811956e632a00f775525f83bcb666ff2ec9d6",
        "torch.cuda.is_available()",
        "experiments.independent_dl.run_campaign",
        "train.csv",
        "trackman_history.csv",
        "independent_dl_campaign_v1",
        "shutil.copytree",
        "INDEPENDENT_DL_CAMPAIGN_CHECKPOINTED",
    )
    for value in required:
        assert value in code

    forbidden = (
        "google.colab",
        "/content/drive",
        "submission.csv",
        "submission.zip",
    )
    for value in forbidden:
        assert value not in code


def test_kaggle_notebook_rejects_ambiguous_inputs_and_preserves_working_copy():
    code = "".join(_load_notebook()["cells"][1]["source"])

    assert "len(matches) != 1" in code
    assert "여러 개 발견" in code
    assert "if CAMPAIGN_OUTPUT_DIR.exists()" in code
    assert "dirs_exist_ok=False" in code
    assert "shutil.rmtree(CAMPAIGN_OUTPUT_DIR)" not in code


def test_kaggle_notebook_uses_embedded_runtime_without_github_authentication():
    code = "".join(_load_notebook()["cells"][1]["source"])

    assert "EMBEDDED_RUNTIME_B64" in code
    assert "EMBEDDED_RUNTIME_SHA256" in code
    assert "base64.b64decode" in code
    assert "tarfile.open" in code
    assert "runtime archive SHA-256 mismatch" in code
    assert "github.com" not in code.lower()
    assert "GITHUB_TOKEN" not in code
    assert "kaggle_secrets" not in code
    assert "git clone" not in code
