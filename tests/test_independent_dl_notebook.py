from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb"
HANDOFF = ROOT / "experiments/independent_dl/COLAB.md"


def _handoff_code() -> str:
    text = HANDOFF.read_text(encoding="utf-8")
    return text.split("```python\n", 1)[1].split("\n```", 1)[0]


def test_independent_dl_notebook_is_clean_single_cell_handoff():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert [cell["cell_type"] for cell in notebook["cells"]] == ["markdown", "code"]
    assert "독립 DL 캠페인" in "".join(notebook["cells"][0]["source"])

    code = notebook["cells"][1]
    assert "".join(code["source"]).rstrip("\n") == _handoff_code().rstrip("\n")
    assert code["execution_count"] is None
    assert code["outputs"] == []

    assert notebook["metadata"]["colab"]["name"] == "INDEPENDENT_DL_CAMPAIGN.ipynb"
    assert notebook["metadata"]["accelerator"] == "GPU"
