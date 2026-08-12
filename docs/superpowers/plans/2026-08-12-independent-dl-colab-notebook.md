# Independent DL Colab Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 독립 DL 단일 실행 셀을 GitHub에서 Colab으로 바로 열 수 있는 노트북으로 제공한다.

**Architecture:** `COLAB.md`를 실행 코드의 단일 원본으로 유지하고, 새 노트북의 코드 셀은 그 Python 코드 블록과 정확히 일치시킨다. 별도 생성기나 실행 로직은 만들지 않고 정적 계약 테스트가 두 파일의 동기화와 Colab 메타데이터를 확인한다.

**Tech Stack:** Jupyter Notebook nbformat 4 JSON, Python 표준 라이브러리 `json`, pytest.

---

## File map

- `notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb`: 안내 Markdown 셀과 완결된 실행 코드 셀 하나.
- `tests/test_independent_dl_notebook.py`: 노트북 구조, 코드 동기화, GPU 메타데이터와 깨끗한 실행 상태를 검증한다.

### Task 1: Add the Colab notebook

**Files:**
- Create: `notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb`
- Create: `tests/test_independent_dl_notebook.py`

- [ ] **Step 1: Write the failing notebook contract test**

```python
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
```

- [ ] **Step 2: Run the focused test and observe the missing notebook failure**

Run:

```bash
uv run --no-project --python python3.11 --with pytest==8.4.1 python -m pytest tests/test_independent_dl_notebook.py -q
```

Expected: fail with `FileNotFoundError` for `notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb`.

- [ ] **Step 3: Create the minimal notebook JSON**

Create nbformat 4.5 JSON with:

```python
{
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 독립 DL 캠페인\n",
                "\n",
                "이 노트북은 공식 데이터와 Colab T4에서 사용자가 실행합니다. "
                "동일한 Drive 결과 경로로 다시 실행하면 checkpoint부터 재개합니다.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": HANDOFF_CODE_AS_LINES,
        },
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "INDEPENDENT_DL_CAMPAIGN.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
```

`HANDOFF_CODE_AS_LINES`는 `COLAB.md`의 유일한 Python 코드 블록을 `splitlines(keepends=True)`한 값으로 채운다. 생성 후 별도 생성 스크립트는 남기지 않는다.

- [ ] **Step 4: Run focused and existing handoff tests**

Run:

```bash
uv run --no-project --python python3.11 --with pytest==8.4.1 python -m pytest tests/test_independent_dl_notebook.py tests/test_independent_dl_handoff.py -q
python3 -m json.tool notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb >/dev/null
git diff --check
```

Expected: all tests pass and both static commands exit `0`. Do not mount Drive, install packages, start CUDA, train a model, edit another notebook, or create a submission artifact.

- [ ] **Step 5: Commit**

```bash
git add notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb tests/test_independent_dl_notebook.py
git commit -m "feat: add independent dl colab notebook"
```

## Completion boundary

Implementation ends with a local commit. Pushing `main` remains a separate user-authorized action. After push, the user opens:

```text
https://colab.research.google.com/github/yh4752/lg-aimers-9th-competition/blob/main/notebooks/INDEPENDENT_DL_CAMPAIGN.ipynb
```
