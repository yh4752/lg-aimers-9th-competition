from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import experiments.tree_expert.s4_production as production


def test_concurrent_confirmation_jobs_initialize_e2_once(tmp_path, monkeypatch):
    runtime = object.__new__(production.ProductionS4Runtime)
    runtime.root = tmp_path / "campaign"
    runtime.verified = SimpleNamespace(
        e2_handoff=tmp_path / "e2_handoff.zip",
        e2_handoff_sha256="a" * 64,
    )
    runtime.e2 = None
    runtime.baselines = {}
    runtime._e2_lock = threading.Lock()
    calls = 0
    calls_lock = threading.Lock()
    verified = SimpleNamespace(fold_predictions={})

    def prepare(*, e2_handoff, output, expected_e2_sha256):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"prepared")

    monkeypatch.setattr(production, "prepare_t3_input", prepare)
    monkeypatch.setattr(
        production, "verify_and_extract_t3_input", lambda *args, **kwargs: verified,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _index: runtime._ensure_e2(), range(2)))

    assert calls == 1
    assert results == (verified, verified)
