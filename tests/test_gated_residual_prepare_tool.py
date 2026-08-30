from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from tools.prepare_gated_residual_final_input import main


def test_prepare_tool_prints_success_marker(tmp_path: Path, capsys) -> None:
    sources = []
    for name in ("e2.zip", "a.zip", "b.zip"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        sources.append(path)
    output = tmp_path / "input.zip"
    hashes = [sha256(path.read_bytes()).hexdigest() for path in sources]

    code = main([
        "--e2-input", str(sources[0]), "--stage-a", str(sources[1]),
        "--stage-b", str(sources[2]), "--output", str(output),
        "--expected-e2-sha256", hashes[0], "--expected-stage-a-sha256", hashes[1],
        "--expected-stage-b-sha256", hashes[2],
    ])

    assert code == 0
    assert output.is_file()
    assert "GATED_RESIDUAL_INPUT_READY" in capsys.readouterr().out
