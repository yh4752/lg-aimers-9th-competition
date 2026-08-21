from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Callable, Mapping, Sequence
from zipfile import BadZipFile, ZipFile
import json

from .artifacts import verify_resume_bundle, write_resume_bundle
from .contracts import load_contract
from .inputs import VerifiedRealignInput, verify_and_extract_input
from .runner import RealignRun, _bindings, _campaign_files, _read_state, run_campaign


class RealignColabError(RuntimeError):
    """Raised when the direct-upload Colab campaign cannot continue."""


@dataclass(frozen=True)
class DownloadEvent:
    phase: str
    path: Path


class EmergencySnapshotMonitor:
    """Store verified active-job resumes without triggering browser downloads."""

    def __init__(
        self,
        *,
        campaign_root: Path,
        snapshot_root: Path,
        verified: VerifiedRealignInput,
        interval_seconds: float = 300.0,
        poll_seconds: float = 2.0,
    ) -> None:
        self.campaign_root = Path(campaign_root)
        self.snapshot_root = Path(snapshot_root)
        self.bindings = _bindings(verified)
        self.interval_seconds = interval_seconds
        self.poll_seconds = poll_seconds
        self.latest: Path | None = None
        self._last_at = 0.0
        self._sequence = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def snapshot(self, *, force: bool = False) -> Path | None:
        now = time.time()
        if not force and now - self._last_at < self.interval_seconds:
            return self.latest
        state_path = self.campaign_root / "state/stage_state.json"
        if not state_path.is_file() or state_path.is_symlink():
            return self.latest
        try:
            state = _read_state(self.campaign_root)
            if state.status not in {"f1_active", "full_fit_active"}:
                return self.latest
            files = _campaign_files(self.campaign_root, state)
            self.snapshot_root.mkdir(parents=True, exist_ok=True)
            candidate = self.snapshot_root / f"active_resume_{self._sequence:04d}.zip"
            self._sequence += 1
            write_resume_bundle(files, candidate, self.bindings)
            verify_resume_bundle(candidate, self.bindings)
            if self.latest is not None and self.latest.is_file():
                def digest(path: Path) -> str:
                    from hashlib import sha256

                    value = sha256()
                    with path.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            value.update(chunk)
                    return value.hexdigest()

                if digest(self.latest) == digest(candidate):
                    candidate.unlink()
                    self._last_at = now
                    return self.latest
            prior = self.latest
            self.latest = candidate
            self._last_at = now
            if prior is not None and prior.parent == self.snapshot_root:
                prior.unlink(missing_ok=True)
            return self.latest
        except Exception:
            return self.latest

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            self.snapshot()

    def start(self) -> None:
        if self._thread is not None:
            raise RealignColabError("emergency monitor is already started")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_seconds * 2))


def _artifact_kind(path: Path) -> str:
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or "manifest.json" not in names:
                raise RealignColabError("uploaded ZIP manifest is missing or duplicated")
            root = json.loads(archive.read("manifest.json"))
    except RealignColabError:
        raise
    except (BadZipFile, OSError, json.JSONDecodeError) as error:
        raise RealignColabError(f"uploaded ZIP is invalid: {error}") from error
    if type(root) is not dict or type(root.get("artifact_kind")) is not str:
        raise RealignColabError("uploaded ZIP artifact kind differs")
    return root["artifact_kind"]


def classify_and_verify_uploads(
    paths: Sequence[Path], *, run_root: Path
) -> tuple[VerifiedRealignInput, Path | None]:
    if len(paths) not in (1, 2):
        raise RealignColabError("upload count must be one or two")
    kinds: dict[str, Path] = {}
    for raw in paths:
        path = Path(raw)
        kind = _artifact_kind(path)
        if kind in kinds:
            raise RealignColabError("uploaded artifact kind is duplicated")
        kinds[kind] = path
    if set(kinds) not in (
        {"catboost_50_50_realign_input_v1"},
        {"catboost_50_50_realign_input_v1", "realign_resume_v1"},
    ):
        raise RealignColabError("uploaded artifact kinds differ")
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=False)
    verified = verify_and_extract_input(
        kinds["catboost_50_50_realign_input_v1"], root / "verified_input", load_contract()
    )
    resume = kinds.get("realign_resume_v1")
    if resume is not None:
        verify_resume_bundle(resume, _bindings(verified))
    return verified, resume


def collect_terminal_downloads(result: RealignRun) -> list[DownloadEvent]:
    if result.status == "completed":
        if result.delivery_bundle is None:
            raise RealignColabError("completed run is missing delivery")
        return [
            DownloadEvent("completed_review", result.review_bundle),
            DownloadEvent("completed_resume", result.resume_bundle),
            DownloadEvent("completed_delivery", result.delivery_bundle),
        ]
    if result.status == "deployment_blocked":
        if result.delivery_bundle is not None:
            raise RealignColabError("blocked run unexpectedly has delivery")
        return [
            DownloadEvent("blocked_review", result.review_bundle),
            DownloadEvent("blocked_resume", result.resume_bundle),
        ]
    return [DownloadEvent("incomplete_resume", result.resume_bundle)]


def run_supervised_campaign(
    *,
    campaign: Callable[..., RealignRun] = run_campaign,
    campaign_kwargs: Mapping[str, object],
    download: Callable[[DownloadEvent], None],
    latest_verified_resume: Callable[[], Path | None],
) -> RealignRun:
    monitor: EmergencySnapshotMonitor | None = None
    verified = campaign_kwargs.get("verified")
    output_dir = campaign_kwargs.get("output_dir")
    if isinstance(verified, VerifiedRealignInput) and output_dir is not None:
        campaign_root = Path(output_dir)
        monitor = EmergencySnapshotMonitor(
            campaign_root=campaign_root,
            snapshot_root=campaign_root.parent / "snapshots",
            verified=verified,
        )
        monitor.start()
    def on_phase_resume(path: Path, phase: str) -> None:
        if phase == "f1_complete":
            try:
                download(DownloadEvent("f1_complete", path))
            except Exception as error:
                raise RealignColabError("download callback failed") from error

    try:
        result = campaign(
            **dict(campaign_kwargs),
            on_phase_resume=on_phase_resume,
        )
        for event in collect_terminal_downloads(result):
            try:
                download(event)
            except Exception as error:
                raise RealignColabError("download callback failed") from error
        return result
    except Exception:
        if monitor is not None:
            monitor.snapshot(force=True)
        latest = monitor.latest if monitor is not None and monitor.latest else latest_verified_resume()
        if latest is not None:
            try:
                download(DownloadEvent("emergency_resume", latest))
            except Exception:
                pass
        raise
    finally:
        if monitor is not None:
            monitor.stop()
