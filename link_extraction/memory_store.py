"""JSON-on-disk persistence for job/batch state across server restarts.

Pragmatic v1 — no SQLite, no schema migrations. Each job/batch terminal
state writes one JSON file under `.memory/` (atomic via tempfile-rename).
On FastAPI startup, scan the directory and rebuild the in-memory dicts.

Why this exists:
  - Jobs survive `Ctrl-C` / process restart — analysts don't lose history
  - The batch CSV/JSON download endpoints work on past runs, not just live
  - Cost meter (Step 14b) can show "you spent $X across N hypotheses today"

What it stores:
  - Full hypothesis dict + final PipelineResult per job
  - Event history (so /events SSE replay still works after restart)
  - Batch member list + relationship to jobs

What it DOES NOT store:
  - In-flight asyncio state (subscribers, completion_event, task) — those
    are session-only. Restored jobs come back as `done` or `error`; we
    never resume an interrupted pipeline.
  - The big `links_by_channel` blob — we keep `triaged_links` and `grouped`
    (the final outputs) but drop discovery raw lists to keep snapshots small.

Layout:
    .memory/
    ├── index.json              # lightweight metadata for fast listing
    ├── jobs/{job_id}.json
    └── batches/{batch_id}.json
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .models import (
    ChannelFit, ChannelId, DiscoveredLink, ShortVideoLink,
    TimeWindow, TypedQuery, Verdict,
)

if TYPE_CHECKING:
    # Avoid circular imports at runtime — these are only used in type hints
    from .job_runner import JobState
    from .batch_runner import BatchState

log = logging.getLogger(__name__)


# Default storage root — overridable via env (e.g. for tests).
_DEFAULT_ROOT = Path(__file__).resolve().parent.parent / ".memory"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write `data` as JSON to `path` atomically (tempfile + rename).

    Atomicity matters because the server can crash mid-write — a partial
    JSON file would poison the next startup's load.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.stem + ".", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


# ─── Serialization helpers ───────────────────────────────────────────────────


def _link_to_dict(link: DiscoveredLink) -> Dict[str, Any]:
    """Full Pydantic dump of a DiscoveredLink (or ShortVideoLink)."""
    return link.model_dump(mode="json")


def _link_from_dict(d: Dict[str, Any]) -> DiscoveredLink:
    """Inverse of `_link_to_dict` — picks the right concrete class."""
    # Heuristic: short-video specific fields → ShortVideoLink, else base
    sv_keys = {"duration_sec", "caption", "view_count", "engagement_score"}
    if any(k in d for k in sv_keys):
        return ShortVideoLink.model_validate(d)
    return DiscoveredLink.model_validate(d)


def _serialize_decomp(decomp) -> Dict[str, Any]:
    return decomp.model_dump(mode="json")


# ─── Snapshot types ──────────────────────────────────────────────────────────


@dataclass
class JobSnapshot:
    """Persistable view of a JobState — drops asyncio internals."""

    job_id: str
    hypothesis_id: str
    hypothesis: Dict[str, Any]
    window: Dict[str, Any]              # TimeWindow.model_dump()
    status: str                          # "done" | "error" | "running" (→ remapped on restore)
    created_at: str
    updated_at: str
    error: Optional[str] = None

    # Final result (None if status == "error" or job died mid-flight)
    result: Optional[Dict[str, Any]] = None
    # Event history (small dicts already; PipelineEvent.to_dict())
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Schema version, for forward compat. Bump on breaking changes.
    schema_version: int = 1

    @classmethod
    def from_state(cls, state: "JobState") -> "JobSnapshot":
        result_payload: Optional[Dict[str, Any]] = None
        if state.result is not None:
            r = state.result
            result_payload = {
                "hypothesis_id":     r.hypothesis_id,
                "elapsed_sec":       r.elapsed_sec,
                "channels_used":     list(r.links_by_channel.keys()),
                "channels_skipped":  list(r.channels_skipped),
                "decomposition":     _serialize_decomp(r.decomposition),
                "channel_fits": [
                    {"channel": f.channel, "fit_score": f.fit_score,
                     "rationale": f.rationale, "expected_signal": f.expected_signal,
                     "sub_scores": f.sub_scores}
                    for f in r.channel_fits
                ],
                "queries_by_channel": {
                    ch: [q.model_dump(mode="json") for q in qs]
                    for ch, qs in r.queries_by_channel.items()
                },
                # Triaged links (the final ranked set) — keep these
                "triaged_links": [_link_to_dict(lk) for lk in r.triaged_links],
                "grouped": {
                    v: [_link_to_dict(lk) for lk in links]
                    for v, links in r.grouped.items()
                },
                # Drop raw links_by_channel to keep snapshots small — the
                # triaged set is what every downstream consumer (UI, CSV,
                # cost analysis) actually reads.
            }
        return cls(
            job_id=state.job_id,
            hypothesis_id=state.hypothesis_id,
            hypothesis=state.hypothesis,
            window=state.window.model_dump(mode="json"),
            status=state.status,
            created_at=state.created_at.isoformat(),
            updated_at=state.updated_at.isoformat(),
            error=state.error,
            result=result_payload,
            history=[ev.to_dict() for ev in state.history],
        )

    def to_state(self) -> "JobState":
        """Rebuild a JobState — note: asyncio fields stay default (no live task)."""
        from .job_runner import JobState  # local to avoid circular import
        from .orchestrator import PipelineEvent, PipelineResult
        from .decomposer import Decomposition

        # Status remap — "running" without a task is effectively errored
        status = self.status
        error = self.error
        if status == "running":
            status = "error"
            if not error:
                error = "session ended before pipeline finished (restored from disk)"

        # Rebuild the TimeWindow
        window = TimeWindow.model_validate(self.window)

        # Rebuild PipelineResult if present
        result: Optional[PipelineResult] = None
        if self.result is not None:
            decomp = Decomposition.model_validate(self.result["decomposition"])
            fits = [ChannelFit.model_validate(f) for f in self.result["channel_fits"]]
            queries_by_channel = {
                ch: [TypedQuery.model_validate(q) for q in qs]
                for ch, qs in (self.result.get("queries_by_channel") or {}).items()
            }
            triaged = [_link_from_dict(d) for d in (self.result.get("triaged_links") or [])]
            grouped = {
                v: [_link_from_dict(d) for d in links]
                for v, links in (self.result.get("grouped") or {}).items()
            }
            result = PipelineResult(
                hypothesis_id=self.result["hypothesis_id"],
                decomposition=decomp,
                channel_fits=fits,
                queries_by_channel=queries_by_channel,
                links_by_channel={},   # dropped from snapshot — empty on restore
                triaged_links=triaged,
                grouped=grouped,
                elapsed_sec=float(self.result.get("elapsed_sec") or 0.0),
                channels_skipped=list(self.result.get("channels_skipped") or []),
                clusters=[],           # dropped from snapshot
            )

        state = JobState(
            job_id=self.job_id,
            hypothesis_id=self.hypothesis_id,
            hypothesis=self.hypothesis,
            window=window,
            status=status,
            created_at=datetime.fromisoformat(self.created_at),
            updated_at=datetime.fromisoformat(self.updated_at),
            error=error,
            result=result,
        )

        # Rebuild event history
        for ev_dict in self.history:
            try:
                state.history.append(PipelineEvent(
                    kind=ev_dict.get("kind", "unknown"),
                    hypothesis_id=ev_dict.get("hypothesis_id", self.hypothesis_id),
                    stage=ev_dict.get("stage"),
                    timestamp=datetime.fromisoformat(ev_dict["timestamp"])
                              if ev_dict.get("timestamp") else datetime.now(timezone.utc),
                    data=ev_dict.get("data") or {},
                ))
            except Exception as e:
                log.debug("skipping malformed history entry: %s", e)

        # Restored jobs are already terminal — set the completion event
        state.completion_event.set()
        return state


@dataclass
class BatchSnapshot:
    """Persistable view of a BatchState. Member jobs are stored separately."""

    batch_id: str
    status: str
    created_at: str
    updated_at: str
    error: Optional[str] = None
    concurrency: int = 3
    core_problems: Dict[str, str] = field(default_factory=dict)
    detected_pairs: List[List[str]] = field(default_factory=list)
    # Per-member metadata (hypothesis_id → row in CSV) — full job lives in
    # its own JobSnapshot file referenced by job_id.
    members: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    schema_version: int = 1

    @classmethod
    def from_state(cls, state: "BatchState") -> "BatchSnapshot":
        return cls(
            batch_id=state.batch_id,
            status=state.status,
            created_at=state.created_at.isoformat(),
            updated_at=state.updated_at.isoformat(),
            error=state.error,
            concurrency=state.concurrency,
            core_problems=dict(state.core_problems),
            detected_pairs=[list(p) for p in state.detected_pairs],
            members=[
                {
                    "row_index": m.row_index,
                    "hypothesis_id": m.hypothesis_id,
                    "core_problem_id": m.core_problem_id,
                    "hypothesis": m.hypothesis,
                    "window_label": m.window_label,
                    "max_triage": m.max_triage,
                    "use_llm": m.use_llm,
                    "status": m.status,
                    "job_id": m.job_id,
                    "started_at": (m.started_at.isoformat() if m.started_at else None),
                    "finished_at": (m.finished_at.isoformat() if m.finished_at else None),
                    "error": m.error,
                }
                for m in state.members
            ],
            history=list(state.history),
        )

    def to_state(self) -> "BatchState":
        from .batch_runner import BatchState, BatchMember
        status = self.status
        if status == "running":
            status = "partial"  # session ended; finished members are still valid
        members = [
            BatchMember(
                row_index=m["row_index"],
                hypothesis_id=m["hypothesis_id"],
                core_problem_id=m["core_problem_id"],
                hypothesis=m["hypothesis"],
                window_label=m["window_label"],
                max_triage=m["max_triage"],
                use_llm=m.get("use_llm", True),
                status=("error" if m["status"] == "running" else m["status"]),
                job_id=m.get("job_id"),
                started_at=(datetime.fromisoformat(m["started_at"])
                            if m.get("started_at") else None),
                finished_at=(datetime.fromisoformat(m["finished_at"])
                             if m.get("finished_at") else None),
                error=m.get("error"),
            )
            for m in self.members
        ]
        state = BatchState(
            batch_id=self.batch_id,
            members=members,
            created_at=datetime.fromisoformat(self.created_at),
            updated_at=datetime.fromisoformat(self.updated_at),
            status=status,
            error=self.error,
            concurrency=self.concurrency,
            core_problems=dict(self.core_problems),
            detected_pairs=[list(p) for p in self.detected_pairs],
        )
        state.history = list(self.history)
        state.completion_event.set()
        return state


# ─── MemoryStore ─────────────────────────────────────────────────────────────


class MemoryStore:
    """File-system backed snapshot store."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else _DEFAULT_ROOT
        self.jobs_dir = self.root / "jobs"
        self.batches_dir = self.root / "batches"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.batches_dir.mkdir(parents=True, exist_ok=True)

    # ── Persistence ───────────────────────────────────────────────────────

    def save_job(self, state: "JobState") -> None:
        try:
            snap = JobSnapshot.from_state(state)
            _atomic_write_json(self.jobs_dir / f"{state.job_id}.json", asdict(snap))
            log.debug("memory_store: saved job %s", state.job_id)
        except Exception as e:
            log.warning("memory_store: save_job(%s) failed: %s", state.job_id, e)

    def save_batch(self, state: "BatchState") -> None:
        try:
            snap = BatchSnapshot.from_state(state)
            _atomic_write_json(self.batches_dir / f"{state.batch_id}.json", asdict(snap))
            log.debug("memory_store: saved batch %s", state.batch_id)
        except Exception as e:
            log.warning("memory_store: save_batch(%s) failed: %s", state.batch_id, e)

    # ── Restore on startup ────────────────────────────────────────────────

    def load_all(self) -> tuple[Dict[str, "JobState"], Dict[str, "BatchState"]]:
        jobs: Dict[str, JobState] = {}
        batches: Dict[str, BatchState] = {}

        for path in sorted(self.jobs_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                snap = JobSnapshot(**data)
                jobs[snap.job_id] = snap.to_state()
            except Exception as e:
                log.warning("memory_store: load job %s failed: %s", path.name, e)

        for path in sorted(self.batches_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                snap = BatchSnapshot(**data)
                batches[snap.batch_id] = snap.to_state()
            except Exception as e:
                log.warning("memory_store: load batch %s failed: %s", path.name, e)

        log.info("memory_store: restored %d jobs + %d batches from %s",
                 len(jobs), len(batches), self.root)
        return jobs, batches

    # ── Listing / search (used by /memory endpoints) ──────────────────────

    def list_job_summaries(
        self, *, hypothesis_id: Optional[str] = None, limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Lightweight metadata list — does NOT hydrate full JobState."""
        out: List[Dict[str, Any]] = []
        for path in sorted(
            self.jobs_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        ):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if hypothesis_id and data.get("hypothesis_id") != hypothesis_id:
                continue
            result = data.get("result") or {}
            grouped = result.get("grouped") or {}
            out.append({
                "job_id": data.get("job_id"),
                "hypothesis_id": data.get("hypothesis_id"),
                "statement": (data.get("hypothesis") or {}).get("statement", "")[:200],
                "status": data.get("status"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "error": data.get("error"),
                "elapsed_sec": result.get("elapsed_sec"),
                "channels_used": result.get("channels_used") or [],
                "verdict_counts": {
                    v: len(grouped.get(v) or [])
                    for v in ("supports", "refutes", "tangential")
                },
            })
            if len(out) >= limit:
                break
        return out

    def list_batch_summaries(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for path in sorted(
            self.batches_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        ):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({
                "batch_id": data.get("batch_id"),
                "status": data.get("status"),
                "created_at": data.get("created_at"),
                "updated_at": data.get("updated_at"),
                "member_count": len(data.get("members") or []),
                "core_problem_count": len(data.get("core_problems") or {}),
                "error": data.get("error"),
            })
            if len(out) >= limit:
                break
        return out


# ─── Module-level singleton ──────────────────────────────────────────────────

_singleton: Optional[MemoryStore] = None


def get_store() -> MemoryStore:
    global _singleton
    if _singleton is None:
        root_env = os.getenv("OUTTLYR_MEMORY_DIR", "")
        root = Path(root_env) if root_env else _DEFAULT_ROOT
        _singleton = MemoryStore(root=root)
    return _singleton


def reset_store() -> None:
    global _singleton
    _singleton = None
