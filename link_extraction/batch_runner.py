"""Batch hypothesis runner — coordinates N pipeline jobs from one CSV upload.

A `Batch` wraps N `JobState`s — each hypothesis row in the uploaded CSV
gets its own pipeline run, and the batch tracks aggregate status, an
async event queue that fans-out across all member jobs, and aggregated
result accessors.

Concurrency: `asyncio.Semaphore(BATCH_CONCURRENCY)` caps how many
pipelines run in parallel. Default 3 — empirically matches YT Data API
quota math (3 × ~106 units in-flight comfortably under the 10K daily
cap) and keeps Gemini rate-limit headroom for the triage batches.

Errored hypotheses (per locked recommendation): skip-and-continue. The
batch is marked `partial` if any member ends in `error`, `done` only if
all members succeeded.

MECE pair pooling (per locked recommendation #3) is **detected** at
parse time (`ParsedBatch.detected_pairs`) and reported, but not yet
**applied** in this v1 — each row runs as its own pipeline. v1.1 would
swap the L2 stage for `synthesize_pair_pooled()` when a pair is detected
in the same batch.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from .csv_io import ParsedBatch, ParsedHypothesis
from .job_runner import JobState, create_job, get_job
from .models import TimeWindow, WindowLabel
from .orchestrator import DiscovererRegistry, PipelineEvent, default_registry

log = logging.getLogger(__name__)


# Default cap on concurrent pipelines (1 hypothesis = 1 pipeline = 1 job).
BATCH_CONCURRENCY = 3


# ─── Batch state ─────────────────────────────────────────────────────────────


BatchStatus = str  # "pending" | "running" | "done" | "partial" | "error"
MemberStatus = str  # "queued" | "running" | "done" | "error" | "skipped"


@dataclass
class BatchMember:
    """One hypothesis inside a batch — wraps a (future) JobState."""

    row_index: int                          # 1-based, from CSV
    hypothesis_id: str
    core_problem_id: str
    hypothesis: Dict[str, Any]              # passed to decompose()
    window_label: WindowLabel               # resolved (batch default ∨ row override)
    max_triage: int                         # resolved
    use_llm: bool
    status: MemberStatus = "queued"
    job_id: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None

    def to_summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "row_index": self.row_index,
            "hypothesis_id": self.hypothesis_id,
            "core_problem_id": self.core_problem_id,
            "statement": self.hypothesis.get("statement", "")[:200],
            "status": self.status,
            "job_id": self.job_id,
            "window_label": self.window_label,
            "max_triage": self.max_triage,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error": self.error,
        }
        # Pull verdict counts + elapsed from the spawned job, if available
        if self.job_id:
            job = get_job(self.job_id)
            if job and job.result is not None:
                out["verdict_counts"] = {k: len(v) for k, v in job.result.grouped.items()}
                out["elapsed_sec"] = round(job.result.elapsed_sec, 3)
        return out


@dataclass
class BatchState:
    batch_id: str
    members: List[BatchMember]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: BatchStatus = "pending"
    error: Optional[str] = None
    concurrency: int = BATCH_CONCURRENCY

    # Source metadata
    core_problems: Dict[str, str] = field(default_factory=dict)  # cp_id → statement
    detected_pairs: List[List[str]] = field(default_factory=list)

    # Event-bus internals (fanned out from all member jobs + batch-level events)
    subscribers: List[asyncio.Queue] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    completion_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = field(default=None)

    # ─── Event handling ───────────────────────────────────────────────────

    def emit(self, kind: str, data: Optional[Dict[str, Any]] = None,
             hypothesis_id: Optional[str] = None) -> None:
        """Push a batch-level event into history + all subscribers."""
        ev: Dict[str, Any] = {
            "kind": kind,
            "batch_id": self.batch_id,
            "hypothesis_id": hypothesis_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {},
        }
        self.history.append(ev)
        self.updated_at = datetime.now(timezone.utc)
        for q in list(self.subscribers):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                log.warning("batch %s subscriber queue full; dropping %s",
                            self.batch_id, kind)

    def relay(self, hypothesis_id: str, ev: PipelineEvent) -> None:
        """Forward a member-job PipelineEvent onto the batch stream."""
        wire = ev.to_dict()
        wire["batch_id"] = self.batch_id
        wire["hypothesis_id"] = hypothesis_id  # ensure tagging
        self.history.append(wire)
        self.updated_at = datetime.now(timezone.utc)
        for q in list(self.subscribers):
            try:
                q.put_nowait(wire)
            except asyncio.QueueFull:
                pass

    def attach_subscriber(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=4000)
        for ev in self.history:
            q.put_nowait(ev)
        if self.status in ("done", "error", "partial"):
            q.put_nowait(None)  # terminal sentinel
        else:
            self.subscribers.append(q)
        return q

    def detach_subscriber(self, q: asyncio.Queue) -> None:
        try:
            self.subscribers.remove(q)
        except ValueError:
            pass

    def _close_subscribers(self) -> None:
        for q in self.subscribers:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass
        self.subscribers.clear()

    # ─── Summary view ─────────────────────────────────────────────────────

    def to_summary(self) -> Dict[str, Any]:
        counts = {"queued": 0, "running": 0, "done": 0, "error": 0, "skipped": 0}
        agg_verdict = {"supports": 0, "refutes": 0, "tangential": 0}
        elapsed = 0.0
        for m in self.members:
            counts[m.status] = counts.get(m.status, 0) + 1
            if m.job_id:
                job = get_job(m.job_id)
                if job and job.result is not None:
                    for v, links in job.result.grouped.items():
                        agg_verdict[v] = agg_verdict.get(v, 0) + len(links)
                    elapsed += job.result.elapsed_sec
        return {
            "batch_id": self.batch_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error": self.error,
            "concurrency": self.concurrency,
            "member_count": len(self.members),
            "core_problem_count": len(self.core_problems),
            "detected_pairs": self.detected_pairs,
            "counts": counts,
            "verdict_counts": agg_verdict,
            "total_pipeline_seconds": round(elapsed, 2),
            "n_events": len(self.history),
        }


# ─── In-memory store ─────────────────────────────────────────────────────────


_batches: Dict[str, BatchState] = {}


def _new_batch_id() -> str:
    return uuid.uuid4().hex[:8]


def get_batch(batch_id: str) -> Optional[BatchState]:
    return _batches.get(batch_id)


def list_batches() -> List[BatchState]:
    return list(_batches.values())


def clear_batches() -> None:
    _batches.clear()


def cancel_batch(batch_id: str) -> bool:
    """Cancel a running batch — kills the asyncio task, marks queued/running
    members as 'skipped', flips state.status to 'partial'. Idempotent.

    Returns True on cancel, False if batch unknown or already terminal.
    """
    state = _batches.get(batch_id)
    if state is None:
        return False
    if state.status in ("done", "error", "partial"):
        return False
    # Cancel the asyncio task (members still in flight will raise
    # CancelledError; their member-status flips to error via the wrapper).
    if state.task is not None and not state.task.done():
        state.task.cancel()
    # Mark queued/running members as skipped + flip state
    now = datetime.now(timezone.utc)
    for m in state.members:
        if m.status in ("queued", "running"):
            m.status = "skipped"
            m.error = "cancelled by user"
            if m.finished_at is None:
                m.finished_at = now
    state.status = "partial"
    state.error = "cancelled by user"
    state.updated_at = now
    state.completion_event.set()
    state._close_subscribers()
    # Persist the cancelled state to disk
    try:
        from .memory_store import get_store
        get_store().save_batch(state)
    except Exception as e:
        log.warning("save_batch on cancel failed for %s: %s", batch_id, e)
    return True


# ─── Batch lifecycle ─────────────────────────────────────────────────────────


def create_batch(
    parsed: ParsedBatch,
    *,
    default_window_label: WindowLabel = "1y",
    default_max_triage: int = 10,
    use_llm: bool = True,
    concurrency: int = BATCH_CONCURRENCY,
    registry: Optional[DiscovererRegistry] = None,
    triage_strictness: str = "liberal",
    backend_preferences: Optional[Any] = None,
    skip_triage: bool = False,
    research_context: Optional[Any] = None,
    skip_synthesis: bool = True,
    max_synthesis_links: int = 12,
) -> str:
    """Register a new batch from a `ParsedBatch` and kick off the runner task.

    Per-row overrides on `window_label` and `max_triage` take precedence
    over the batch-wide defaults. `triage_strictness` and `backend_preferences`
    are applied uniformly to every member.
    """
    if not parsed.hypotheses:
        raise ValueError("ParsedBatch is empty — no hypotheses to run")

    batch_id = _new_batch_id()
    members: List[BatchMember] = []
    for ph in parsed.hypotheses:
        members.append(BatchMember(
            row_index=ph.row_index,
            hypothesis_id=ph.hypothesis["hypothesis_id"],
            core_problem_id=ph.core_problem_id,
            hypothesis=ph.hypothesis,
            window_label=(ph.window_label_override or default_window_label),
            max_triage=(ph.max_triage_override
                        if ph.max_triage_override is not None
                        else default_max_triage),
            use_llm=use_llm,
        ))

    state = BatchState(
        batch_id=batch_id,
        members=members,
        concurrency=max(1, int(concurrency)),
        core_problems={cp_id: parsed.core_problem_statements.get(cp_id, "")
                       for cp_id in parsed.core_problems},
        detected_pairs=[list(p) for p in parsed.detected_pairs],
    )
    # Stash batch-wide settings for _run_member to read (they're not stored
    # per-BatchMember because they don't have per-row overrides today).
    state._triage_strictness = triage_strictness  # type: ignore[attr-defined]
    state._backend_preferences = backend_preferences  # type: ignore[attr-defined]
    state._skip_triage = skip_triage  # type: ignore[attr-defined]
    state._research_context = research_context  # type: ignore[attr-defined]
    state._skip_synthesis = skip_synthesis  # type: ignore[attr-defined]
    state._max_synthesis_links = max_synthesis_links  # type: ignore[attr-defined]
    _batches[batch_id] = state

    state.task = asyncio.create_task(
        _run_batch(state, registry=registry),
        name=f"batch-{batch_id}",
    )
    return batch_id


async def _run_member(
    state: BatchState,
    member: BatchMember,
    sem: asyncio.Semaphore,
    registry: Optional[DiscovererRegistry],
) -> None:
    """Run one batch member under the global concurrency semaphore."""
    async with sem:
        member.status = "running"
        member.started_at = datetime.now(timezone.utc)
        state.emit(
            "member_start",
            data={"row_index": member.row_index,
                  "core_problem_id": member.core_problem_id,
                  "window_label": member.window_label,
                  "max_triage": member.max_triage},
            hypothesis_id=member.hypothesis_id,
        )

        try:
            window = TimeWindow.from_label(member.window_label)
        except Exception as e:
            member.status = "error"
            member.error = f"bad window_label: {e}"
            member.finished_at = datetime.now(timezone.utc)
            state.emit("member_error", data={"error": member.error},
                       hypothesis_id=member.hypothesis_id)
            return

        try:
            job_id = create_job(
                member.hypothesis, window,
                registry=registry,
                use_llm=member.use_llm,
                max_triage=member.max_triage,
                triage_strictness=getattr(state, "_triage_strictness", "liberal"),
                backend_preferences=getattr(state, "_backend_preferences", None),
                skip_triage=getattr(state, "_skip_triage", False),
                research_context=getattr(state, "_research_context", None),
                skip_synthesis=getattr(state, "_skip_synthesis", True),
                max_synthesis_links=getattr(state, "_max_synthesis_links", 12),
            )
        except Exception as e:
            member.status = "error"
            member.error = f"create_job failed: {type(e).__name__}: {e}"
            member.finished_at = datetime.now(timezone.utc)
            state.emit("member_error", data={"error": member.error},
                       hypothesis_id=member.hypothesis_id)
            return

        member.job_id = job_id
        job = get_job(job_id)
        if job is None:  # shouldn't happen — create_job always registers
            member.status = "error"
            member.error = "job vanished after create_job"
            member.finished_at = datetime.now(timezone.utc)
            state.emit("member_error", data={"error": member.error},
                       hypothesis_id=member.hypothesis_id)
            return

        # Drain the job's events into the batch event stream until terminal.
        q = job.attach_subscriber()
        try:
            while True:
                ev = await q.get()
                if ev is None:
                    break
                state.relay(member.hypothesis_id, ev)
        finally:
            job.detach_subscriber(q)

        # Resolve final member status from the job's terminal status
        member.finished_at = datetime.now(timezone.utc)
        if job.status == "done":
            member.status = "done"
        else:
            member.status = "error"
            member.error = job.error or "job ended without result"
        verdict_counts: Dict[str, int] = {}
        if job.result is not None:
            verdict_counts = {k: len(v) for k, v in job.result.grouped.items()}
        state.emit(
            "member_done",
            data={"status": member.status, "verdict_counts": verdict_counts,
                  "elapsed_sec": (
                      round(job.result.elapsed_sec, 2)
                      if job.result is not None else None
                  ),
                  "error": member.error},
            hypothesis_id=member.hypothesis_id,
        )


async def _run_batch(
    state: BatchState,
    *,
    registry: Optional[DiscovererRegistry],
) -> None:
    """Background task driving the whole batch to completion."""
    state.status = "running"
    state.updated_at = datetime.now(timezone.utc)
    reg = registry or default_registry()
    state.emit("batch_start", data={
        "member_count": len(state.members),
        "core_problem_count": len(state.core_problems),
        "concurrency": state.concurrency,
        "detected_pairs": state.detected_pairs,
        "available_channels": list(reg.available_channels()),
    })

    sem = asyncio.Semaphore(state.concurrency)
    try:
        await asyncio.gather(*(
            _run_member(state, m, sem, reg) for m in state.members
        ))
    except Exception as e:
        log.exception("batch %s crashed", state.batch_id)
        state.error = f"{type(e).__name__}: {e}"

    # Resolve batch terminal status
    n_errored = sum(1 for m in state.members if m.status == "error")
    n_done = sum(1 for m in state.members if m.status == "done")
    if n_errored == 0:
        state.status = "done"
    elif n_done == 0:
        state.status = "error"
    else:
        state.status = "partial"

    state.updated_at = datetime.now(timezone.utc)
    state.emit("batch_complete", data={
        "status": state.status,
        "n_done": n_done,
        "n_error": n_errored,
        "verdict_counts": state.to_summary()["verdict_counts"],
        "total_pipeline_seconds": state.to_summary()["total_pipeline_seconds"],
    })
    state.completion_event.set()
    state._close_subscribers()
    # Persist terminal-state snapshot (Step 14)
    try:
        from .memory_store import get_store
        get_store().save_batch(state)
    except Exception as e:
        log.warning("memory_store.save_batch failed for %s: %s", state.batch_id, e)


# ─── Async subscribe / wait helpers ──────────────────────────────────────────


async def subscribe_batch(batch_id: str) -> AsyncIterator[Dict[str, Any]]:
    """Async generator yielding every event for `batch_id` until terminal.

    Mixes batch-level events (`batch_start`, `member_start`, `member_done`,
    `batch_complete`) with relayed per-member PipelineEvents (each tagged
    with `hypothesis_id` + `batch_id`).
    """
    state = _batches.get(batch_id)
    if state is None:
        raise KeyError(f"unknown batch_id: {batch_id}")
    q = state.attach_subscriber()
    try:
        while True:
            ev = await q.get()
            if ev is None:
                return
            yield ev
    finally:
        state.detach_subscriber(q)


async def await_batch_completion(
    batch_id: str, timeout: Optional[float] = None,
) -> BatchState:
    state = _batches.get(batch_id)
    if state is None:
        raise KeyError(f"unknown batch_id: {batch_id}")
    if state.status in ("done", "error", "partial"):
        return state
    if timeout is not None:
        await asyncio.wait_for(state.completion_event.wait(), timeout=timeout)
    else:
        await state.completion_event.wait()
    return state
