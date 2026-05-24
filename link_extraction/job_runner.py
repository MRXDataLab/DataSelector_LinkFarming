"""Step 8 — In-memory job store + SSE-ready event fanout.

Decouples the orchestrator (one hypothesis → one PipelineResult) from the
API layer (Step 9) that exposes status / results / event-stream endpoints.

Design choices:
- **In-memory dict store** keyed by short UUID — matches host convention
  (`uuid.uuid4().hex[:8]`). No Redis required for v1.
- **Per-subscriber asyncio.Queue** fanout — each SSE client gets its own
  queue. Late subscribers replay the full event history first, then stream
  live. A terminal `None` is enqueued when the job completes so subscribers
  can exit their `async for` loop cleanly.
- **One job per hypothesis** — multiple hypotheses run in parallel as
  separate jobs. Per-hypothesis serialization happens inside the orchestrator.

Public API:
    create_job(hypothesis, window, *, registry=None, use_llm=True) -> job_id
    get_job(job_id) -> JobState | None
    list_jobs() -> List[JobState]
    subscribe(job_id) -> AsyncIterator[PipelineEvent]   # async generator
    await_completion(job_id) -> JobState
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from .models import TimeWindow
from .orchestrator import (
    DiscovererRegistry,
    PipelineEvent,
    PipelineResult,
    default_registry,
    run_pipeline,
)

log = logging.getLogger(__name__)


# ─── Job state ───────────────────────────────────────────────────────────────


JobStatus = str  # "pending" | "running" | "done" | "error"


@dataclass
class JobState:
    """Server-side view of one pipeline job."""

    job_id: str
    hypothesis_id: str
    hypothesis: Dict[str, Any]
    window: TimeWindow
    status: JobStatus = "pending"
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    error: Optional[str] = None
    result: Optional[PipelineResult] = None

    # Event-bus internals
    history: List[PipelineEvent] = field(default_factory=list)
    subscribers: List[asyncio.Queue] = field(default_factory=list)
    completion_event: asyncio.Event = field(default_factory=asyncio.Event)

    # The actual asyncio task driving the job (so callers can cancel)
    task: Optional[asyncio.Task] = field(default=None)

    def emit(self, ev: PipelineEvent) -> None:
        """Record an event in history AND fanout to all live subscribers."""
        self.history.append(ev)
        self.updated_at = datetime.now(timezone.utc)
        for q in list(self.subscribers):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                log.warning("subscriber queue full on job %s; dropping event %s",
                            self.job_id, ev.kind)

    def attach_subscriber(self) -> asyncio.Queue:
        """Create + register a queue, replaying history first.

        Caller should `await q.get()` in a loop until it receives `None`
        (terminal sentinel pushed when the job finishes).
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        for ev in self.history:
            q.put_nowait(ev)
        if self.status in ("done", "error"):
            q.put_nowait(None)  # terminal — no more events coming
        else:
            self.subscribers.append(q)
        return q

    def detach_subscriber(self, q: asyncio.Queue) -> None:
        try:
            self.subscribers.remove(q)
        except ValueError:
            pass

    def _close_subscribers(self) -> None:
        """Push the terminal sentinel to all subscribers; clear the list."""
        for q in self.subscribers:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass
        self.subscribers.clear()

    def to_summary(self) -> Dict[str, Any]:
        """Lightweight status payload (no link blobs) for `/jobs/{id}` polling."""
        summary: Dict[str, Any] = {
            "job_id": self.job_id,
            "hypothesis_id": self.hypothesis_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "n_events": len(self.history),
            "error": self.error,
        }
        if self.result is not None:
            summary["verdict_counts"] = {
                k: len(v) for k, v in self.result.grouped.items()
            }
            summary["elapsed_sec"] = round(self.result.elapsed_sec, 3)
            summary["channels_used"] = list(self.result.links_by_channel.keys())
            summary["channels_skipped"] = list(self.result.channels_skipped)
        return summary


# ─── In-memory store ─────────────────────────────────────────────────────────

_jobs: Dict[str, JobState] = {}


def _new_job_id() -> str:
    return uuid.uuid4().hex[:8]


def get_job(job_id: str) -> Optional[JobState]:
    return _jobs.get(job_id)


def list_jobs() -> List[JobState]:
    return list(_jobs.values())


def clear_jobs() -> None:
    """Drop all jobs (used by tests/restarts; never call from prod paths)."""
    _jobs.clear()


# ─── Job lifecycle ───────────────────────────────────────────────────────────


def create_job(
    hypothesis: Dict[str, Any],
    window: TimeWindow,
    *,
    registry: Optional[DiscovererRegistry] = None,
    use_llm: bool = True,
    max_triage: int = 30,
) -> str:
    """Register a new job and kick off the pipeline as a background task.

    Returns the job_id; status starts at 'pending' then flips to 'running'
    on the first event. Caller can immediately `subscribe(job_id)` to stream
    events without missing any (history is replayed on attach).
    """
    job_id = _new_job_id()
    hyp_id = (
        hypothesis.get("hypothesis_id")
        or hypothesis.get("id")
        or "unknown"
    )
    state = JobState(
        job_id=job_id,
        hypothesis_id=hyp_id,
        hypothesis=hypothesis,
        window=window,
    )
    _jobs[job_id] = state

    state.task = asyncio.create_task(
        _run_job(
            state,
            registry=registry,
            use_llm=use_llm,
            max_triage=max_triage,
        ),
        name=f"pipeline-{job_id}",
    )
    return job_id


async def _run_job(
    state: JobState,
    *,
    registry: Optional[DiscovererRegistry],
    use_llm: bool,
    max_triage: int,
) -> None:
    """Background task body — drives one PipelineResult to completion."""
    state.status = "running"
    state.updated_at = datetime.now(timezone.utc)

    def _emit(ev: PipelineEvent) -> None:
        state.emit(ev)

    try:
        result = await run_pipeline(
            state.hypothesis,
            state.window,
            registry=registry or default_registry(),
            emit=_emit,
            use_llm=use_llm,
            max_triage=max_triage,
            job_id=state.job_id,
        )
        state.result = result
        if result.error:
            state.status = "error"
            state.error = result.error
        else:
            state.status = "done"
    except Exception as e:
        log.exception("job %s crashed", state.job_id)
        state.status = "error"
        state.error = f"{type(e).__name__}: {e}"
        # Surface as a pipeline_error event for any subscribers still listening
        state.emit(PipelineEvent(
            kind="pipeline_error",
            hypothesis_id=state.hypothesis_id,
            data={"error": state.error},
        ))
    finally:
        state.updated_at = datetime.now(timezone.utc)
        state.completion_event.set()
        state._close_subscribers()
        # Persist terminal-state snapshot for cross-restart durability (Step 14)
        try:
            from .memory_store import get_store
            get_store().save_job(state)
        except Exception as e:
            log.warning("memory_store.save_job failed for %s: %s", state.job_id, e)


async def await_completion(job_id: str, timeout: Optional[float] = None) -> JobState:
    """Block (asynchronously) until the job reaches a terminal state."""
    state = _jobs.get(job_id)
    if state is None:
        raise KeyError(f"unknown job_id: {job_id}")
    if state.status in ("done", "error"):
        return state
    if timeout is not None:
        await asyncio.wait_for(state.completion_event.wait(), timeout=timeout)
    else:
        await state.completion_event.wait()
    return state


# ─── Event subscription ──────────────────────────────────────────────────────


async def subscribe(job_id: str) -> AsyncIterator[PipelineEvent]:
    """Async generator yielding every event for `job_id`, in order.

    Replays history first, then streams live events. Ends when the job
    completes (a terminal `None` sentinel is enqueued in `_close_subscribers`).

    Usage:
        async for ev in subscribe(job_id):
            yield f"data: {json.dumps(ev.to_dict())}\\n\\n"
    """
    state = _jobs.get(job_id)
    if state is None:
        raise KeyError(f"unknown job_id: {job_id}")
    q = state.attach_subscriber()
    try:
        while True:
            ev = await q.get()
            if ev is None:
                return  # terminal sentinel — job done
            yield ev
    finally:
        state.detach_subscriber(q)
