"""FastAPI app exposing the Outtlyr Data Selection pipeline.

Endpoints
---------

POST /api/data-selection/start
    body: { "hypothesis": {...}, "window_label": "7d|30d|90d|1y|5y",
            "use_llm": true, "max_triage": 10 }
    → { "job_id": "abcd1234" }

GET  /api/data-selection/jobs/{job_id}
    → JobState.to_summary() — status, verdict_counts, elapsed_sec, channels_used

GET  /api/data-selection/jobs/{job_id}/events
    → text/event-stream; one SSE event per PipelineEvent, named by `kind`
    History is replayed on attach, then live events stream until terminal.

GET  /api/data-selection/jobs/{job_id}/results
    → grouped triaged links (supports / refutes / tangential) with full
      ShortVideoLink metadata for the frontend grid.
    ?wait=true   → blocks until the job is done
    ?download=1  → sets Content-Disposition: attachment for browser save

GET  /api/data-selection/jobs/{job_id}/results.csv
    → flat 29-column CSV (UTF-8 with BOM); one row per triaged link.
    ?wait=true   → blocks until the job is done

GET  /api/data-selection/jobs/{job_id}/discovered.csv
    → flat CSV of EVERY discovered link (not just the triaged top-N).
    Same 29-column schema; below-cut links have empty verdict cells.
    ?wait=true   → blocks until the job is done

POST /api/data-selection/batch/preview
    multipart: file=<CSV upload>
    → preview {hypothesis_count, core_problems[], detected_pairs[], errors[]}
    Does NOT start jobs. Use to validate a CSV before committing.

POST /api/data-selection/batch/start
    body: { csv_text, window_label, use_llm, max_triage, concurrency }
    → { batch_id, member_count, core_problem_count, ... }

GET  /api/data-selection/batch/{batch_id}
    → batch summary + member list with per-hypothesis status

GET  /api/data-selection/batch/{batch_id}/events
    → SSE; merged stream of batch-level + per-member pipeline events

GET  /api/data-selection/batch/{batch_id}/results.csv
    → aggregated CSV across every hypothesis in the batch
    (28 link columns + 2 batch columns: core_problem_id, core_problem_statement)
    ?wait=true   → blocks until batch finishes

GET  /api/data-selection/batch/{batch_id}/results.json
    → nested batch result (core_problem → hypothesis → grouped links)
    ?wait=true   → blocks
    ?download=1  → attachment Content-Disposition

GET  /api/data-selection/registry
    → currently-available discoverer channels (for the demo's status badge)

GET  /  (static)
    → the single-page demo at api/static/index.html

Run with:
    "../MRX_Module_1 Claude/backend/venv/bin/uvicorn" api.app:app --port 8080
"""
from __future__ import annotations

import csv
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Sequence

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load .env from MRX_DataSelector first, then host repo (mirrors smoke test).
_HERE = Path(__file__).resolve().parent
for _env in (_HERE.parent / ".env",
             _HERE.parent.parent / "MRX_Module_1 Claude" / "backend" / ".env"):
    if _env.exists():
        load_dotenv(_env)
        break

from link_extraction import (  # noqa: E402
    BATCH_CONCURRENCY,
    BatchState,
    DiscoveredLink,
    ShortVideoLink,
    TimeWindow,
    WindowLabel,
    await_batch_completion,
    await_completion,
    create_batch,
    create_job,
    default_registry,
    get_batch,
    list_batches,
    parse_hypothesis_csv,
    preview_summary,
    subscribe_batch,
    get_job,
    list_jobs,
    subscribe,
)
from link_extraction.backends.preferences import (  # noqa: E402
    ALL_BACKENDS,
    BackendPreferences,
)
from link_extraction.memory_store import get_store  # noqa: E402
from link_extraction.job_runner import _jobs as _live_jobs  # noqa: E402
from link_extraction.batch_runner import _batches as _live_batches  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="Outtlyr Data Selection",
    description="Hypothesis-driven multi-channel link discovery (Day-10 demo)",
    version="0.1.0",
)

# Permissive CORS for the standalone demo (same-origin static serve is fine,
# but this lets you point a separate frontend at the API if needed).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _restore_from_memory() -> None:
    """Step 14: hydrate in-memory dicts from disk snapshots on boot.

    Any "running" jobs/batches are remapped to "error"/"partial" — we don't
    resume interrupted pipelines (asyncio tasks die at process boundary).
    """
    try:
        jobs, batches = get_store().load_all()
        _live_jobs.update(jobs)
        _live_batches.update(batches)
        logging.getLogger(__name__).info(
            "startup: restored %d jobs + %d batches from disk",
            len(jobs), len(batches),
        )
    except Exception as e:
        logging.getLogger(__name__).warning("memory restore failed: %s", e)


# ─── Request models ──────────────────────────────────────────────────────────


class StartRequest(BaseModel):
    hypothesis: Dict[str, Any] = Field(
        ...,
        description="Hypothesis dict; required keys: hypothesis_id (or id), "
                    "statement; optional: dimension, force_assignment, "
                    "investigation_priority, expected_signals, "
                    "expected_counter_signals, rationale, contrarian_pair_id, "
                    "core_problem_statement.",
    )
    window_label: WindowLabel = Field("1y", description="TimeWindow preset")
    use_llm: bool = Field(True, description="Toggle Gemini slot-fill + triage verdict")
    max_triage: int = Field(30, ge=1, le=100, description="Top-N triage budget")
    triage_strictness: Literal["strict", "balanced", "liberal"] = Field(
        "liberal",
        description="How aggressive the LLM is about calling supports/refutes "
                    "vs tangential. 'liberal' = lean toward decisive verdicts; "
                    "'strict' = original conservative behaviour.",
    )
    backends: Optional[List[str]] = Field(
        None,
        description="Search backends to enable, in priority order. Subset of "
                    "['google_free', 'brave', 'duckduckgo']. Default = all "
                    "enabled, google_free first. Channels that use official "
                    "APIs (youtube_shorts, youtube, trends) are unaffected.",
    )


class StartResponse(BaseModel):
    job_id: str
    status: str
    hypothesis_id: str
    window_label: WindowLabel


# ─── Serialisation helpers ───────────────────────────────────────────────────


def _link_to_dict(link: DiscoveredLink) -> Dict[str, Any]:
    """Compact JSON-safe view of a DiscoveredLink (with ShortVideoLink extras)."""
    out: Dict[str, Any] = {
        "url": link.url,
        "canonical_url": link.canonical_url or link.url,
        "title": link.title,
        "snippet": link.snippet,
        "channel": link.channel,
        "also_found_on": list(link.also_found_on),
        "geo_scope": link.geo_scope,
        "hypothesis_id": link.hypothesis_id,
        "backend_used": link.backend_used,
        "observed_at": link.observed_at.isoformat() if link.observed_at else None,
        "verdict": link.supports_or_refutes,
        "confidence": link.confidence,
        "signal_tags": link.signal_tags,
        "query": {
            "text": link.query.text,
            "archetype": link.query.archetype,
            "archetype_name": link.query.archetype_name,
            "falsifier": link.query.falsifier,
        },
    }
    if isinstance(link, ShortVideoLink):
        out.update({
            "duration_sec": link.duration_sec,
            "caption": link.caption[:600] if link.caption else "",
            "hashtags": link.hashtags,
            "creator": link.creator,
            "view_count": link.view_count,
            "like_count": link.like_count,
            "comment_count": link.comment_count,
            "share_count": link.share_count,
            "thumbnail_url": link.thumbnail_url,
            "top_comments": link.top_comments[:5],
            "engagement_score": link.engagement_score,
            "is_short_video": True,
        })
    else:
        out["is_short_video"] = False
    return out


# ─── CSV export ──────────────────────────────────────────────────────────────


CSV_COLUMNS: tuple[str, ...] = (
    # Identity
    "hypothesis_id", "verdict", "confidence", "channel", "is_short_video",
    "geo_scope",                          # india | row | unknown
    # URL
    "url", "canonical_url",
    # Content
    "title", "snippet",
    # Short-video metadata (None/blank for long-form links)
    "creator", "duration_sec",
    "view_count", "like_count", "comment_count", "share_count",
    "engagement_score",
    "hashtags", "top_comments", "transcript_preview",
    # Verdict provenance
    "signal_tags",
    # L5 cross-platform provenance (Step 13)
    "also_found_on",
    # Query provenance (L2 → L3)
    "query_text", "archetype", "archetype_name", "falsifier", "target_signal",
    # Pipeline provenance
    "backend_used", "discovered_at", "observed_at",
)

# Inside-cell joiner for list fields (hashtags, top_comments, signal_tags).
# Pipe is robust against commas, newlines, and the CSV escaping process.
_LIST_DELIM = " | "

# Hard cap on transcript preview so Excel doesn't choke on a single 8K-char cell.
_TRANSCRIPT_CHARS = 500


def _join_list(items: Optional[Sequence[str]]) -> str:
    if not items:
        return ""
    return _LIST_DELIM.join(str(x).replace("\r", " ").replace("\n", " ").strip()
                            for x in items if x)


def _fmt_bool(v: Optional[bool]) -> str:
    if v is None:
        return ""
    return "true" if v else "false"


def _fmt_num(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        # Engagement scores can be 6 decimals; trim trailing zeros for readability.
        return f"{v:.6f}".rstrip("0").rstrip(".") or "0"
    return str(v)


def _fmt_dt(v: Optional[datetime]) -> str:
    if v is None:
        return ""
    return v.isoformat()


def _link_to_csv_row(link: DiscoveredLink) -> Dict[str, str]:
    """Flatten a DiscoveredLink (or ShortVideoLink) into the 28-column row.

    All values are strings — `csv.DictWriter` will quote any field containing
    commas/quotes/newlines. None → "". Lists → pipe-joined.
    """
    is_sv = isinstance(link, ShortVideoLink)
    row: Dict[str, str] = {
        "hypothesis_id":   link.hypothesis_id or "",
        "verdict":         link.supports_or_refutes or "",
        "confidence":      _fmt_num(link.confidence),
        "channel":         link.channel,
        "is_short_video":  _fmt_bool(is_sv),
        "geo_scope":       link.geo_scope or "unknown",
        "url":             link.url,
        "canonical_url":   link.canonical_url or link.url,
        "title":           (link.title or "").replace("\r", " ").replace("\n", " "),
        "snippet":         (link.snippet or "").replace("\r", " ").replace("\n", " "),
        "signal_tags":     _join_list(link.signal_tags),
        "also_found_on":   _join_list(link.also_found_on),
        "query_text":      link.query.text if link.query else "",
        "archetype":       str(link.query.archetype) if link.query else "",
        "archetype_name":  link.query.archetype_name if link.query else "",
        "falsifier":       _fmt_bool(link.query.falsifier) if link.query else "",
        "target_signal":   link.query.target_signal if link.query else "",
        "backend_used":    link.backend_used or "",
        "discovered_at":   _fmt_dt(link.discovered_at),
        "observed_at":     _fmt_dt(link.observed_at),
        # Short-video columns default to "" for long-form links
        "creator":            "",
        "duration_sec":       "",
        "view_count":         "",
        "like_count":         "",
        "comment_count":      "",
        "share_count":        "",
        "engagement_score":   "",
        "hashtags":           "",
        "top_comments":       "",
        "transcript_preview": "",
    }
    if is_sv:
        sv = link  # type: ignore[assignment]
        row.update({
            "creator":          sv.creator or "",
            "duration_sec":     _fmt_num(sv.duration_sec),
            "view_count":       _fmt_num(sv.view_count),
            "like_count":       _fmt_num(sv.like_count),
            "comment_count":    _fmt_num(sv.comment_count),
            "share_count":      _fmt_num(sv.share_count),
            "engagement_score": _fmt_num(sv.engagement_score),
            "hashtags":         _join_list(sv.hashtags),
            "top_comments":     _join_list(sv.top_comments),
            "transcript_preview": (
                (sv.transcript or "")[:_TRANSCRIPT_CHARS]
                .replace("\r", " ").replace("\n", " ").strip()
            ),
        })
    return row


def _results_to_csv(links: Sequence[DiscoveredLink]) -> str:
    """Render the 28-column CSV for a sequence of triaged links.

    Begins with a UTF-8 BOM so Excel renders Hindi/Hebrew/CJK characters
    correctly. Empty link list → header row only.
    """
    buf = io.StringIO()
    # UTF-8 BOM for Excel compatibility
    buf.write("﻿")
    writer = csv.DictWriter(
        buf, fieldnames=list(CSV_COLUMNS), extrasaction="ignore",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for lk in links:
        writer.writerow(_link_to_csv_row(lk))
    return buf.getvalue()


def _safe_filename(*parts: str) -> str:
    """Build a filesystem-safe filename component from arbitrary strings."""
    out = []
    for p in parts:
        cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in (p or ""))
        if cleaned:
            out.append(cleaned)
    return "_".join(out) or "outtlyr"


# ─── Routes ──────────────────────────────────────────────────────────────────


@app.post("/api/data-selection/start", response_model=StartResponse)
async def start_job(req: StartRequest) -> StartResponse:
    """Kick off a pipeline job. Returns immediately with a job_id."""
    try:
        window = TimeWindow.from_label(req.window_label)
    except ValueError as e:
        raise HTTPException(400, str(e))

    prefs = None
    if req.backends is not None:
        # Sanitise: only known backend ids, preserve user's order.
        clean = [b for b in req.backends if b in ALL_BACKENDS]
        if not clean:
            raise HTTPException(400, "At least one backend must be enabled.")
        prefs = BackendPreferences(enabled=clean)

    job_id = create_job(
        req.hypothesis,
        window,
        use_llm=req.use_llm,
        max_triage=req.max_triage,
        triage_strictness=req.triage_strictness,
        backend_preferences=prefs,
    )
    state = get_job(job_id)
    assert state is not None
    return StartResponse(
        job_id=job_id,
        status=state.status,
        hypothesis_id=state.hypothesis_id,
        window_label=req.window_label,
    )


@app.get("/api/data-selection/jobs")
async def list_all_jobs() -> Dict[str, Any]:
    """Return summaries for every job in memory (for demo status panel)."""
    return {"jobs": [j.to_summary() for j in list_jobs()]}


@app.get("/api/data-selection/jobs/{job_id}")
async def job_status(job_id: str) -> Dict[str, Any]:
    state = get_job(job_id)
    if state is None:
        raise HTTPException(404, f"unknown job_id: {job_id}")
    return state.to_summary()


@app.get("/api/data-selection/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    """SSE event stream — replays history then streams live events."""
    state = get_job(job_id)
    if state is None:
        raise HTTPException(404, f"unknown job_id: {job_id}")

    async def _gen() -> AsyncIterator[str]:
        async for ev in subscribe(job_id):
            payload = json.dumps(ev.to_dict(), ensure_ascii=False)
            yield f"event: {ev.kind}\ndata: {payload}\n\n"
        # Tell the client cleanly we're done — most browsers will auto-reconnect
        # without this, which would spam the server on a completed job.
        yield "event: stream_end\ndata: {}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering for SSE
        },
    )


async def _resolve_completed_job(job_id: str, wait: bool):
    """Shared lookup + readiness check used by /results and /results.csv."""
    state = get_job(job_id)
    if state is None:
        raise HTTPException(404, f"unknown job_id: {job_id}")
    if state.status not in ("done", "error"):
        if wait:
            state = await await_completion(job_id, timeout=300)
        else:
            raise HTTPException(409, f"job not done yet (status={state.status})")
    if state.status == "error" or state.result is None:
        raise HTTPException(
            500 if state.status == "error" else 404,
            state.error or "no result on a non-error job",
        )
    return state


@app.get("/api/data-selection/jobs/{job_id}/results")
async def job_results(
    job_id: str,
    wait: bool = False,
    download: bool = False,
) -> Response:
    """Return triaged + grouped links once the job has completed.

    If `wait=true`, blocks (asynchronously) until the job is done. Otherwise
    returns 409 if the job is still running.

    If `download=true`, the response carries a `Content-Disposition: attachment`
    header so the browser saves the file instead of rendering it.
    """
    state = await _resolve_completed_job(job_id, wait)
    result = state.result
    assert result is not None  # type narrowing — _resolve_completed_job guarantees

    payload = {
        "job_id": state.job_id,
        "hypothesis_id": state.hypothesis_id,
        "elapsed_sec": round(result.elapsed_sec, 3),
        "verdict_counts": {k: len(v) for k, v in result.grouped.items()},
        "channels_used": list(result.links_by_channel.keys()),
        "channels_skipped": list(result.channels_skipped),
        "cost": result.cost,
        "channel_fits": [
            {"channel": f.channel, "fit_score": f.fit_score,
             "rationale": f.rationale, "expected_signal": f.expected_signal}
            for f in result.channel_fits
        ],
        "queries_by_channel": {
            ch: [
                {"text": q.text, "archetype": q.archetype,
                 "archetype_name": q.archetype_name, "falsifier": q.falsifier}
                for q in qs
            ]
            for ch, qs in result.queries_by_channel.items()
        },
        "decomposition": {
            "primary_entity": result.decomposition.primary_entity,
            "entities": result.decomposition.entities,
            "competitor_anchors": result.decomposition.competitor_anchors,
            "pains": result.decomposition.pains,
            "aspirations": result.decomposition.aspirations,
            "identity_claims": result.decomposition.identity_claims,
            "geo_hints": result.decomposition.geo_hints,
        },
        "grouped": {
            verdict: [_link_to_dict(lk) for lk in links]
            for verdict, links in result.grouped.items()
        },
    }

    if download:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"outtlyr_{_safe_filename(state.hypothesis_id, state.job_id, ts)}.json"
        return Response(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    return Response(
        content=json.dumps(payload, ensure_ascii=False),
        media_type="application/json",
    )


@app.get("/api/data-selection/jobs/{job_id}/results.csv")
async def job_results_csv(job_id: str, wait: bool = False) -> Response:
    """Flat CSV of every triaged link for a completed job.

    29 columns, UTF-8 with BOM (Excel-friendly), list cells pipe-joined.
    Browsers + `curl -O` save it via `Content-Disposition: attachment`.

    Verdicts come back in pipeline order (supports → refutes → tangential,
    each sub-bucket conf-desc) since `result.triaged_links` is the already-
    ranked sequence.
    """
    state = await _resolve_completed_job(job_id, wait)
    result = state.result
    assert result is not None

    csv_text = _results_to_csv(result.triaged_links)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"outtlyr_{_safe_filename(state.hypothesis_id, state.job_id, ts)}.csv"
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/data-selection/jobs/{job_id}/discovered.csv")
async def job_discovered_csv(job_id: str, wait: bool = False) -> Response:
    """Flat CSV of EVERY discovered link, not just the triaged top-N.

    Same 29-column schema as `/results.csv`. Links below the triage budget
    cut have empty `verdict`/`confidence`/`signal_tags` cells (the LLM
    never saw them). Cluster-member links carry the propagated verdict
    from their cluster's representative.

    Rationale: Change #3 — analysts want the BROADER set for downstream
    review/spot-checks, not just the top-N the triage prioritised.
    """
    state = await _resolve_completed_job(job_id, wait)
    result = state.result
    assert result is not None

    # Dedup by URL across channels — a link discovered via 2 channels would
    # appear twice in links_by_channel; clusters already merged it once for
    # the triaged_links path, but `links_by_channel` keeps the originals.
    seen: set[str] = set()
    all_links = []
    for channel_links in result.links_by_channel.values():
        for lk in channel_links:
            key = (lk.canonical_url or lk.url).lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            all_links.append(lk)

    csv_text = _results_to_csv(all_links)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"outtlyr_{_safe_filename(state.hypothesis_id, state.job_id, ts)}_all_discovered.csv"
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/data-selection/registry")
async def registry_status() -> Dict[str, Any]:
    reg = default_registry()
    return {
        "available_channels": list(reg.available_channels()),
    }


# ─── Memory endpoints (Step 14a) — surface persisted history ────────────────


@app.get("/api/data-selection/memory/jobs")
async def memory_jobs(
    hypothesis_id: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """List job snapshots from disk.

    Filters available: `hypothesis_id` (only show runs for a given hyp).
    Sorted by mtime descending (most recent first). Cheap — no full hydration.
    """
    summaries = get_store().list_job_summaries(
        hypothesis_id=hypothesis_id, limit=limit,
    )
    return {"jobs": summaries, "count": len(summaries)}


@app.get("/api/data-selection/memory/batches")
async def memory_batches(limit: int = 50) -> Dict[str, Any]:
    summaries = get_store().list_batch_summaries(limit=limit)
    return {"batches": summaries, "count": len(summaries)}


@app.get("/api/data-selection/memory/hypotheses/{hypothesis_id}")
async def memory_hypothesis_history(hypothesis_id: str) -> Dict[str, Any]:
    """All historical runs for a specific hypothesis_id."""
    summaries = get_store().list_job_summaries(
        hypothesis_id=hypothesis_id, limit=200,
    )
    return {
        "hypothesis_id": hypothesis_id,
        "run_count": len(summaries),
        "runs": summaries,
    }


# ─── Batch upload endpoints ──────────────────────────────────────────────────


async def _read_upload(file: UploadFile) -> str:
    """Decode an uploaded CSV file as UTF-8 (with BOM tolerance)."""
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "uploaded file is empty")
    # Tolerate UTF-8 BOM + decode any plausible encoding the analyst's
    # spreadsheet might have written.
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise HTTPException(400, "unable to decode CSV — UTF-8 expected")


@app.post("/api/data-selection/batch/preview")
async def batch_preview(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Parse a CSV and return a preview WITHOUT starting any jobs.

    UI uses this to show the analyst:
      - how many hypotheses were parsed
      - which core problems they're grouped under
      - which contrarian pairs were detected
      - any row-level parse errors
    User confirms before calling /batch/start.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "expected a .csv file")
    text = await _read_upload(file)
    parsed = parse_hypothesis_csv(text)
    if parsed.hypothesis_count == 0 and parsed.errors:
        # Header-level failure (missing required column, bad CSV) is 400
        # so the UI can show it inline; row-level errors with some hypotheses
        # parsed are 200 with the errors list so the user can decide.
        if any(e.row_index == 0 for e in parsed.errors):
            raise HTTPException(400, parsed.errors[0].message)
    return preview_summary(parsed)


class BatchStartRequest(BaseModel):
    csv_text: str = Field(..., description="Raw CSV content (already validated via /batch/preview)")
    window_label: WindowLabel = "1y"
    use_llm: bool = True
    max_triage: int = Field(10, ge=1, le=30)
    concurrency: int = Field(BATCH_CONCURRENCY, ge=1, le=10)


@app.post("/api/data-selection/batch/start")
async def batch_start(req: BatchStartRequest) -> Dict[str, Any]:
    """Parse + commit. Returns a batch_id immediately; jobs run in background.

    Re-parses the CSV server-side (the preview endpoint can't be trusted
    to have used the same content — keeps it stateless).
    """
    parsed = parse_hypothesis_csv(req.csv_text)
    if parsed.hypothesis_count == 0:
        raise HTTPException(
            400,
            parsed.errors[0].message if parsed.errors else "no hypotheses parsed",
        )
    try:
        batch_id = create_batch(
            parsed,
            default_window_label=req.window_label,
            default_max_triage=req.max_triage,
            use_llm=req.use_llm,
            concurrency=req.concurrency,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    state = get_batch(batch_id)
    assert state is not None
    return {
        "batch_id": batch_id,
        "status": state.status,
        "member_count": len(state.members),
        "core_problem_count": len(state.core_problems),
        "concurrency": state.concurrency,
        "detected_pairs": state.detected_pairs,
        "errors": [
            {"row_index": e.row_index, "message": e.message}
            for e in parsed.errors
        ],
    }


@app.get("/api/data-selection/batches")
async def list_all_batches() -> Dict[str, Any]:
    return {"batches": [b.to_summary() for b in list_batches()]}


@app.get("/api/data-selection/batch/{batch_id}")
async def batch_status(batch_id: str) -> Dict[str, Any]:
    state = get_batch(batch_id)
    if state is None:
        raise HTTPException(404, f"unknown batch_id: {batch_id}")
    summary = state.to_summary()
    summary["members"] = [m.to_summary() for m in state.members]
    summary["core_problems"] = {cp_id: stmt for cp_id, stmt in state.core_problems.items()}
    return summary


@app.get("/api/data-selection/batch/{batch_id}/events")
async def batch_events(batch_id: str) -> StreamingResponse:
    """SSE — merged stream across batch-level events + every member's pipeline events.

    Each event carries a `batch_id` and (where applicable) `hypothesis_id`
    so the client can fan out by member.
    """
    state = get_batch(batch_id)
    if state is None:
        raise HTTPException(404, f"unknown batch_id: {batch_id}")

    async def _gen() -> AsyncIterator[str]:
        async for ev in subscribe_batch(batch_id):
            payload = json.dumps(ev, ensure_ascii=False)
            kind = ev.get("kind", "unknown")
            yield f"event: {kind}\ndata: {payload}\n\n"
        yield "event: stream_end\ndata: {}\n\n"

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _aggregate_batch_links(state: BatchState):
    """Walk every member's job and collect its triaged links + provenance.

    Returns a list of (member, link) tuples in member order, then link
    order within each member.
    """
    out = []
    for member in state.members:
        if not member.job_id:
            continue
        job = get_job(member.job_id)
        if job is None or job.result is None:
            continue
        for link in job.result.triaged_links:
            out.append((member, link))
    return out


@app.get("/api/data-selection/batch/{batch_id}/results.csv")
async def batch_results_csv(batch_id: str, wait: bool = False) -> Response:
    """Aggregated CSV across every hypothesis in the batch.

    Schema = the same 28 columns as /jobs/.../results.csv, plus 2 batch
    columns prepended: `core_problem_id` and `core_problem_statement`.
    Rows are emitted in CSV order (the order the analyst uploaded).
    """
    state = get_batch(batch_id)
    if state is None:
        raise HTTPException(404, f"unknown batch_id: {batch_id}")
    if state.status not in ("done", "error", "partial"):
        if wait:
            state = await await_batch_completion(batch_id, timeout=3600)
        else:
            raise HTTPException(409, f"batch not done yet (status={state.status})")

    # Build the CSV manually so we can prepend the 2 batch-level columns
    pairs = await _aggregate_batch_links(state)
    buf = io.StringIO()
    buf.write("﻿")  # UTF-8 BOM
    batch_cols = ("core_problem_id", "core_problem_statement")
    writer = csv.DictWriter(
        buf, fieldnames=list(batch_cols) + list(CSV_COLUMNS),
        extrasaction="ignore", quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for member, link in pairs:
        row = {
            "core_problem_id": member.core_problem_id,
            "core_problem_statement": state.core_problems.get(member.core_problem_id, ""),
        }
        row.update(_link_to_csv_row(link))
        writer.writerow(row)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"outtlyr_batch_{_safe_filename(state.batch_id, ts)}.csv"
    return Response(
        content=buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/data-selection/batch/{batch_id}/results.json")
async def batch_results_json(
    batch_id: str, wait: bool = False, download: bool = False,
) -> Response:
    """Full batch result JSON — nested by core problem → hypothesis → grouped links."""
    state = get_batch(batch_id)
    if state is None:
        raise HTTPException(404, f"unknown batch_id: {batch_id}")
    if state.status not in ("done", "error", "partial"):
        if wait:
            state = await await_batch_completion(batch_id, timeout=3600)
        else:
            raise HTTPException(409, f"batch not done yet (status={state.status})")

    # Group: core_problem_id → hypothesis_id → {summary, grouped}
    by_cp: Dict[str, Any] = {}
    for member in state.members:
        cp = member.core_problem_id
        by_cp.setdefault(cp, {
            "core_problem_id": cp,
            "core_problem_statement": state.core_problems.get(cp, ""),
            "hypotheses": [],
        })
        hyp_block: Dict[str, Any] = {
            "hypothesis_id": member.hypothesis_id,
            "statement": member.hypothesis.get("statement", ""),
            "status": member.status,
            "error": member.error,
            "window_label": member.window_label,
            "max_triage": member.max_triage,
            "job_id": member.job_id,
        }
        if member.job_id:
            job = get_job(member.job_id)
            if job and job.result is not None:
                hyp_block["verdict_counts"] = {
                    k: len(v) for k, v in job.result.grouped.items()
                }
                hyp_block["elapsed_sec"] = round(job.result.elapsed_sec, 3)
                hyp_block["channels_used"] = list(job.result.links_by_channel.keys())
                hyp_block["grouped"] = {
                    verdict: [_link_to_dict(lk) for lk in links]
                    for verdict, links in job.result.grouped.items()
                }
        by_cp[cp]["hypotheses"].append(hyp_block)

    payload = {
        "batch_id": state.batch_id,
        "status": state.status,
        "created_at": state.created_at.isoformat(),
        "updated_at": state.updated_at.isoformat(),
        "summary": state.to_summary(),
        "detected_pairs": state.detected_pairs,
        "core_problems": list(by_cp.values()),
    }

    if download:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"outtlyr_batch_{_safe_filename(state.batch_id, ts)}.json"
        return Response(
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    return Response(
        content=json.dumps(payload, ensure_ascii=False),
        media_type="application/json",
    )


# ─── Static demo page ────────────────────────────────────────────────────────


_STATIC_DIR = _HERE / "static"
if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
