"""Auto-recovery probe loop for backends (Phase 1.7-B).

A single asyncio background task started at app boot. Every
`PROBE_INTERVAL_SEC` seconds, it inspects every backend whose
`cooldown_seconds_remaining` has hit 0 and dispatches a lightweight probe
query to confirm recovery. On success the backend's health flips to ``ok``
and the global recovery event is signalled (so any hard-waiting caller
wakes up immediately). On failure the cooldown is extended via the
existing exponential-backoff logic in ``backend_health.report``.

Why a background loop instead of inline-on-query checks?
  • Inline checks burn user-facing latency: each probe takes 5-15 sec.
  • A timer-driven loop probes ONCE per backend per cooldown cycle, so
    recovery has near-zero impact on the next real query's response time.
  • Lets the UI show "auto-retrying every 30s" with confidence.

Probe queries are intentionally minimal and content-neutral:
  • Brave        → ``"test"`` against `/web/search` with `count=1`
  • DuckDuckGo   → ``ddgs.text("test", max_results=1)``
  • Headless     → ``"test"`` against `/search?q=test` (no JS-heavy verticals)

These queries cost ~0 against quotas and are clearly distinguishable from
real-user traffic in the logs (we tag the user-agent).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from . import backend_health
from .models import TimeWindow

log = logging.getLogger(__name__)


PROBE_INTERVAL_SEC = 30
PROBE_QUERY_TEXT = "test"

# Backends we know how to probe. `youtube` is excluded — its quota is daily
# and recovers on UTC midnight, not via probe queries.
PROBE_TARGETS: tuple[str, ...] = ("brave", "duckduckgo", "headless_google")


# Module-level handle to the running task so we can cancel it on shutdown
_recovery_task: Optional[asyncio.Task] = None


async def _probe_backend(backend_id: str) -> bool:
    """Fire one minimal query at the backend. Return True on success."""
    try:
        from .backends.registry import get_brave, get_ddg, get_headless
        if backend_id == "brave":
            b = get_brave()
            if not b.api_key:
                return False
            out = await b.search(PROBE_QUERY_TEXT, vertical="web", count=1)
            return len(out) > 0
        elif backend_id == "duckduckgo":
            d = get_ddg()
            if not d.available:
                return False
            out = await d.search(PROBE_QUERY_TEXT, vertical="web", count=1)
            return len(out) > 0
        elif backend_id == "headless_google":
            h = get_headless()
            if not h.available:
                return False
            out = await h.search(PROBE_QUERY_TEXT, vertical="web", count=1)
            return len(out) > 0
    except Exception as e:
        log.info("recovery probe %s failed: %s", backend_id, e)
        return False
    return False


async def _recovery_loop() -> None:
    """Main loop — sleeps `PROBE_INTERVAL_SEC` between scans."""
    log.info(
        "backend_recovery: loop started (probe interval %ds, targets %s)",
        PROBE_INTERVAL_SEC, list(PROBE_TARGETS),
    )
    while True:
        try:
            await _scan_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("backend_recovery: scan iteration failed: %s", e)
        await asyncio.sleep(PROBE_INTERVAL_SEC)


async def _scan_once() -> None:
    """One pass — probe every backend whose cooldown has expired."""
    snap = backend_health.snapshot()
    for backend_id in PROBE_TARGETS:
        h = snap.get("backends", {}).get(backend_id)
        if not h:
            continue
        status = h.get("status")
        # Skip already-healthy backends — nothing to probe.
        if status == "ok":
            continue
        # Skip backends that aren't recoverable via probe (init-time
        # failures need operator action).
        if status in ("missing_key", "not_installed"):
            continue
        # Only probe when cooldown has elapsed. If it hasn't, leave the
        # backend in its blocked state — the cooldown timer is the source
        # of truth.
        remaining = h.get("cooldown_seconds_remaining", 0)
        if remaining > 0:
            continue
        log.info("backend_recovery: probing %s (was %s)", backend_id, status)
        ok = await _probe_backend(backend_id)
        if ok:
            log.info("backend_recovery: %s recovered", backend_id)
            backend_health.report(
                backend_id, "ok",
                message=f"auto-recovery probe succeeded at {time.strftime('%H:%M:%S')}",
            )
        else:
            log.info(
                "backend_recovery: %s still failing; cooldown extended", backend_id,
            )
            # The probe's failure path inside the backend already called
            # backend_health.report(...) with an extended cooldown via
            # exponential backoff. Nothing more to do.


def start(loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    """Install the recovery event + start the loop. Idempotent."""
    global _recovery_task
    if _recovery_task is not None and not _recovery_task.done():
        return
    if loop is None:
        loop = asyncio.get_event_loop()
    backend_health.init_recovery_event(loop)
    _recovery_task = loop.create_task(_recovery_loop(), name="backend_recovery")
    log.info("backend_recovery: task registered")


def stop() -> None:
    """Cancel the recovery loop (called at app shutdown)."""
    global _recovery_task
    if _recovery_task is not None and not _recovery_task.done():
        _recovery_task.cancel()
        _recovery_task = None
