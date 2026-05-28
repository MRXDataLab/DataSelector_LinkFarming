"""Per-backend health tracker — singleton state for Brave / DDG / Headless.

Plumbed into the pipeline so the UI can show *why* a channel returned 0
links instead of guessing. Previously when Brave hit 402 (monthly quota
exhausted) or headless entered CAPTCHA cooldown, the failure was logged
to stderr and the channel silently produced empty results — a terrible
experience for analysts uploading their first manifest.

Status values
-------------
- ``"ok"``                 — backend is healthy, last call succeeded
- ``"missing_key"``        — API key not set (Brave with no BRAVE_API_KEY)
- ``"not_installed"``      — dependency missing (playwright for headless)
- ``"quota_exhausted"``    — 402 Payment Required (Brave free-tier monthly)
- ``"rate_limited"``       — 429 Too Many Requests
- ``"captcha_cooldown"``   — Google flagged headless; cooldown active
- ``"unavailable"``        — disabled by user preferences or unknown failure
- ``"unknown"``            — never called yet this session

The store is a process-global dict (no Redis). Reset on process restart
which is fine — quota / cooldown state is the operator's concern, not
something we persist across deploys.

Public API:
    report(backend_id, status, **detail)   — backends call this on every call
    snapshot()                              — dict for the /backend-health endpoint
    overall_ok() -> bool                    — quick "are any backends healthy?" check
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

log = logging.getLogger(__name__)

BackendStatus = Literal[
    "ok",
    "missing_key",
    "not_installed",
    "quota_exhausted",
    "rate_limited",
    "captcha_cooldown",
    "unavailable",
    "unknown",
]

# Backend ids tracked. Keep this aligned with `backends/registry.py`
# (`_BACKEND_GETTERS` keys + any official-API backends like YouTube).
_TRACKED_BACKENDS: tuple[str, ...] = (
    "brave",
    "duckduckgo",
    "headless_google",
    "youtube",         # official API — tracked separately for quota panel
)


# ─── State record ────────────────────────────────────────────────────────────


@dataclass
class BackendHealth:
    backend_id: str
    status: BackendStatus = "unknown"
    last_message: str = ""
    last_updated: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # When status is `captcha_cooldown` or `rate_limited`, when (epoch
    # seconds) the backend should be retryable.
    cooldown_until_ts: Optional[float] = None
    # Successful call counters for the session (cheap diagnostics)
    success_count: int = 0
    failure_count: int = 0
    # Backend-specific detail: HTTP code, response body excerpt, etc.
    extra: Dict[str, Any] = field(default_factory=dict)
    # Phase 1.7-D — exponential backoff. Count of consecutive failures
    # since the last successful call. Each consecutive failure doubles
    # the cooldown duration (capped at 60 minutes). Resets to 0 on
    # the first successful probe.
    consecutive_failures: int = 0
    # Last cooldown duration that was applied (so callers / UI can show
    # "next retry uses 600s cooldown" hints).
    last_cooldown_seconds: Optional[int] = None

    def cooldown_seconds_remaining(self) -> int:
        if self.cooldown_until_ts is None:
            return 0
        rem = int(self.cooldown_until_ts - time.time())
        return max(0, rem)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "status": self.status,
            "message": self.last_message,
            "last_updated": self.last_updated.isoformat(),
            "cooldown_until_ts": self.cooldown_until_ts,
            "cooldown_seconds_remaining": self.cooldown_seconds_remaining(),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "consecutive_failures": self.consecutive_failures,
            "last_cooldown_seconds": self.last_cooldown_seconds,
            "extra": self.extra,
        }


# ─── In-process store ────────────────────────────────────────────────────────


_STORE: Dict[str, BackendHealth] = {
    bid: BackendHealth(backend_id=bid) for bid in _TRACKED_BACKENDS
}


def _get(backend_id: str) -> BackendHealth:
    if backend_id not in _STORE:
        _STORE[backend_id] = BackendHealth(backend_id=backend_id)
    return _STORE[backend_id]


# ─── Reporting API ───────────────────────────────────────────────────────────


# Phase 1.7-D — exponential backoff. Doubles per consecutive failure,
# capped. `quota_exhausted` keeps the long monthly cooldown (1hr probe).
_BACKOFF_CAP_SEC = 3600  # 1 hour ceiling
_BACKOFF_MULTIPLIER = 2

# asyncio.Event signalled whenever ANY backend flips to "ok". The hard-wait
# coordinator in `search_with_fallback` awaits on this when all backends
# are blocked, so it wakes up immediately on recovery (no polling needed).
_RECOVERY_EVENT_LOOP: Optional[Any] = None
_RECOVERY_EVENT: Optional[Any] = None


def _signal_recovery() -> None:
    """Pulse the recovery event — wakes any caller hard-waiting in the pool."""
    try:
        import asyncio
        ev = _RECOVERY_EVENT
        loop = _RECOVERY_EVENT_LOOP
        if ev is not None and loop is not None and not loop.is_closed():
            # Must set the event from the loop's thread; if we're already
            # on it, plain .set() works; otherwise schedule via call_soon.
            try:
                if asyncio.get_running_loop() is loop:
                    ev.set()
                    # Immediately reset so the next blocked period can wait
                    # on a fresh event without a stale set state.
                    loop.call_soon(ev.clear)
                    return
            except RuntimeError:
                pass
            loop.call_soon_threadsafe(_pulse_event_callback)
    except Exception:
        pass


def _pulse_event_callback() -> None:
    """Set then immediately clear the recovery event (one wakeup pulse)."""
    if _RECOVERY_EVENT is not None:
        _RECOVERY_EVENT.set()
        _RECOVERY_EVENT.clear()


def init_recovery_event(loop: Any) -> None:
    """Install the per-loop recovery event. Called from app startup."""
    global _RECOVERY_EVENT, _RECOVERY_EVENT_LOOP
    import asyncio
    _RECOVERY_EVENT_LOOP = loop
    _RECOVERY_EVENT = asyncio.Event()


async def wait_for_any_recovery(timeout: Optional[float] = None) -> bool:
    """Block until some backend flips to ok or the timeout elapses.

    Returns True if recovery was signalled, False on timeout.
    """
    import asyncio
    if _RECOVERY_EVENT is None:
        # No event installed (rare) — fall back to polling.
        deadline = time.time() + (timeout or 60.0)
        while time.time() < deadline:
            if overall_ok():
                return True
            await asyncio.sleep(2)
        return False
    try:
        await asyncio.wait_for(
            _RECOVERY_EVENT.wait(),
            timeout=timeout,
        )
        return True
    except asyncio.TimeoutError:
        return False


def report(
    backend_id: str,
    status: BackendStatus,
    *,
    message: str = "",
    cooldown_seconds: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Update a backend's status. Cheap — never raises.

    Backends call this:
      • on construction (``status="missing_key"`` / ``"not_installed"`` / ``"ok"``)
      • on every successful call (``status="ok"``)
      • on HTTP errors (``status="quota_exhausted"`` / ``"rate_limited"``)
      • on CAPTCHA (``status="captcha_cooldown"`` with cooldown_seconds)

    Phase 1.7-D adds exponential backoff: each consecutive failure
    doubles the effective cooldown (capped at 1hr). Resets to base on
    the first successful call.
    """
    try:
        h = _get(backend_id)
        prev_status = h.status
        h.status = status
        h.last_message = message
        h.last_updated = datetime.now(timezone.utc)
        if status == "ok":
            # Successful call → clear cooldown + reset backoff counter.
            h.cooldown_until_ts = None
            h.consecutive_failures = 0
            h.last_cooldown_seconds = None
            h.success_count += 1
            # If we just recovered from a blocked state, signal the pool
            # so any caller hard-waiting can wake up immediately.
            if prev_status in ("captcha_cooldown", "rate_limited",
                               "quota_exhausted", "unavailable",
                               "missing_key", "not_installed", "unknown"):
                _signal_recovery()
        elif status in ("quota_exhausted", "rate_limited",
                        "captcha_cooldown", "unavailable"):
            h.consecutive_failures += 1
            h.failure_count += 1
            # Apply exponential backoff. `cooldown_seconds` is the BASE; we
            # multiply by 2^(consecutive_failures-1), capped at 1hr.
            if cooldown_seconds is not None and cooldown_seconds > 0:
                effective = min(
                    _BACKOFF_CAP_SEC,
                    int(cooldown_seconds * (_BACKOFF_MULTIPLIER ** max(
                        0, h.consecutive_failures - 1))),
                )
                h.cooldown_until_ts = time.time() + effective
                h.last_cooldown_seconds = effective
        # Init-time states ("missing_key", "not_installed") don't count
        # as failures (no successful probe was attempted) — leave counters.
        if extra:
            h.extra.update(extra)
    except Exception as e:  # never let telemetry crash the pipeline
        log.warning("backend_health.report(%s) failed: %s", backend_id, e)


def is_blocked(backend_id: str) -> bool:
    """True when this backend should NOT be called right now (cooldown active
    or hard-failed in this session)."""
    h = _get(backend_id)
    if h.status in ("quota_exhausted", "missing_key", "not_installed"):
        return True
    if h.status in ("captcha_cooldown", "rate_limited"):
        return h.cooldown_seconds_remaining() > 0
    return False


def snapshot() -> Dict[str, Any]:
    """Return a JSON-serialisable view of all backend statuses + summary."""
    backends = {bid: h.to_dict() for bid, h in _STORE.items()}
    ok = any(h.status == "ok" for h in _STORE.values())
    any_blocked = any(
        h.status in ("quota_exhausted", "captcha_cooldown",
                     "rate_limited", "missing_key", "not_installed")
        for h in _STORE.values()
    )
    return {
        "any_healthy": ok,
        "any_blocked": any_blocked,
        "backends": backends,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def overall_ok() -> bool:
    """At least one backend is in ok state and not all blocked."""
    return any(h.status == "ok" for h in _STORE.values())


def list_unavailable_reasons() -> List[Dict[str, str]]:
    """Convenience for the orchestrator's pipeline_start event — list of
    {backend_id, status, message} tuples for backends that ARE NOT ok."""
    out: List[Dict[str, str]] = []
    for bid, h in _STORE.items():
        if h.status in ("quota_exhausted", "captcha_cooldown",
                        "rate_limited", "missing_key", "not_installed",
                        "unavailable"):
            out.append({
                "backend_id": bid,
                "status": h.status,
                "message": h.last_message,
                "cooldown_seconds_remaining": str(h.cooldown_seconds_remaining()),
            })
    return out


def reset() -> None:
    """Test-only — clear all health state."""
    global _STORE
    _STORE = {bid: BackendHealth(backend_id=bid) for bid in _TRACKED_BACKENDS}
