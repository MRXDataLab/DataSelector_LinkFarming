"""Backend registry + fallback chain.

Per-job priority order is determined by the ambient `BackendPreferences`
(set by the orchestrator before the pipeline runs). Default order is
**Google → Brave → DuckDuckGo** so the free-tier scrape gets first crack;
users can flip individual backends off via the demo UI, in which case
disabled backends are skipped entirely.

Per-vertical override (added 2026-05-29) — for the `web`, `news`,
`videos`, and `forums` verticals, **Brave is promoted to the front of
the chain** regardless of the user's default ordering. Brave's API is
faster and immune to the CAPTCHA wall that randomly knocks headless
Google offline mid-batch. For the remaining verticals
(`paa`, `related`, `scholar`, `local`) only headless Google is capable,
so no override applies.

Verticals only Google exposes (`paa`, `related`) go straight to headless
even when Google is disabled in prefs — the discoverer is responsible
for handling that case (will simply return [] when Google is off).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .. import backend_health
from ..models import QueryVertical, RawResult, TimeWindow
from . import _query_cache
from .base import SearchBackend
from .brave import BraveBackend
from .duckduckgo import DuckDuckGoBackend
from .headless_google import HeadlessGoogleBackend
from .preferences import current_preferences
from .serpapi_stub import SerpApiStubBackend

log = logging.getLogger(__name__)

# Verticals exclusive to Google (no Brave/DDG equivalent surface).
# Phase 1.6 adds `scholar` (scholar.google.com) and `local` (Google local
# 3-pack inline in web SERPs).
HEADLESS_ONLY: set[str] = {"paa", "related", "scholar", "local"}

# Verticals where Brave's API beats headless Google on speed + reliability
# (no CAPTCHA risk, structured JSON). User directive 2026-05-29: prefer
# Brave first for these, then headless Google, then DDG. Other verticals
# (PAA/related/scholar/local) are headless-only anyway, so no override
# needed there.
BRAVE_PREFERRED_VERTICALS: set[str] = {"web", "news", "videos", "forums"}


def _reorder_for_vertical(enabled_ids: List[str], vertical: QueryVertical) -> List[str]:
    """Promote `brave` to the front of the chain for Brave-preferred verticals.

    Only reshuffles when Brave is actually enabled; respects the user's full
    on/off selection. Other backends retain their relative order behind Brave.
    """
    if vertical not in BRAVE_PREFERRED_VERTICALS:
        return enabled_ids
    if "brave" not in enabled_ids:
        return enabled_ids
    rest = [b for b in enabled_ids if b != "brave"]
    return ["brave", *rest]

# ── Singletons ──────────────────────────────────────────────────────────────

_BRAVE: Optional[BraveBackend] = None
_DDG: Optional[DuckDuckGoBackend] = None
_HEADLESS: Optional[HeadlessGoogleBackend] = None
_SERPAPI: Optional[SerpApiStubBackend] = None


def get_brave() -> BraveBackend:
    global _BRAVE
    if _BRAVE is None:
        _BRAVE = BraveBackend()
    return _BRAVE


def get_ddg() -> DuckDuckGoBackend:
    global _DDG
    if _DDG is None:
        _DDG = DuckDuckGoBackend()
    return _DDG


def get_headless() -> HeadlessGoogleBackend:
    global _HEADLESS
    if _HEADLESS is None:
        _HEADLESS = HeadlessGoogleBackend()
    return _HEADLESS


def get_serpapi() -> SerpApiStubBackend:
    global _SERPAPI
    if _SERPAPI is None:
        _SERPAPI = SerpApiStubBackend()
    return _SERPAPI


# ── Fallback chain ──────────────────────────────────────────────────────────


_BACKEND_GETTERS: Dict[str, callable] = {
    "headless_google": get_headless,
    "brave":           get_brave,
    "duckduckgo":      get_ddg,
}


async def search_with_fallback(
    query: str,
    vertical: QueryVertical = "web",
    count: int = 10,
    window: Optional[TimeWindow] = None,
    min_results: int = 3,
    *,
    hard_wait_recovery: bool = True,
    hard_wait_max_seconds: Optional[float] = None,
) -> List[RawResult]:
    """Run enabled backends in priority order until `min_results` accumulated.

    Phase 1.7-C — when ALL backends are blocked at call time,
    ``hard_wait_recovery=True`` (default) makes this coroutine pause on
    the global recovery event until *some* backend flips back to ``ok``,
    then retries. ``hard_wait_max_seconds`` caps the wait (None ⇒ no cap,
    matching the user's "hard-wait indefinitely" choice).

    Honours the ambient `BackendPreferences` ContextVar (set by the
    orchestrator). Default = Google → Brave → DDG. Disabled backends are
    skipped entirely.

    PAA / Related / Scholar / Local verticals are headless-only — we
    still hard-wait on headless recovery when it's blocked.
    """
    prefs = current_preferences()
    enabled_ids = prefs.backend_ids_in_order  # priority order
    # Per-vertical override: Brave-first for web/news/videos/forums.
    enabled_ids = _reorder_for_vertical(enabled_ids, vertical)

    # ── Cache lookup ────────────────────────────────────────────────────────
    # Within a batch the same (query, vertical, window, count, priority)
    # tuple is hit many times by different discoverers. The cache returns
    # a copy on hit so callers can mutate freely. Key includes the
    # priority-chain tuple so a Safe-mode result doesn't satisfy a
    # Full-mode lookup (different chain → potentially different links).
    _cache_key = _query_cache.make_key(
        query=query,
        vertical=vertical,
        count=count,
        window=window,
        priority_chain=tuple(enabled_ids),
    )
    _cached = _query_cache.get(_cache_key)
    if _cached is not None:
        log.debug(
            "cache HIT: %s/%s (n=%d) — saved one backend call",
            vertical, query[:40], len(_cached),
        )
        return _cached

    def _emit(results: List[RawResult]) -> List[RawResult]:
        """Store on cache miss, then return. Used at every return point."""
        _query_cache.put(_cache_key, results)
        return results

    # Headless-only verticals — wait for headless to come back if it's blocked.
    if vertical in HEADLESS_ONLY:
        if "headless_google" not in enabled_ids:
            log.info("%s requested but google_free is disabled — returning []", vertical)
            return _emit([])
        while True:
            if not backend_health.is_blocked("headless_google"):
                return _emit(
                    await get_headless().search(query, vertical, count, window)
                )
            if not hard_wait_recovery:
                return _emit([])
            log.info(
                "headless-only vertical %s waiting for headless recovery", vertical,
            )
            recovered = await backend_health.wait_for_any_recovery(
                timeout=hard_wait_max_seconds,
            )
            if not recovered and hard_wait_max_seconds is not None:
                # Timed out — give up rather than spinning forever.
                return _emit([])

    accumulated: list[RawResult] = []
    seen_urls: set[str] = set()
    backends_used: list[str] = []

    async def _try(backend: SearchBackend) -> bool:
        """Run backend and merge into accumulated. Return True if quota met."""
        if not backend.available:
            return False
        # Phase 1.5 fix: respect backend_health.is_blocked() to skip backends
        # that hit a hard failure earlier this session.
        if backend_health.is_blocked(backend.id):
            return False
        out = await backend.search(query, vertical, count, window)
        backends_used.append(f"{backend.id}({len(out)})")
        for r in out:
            if r.url and r.url in seen_urls:
                continue
            if r.url:
                seen_urls.add(r.url)
            accumulated.append(r)
        return len(accumulated) >= min_results

    # Phase 1.7-C — hard-wait loop. Try the priority chain; if every
    # eligible backend is blocked, sleep on the recovery event and retry.
    # `loops_without_progress` guards against pathological cases where
    # the recovery event keeps firing but no backend ever returns data
    # (e.g. all keys missing). After 5 such loops, give up.
    loops_without_progress = 0
    while True:
        starting_count = len(accumulated)
        any_attempted = False
        for backend_id in enabled_ids:
            getter = _BACKEND_GETTERS.get(backend_id)
            if getter is None:
                continue
            backend = getter()
            if not backend.available or backend_health.is_blocked(backend.id):
                continue
            any_attempted = True
            if await _try(backend):
                return _emit(accumulated)

        # If we accumulated anything (even partial), return it.
        if len(accumulated) > 0:
            return _emit(accumulated)

        # Nothing attempted means EVERY eligible backend was blocked.
        if not any_attempted:
            if not hard_wait_recovery:
                break
            if loops_without_progress >= 5:
                log.warning(
                    "search_with_fallback giving up after 5 recovery cycles "
                    "with no progress (all eligible backends still failing)",
                )
                break
            log.info(
                "all backends blocked — waiting for recovery (query=%r, "
                "vertical=%s, attempted=%s)",
                query[:60], vertical, [b.id for b in (
                    getter() for getter in (
                        _BACKEND_GETTERS.get(bid) for bid in enabled_ids
                    ) if getter is not None
                )],
            )
            recovered = await backend_health.wait_for_any_recovery(
                timeout=hard_wait_max_seconds,
            )
            if not recovered and hard_wait_max_seconds is not None:
                break
            loops_without_progress += 1
            continue

        # Some backend was tried, returned 0 or non-quota — bail (not blocked,
        # just empty).
        break

    log.debug(
        "Backend chain exhausted [%s]: %s → %d results for %r [%s]",
        ",".join(enabled_ids) or "(none enabled)",
        " → ".join(backends_used) or "(no backend available)",
        len(accumulated),
        query[:50],
        vertical,
    )
    return _emit(accumulated)
