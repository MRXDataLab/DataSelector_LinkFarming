"""Backend registry + fallback chain.

Per-job priority order is determined by the ambient `BackendPreferences`
(set by the orchestrator before the pipeline runs). Default order is
**Google → Brave → DuckDuckGo** so the free-tier scrape gets first crack;
users can flip individual backends off via the demo UI, in which case
disabled backends are skipped entirely.

Verticals only Google exposes (`paa`, `related`) go straight to headless
even when Google is disabled in prefs — the discoverer is responsible
for handling that case (will simply return [] when Google is off).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ..models import QueryVertical, RawResult, TimeWindow
from .base import SearchBackend
from .brave import BraveBackend
from .duckduckgo import DuckDuckGoBackend
from .headless_google import HeadlessGoogleBackend
from .preferences import current_preferences
from .serpapi_stub import SerpApiStubBackend

log = logging.getLogger(__name__)

# Verticals exclusive to Google (no Brave/DDG equivalent surface)
HEADLESS_ONLY: set[str] = {"paa", "related"}

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
) -> List[RawResult]:
    """Run enabled backends in the user's priority order until `min_results`
    are accumulated.

    Honours the ambient `BackendPreferences` ContextVar (set by the
    orchestrator). Default = Google → Brave → DDG. Disabled backends are
    skipped entirely.

    Results across tiers are merged (deduplicated by URL) so partial
    returns from multiple backends can combine to clear the threshold.

    PAA/Related verticals: only Google exposes them. We try headless if
    it's in the enabled set, otherwise return [] — no alternative source.
    """
    prefs = current_preferences()
    enabled_ids = prefs.backend_ids_in_order  # in user-specified priority order

    if vertical in HEADLESS_ONLY:
        if "headless_google" in enabled_ids:
            return await get_headless().search(query, vertical, count, window)
        log.info("PAA/related requested but google_free is disabled — returning []")
        return []

    accumulated: list[RawResult] = []
    seen_urls: set[str] = set()
    backends_used: list[str] = []

    async def _try(backend: SearchBackend) -> bool:
        """Run backend and merge into accumulated. Return True if quota met."""
        if not backend.available:
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

    for backend_id in enabled_ids:
        getter = _BACKEND_GETTERS.get(backend_id)
        if getter is None:
            continue
        if await _try(getter()):
            return accumulated

    log.debug(
        "Backend chain exhausted [%s]: %s → %d results for %r [%s]",
        ",".join(enabled_ids) or "(none enabled)",
        " → ".join(backends_used) or "(no backend available)",
        len(accumulated),
        query[:50],
        vertical,
    )
    return accumulated
