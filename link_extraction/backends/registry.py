"""Backend registry + Brave → DDG → Headless fallback chain.

The single public function `search_with_fallback()` is what discoverers will
call. Verticals that only Google exposes (`paa`, `related`) bypass tiers 1-2
and go straight to headless.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..models import QueryVertical, RawResult, TimeWindow
from .base import SearchBackend
from .brave import BraveBackend
from .duckduckgo import DuckDuckGoBackend
from .headless_google import HeadlessGoogleBackend
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


async def search_with_fallback(
    query: str,
    vertical: QueryVertical = "web",
    count: int = 10,
    window: Optional[TimeWindow] = None,
    min_results: int = 3,
) -> List[RawResult]:
    """Run Brave → DDG → Headless until at least `min_results` are accumulated.

    Results across tiers are merged (deduplicated by URL) so partial returns
    from Brave + DDG can combine to clear the threshold without escalating to
    headless.

    PAA/Related verticals go straight to headless — neither Brave nor DDG
    expose those surfaces.
    """
    if vertical in HEADLESS_ONLY:
        return await get_headless().search(query, vertical, count, window)

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

    if await _try(get_brave()):
        return accumulated
    if await _try(get_ddg()):
        return accumulated
    if await _try(get_headless()):
        return accumulated

    log.debug(
        "Fallback chain exhausted: %s → %d results for %r [%s]",
        " → ".join(backends_used) or "(no backend available)",
        len(accumulated),
        query[:50],
        vertical,
    )
    return accumulated
