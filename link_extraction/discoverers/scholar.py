"""Google Scholar discoverer (Phase 1.6).

Wraps `search_with_fallback(vertical="scholar")` which routes through the
headless Chromium backend to `scholar.google.com`. Returns academic /
research-paper links for hypotheses where citation evidence matters
(wellness, public health, cultural identity, policy, behavioral econ).

Free — no API key. Rate-limited like the rest of headless Google
(5-minute CAPTCHA cooldown shared across all headless callers).

The signal tag `scholar` is attached to every result so analysts can
filter / weight academic citations differently in downstream triage.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..backends import search_with_fallback, get_headless
from ..models import DiscoveredLink, TimeWindow, TypedQuery
from ._common import raw_result_to_link
from .base import Discoverer

log = logging.getLogger(__name__)


class ScholarDiscoverer(Discoverer):
    channel_id = "scholar"

    def __init__(self) -> None:
        # Headless-only — Brave / DDG don't expose Scholar.
        self.available = get_headless().available

    async def discover(
        self,
        query: TypedQuery,
        window: TimeWindow,
        count: int = 10,
    ) -> List[DiscoveredLink]:
        if not self.available:
            return []
        raw = await search_with_fallback(
            query.text,
            vertical="scholar",
            count=count,
            window=window,
            min_results=1,
        )
        out: List[DiscoveredLink] = []
        seen: set[str] = set()
        for r in raw:
            if not r.url:
                continue
            canonical = r.url.split("#", 1)[0].rstrip("/")
            if canonical in seen:
                continue
            seen.add(canonical)
            link = raw_result_to_link(
                r, query, self.channel_id, backend_used=f"{r.backend}+scholar",
            )
            link.url = canonical
            link.canonical_url = canonical
            link.signal_tags = list(link.signal_tags) + ["academic"]
            # Citation meta (authors, year, journal) lands in raw_metadata.
            cm = (r.raw_metadata or {}).get("citation_meta")
            if cm:
                link.signal_tags.append(f"citation:{cm[:80]}")
            out.append(link)
        return out


_singleton: Optional[ScholarDiscoverer] = None


def get_scholar() -> ScholarDiscoverer:
    global _singleton
    if _singleton is None:
        _singleton = ScholarDiscoverer()
    return _singleton


def reset_scholar() -> None:
    global _singleton
    _singleton = None
