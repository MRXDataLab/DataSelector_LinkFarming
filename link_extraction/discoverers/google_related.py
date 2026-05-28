"""Google "Related Searches" discoverer.

Sibling of `google_paa.py`. Where PAA exposes *questions* consumers type
about a topic, the Related-Searches strip at the bottom of a Google SERP
exposes *the next-most-popular query reformulations*. Both are gold for
hypothesis investigation because the language is consumer-authored, not
analyst-authored.

Wraps the existing headless Chromium backend at `vertical="related"`. The
backend's `_extract_related()` returns each related-search phrase as a
`RawResult` with `title=<phrase>` and `url=<a href value>` (Google's
related-search anchors point at a `/search?q=<phrase>` URL — clickable).

Each link is tagged `signal_tags=["related_search"]` so downstream
triage / dedup / CSV-export can identify them as query-reformulation
signals rather than article links.
"""
from __future__ import annotations

import logging
from typing import List, Optional
from urllib.parse import quote_plus

from ..backends import search_with_fallback, get_headless
from ..models import DiscoveredLink, TimeWindow, TypedQuery
from .base import Discoverer

log = logging.getLogger(__name__)


def _synthesize_related_url(phrase: str) -> str:
    """Mint a clickable Google search URL for a related-search phrase."""
    return f"https://www.google.com/search?q={quote_plus(phrase)}"


class GoogleRelatedDiscoverer(Discoverer):
    channel_id = "google_related"

    def __init__(self) -> None:
        self.available = get_headless().available

    async def discover(
        self,
        query: TypedQuery,
        window: TimeWindow,
        count: int = 10,
    ) -> List[DiscoveredLink]:
        if not get_headless().available:
            return []
        raw = await search_with_fallback(
            query.text,
            vertical="related",
            count=count,
            window=window,
            min_results=1,
        )

        out: List[DiscoveredLink] = []
        seen: set[str] = set()
        for r in raw:
            phrase = (r.title or "").strip()
            if not phrase or len(phrase) < 3 or len(phrase) > 120:
                continue
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            url = r.url or _synthesize_related_url(phrase)
            link = DiscoveredLink(
                url=url,
                canonical_url=url,
                title=phrase,
                snippet=r.snippet or f"Related search: {phrase}",
                channel=self.channel_id,
                hypothesis_id=query.hypothesis_id,
                query=query,
                backend_used=f"{r.backend}+related",
                signal_tags=["related_search"],
            )
            out.append(link)
        return out


_singleton: Optional[GoogleRelatedDiscoverer] = None


def get_google_related() -> GoogleRelatedDiscoverer:
    global _singleton
    if _singleton is None:
        _singleton = GoogleRelatedDiscoverer()
    return _singleton


def reset_google_related() -> None:
    global _singleton
    _singleton = None
