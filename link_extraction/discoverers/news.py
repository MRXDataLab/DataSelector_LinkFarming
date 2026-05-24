"""News discoverer — wraps Brave News (primary).

For windows ≤ 1y, Brave's news vertical gives strong recency-weighted
coverage with `freshness=pd|pw|pm|py`. For windows > 1y the build plan
calls for a GDELT 2.0 client (no key needed) but that's punted to v1.1 —
this v1 implementation degrades to whatever Brave returns without a
freshness filter (still useful, just less time-bounded).

Returns plain `DiscoveredLink`. Triage's long-form path body-fetches the
article URL when verdict-classifying.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..backends import search_with_fallback, get_brave
from ..models import DiscoveredLink, TimeWindow, TypedQuery
from ._common import raw_result_to_link
from .base import Discoverer

log = logging.getLogger(__name__)


class NewsDiscoverer(Discoverer):
    channel_id = "news"

    def __init__(self) -> None:
        # Brave is the primary news source; without it we can't run at all
        # (headless's `news` vertical works in theory but is CAPTCHA-flaky).
        self.available = get_brave().available

    async def discover(
        self,
        query: TypedQuery,
        window: TimeWindow,
        count: int = 10,
    ) -> List[DiscoveredLink]:
        if not get_brave().available:
            return []
        raw = await search_with_fallback(
            query.text,
            vertical="news",
            count=count,
            window=window,
            min_results=1,
        )
        return [
            raw_result_to_link(r, query, self.channel_id)
            for r in raw
            if r.url
        ]


_singleton: Optional[NewsDiscoverer] = None


def get_news() -> NewsDiscoverer:
    global _singleton
    if _singleton is None:
        _singleton = NewsDiscoverer()
    return _singleton


def reset_news() -> None:
    global _singleton
    _singleton = None
