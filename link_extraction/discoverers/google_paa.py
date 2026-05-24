"""Google PAA (People Also Ask) discoverer.

Wraps the existing headless Chromium backend at `vertical="paa"`. PAA is the
only Google SERP surface that exposes *questions consumers actually type
about a topic* — it's gold for hypothesis investigation because each PAA
node carries an accordion answer snippet plus a "Search for" link.

Quirks:
- Headless-only; the registry routes `vertical="paa"` straight to Chromium.
- Slow (~3-6 sec per query depending on PAA depth + CAPTCHA risk).
- Backend's global concurrency semaphore (default 3) caps parallel pages.
- On CAPTCHA, the backend self-disables for 30s and returns []; we propagate
  that as an empty discovery rather than retrying — the cooldown is shared
  across all headless callers (TikTok / Quora to come) so any retry would
  just lengthen the disable window.

Returns plain `DiscoveredLink`. Triage's long-form path will body-fetch the
search-result URL for the LLM verdict pass.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..backends import search_with_fallback, get_headless
from ..models import DiscoveredLink, TimeWindow, TypedQuery
from ._common import raw_result_to_link
from .base import Discoverer

log = logging.getLogger(__name__)


class GooglePAADiscoverer(Discoverer):
    channel_id = "google_paa"

    def __init__(self) -> None:
        # Availability mirrors the headless backend (Chromium installed +
        # not in CAPTCHA cooldown). Re-checked on each discover() call.
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
            vertical="paa",
            count=count,
            window=window,
            min_results=1,
        )
        return [
            raw_result_to_link(r, query, self.channel_id)
            for r in raw
            if r.url
        ]


_singleton: Optional[GooglePAADiscoverer] = None


def get_google_paa() -> GooglePAADiscoverer:
    global _singleton
    if _singleton is None:
        _singleton = GooglePAADiscoverer()
    return _singleton


def reset_google_paa() -> None:
    global _singleton
    _singleton = None
