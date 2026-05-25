"""General web articles discoverer — fills the `google_web` channel.

The `google_web` channel was reserved in `ChannelId` from day 1 but had no
discoverer wired up — so general web articles (blog posts, brand pages,
explainer sites, niche review blogs) never made it into the pipeline.
The `news` channel only covers news-vertical results, and `marketplace`
only review-rich domains. This fills the gap.

Routes through `search_with_fallback(vertical="web")` so it honours the
user's backend preferences (Google free → Brave → DDG). When the hypothesis
geo includes India, Brave's `country=IN` bias is automatically applied
(see `backends/brave.py`).

Returns plain `DiscoveredLink`. Triage's long-form path body-fetches the
URL for verdict classification.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from ..backends import get_brave, search_with_fallback
from ..models import DiscoveredLink, TimeWindow, TypedQuery
from ._common import raw_result_to_link
from .base import Discoverer

log = logging.getLogger(__name__)


# Hosts we explicitly DON'T want in this bucket (they belong to other
# channels). Drops them post-search to keep the channels clean.
_EXCLUDED_HOSTS: tuple[str, ...] = (
    "reddit.com",         # → reddit channel
    "quora.com",          # → quora channel
    "youtube.com",        # → youtube / youtube_shorts
    "tiktok.com",         # → tiktok channel
    "substack.com",       # → substack channel
    "trustpilot.com",     # → marketplace
    "amazon.in", "amazon.com",  # → marketplace
    "flipkart.com",       # → marketplace
)


def _host_excluded(url: str) -> bool:
    if not url:
        return True
    try:
        host = re.split(r"^https?://", url, maxsplit=1)[-1].split("/")[0].lower()
    except Exception:
        return False
    if host.startswith("www."):
        host = host[4:]
    return any(host == h or host.endswith("." + h) for h in _EXCLUDED_HOSTS)


class GoogleWebDiscoverer(Discoverer):
    """General web search results — the catch-all for articles + brand sites."""

    channel_id = "google_web"

    def __init__(self) -> None:
        # We're available as long as ONE of Brave / DDG / headless is configured.
        # `search_with_fallback` handles the routing per preferences.
        self.available = get_brave().available  # most common gate

    async def discover(
        self,
        query: TypedQuery,
        window: TimeWindow,
        count: int = 10,
    ) -> List[DiscoveredLink]:
        # Over-fetch then exclude channel-mismatched hosts.
        raw = await search_with_fallback(
            query.text,
            vertical="web",
            count=count * 2,
            window=window,
            min_results=1,
        )
        out: List[DiscoveredLink] = []
        seen: set[str] = set()
        for r in raw:
            if not r.url or _host_excluded(r.url):
                continue
            canonical = r.url.split("#", 1)[0].rstrip("/")
            if canonical in seen:
                continue
            seen.add(canonical)
            link = raw_result_to_link(r, query, self.channel_id, backend_used=r.backend)
            link.url = canonical
            link.canonical_url = canonical
            out.append(link)
            if len(out) >= count:
                break
        return out


_singleton: Optional[GoogleWebDiscoverer] = None


def get_google_web() -> GoogleWebDiscoverer:
    global _singleton
    if _singleton is None:
        _singleton = GoogleWebDiscoverer()
    return _singleton


def reset_google_web() -> None:
    global _singleton
    _singleton = None
