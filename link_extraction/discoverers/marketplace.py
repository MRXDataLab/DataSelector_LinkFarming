"""Marketplace discoverer (Step 12) — review-rich domain search via Brave.

The build plan calls for Brave shopping + Trustpilot scrape; we ship a
simpler v1 that lets Brave's organic web index do the heavy lifting via
`site:` filtering to a curated list of review/marketplace domains.

The legacy pipeline had Amazon in `DOMAIN_BLACKLIST` (per
DATA_SELECTION_MODULE_CONTEXT.md §6). Per locked decision §6.8 marketplace
SHOULD override that blacklist for its own results — Amazon reviews are
prime taste-complaint surface. We include amazon.in (India-first, since
the demo runs against Indian Kellogg's hypotheses) and amazon.com.

Returns plain `DiscoveredLink`. Triage's long-form path body-fetches the
review page (Trustpilot + amazon reviews are HTML-readable).
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

# Order matters — earlier hosts get priority when we cap result count.
# Trustpilot first (purpose-built reviews), then ecommerce.
_MARKETPLACE_HOSTS = (
    "trustpilot.com",
    "amazon.in", "amazon.com",
    "flipkart.com",
    "bestproducts.com",
    "consumerreports.org",
    "influenster.com",
    "makeupalley.com",
)

# Brave honours OR of `site:` operators (max ~6 for index reliability).
# We split into two passes if we have more than that — for v1 keep ≤6.
_BRAVE_SITES = (
    "trustpilot.com",
    "amazon.in", "amazon.com",
    "flipkart.com",
    "consumerreports.org",
    "influenster.com",
)
_SITE_QUERY = " OR ".join(f"site:{h}" for h in _BRAVE_SITES)

_HOST_FROM_URL_RE = re.compile(r"^https?://([^/]+)/", re.IGNORECASE)


def _host_of(url: str) -> str:
    m = _HOST_FROM_URL_RE.match(url or "")
    return m.group(1).lower() if m else ""


def _is_marketplace_url(url: str) -> bool:
    host = _host_of(url)
    if not host:
        return False
    return any(host.endswith(h) for h in _MARKETPLACE_HOSTS)


class MarketplaceDiscoverer(Discoverer):
    """Brave-backed review/marketplace domain search."""

    channel_id = "marketplace"

    def __init__(self) -> None:
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
            f"{query.text} ({_SITE_QUERY})",
            vertical="web",
            count=count * 2,  # over-fetch since we'll host-filter
            window=window,
            min_results=1,
        )

        out: List[DiscoveredLink] = []
        seen: set[str] = set()
        for r in raw:
            if not _is_marketplace_url(r.url):
                continue
            canonical = r.url.split("#", 1)[0].rstrip("/")
            if canonical in seen:
                continue
            seen.add(canonical)
            link = raw_result_to_link(
                r, query, self.channel_id,
                backend_used=f"{r.backend}+marketplace_hosts",
            )
            link.url = canonical
            link.canonical_url = canonical
            # Tag the host so triage signal_tags carry the source brand
            link.signal_tags = list({*link.signal_tags, f"host:{_host_of(r.url)}"})
            out.append(link)
            if len(out) >= count:
                break
        return out


_singleton: Optional[MarketplaceDiscoverer] = None


def get_marketplace() -> MarketplaceDiscoverer:
    global _singleton
    if _singleton is None:
        _singleton = MarketplaceDiscoverer()
    return _singleton


def reset_marketplace() -> None:
    global _singleton
    _singleton = None
