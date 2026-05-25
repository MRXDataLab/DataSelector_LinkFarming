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

# Three categories, each tagged separately so analysts can filter the CSV.
# Categories are exposed as `host_category:<name>` signal tags on each link.

_REVIEW_HOSTS = (
    "trustpilot.com",
    "consumerreports.org", "bestproducts.com",
    "influenster.com", "makeupalley.com",
)

# Ecommerce — Indian sites first (most relevant when geo=india), then global.
_ECOM_HOSTS_INDIA = (
    "flipkart.com", "myntra.com", "nykaa.com", "ajio.com",
    "tatacliq.com", "jiomart.com", "croma.com", "snapdeal.com",
    "meesho.com", "firstcry.com", "lenskart.com", "pepperfry.com",
    "purplle.com", "limeroad.com", "shopclues.com",
)
_ECOM_HOSTS_GLOBAL = (
    "amazon.in", "amazon.com", "amazon.co.uk",
    "etsy.com", "ebay.com", "walmart.com", "target.com", "bestbuy.com",
)

# Quick commerce / grocery / food delivery — Indian sites dominate this
# category (US has Instacart but the analyst's stated need is India).
_QUICK_COMMERCE_HOSTS = (
    "zepto.in", "blinkit.com", "swiggy.com", "instamart.com",
    "dunzo.com", "bigbasket.com", "zomato.com",
)

# Union for the host-allow filter (anything in any category passes).
_MARKETPLACE_HOSTS = (
    *_REVIEW_HOSTS,
    *_ECOM_HOSTS_INDIA,
    *_ECOM_HOSTS_GLOBAL,
    *_QUICK_COMMERCE_HOSTS,
)


def _host_category(host: str) -> str:
    """Return a signal-tag-friendly category for a host."""
    if any(host.endswith(h) for h in _QUICK_COMMERCE_HOSTS):
        return "quick_commerce"
    if any(host.endswith(h) for h in _ECOM_HOSTS_INDIA):
        return "ecom_india"
    if any(host.endswith(h) for h in _ECOM_HOSTS_GLOBAL):
        return "ecom_global"
    if any(host.endswith(h) for h in _REVIEW_HOSTS):
        return "reviews"
    return "other"


# Brave's `site:` OR clauses hit index reliability past ~6 alternations.
# We split the search into multiple passes: one per category, then merge.
_BRAVE_SITE_GROUPS = (
    ("reviews",        _REVIEW_HOSTS[:5]),
    ("ecom_india",     _ECOM_HOSTS_INDIA[:6]),
    ("ecom_global",    _ECOM_HOSTS_GLOBAL[:5]),
    ("quick_commerce", _QUICK_COMMERCE_HOSTS[:6]),
)


def _build_site_query(hosts: tuple[str, ...]) -> str:
    return " OR ".join(f"site:{h}" for h in hosts)

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
    """Brave-backed reviews + ecommerce + quick-commerce domain search.

    Fires up to 4 Brave queries (one per category group) so the `site:OR`
    chain stays under Brave's reliable-alternation count. Results are merged,
    each link gets `host_category:<reviews|ecom_india|ecom_global|quick_commerce>`
    in its signal_tags so analysts can filter / split in the CSV.

    The total result count is split proportionally across category groups
    (default: count/4 per group, then over-fetch by 2× and trim).
    """

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

        per_group_count = max(2, count // len(_BRAVE_SITE_GROUPS))
        out: List[DiscoveredLink] = []
        seen: set[str] = set()
        # Run categories sequentially so we don't burst Brave; each category
        # query is small.
        for cat_name, host_list in _BRAVE_SITE_GROUPS:
            site_q = _build_site_query(host_list)
            raw = await search_with_fallback(
                f"{query.text} ({site_q})",
                vertical="web",
                count=per_group_count * 2,
                window=window,
                min_results=1,
            )
            for r in raw:
                if not _is_marketplace_url(r.url):
                    continue
                canonical = r.url.split("#", 1)[0].rstrip("/")
                if canonical in seen:
                    continue
                seen.add(canonical)
                link = raw_result_to_link(
                    r, query, self.channel_id,
                    backend_used=f"{r.backend}+{cat_name}",
                )
                link.url = canonical
                link.canonical_url = canonical
                host = _host_of(r.url)
                # Tag both the specific host AND the category for filtering.
                link.signal_tags = list({
                    *link.signal_tags,
                    f"host:{host}",
                    f"host_category:{_host_category(host)}",
                })
                out.append(link)
                if len(out) >= count:
                    return out
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
