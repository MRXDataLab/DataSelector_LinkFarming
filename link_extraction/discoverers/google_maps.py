"""Google Maps "local-pack" discoverer (Phase 1.6).

Wraps `search_with_fallback(vertical="local")`. The local-pack is the
3-card place box inline on a regular Google web SERP for location-intent
queries — it surfaces brick-and-mortar businesses (Brigade sales offices,
hotels, restaurants, retail stores) with name + rating + reviews + address
+ a "Website" link.

When the local-pack is empty for a query, headless falls through to the
regular organic results — so this discoverer always returns *something*
useful for the query, never silently dark.

Free — no API key. Headless-only (CAPTCHA-rate limited).

Why this matters for Outtlyr: hypotheses about consumer brands with
physical presence get a layer of unfiltered first-party reviews that
neither Reddit nor Substack carry. For real-estate hypotheses, this
surfaces builder sales-office reviews and project-site reviews that
99acres / NoBroker etc don't aggregate.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from ..backends import search_with_fallback, get_headless
from ..models import DiscoveredLink, TimeWindow, TypedQuery
from ._common import raw_result_to_link
from .base import Discoverer

log = logging.getLogger(__name__)


class GoogleMapsDiscoverer(Discoverer):
    channel_id = "google_maps"

    def __init__(self) -> None:
        # Headless-only.
        self.available = get_headless().available

    async def discover(
        self,
        query: TypedQuery,
        window: TimeWindow,
        count: int = 10,
    ) -> List[DiscoveredLink]:
        if not self.available:
            return []
        # Local-intent queries surface better with "reviews" / "near me" /
        # location appended. Don't mutate the query text if it already has
        # one of those words.
        q_text = (query.text or "").strip()
        if not any(w in q_text.lower() for w in ("review", "near me", "location")):
            q_text = f"{q_text} reviews"
        raw = await search_with_fallback(
            q_text,
            vertical="local",
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
                r, query, self.channel_id, backend_used=f"{r.backend}+local",
            )
            link.url = canonical
            link.canonical_url = canonical
            link.signal_tags = list(link.signal_tags) + ["local_business"]
            out.append(link)
        return out


_singleton: Optional[GoogleMapsDiscoverer] = None


def get_google_maps() -> GoogleMapsDiscoverer:
    global _singleton
    if _singleton is None:
        _singleton = GoogleMapsDiscoverer()
    return _singleton


def reset_google_maps() -> None:
    global _singleton
    _singleton = None
