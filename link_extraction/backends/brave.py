"""Brave Search API backend — PRIMARY for web/news/forums/videos.

Free tier: 2 000 queries/month. Rate limit ~1 req/sec.
Endpoints used:
    web    : /res/v1/web/search
    news   : /res/v1/news/search   (+freshness)
    videos : /res/v1/videos/search
    forums : web with appended `site:` operator
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional
from urllib.parse import quote_plus

import requests

from .. import backend_health
from ..models import QueryVertical, RawResult, TimeWindow
from ..temporal import to_brave_params
from .base import SearchBackend

# Map decomposer geo_hints → Brave's `country` parameter (ISO 3166-1 alpha-2).
# Conservative — only includes countries whose Brave-search index is rich
# enough that biasing yields more relevant results vs the default global mix.
_GEO_HINT_TO_BRAVE_COUNTRY: dict[str, str] = {
    "india": "IN", "indian": "IN", "bharat": "IN",
    "us": "US", "usa": "US", "american": "US",
    "uk": "GB", "british": "GB", "england": "GB",
    "canada": "CA", "canadian": "CA",
    "australia": "AU", "australian": "AU",
    "germany": "DE", "german": "DE",
    "france": "FR", "french": "FR",
    "japan": "JP", "japanese": "JP",
}


def _brave_country_for_geo_hints(geo_hints: Optional[List[str]]) -> Optional[str]:
    """Pick the first Brave-supported country code from a list of geo hints."""
    if not geo_hints:
        return None
    for h in geo_hints:
        code = _GEO_HINT_TO_BRAVE_COUNTRY.get((h or "").strip().lower())
        if code:
            return code
    return None

log = logging.getLogger(__name__)


class BraveBackend(SearchBackend):
    id = "brave"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("BRAVE_API_KEY", "")
        self.available = bool(self.api_key)
        # Report initial health state so the UI shows missing-key
        # before any search is fired.
        if not self.available:
            backend_health.report(
                "brave", "missing_key",
                message="BRAVE_API_KEY env var not set",
            )
        else:
            backend_health.report(
                "brave", "ok",
                message="API key configured; no calls yet",
            )

    async def search(
        self,
        query: str,
        vertical: QueryVertical = "web",
        count: int = 10,
        window: Optional[TimeWindow] = None,
    ) -> List[RawResult]:
        if not self.available:
            return []
        return await asyncio.to_thread(self._search_sync, query, vertical, count, window)

    # ── private ────────────────────────────────────────────────────────────

    def _search_sync(
        self,
        query: str,
        vertical: QueryVertical,
        count: int,
        window: Optional[TimeWindow],
    ) -> List[RawResult]:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        freshness = (to_brave_params(window) or {}).get("freshness") if window else None

        try:
            if vertical == "news":
                results = self._news(query, count, freshness, headers)
            elif vertical == "videos":
                results = self._videos(query, count, headers)
            else:
                results = self._web(query, vertical, count, freshness, headers)
            backend_health.report(
                "brave", "ok",
                message=f"{vertical}: {len(results)} results",
            )
            return results
        except requests.HTTPError as e:
            code = e.response.status_code
            # 402 Payment Required = free-tier monthly quota exhausted.
            # 429 = rate limit. Surface these distinctly so the operator
            # knows it's a quota / billing issue, not a query problem.
            hint = ""
            if code == 402:
                hint = " — Brave free-tier MONTHLY quota exhausted; upgrade plan or switch backend"
                # Quota is monthly — block for 1 hour, then re-probe.
                # If the operator upgrades the plan mid-session a fresh
                # call will refresh health to ok on first success.
                backend_health.report(
                    "brave", "quota_exhausted",
                    message=f"HTTP 402 on {vertical} query{hint}",
                    cooldown_seconds=3600,
                    extra={"http_code": 402, "vertical": vertical},
                )
            elif code == 429:
                hint = " — Brave rate-limited; slow request rate"
                backend_health.report(
                    "brave", "rate_limited",
                    message=f"HTTP 429 on {vertical} query",
                    cooldown_seconds=60,
                    extra={"http_code": 429, "vertical": vertical},
                )
            else:
                backend_health.report(
                    "brave", "unavailable",
                    message=f"HTTP {code} on {vertical}",
                    extra={"http_code": code},
                )
            log.warning("Brave %s HTTP %s for %r%s", vertical, code, query[:60], hint)
            return []
        except Exception as e:
            backend_health.report(
                "brave", "unavailable",
                message=f"{type(e).__name__}: {e}",
            )
            log.warning("Brave %s failed for %r: %s", vertical, query[:60], e)
            return []

    def _web(
        self,
        query: str,
        vertical: QueryVertical,
        count: int,
        freshness: Optional[str],
        headers: dict,
    ) -> List[RawResult]:
        q = query
        if vertical == "forums":
            q = f"{query} (site:reddit.com OR site:quora.com)"
        url = f"https://api.search.brave.com/res/v1/web/search?q={quote_plus(q)}&count={count}"
        if freshness:
            url += f"&freshness={freshness}"
        # Country bias: when the hypothesis's geo_hints map to a Brave-
        # supported country code, append `country=XX` so the index biases
        # toward region-specific results. India-first hypotheses get many
        # more .in / hindustantimes / flipkart / etc. results this way.
        try:
            from ..geo import current_geo_hints
            country = _brave_country_for_geo_hints(current_geo_hints())
            if country:
                url += f"&country={country}"
        except Exception:
            pass
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        return [
            RawResult(
                url=it.get("url", ""),
                title=it.get("title", ""),
                snippet=it.get("description", ""),
                backend=self.id,
                vertical=vertical,
                raw_metadata={"page_age": it.get("page_age", "")},
            )
            for it in data.get("web", {}).get("results", [])[:count]
            if it.get("url")
        ]

    def _news(
        self,
        query: str,
        count: int,
        freshness: Optional[str],
        headers: dict,
    ) -> List[RawResult]:
        url = (
            f"https://api.search.brave.com/res/v1/news/search"
            f"?q={quote_plus(query)}&count={count}"
        )
        if freshness:
            url += f"&freshness={freshness}"
        try:
            from ..geo import current_geo_hints
            country = _brave_country_for_geo_hints(current_geo_hints())
            if country:
                url += f"&country={country}"
        except Exception:
            pass
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        return [
            RawResult(
                url=it.get("url", ""),
                title=it.get("title", ""),
                snippet=it.get("description", ""),
                backend=self.id,
                vertical="news",
                raw_metadata={
                    "age": it.get("age", ""),
                    "page_age": it.get("page_age", ""),
                },
            )
            for it in data.get("results", [])[:count]
            if it.get("url")
        ]

    def _videos(self, query: str, count: int, headers: dict) -> List[RawResult]:
        url = (
            f"https://api.search.brave.com/res/v1/videos/search"
            f"?q={quote_plus(query)}&count={count}"
        )
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        return [
            RawResult(
                url=it.get("url", ""),
                title=it.get("title", ""),
                snippet=it.get("description", ""),
                backend=self.id,
                vertical="videos",
                raw_metadata={"duration": (it.get("video") or {}).get("duration", "")},
            )
            for it in data.get("results", [])[:count]
            if it.get("url")
        ]
