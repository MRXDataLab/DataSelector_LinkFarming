"""DuckDuckGo backend — FALLBACK (off if USE_DUCKDUCKGO=0).

Requires:  pip install duckduckgo-search
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional

from ..models import QueryVertical, RawResult, TimeWindow
from ..temporal import to_ddg_timelimit
from .base import SearchBackend

log = logging.getLogger(__name__)


class DuckDuckGoBackend(SearchBackend):
    id = "duckduckgo"

    def __init__(self) -> None:
        enabled = os.getenv("USE_DUCKDUCKGO", "1") not in ("0", "false", "False")
        try:
            from duckduckgo_search import DDGS  # noqa: F401

            self.available = enabled
        except ImportError:
            log.info(
                "duckduckgo-search not installed; DDG backend disabled "
                "(pip install duckduckgo-search)"
            )
            self.available = False

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
        from duckduckgo_search import DDGS

        timelimit = to_ddg_timelimit(window) if window else None
        try:
            with DDGS() as ddgs:
                if vertical == "news":
                    rows = list(ddgs.news(query, max_results=count, timelimit=timelimit))
                    return [
                        RawResult(
                            url=r.get("url", ""),
                            title=r.get("title", ""),
                            snippet=r.get("body", ""),
                            backend=self.id,
                            vertical="news",
                            raw_metadata={"date": r.get("date", "")},
                        )
                        for r in rows
                        if r.get("url")
                    ]
                if vertical == "videos":
                    rows = list(ddgs.videos(query, max_results=count, timelimit=timelimit))
                    return [
                        RawResult(
                            url=r.get("content") or r.get("url", ""),
                            title=r.get("title", ""),
                            snippet=r.get("description", ""),
                            backend=self.id,
                            vertical="videos",
                            raw_metadata={"duration": r.get("duration", "")},
                        )
                        for r in rows
                        if (r.get("content") or r.get("url"))
                    ]
                q = query
                if vertical == "forums":
                    q = f"{query} site:reddit.com OR site:quora.com"
                rows = list(ddgs.text(q, max_results=count, timelimit=timelimit))
                return [
                    RawResult(
                        url=r.get("href", ""),
                        title=r.get("title", ""),
                        snippet=r.get("body", ""),
                        backend=self.id,
                        vertical=vertical,
                    )
                    for r in rows
                    if r.get("href")
                ]
        except Exception as e:
            log.warning("DDG %s failed for %r: %s", vertical, query[:60], e)
            return []
