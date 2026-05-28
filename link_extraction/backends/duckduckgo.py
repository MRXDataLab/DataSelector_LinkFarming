"""DuckDuckGo backend — FALLBACK (off if USE_DUCKDUCKGO=0).

Requires:  pip install ddgs    (was `duckduckgo-search`, renamed upstream)

The `ddgs` package re-exports the same `DDGS` class with identical
`.text()` / `.news()` / `.videos()` method signatures. We import from
`ddgs` first and fall back to `duckduckgo_search` for backward
compatibility on older installs.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from .. import backend_health
from ..models import QueryVertical, RawResult, TimeWindow
from ..temporal import to_ddg_timelimit
from .base import SearchBackend

log = logging.getLogger(__name__)


class DuckDuckGoBackend(SearchBackend):
    id = "duckduckgo"

    def __init__(self) -> None:
        enabled = os.getenv("USE_DUCKDUCKGO", "1") not in ("0", "false", "False")
        try:
            # New `ddgs` package (renamed from `duckduckgo-search`)
            from ddgs import DDGS  # noqa: F401
            self._pkg_name = "ddgs"
        except ImportError:
            try:
                from duckduckgo_search import DDGS  # noqa: F401
                self._pkg_name = "duckduckgo_search"
            except ImportError:
                log.info(
                    "ddgs not installed; DDG backend disabled "
                    "(pip install ddgs)"
                )
                self.available = False
                backend_health.report(
                    "duckduckgo", "not_installed",
                    message="ddgs package not installed (pip install ddgs)",
                )
                return

        self.available = enabled
        if enabled:
            backend_health.report(
                "duckduckgo", "ok",
                message=f"{self._pkg_name} installed; ready",
            )
        else:
            backend_health.report(
                "duckduckgo", "unavailable",
                message="disabled via USE_DUCKDUCKGO=0",
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
        # Prefer the new `ddgs` package; fall back for legacy installs.
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        # Phase 1.7-A — `verify=False` to bypass corporate-proxy MITM TLS
        # interception. On a clean direct connection this is a no-op; on a
        # corp network the env will inject a self-signed CA into the chain
        # and DDG's `primp` HTTP client refuses it with
        # `[SSL: CERTIFICATE_VERIFY_FAILED]`. Bypass is opt-out via env var
        # `DDG_SSL_VERIFY=1`. Default is bypass-on because most corp envs
        # need it and the search data itself is non-sensitive.
        verify_ssl = os.getenv("DDG_SSL_VERIFY", "0") in ("1", "true", "True")
        ddg_timeout = int(os.getenv("DDG_TIMEOUT_SEC", "20"))
        ddg_kwargs: Dict[str, Any] = {"timeout": ddg_timeout}
        if not verify_ssl:
            ddg_kwargs["verify"] = False

        timelimit = to_ddg_timelimit(window) if window else None
        results: List[RawResult] = []
        try:
            with DDGS(**ddg_kwargs) as ddgs:
                if vertical == "news":
                    rows = list(ddgs.news(query, max_results=count, timelimit=timelimit))
                    results = [
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
                elif vertical == "videos":
                    rows = list(ddgs.videos(query, max_results=count, timelimit=timelimit))
                    results = [
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
                else:
                    q = query
                    if vertical == "forums":
                        q = f"{query} site:reddit.com OR site:quora.com"
                    rows = list(ddgs.text(q, max_results=count, timelimit=timelimit))
                    results = [
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
                backend_health.report(
                    "duckduckgo", "ok",
                    message=f"{vertical}: {len(results)} results",
                )
                return results
        except Exception as e:
            # DDG can throw `RatelimitException` from duckduckgo-search when
            # they detect scraping. Treat that as rate-limited with a short
            # cooldown so subsequent queries fall through to other backends.
            err_str = f"{type(e).__name__}: {e}"
            err_lower = err_str.lower()
            if "ratelimit" in err_lower or "rate limit" in err_lower:
                backend_health.report(
                    "duckduckgo", "rate_limited",
                    message=err_str[:200],
                    cooldown_seconds=120,
                )
            elif "ssl" in err_lower or "certificate" in err_lower:
                # Surface SSL issues with an actionable hint. Auto-retrying
                # is useless until the operator either flips DDG_SSL_VERIFY=0
                # or installs the corp CA bundle, so we use a longer cooldown.
                backend_health.report(
                    "duckduckgo", "unavailable",
                    message=("SSL cert verify failed — corporate proxy likely. "
                             "Set DDG_SSL_VERIFY=0 to bypass (already on by "
                             "default; check env override)."),
                    cooldown_seconds=600,
                )
            else:
                backend_health.report(
                    "duckduckgo", "unavailable",
                    message=err_str[:200],
                    cooldown_seconds=300,
                )
            log.warning("DDG %s failed for %r: %s", vertical, query[:60], e)
            return []
