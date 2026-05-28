"""Substack discoverer (Step 12) — Brave-backed long-form essay search.

Substack publications live at `<author>.substack.com/p/<slug>`. There's
no official search API; we use Brave with a `site:substack.com` rewrite
(which matches `*.substack.com` subdomains too) and post-filter to ensure
results are actual essay URLs, not author homepages.

Returns plain `DiscoveredLink`. Triage's long-form path body-fetches the
essay for the LLM verdict pass (Substack essays are public-readable, no
paywall on the HTML).
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

# Matches any substack-hosted host: *.substack.com (newsletters) + main
# substack.com (rare, mostly discovery pages).
_SUBSTACK_HOST_RE = re.compile(
    r"^https?://(?:[a-z0-9-]+\.)?substack\.com/", re.IGNORECASE
)
# Essay URL pattern: /<author>.substack.com/p/<slug>
# Author homepages (no /p/) and About pages are filtered out.
_ESSAY_PATH_RE = re.compile(r"/p/[^/?#]+", re.IGNORECASE)


class SubstackDiscoverer(Discoverer):
    """Backend-agnostic Substack essay search via `site:*.substack.com`."""

    channel_id = "substack"

    def __init__(self) -> None:
        # Phase 1 fix: any web-search backend can do `site:` rewriting.
        from ..backends.registry import get_ddg, get_headless
        self.available = (
            get_brave().available
            or get_ddg().available
            or get_headless().available
        )

    async def discover(
        self,
        query: TypedQuery,
        window: TimeWindow,
        count: int = 10,
    ) -> List[DiscoveredLink]:
        if not get_brave().available:
            return []

        raw = await search_with_fallback(
            f"{query.text} site:substack.com",
            vertical="web",
            count=count * 2,
            window=window,
            min_results=1,
        )

        out: List[DiscoveredLink] = []
        seen: set[str] = set()
        for r in raw:
            if not _SUBSTACK_HOST_RE.match(r.url):
                continue
            if not _ESSAY_PATH_RE.search(r.url):
                # Author homepage / About / Archive — not an essay
                continue
            canonical = r.url.split("#", 1)[0].split("?", 1)[0].rstrip("/")
            if canonical in seen:
                continue
            seen.add(canonical)
            link = raw_result_to_link(
                r, query, self.channel_id,
                backend_used=f"{r.backend}+site:substack.com",
            )
            link.url = canonical
            link.canonical_url = canonical
            out.append(link)
            if len(out) >= count:
                break
        return out


_singleton: Optional[SubstackDiscoverer] = None


def get_substack() -> SubstackDiscoverer:
    global _singleton
    if _singleton is None:
        _singleton = SubstackDiscoverer()
    return _singleton


def reset_substack() -> None:
    global _singleton
    _singleton = None
