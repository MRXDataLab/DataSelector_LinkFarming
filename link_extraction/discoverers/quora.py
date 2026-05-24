"""Quora discoverer (Step 12) — Brave `site:quora.com` path.

Quora has no public search API. Brave's web vertical already biases its
`forums` mode to Reddit + Quora, but `forums` mixes both — we use an
explicit `site:quora.com` rewrite so this discoverer only returns Quora
content (Reddit content goes through `discoverers/reddit.py`).

Quora URLs canonical form:
    https://www.quora.com/<Question-Title-With-Hyphens>
    https://www.quora.com/<Question-Title>/answer/<Author-Name>

We accept both; the canonical key for dedup is the URL minus any
`/answer/...` suffix (so multiple top answers to the same question
collapse to one DiscoveredLink — the search result already picks the
top-ranked answer's body as the snippet).

v1.1 candidate: headless scrape of the Related Questions sidebar would
expand the per-question graph (1 question → 3-5 related). Punted to
Step 14 polish since it adds another headless-concurrency consumer.
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

_QUORA_HOST_RE = re.compile(r"^https?://(?:www\.|[a-z]{2}\.)?quora\.com/", re.IGNORECASE)
# Strip /answer/<author> suffix so the same Q with N top answers dedups
_ANSWER_SUFFIX_RE = re.compile(r"/answer/[^/?#]+(?=/?$|/?\?|/?#)", re.IGNORECASE)


def _canonical_quora_url(url: str) -> str:
    """Normalise URL: lower-host, drop trailing /answer/<author>, drop fragment."""
    if not url:
        return url
    # Drop fragment
    url = url.split("#", 1)[0]
    # Drop /answer/<author> suffix
    url = _ANSWER_SUFFIX_RE.sub("", url)
    return url.rstrip("/")


class QuoraDiscoverer(Discoverer):
    """Brave-backed Quora discovery."""

    channel_id = "quora"

    def __init__(self) -> None:
        # Brave is the only path; without it we can't do anything useful.
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
            f"{query.text} site:quora.com",
            vertical="web",
            count=count * 2,  # over-fetch since /answer/ dupes collapse
            window=window,
            min_results=1,
        )

        out: List[DiscoveredLink] = []
        seen: set[str] = set()
        for r in raw:
            if not _QUORA_HOST_RE.match(r.url):
                continue
            canonical = _canonical_quora_url(r.url)
            if canonical in seen:
                continue
            seen.add(canonical)
            link = raw_result_to_link(r, query, self.channel_id, backend_used=f"{r.backend}+site:quora.com")
            link.url = canonical
            link.canonical_url = canonical
            out.append(link)
            if len(out) >= count:
                break
        return out


_singleton: Optional[QuoraDiscoverer] = None


def get_quora() -> QuoraDiscoverer:
    global _singleton
    if _singleton is None:
        _singleton = QuoraDiscoverer()
    return _singleton


def reset_quora() -> None:
    global _singleton
    _singleton = None
