"""Reddit discoverer — PRAW when credentials are present, Brave fallback otherwise.

Two-mode design so the demo works whether or not the user has provisioned
Reddit API credentials.

**Rich mode (PRAW path)** — preferred when `REDDIT_CLIENT_ID`,
`REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` are all set:

    1. For each TypedQuery, call `reddit.subreddit("all").search(...)` with
       `time_filter` derived from TimeWindow + `sort="relevance"` +
       `limit=count`.
    2. Convert each Submission → DiscoveredLink with title + selftext snippet
       + url + score + comment_count + observed_at (created_utc as datetime).
    3. For the top-N submissions (default 5), fetch up to 3 top-level
       comments and attach them as a snippet extension so triage can read
       discussion sentiment without a body fetch.

**Fallback mode (Brave + site:reddit.com)** — engaged when PRAW is unavailable
or credentials missing:

    Routes through the shared backend chain with vertical="forums" which
    Brave already biases toward Reddit + Quora via the `site:` operator
    rewrite in `backends/brave.py:_web`. No body fetch — title + snippet only.

Both modes return `List[DiscoveredLink]` (not ShortVideoLink). Triage's
long-form path will then fetch URL bodies for the LLM verdict pass.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

from ..backends import search_with_fallback
from ..models import DiscoveredLink, TimeWindow, TypedQuery
from ..temporal import to_reddit_t
from ._common import raw_result_to_link
from .base import Discoverer

log = logging.getLogger(__name__)


# Snippet length cap so PRAW-fetched selftext + comments don't blow past
# what triage's body_char_budget would have asked for anyway.
_SNIPPET_CAP = 1500
_TOP_N_FOR_COMMENTS = 5
_COMMENTS_PER_POST = 3


class RedditDiscoverer(Discoverer):
    """Reddit search with optional PRAW enrichment.

    `mode` reports "praw" if rich mode is wired, "brave_fallback" otherwise.
    `available` is True in either mode — fallback always works as long as
    Brave is configured.
    """

    channel_id = "reddit"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        *,
        force_fallback: bool = False,
    ) -> None:
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID", "").strip()
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET", "").strip()
        self.user_agent = user_agent or os.getenv(
            "REDDIT_USER_AGENT", "outtlyr-data-selection/0.1"
        ).strip()
        self._praw = None  # lazy

        self.mode = "brave_fallback"
        if not force_fallback and self.client_id and self.client_secret:
            try:
                self._praw_module = __import__("praw")
                self.mode = "praw"
            except ImportError:
                log.info("praw not installed — Reddit will use Brave fallback")
                self._praw_module = None
        else:
            self._praw_module = None

        # We're always "available" — the Brave fallback works without keys
        # as long as BRAVE_API_KEY is set (checked at search time).
        self.available = True

    # ── Public API ─────────────────────────────────────────────────────────

    async def discover(
        self,
        query: TypedQuery,
        window: TimeWindow,
        count: int = 10,
    ) -> List[DiscoveredLink]:
        if self.mode == "praw":
            return await asyncio.to_thread(self._discover_praw, query, window, count)
        return await self._discover_brave(query, window, count)

    # ── PRAW (rich) path ───────────────────────────────────────────────────

    def _get_praw_client(self):
        if self._praw is not None:
            return self._praw
        if not self._praw_module:
            return None
        try:
            self._praw = self._praw_module.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
                check_for_async=False,
            )
            self._praw.read_only = True
            return self._praw
        except Exception as e:
            log.warning("praw client init failed: %s — degrading to Brave fallback", e)
            self.mode = "brave_fallback"
            return None

    def _discover_praw(
        self, query: TypedQuery, window: TimeWindow, count: int
    ) -> List[DiscoveredLink]:
        client = self._get_praw_client()
        if client is None:
            return []
        time_filter = to_reddit_t(window)
        try:
            subs = list(
                client.subreddit("all").search(
                    query.text,
                    sort="relevance",
                    time_filter=time_filter,
                    limit=count,
                )
            )
        except Exception as e:
            log.warning("PRAW search failed for %r: %s", query.text[:60], e)
            return []

        links: List[DiscoveredLink] = []
        for i, sub in enumerate(subs):
            try:
                snippet_bits: List[str] = []
                if sub.selftext:
                    snippet_bits.append(sub.selftext[:600])
                snippet_bits.append(
                    f"[r/{sub.subreddit.display_name} · "
                    f"{sub.score} upvotes · {sub.num_comments} comments]"
                )

                # Enrich top-N with top comments (cheap; ~1 API call each)
                if i < _TOP_N_FOR_COMMENTS:
                    try:
                        sub.comment_sort = "top"
                        sub.comments.replace_more(limit=0)  # drop "load more" stubs
                        top_comments = [
                            c.body.strip()
                            for c in sub.comments[:_COMMENTS_PER_POST]
                            if getattr(c, "body", None)
                        ]
                        if top_comments:
                            snippet_bits.append(
                                "Top comments: " + " // ".join(top_comments)
                            )
                    except Exception as e:
                        log.debug("PRAW comment fetch failed for %s: %s", sub.id, e)

                snippet = "\n".join(snippet_bits)[:_SNIPPET_CAP]
                observed = datetime.fromtimestamp(sub.created_utc, tz=timezone.utc)

                links.append(DiscoveredLink(
                    url=f"https://www.reddit.com{sub.permalink}",
                    canonical_url=f"https://www.reddit.com{sub.permalink}",
                    title=sub.title or "",
                    snippet=snippet,
                    channel=self.channel_id,
                    hypothesis_id=query.hypothesis_id,
                    query=query,
                    observed_at=observed,
                    backend_used="praw",
                    signal_tags=[
                        f"subreddit:{sub.subreddit.display_name.lower()}",
                    ],
                ))
            except Exception as e:
                log.debug("PRAW submission convert failed: %s", e)
                continue
        return links

    # ── Brave (fallback) path ──────────────────────────────────────────────

    async def _discover_brave(
        self, query: TypedQuery, window: TimeWindow, count: int
    ) -> List[DiscoveredLink]:
        # Brave's "forums" vertical biases the query to `site:reddit.com OR
        # site:quora.com`. We further filter on the Reddit host so this
        # discoverer doesn't leak Quora results into the Reddit bucket.
        raw = await search_with_fallback(
            query.text,
            vertical="forums",
            count=count * 2,  # over-fetch since we'll filter to reddit only
            window=window,
            min_results=count,
        )
        links: List[DiscoveredLink] = []
        for r in raw:
            host = r.url.split("/")[2].lower() if "://" in r.url else ""
            if "reddit.com" not in host:
                continue
            links.append(raw_result_to_link(
                r, query, self.channel_id,
                backend_used=f"{r.backend}+site:reddit.com",
            ))
            if len(links) >= count:
                break
        return links


# ─── Singleton ───────────────────────────────────────────────────────────────


_singleton: Optional[RedditDiscoverer] = None


def get_reddit() -> RedditDiscoverer:
    global _singleton
    if _singleton is None:
        _singleton = RedditDiscoverer()
    return _singleton


def reset_reddit() -> None:
    global _singleton
    _singleton = None
