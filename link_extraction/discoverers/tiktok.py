"""TikTok discoverer (Step 11) — Brave-search backed `site:tiktok.com` path.

TikTok has **no official public search API** for general consumer content.
We use the existing Brave backend with a `site:tiktok.com` query modifier
and parse what the search index already returns: canonical video URLs,
titles/captions, and hashtags embedded in snippets.

What we DO get from search snippets:
  • canonical URL → `creator` + `video_id` (regex-parsed from `@user/video/id`)
  • title (often the literal caption)
  • snippet (often caption + hashtags)
  • hashtags (regex from title+snippet)

What we DON'T get without scraping the page:
  • view_count / like_count / comment_count / share_count
  • top_comments / transcript

→ Basic mode (v1, shipped here): rich text + URL metadata, no counts.
   Triage's short-video path will use title + caption + hashtags as the
   evidence blob for the Gemini verdict.

→ Enriched mode (v1.1, optional): headless Chromium page fetch, parse
   `__UNIVERSAL_DATA_FOR_REHYDRATION__` JSON for live counts. Punted to
   Step 14 polish so we don't add a third tier of headless concurrency
   contention to the demo.

**Instagram Reels** is intentionally NOT built in v1 per the locked
decision in HANDOFF §6.5: lowest-yield short-video surface + most
CAPTCHA-prone. Revisit in v1.1 after TikTok validates the pattern.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from ..backends import get_brave, search_with_fallback
from ..models import ShortVideoLink, TimeWindow, TypedQuery
from .base import ShortVideoDiscoverer

log = logging.getLogger(__name__)


# Canonical TikTok video URL: /@username/video/123456789
_VIDEO_RE = re.compile(
    r"^https?://(?:www\.|m\.)?tiktok\.com/@([A-Za-z0-9._-]+)/video/(\d+)",
    re.IGNORECASE,
)
# TikTok Discover topic page: /discover/topic-slug — Brave indexes these heavily
_DISCOVER_RE = re.compile(
    r"^https?://(?:www\.)?tiktok\.com/discover/([a-zA-Z0-9][a-zA-Z0-9-]*)",
    re.IGNORECASE,
)
# TikTok hashtag page: /tag/hashtagname
_TAG_RE = re.compile(
    r"^https?://(?:www\.)?tiktok\.com/tag/([a-zA-Z0-9_-]+)",
    re.IGNORECASE,
)
# Allow trailing query/fragment but match host first.
_HOST_OK_RE = re.compile(r"^https?://(?:www\.|m\.)?tiktok\.com/", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"#([A-Za-z0-9_À-ɏḀ-ỿ]+)")


class TikTokURLKind:
    """URL-kind discriminator for `_parse_tiktok_url` results."""
    VIDEO = "video"           # /@user/video/id — single canonical video
    DISCOVER = "discover"     # /discover/topic-slug — topical aggregate page
    TAG = "tag"               # /tag/hashtagname — hashtag aggregate page


def _parse_tiktok_url(url: str) -> Optional[tuple[str, str, str]]:
    """Classify a TikTok URL.

    Returns `(kind, id_a, id_b)`:
      - VIDEO    → (creator, video_id)
      - DISCOVER → (slug, "")
      - TAG      → (tagname, "")
    None if unrecognised (profile pages, shortlinks, foreign hosts).
    """
    if not url:
        return None
    m = _VIDEO_RE.match(url)
    if m:
        return (TikTokURLKind.VIDEO, m.group(1), m.group(2))
    m = _DISCOVER_RE.match(url)
    if m:
        return (TikTokURLKind.DISCOVER, m.group(1), "")
    m = _TAG_RE.match(url)
    if m:
        return (TikTokURLKind.TAG, m.group(1), "")
    return None


def _canonical(kind: str, a: str, b: str) -> str:
    """Normalised canonical form — strips query/fragment for dedup keys."""
    if kind == TikTokURLKind.VIDEO:
        return f"https://www.tiktok.com/@{a}/video/{b}"
    if kind == TikTokURLKind.DISCOVER:
        return f"https://www.tiktok.com/discover/{a}"
    if kind == TikTokURLKind.TAG:
        return f"https://www.tiktok.com/tag/{a}"
    return ""


def _extract_hashtags(text: str) -> List[str]:
    if not text:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for h in _HASHTAG_RE.findall(text):
        low = h.lower()
        if low not in seen:
            seen.add(low)
            out.append(low)
    return out


class TikTokDiscoverer(ShortVideoDiscoverer):
    """Brave-backed TikTok discovery (no official API).

    Args:
        strict_videos_only: if True, only emit canonical `@user/video/id`
            URLs. Default False — also emits `/discover/<topic>` and
            `/tag/<name>` aggregate pages, because **Brave's TikTok index
            is dominated by Discover pages** (almost no canonical video
            URLs in practice). Aggregate pages still carry strong topical
            signal for the triage LLM via title + snippet.
    """

    channel_id = "tiktok"

    def __init__(self, strict_videos_only: bool = False) -> None:
        self.strict_videos_only = strict_videos_only
        # Phase 1 fix: `site:tiktok.com` works on any web-search backend.
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
    ) -> List[ShortVideoLink]:
        if not get_brave().available:
            return []

        # Rewrite the query to force TikTok results. Brave honours `site:`
        # in its web vertical. Over-fetch 3x since many results are profile
        # pages / shortlinks that we filter out.
        raw = await search_with_fallback(
            f"{query.text} site:tiktok.com",
            vertical="web",
            count=count * 3,
            window=window,
            min_results=1,
        )

        out: List[ShortVideoLink] = []
        seen_canon: set[str] = set()
        for r in raw:
            if not _HOST_OK_RE.match(r.url):
                continue
            parsed = _parse_tiktok_url(r.url)
            if parsed is None:
                # Profile page, shortlink, or unknown form — skip
                continue
            kind, a, b = parsed
            if self.strict_videos_only and kind != TikTokURLKind.VIDEO:
                continue
            canonical = _canonical(kind, a, b)
            if canonical in seen_canon:
                continue
            seen_canon.add(canonical)

            text_blob = f"{r.title}\n{r.snippet}"
            hashtags = _extract_hashtags(text_blob)
            # For discover/tag pages, harvest the slug as an extra hashtag
            # so triage sees the topic anchor in signal_tags.
            if kind in (TikTokURLKind.DISCOVER, TikTokURLKind.TAG):
                slug = a.replace("-", "").replace("_", "").lower()
                if slug and slug not in hashtags:
                    hashtags.insert(0, slug)

            # Creator field varies by URL kind:
            #   VIDEO    → "@username"
            #   DISCOVER → "@tiktok/discover" (synthetic — aggregate page)
            #   TAG      → "@tiktok/tag"     (synthetic — aggregate page)
            if kind == TikTokURLKind.VIDEO:
                creator = f"@{a}"
            elif kind == TikTokURLKind.DISCOVER:
                creator = "@tiktok/discover"
            else:
                creator = "@tiktok/tag"

            out.append(ShortVideoLink(
                url=canonical,
                canonical_url=canonical,
                title=r.title,
                snippet=r.snippet,
                channel=self.channel_id,
                hypothesis_id=query.hypothesis_id,
                query=query,
                observed_at=r.observed_at,
                backend_used=f"{r.backend}+site:tiktok.com:{kind}",
                # Short-video payload — what we *can* fill without scraping:
                caption=r.snippet or r.title,
                hashtags=hashtags,
                creator=creator,
                # Everything below is None in basic mode (v1.1 enriched mode
                # would fetch the page for view/like/comment counts).
                duration_sec=None,
                view_count=None,
                like_count=None,
                comment_count=None,
                share_count=None,
                thumbnail_url="",
                top_comments=[],
                transcript=None,
                engagement_score=None,
            ))
            if len(out) >= count:
                break
        return out


# ─── Singleton ───────────────────────────────────────────────────────────────

_singleton: Optional[TikTokDiscoverer] = None


def get_tiktok() -> TikTokDiscoverer:
    global _singleton
    if _singleton is None:
        _singleton = TikTokDiscoverer()
    return _singleton


def reset_tiktok() -> None:
    global _singleton
    _singleton = None
