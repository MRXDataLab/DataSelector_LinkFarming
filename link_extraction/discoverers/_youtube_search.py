"""Search-backed YouTube discovery — zero YT API quota burn.

Used by both `youtube_shorts.py` and `youtube.py` as the *primary* discovery
path. Uses `search_with_fallback(vertical="videos")` to ride free headless
Google + DDG + Brave video search instead of the YouTube Data API. YT API
remains available as an enrichment hook (comments / transcript for the
top-N triaged candidates only — not for discovery).

The trade-off: search results lack view/like/comment counts. For discovery
purposes — and especially for `skip_triage=True` runs — that's enough.

URL shapes we recognize:
    https://www.youtube.com/watch?v=<11-char-id>
    https://www.youtube.com/shorts/<11-char-id>
    https://youtu.be/<11-char-id>
    https://m.youtube.com/watch?v=<11-char-id>
    https://www.youtube.com/embed/<11-char-id>

The thumbnail URL is deterministic — `https://i.ytimg.com/vi/<id>/hqdefault.jpg`
always resolves for any public video without an API call.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from ..backends import search_with_fallback
from ..models import ShortVideoLink, TimeWindow, TypedQuery

log = logging.getLogger(__name__)


# Match the 11-char YT video id in any common URL shape.
_VIDEO_ID_RE = re.compile(
    r"(?:"
    r"youtube\.com/watch\?(?:[^#]*?&)?v=([A-Za-z0-9_-]{11})"
    r"|youtube\.com/shorts/([A-Za-z0-9_-]{11})"
    r"|youtube\.com/embed/([A-Za-z0-9_-]{11})"
    r"|youtu\.be/([A-Za-z0-9_-]{11})"
    r")",
    re.IGNORECASE,
)

# Match a duration string in a search snippet ("0:48", "12:34", "1:02:34")
# Some Brave / DDG snippets prefix it with "Duration: " or wrap in parens.
_DURATION_RE = re.compile(
    r"(?:^|[\s(])(\d{1,2}):(\d{2})(?::(\d{2}))?(?=[\s)]|$)"
)


def extract_video_id(url: str) -> Optional[str]:
    """Return the 11-char video id, or None if the URL is not a YT video."""
    if not url:
        return None
    m = _VIDEO_ID_RE.search(url)
    if not m:
        return None
    return next(g for g in m.groups() if g)


def parse_duration_sec(snippet: str, raw_meta: Optional[dict] = None) -> Optional[int]:
    """Best-effort duration extraction from a search-result snippet / metadata.

    Brave's `videos` vertical returns `duration` in raw_metadata; headless
    Google often inlines `"3:42 ·"` at the start of the snippet; DDG returns
    a `duration` key in its raw row. Try the structured field first.
    """
    if raw_meta:
        d = raw_meta.get("duration")
        if isinstance(d, str):
            m = _DURATION_RE.search(d)
            if m:
                return _hms_to_sec(m.group(1), m.group(2), m.group(3))
        elif isinstance(d, (int, float)):
            return int(d)
    if snippet:
        m = _DURATION_RE.search(snippet)
        if m:
            return _hms_to_sec(m.group(1), m.group(2), m.group(3))
    return None


def _hms_to_sec(h_or_m: str, m_or_s: str, s: Optional[str]) -> int:
    """Convert a matched 2- or 3-component time to seconds.

    The first regex group is hours if 3 components are present, otherwise
    minutes. So we normalize by checking whether `s` is set.
    """
    if s is not None:
        h, m, sec = int(h_or_m), int(m_or_s), int(s)
        return h * 3600 + m * 60 + sec
    m, sec = int(h_or_m), int(m_or_s)
    return m * 60 + sec


def _shorts_url(video_id: str) -> str:
    return f"https://www.youtube.com/shorts/{video_id}"


def _watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _thumbnail_url(video_id: str) -> str:
    """Deterministic thumbnail URL — works without any API call."""
    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"


# ─── Public discovery entry ─────────────────────────────────────────────────


async def youtube_via_search(
    query: TypedQuery,
    window: TimeWindow,
    count: int = 10,
    *,
    shorts_only: bool = False,
    long_only: bool = False,
    channel_id: str = "youtube",
) -> List[ShortVideoLink]:
    """Discover YouTube videos via search backends — no YT API call.

    Args:
        query: the typed query (text gets a `site:youtube.com[/shorts]` filter
            appended).
        window: time window passed to the search backend.
        count: target number of unique videos to return.
        shorts_only: append `site:youtube.com/shorts` to bias to Shorts URLs.
        long_only: append `site:youtube.com/watch -inurl:shorts` to bias to
            long-form (Watch) URLs. Mutually exclusive with `shorts_only`.
        channel_id: the channel label to stamp on returned links
            (`youtube_shorts` or `youtube`).

    Returns:
        A list of `ShortVideoLink` (no view/like/comment counts — those come
        from the YT API enrichment hook only when triage is on). Thumbnail
        URL is filled in deterministically.
    """
    if shorts_only and long_only:
        raise ValueError("shorts_only and long_only are mutually exclusive")

    # IMPORTANT — we do NOT append a `site:youtube.com[/shorts]` filter to the
    # query. DDG's videos vertical doesn't honor the `site:` operator at all
    # and returns 0 results. Headless Google DOES honor it but biasing
    # toward Shorts via URL pattern post-filter works better across all
    # backends. So: send the plain query, get any videos, then filter to
    # YouTube hosts + the desired URL shape (watch vs shorts) below.
    q_text = (query.text or "").strip()

    raw = await search_with_fallback(
        q_text,
        vertical="videos",
        count=count * 3,   # over-fetch since YT-host + shape filter trims hard
        window=window,
        min_results=1,
    )

    # Shorts cutoff — YouTube counts videos ≤ 60s as Shorts. We use 180s
    # for the search-mode filter because (a) some 60-180s vlogs are
    # genuinely shorts-format (vertical, snack content), and (b) durations
    # parsed from search snippets are sometimes rounded. The L4 client-
    # side filter in the orchestrator can re-tighten if needed.
    SHORTS_MAX_DURATION = 180

    out: List[ShortVideoLink] = []
    seen_ids: set[str] = set()
    for r in raw:
        vid = extract_video_id(r.url)
        if not vid:
            # Some backends return non-YT video sources (Vimeo, etc.) in the
            # videos vertical — skip.
            continue
        if vid in seen_ids:
            continue

        # Detect shape — Shorts URLs vs Watch URLs, plus duration heuristic.
        url_says_shorts = "/shorts/" in (r.url or "")
        duration_sec = parse_duration_sec(r.snippet, r.raw_metadata)
        # Phase 4 fix — when shorts_only is requested, accept the video if
        # ANY of these is true:
        #   • URL is /shorts/<id> (explicit Shorts URL)
        #   • duration ≤ SHORTS_MAX_DURATION (whatever the URL shape)
        #   • duration unknown (no signal in snippet — give it a chance;
        #     L4 / triage can drop later if it turns out to be long-form)
        # That's the inverse of the previous URL-only filter that dropped
        # 100% of DDG-returned videos (DDG mostly returns watch?v= URLs
        # for Shorts content).
        if shorts_only:
            looks_short = (
                url_says_shorts
                or (duration_sec is not None and duration_sec <= SHORTS_MAX_DURATION)
                or duration_sec is None
            )
            if not looks_short:
                continue
        # Long-only: drop URLs that are clearly Shorts (by URL OR by
        # duration ≤ 60s). Leave unknown durations as long-form
        # candidates.
        if long_only:
            if url_says_shorts:
                continue
            if duration_sec is not None and duration_sec <= 60:
                continue

        seen_ids.add(vid)

        # Canonical URL — collapse all URL shapes to a single form per
        # video so L5 dedup correctly merges duplicates from different
        # backends. We use the canonical Shorts URL when EITHER the
        # source URL or duration suggests Shorts content.
        is_short_canonical = (
            shorts_only
            or url_says_shorts
            or (duration_sec is not None and duration_sec <= SHORTS_MAX_DURATION)
        )
        canonical = _shorts_url(vid) if is_short_canonical else _watch_url(vid)

        link = ShortVideoLink(
            url=canonical,
            canonical_url=canonical,
            title=r.title or "",
            snippet=r.snippet or "",
            channel=channel_id,
            hypothesis_id=query.hypothesis_id,
            query=query,
            backend_used=f"{r.backend}+yt_search",
            duration_sec=duration_sec,
            thumbnail_url=_thumbnail_url(vid),
            signal_tags=["yt_search_mode"],  # marks "not API-enriched"
        )
        out.append(link)
        if len(out) >= count:
            break

    log.debug(
        "youtube_via_search(%s, shorts=%s, long=%s) → %d/%d unique videos",
        query.text[:50], shorts_only, long_only, len(out), count,
    )
    return out
