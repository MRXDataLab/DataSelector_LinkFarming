"""Step 6 — YouTube Shorts discoverer (PRIORITY #1).

This is the hero channel for the Day-10 demo: highest cultural-signal density
for Gen-Z identity/aspiration hypotheses, richest engagement metadata, and
the surface where falsifier/counter queries reveal the most.

Wire pipeline per query:

    1. youtube.search.list (cost 100)        — q + videoDuration=short
                                                 + publishedAfter/Before
                                                 + (optional regionCode)
    2. youtube.videos.list (cost 1 / call)   — snippet + contentDetails
                                                 + statistics for all IDs
                                                 (batched, 50 per call)
    3. youtube.commentThreads.list (cost 1)  — top relevant comments per
                                                 top-N enrichment videos
    4. youtube-transcript-api                — caption transcript (no quota)
                                                 EN preferred, HI fallback,
                                                 then any available lang

Quota math (default settings, 1 query): ~106 units. With 10K daily quota
that's ~94 queries/day before throttling. The enrichment pass is capped at
`top_n_enrichment=5` to keep cost bounded.

Returns `ShortVideoLink` with the full metadata payload:
    duration_sec, caption, hashtags, creator, view/like/comment counts,
    thumbnail_url, top_comments, transcript, engagement_score.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import requests

from ..models import ShortVideoLink, TimeWindow, TypedQuery
from ..temporal import to_youtube_params
from .base import ShortVideoDiscoverer

log = logging.getLogger(__name__)


# ─── Constants ───────────────────────────────────────────────────────────────

_YT_API = "https://www.googleapis.com/youtube/v3"

# Geo demonym → ISO alpha-2 (same shape as trends_seed._GEO_ALIASES but
# narrower — only the ones supported by YT Data API regionCode).
_GEO_TO_REGION: Dict[str, str] = {
    "india": "IN", "indian": "IN",
    "us": "US", "usa": "US", "america": "US", "american": "US",
    "uk": "GB", "britain": "GB", "british": "GB", "english": "GB",
    "canada": "CA", "canadian": "CA",
    "australia": "AU", "australian": "AU",
    "germany": "DE", "german": "DE",
    "france": "FR", "french": "FR",
    "japan": "JP", "japanese": "JP",
    "brazil": "BR", "brazilian": "BR",
    "mexico": "MX", "mexican": "MX",
}

# Shorts URL canonical form.
_SHORTS_URL = "https://www.youtube.com/shorts/{video_id}"

# ISO 8601 duration: PT#H#M#S — any component may be absent.
_DURATION_RE = re.compile(
    r"P(?:T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?)?$"
)

_HASHTAG_RE = re.compile(r"#([A-Za-z0-9_À-ɏḀ-ỿ]+)")

# Shorts are ≤60s historically; Shorts uploader allows up to 180s (3 min)
# on new uploads. Keep the cap permissive but exclude full-length videos
# that slip through the API's `videoDuration=short` (<4min) filter.
MAX_SHORT_DURATION_SEC = 181


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _parse_iso_duration(iso: str) -> Optional[int]:
    """`PT1M30S` → 90. None on parse failure."""
    if not iso:
        return None
    m = _DURATION_RE.match(iso.strip())
    if not m:
        return None
    h = int(m.group("h") or 0)
    mi = int(m.group("m") or 0)
    s = int(m.group("s") or 0)
    total = h * 3600 + mi * 60 + s
    return total if total > 0 else None


def _extract_hashtags(text: str) -> List[str]:
    if not text:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for h in _HASHTAG_RE.findall(text):
        lower = h.lower()
        if lower not in seen:
            seen.add(lower)
            out.append(lower)
    return out


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _engagement_score(
    *,
    views: Optional[int],
    likes: Optional[int],
    comments: Optional[int],
) -> Optional[float]:
    """Comment-heavy weighted engagement rate (no shares from YT Data API).

    Formula: (likes + comments*3) / max(views, 1). Returns None if views
    is missing or zero (engagement rate is undefined on unviewed videos).
    Capped at 1.0 to keep the value in a comparable range.
    """
    if not views or views <= 0:
        return None
    l = likes or 0
    c = comments or 0
    score = (l + c * 3) / views
    return min(round(score, 6), 1.0)


def _parse_published_at(iso: str) -> Optional[datetime]:
    if not iso:
        return None
    try:
        # YT returns 'YYYY-MM-DDTHH:MM:SSZ'
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _region_from_query(query: TypedQuery) -> Optional[str]:
    for g in query.geo_proxies:
        code = _GEO_TO_REGION.get((g or "").strip().lower())
        if code:
            return code
    return None


# ─── Discoverer ──────────────────────────────────────────────────────────────


class YouTubeShortsDiscoverer(ShortVideoDiscoverer):
    """YouTube Data API v3 + youtube-transcript-api driven Shorts pull."""

    channel_id = "youtube_shorts"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        top_n_enrichment: int = 5,
        comments_per_video: int = 5,
        request_timeout: int = 15,
        enable_transcripts: bool = True,
        transcript_languages: Tuple[str, ...] = ("en", "en-US", "en-IN", "hi", "hi-IN"),
    ) -> None:
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY", "").strip()
        self.available = bool(self.api_key)
        self.top_n_enrichment = top_n_enrichment
        self.comments_per_video = comments_per_video
        self.request_timeout = request_timeout
        self.enable_transcripts = enable_transcripts
        self.transcript_languages = transcript_languages

    # ── Public API ─────────────────────────────────────────────────────────

    async def discover(
        self,
        query: TypedQuery,
        window: TimeWindow,
        count: int = 10,
    ) -> List[ShortVideoLink]:
        if not self.available:
            return []
        return await asyncio.to_thread(
            self._discover_sync, query, window, count
        )

    # ── Sync pipeline (run in to_thread) ───────────────────────────────────

    def _discover_sync(
        self,
        query: TypedQuery,
        window: TimeWindow,
        count: int,
    ) -> List[ShortVideoLink]:
        try:
            video_ids = self._search_video_ids(query, window, count)
        except Exception as e:
            log.warning("YT Shorts search failed for %r: %s", query.text[:60], e)
            return []
        if not video_ids:
            return []

        try:
            details = self._fetch_video_details(video_ids)
        except Exception as e:
            log.warning("YT Shorts videos.list failed: %s", e)
            return []

        # Filter to actual Shorts (≤ 3 min). The `videoDuration=short` API
        # filter is <4min and occasionally leaks 60-180s vlogs through — but
        # those are arguably "shorts" by YouTube's newer definition, so we
        # keep them (boundary chosen above).
        kept: List[Dict[str, Any]] = []
        for vid in video_ids:
            d = details.get(vid)
            if not d:
                continue
            dur = d.get("duration_sec")
            if dur is not None and dur > MAX_SHORT_DURATION_SEC:
                continue
            kept.append(d)

        # Enrich top-N with comments + transcripts (the rest gets minimal payload).
        top_n = min(self.top_n_enrichment, len(kept))
        for d in kept[:top_n]:
            try:
                d["top_comments"] = self._fetch_top_comments(d["video_id"])
            except Exception as e:
                log.info("YT comments failed for %s: %s", d["video_id"], e)
                d["top_comments"] = []
            if self.enable_transcripts:
                d["transcript"] = self._fetch_transcript(d["video_id"])
            else:
                d["transcript"] = None
        for d in kept[top_n:]:
            d["top_comments"] = []
            d["transcript"] = None

        return [self._build_link(d, query) for d in kept]

    # ── Step 1: search.list ────────────────────────────────────────────────

    def _search_video_ids(
        self,
        query: TypedQuery,
        window: TimeWindow,
        count: int,
    ) -> List[str]:
        time_params = to_youtube_params(window)
        params = {
            "part": "id",
            "type": "video",
            "videoDuration": "short",
            "maxResults": str(min(max(count, 1), 50)),
            "q": query.text,
            "order": "relevance",
            "key": self.api_key,
            "publishedAfter": time_params["publishedAfter"],
            "publishedBefore": time_params["publishedBefore"],
        }
        region = _region_from_query(query)
        if region:
            params["regionCode"] = region
            params["relevanceLanguage"] = "en"

        r = requests.get(
            f"{_YT_API}/search",
            params=params,
            timeout=self.request_timeout,
        )
        # Cost meter: charge BEFORE raise_for_status so quota counted even on
        # 4xx (Google deducts on most non-auth errors). Skip 429 since
        # rate-limit rejections don't deduct quota.
        if r.status_code != 429:
            try:
                from ..cost_meter import current_job_id, get_meter
                jid = current_job_id()
                if jid:
                    get_meter().charge_yt_quota(jid, "search.list")
            except Exception:
                pass
        r.raise_for_status()
        data = r.json()
        ids: List[str] = []
        for item in data.get("items", []):
            vid = (item.get("id") or {}).get("videoId")
            if vid:
                ids.append(vid)
        return ids

    # ── Step 2: videos.list (batched) ──────────────────────────────────────

    def _fetch_video_details(self, video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for chunk_start in range(0, len(video_ids), 50):
            chunk = video_ids[chunk_start: chunk_start + 50]
            params = {
                "part": "snippet,contentDetails,statistics",
                "id": ",".join(chunk),
                "key": self.api_key,
            }
            r = requests.get(
                f"{_YT_API}/videos",
                params=params,
                timeout=self.request_timeout,
            )
            r.raise_for_status()
            try:
                from ..cost_meter import current_job_id, get_meter
                jid = current_job_id()
                if jid:
                    get_meter().charge_yt_quota(jid, "videos.list")
            except Exception:
                pass
            data = r.json()
            for item in data.get("items", []):
                vid = item.get("id")
                if not vid:
                    continue
                snippet = item.get("snippet") or {}
                content = item.get("contentDetails") or {}
                stats = item.get("statistics") or {}
                description = snippet.get("description") or ""
                title = snippet.get("title") or ""

                thumb = (snippet.get("thumbnails") or {})
                thumb_url = (
                    (thumb.get("maxres") or thumb.get("high") or thumb.get("medium")
                     or thumb.get("default") or {}).get("url", "")
                )

                out[vid] = {
                    "video_id": vid,
                    "url": _SHORTS_URL.format(video_id=vid),
                    "title": title,
                    "snippet": description[:280],
                    "caption": description,
                    "hashtags": _extract_hashtags(f"{title}\n{description}"),
                    "duration_sec": _parse_iso_duration(content.get("duration", "")),
                    "creator": snippet.get("channelTitle") or "",
                    "creator_id": snippet.get("channelId") or "",
                    "thumbnail_url": thumb_url,
                    "view_count": _safe_int(stats.get("viewCount")),
                    "like_count": _safe_int(stats.get("likeCount")),
                    "comment_count": _safe_int(stats.get("commentCount")),
                    "observed_at": _parse_published_at(snippet.get("publishedAt", "")),
                }
        return out

    # ── Step 3: commentThreads.list ────────────────────────────────────────

    def _fetch_top_comments(self, video_id: str) -> List[str]:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "order": "relevance",
            "maxResults": str(self.comments_per_video),
            "textFormat": "plainText",
            "key": self.api_key,
        }
        r = requests.get(
            f"{_YT_API}/commentThreads",
            params=params,
            timeout=self.request_timeout,
        )
        if r.status_code == 403:
            # Comments disabled on the video — common, not an error.
            return []
        r.raise_for_status()
        try:
            from ..cost_meter import current_job_id, get_meter
            jid = current_job_id()
            if jid:
                get_meter().charge_yt_quota(jid, "commentThreads.list")
        except Exception:
            pass
        data = r.json()
        out: List[str] = []
        for item in data.get("items", []):
            top = (
                (((item.get("snippet") or {}).get("topLevelComment") or {})
                 .get("snippet") or {}).get("textDisplay")
            )
            if top:
                out.append(str(top).strip())
        return out

    # ── Step 4: youtube-transcript-api ────────────────────────────────────

    def _fetch_transcript(self, video_id: str) -> Optional[str]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
        except ImportError:
            return None

        try:
            api = YouTubeTranscriptApi()
            transcript_obj = api.fetch(
                video_id, languages=list(self.transcript_languages)
            )
            segments = transcript_obj.to_raw_data()
        except Exception as e:
            log.debug("transcript fetch failed for %s: %s", video_id, e)
            return None

        # Concatenate segment text into one string (cap length so triage L6
        # doesn't end up with a 50KB blob for a single short).
        text = " ".join(
            (seg.get("text") or "").strip()
            for seg in segments
            if seg.get("text")
        ).strip()
        if not text:
            return None
        return text[:8000]

    # ── ShortVideoLink assembly ────────────────────────────────────────────

    def _build_link(self, d: Dict[str, Any], query: TypedQuery) -> ShortVideoLink:
        return ShortVideoLink(
            url=d["url"],
            canonical_url=d["url"],
            title=d["title"],
            snippet=d["snippet"],
            channel="youtube_shorts",
            hypothesis_id=query.hypothesis_id,
            query=query,
            observed_at=d["observed_at"],
            backend_used="youtube_data_api_v3",
            duration_sec=d["duration_sec"],
            caption=d["caption"],
            hashtags=d["hashtags"],
            creator=d["creator"],
            view_count=d["view_count"],
            like_count=d["like_count"],
            comment_count=d["comment_count"],
            thumbnail_url=d["thumbnail_url"],
            top_comments=d["top_comments"],
            transcript=d["transcript"],
            engagement_score=_engagement_score(
                views=d["view_count"],
                likes=d["like_count"],
                comments=d["comment_count"],
            ),
        )


# ─── Singleton accessor (matches backend pattern) ────────────────────────────

_singleton: Optional[YouTubeShortsDiscoverer] = None


def get_youtube_shorts() -> YouTubeShortsDiscoverer:
    """Process-wide singleton — avoids re-reading env on every call."""
    global _singleton
    if _singleton is None:
        _singleton = YouTubeShortsDiscoverer()
    return _singleton


def reset_youtube_shorts() -> None:
    """Drop the cached instance (used after rotating env vars in tests)."""
    global _singleton
    _singleton = None
