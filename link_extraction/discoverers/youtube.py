"""YouTube long-form discoverer (Step 12).

Reuses the YouTube Data API v3 plumbing from `youtube_shorts.py` but with
`videoDuration=medium` (4-20 min) — the sweet spot for expert reviews,
deep-dives, and creator essays. Skips videos ≤ 181s (those are Shorts and
the `YouTubeShortsDiscoverer` already covers them).

Same emission shape (`ShortVideoLink`) because every field — duration,
caption, hashtags, view/like counts, top_comments, transcript,
engagement_score — applies to long-form video too. The "short-video" name
on the class is historical; the channel discriminator (`SHORT_VIDEO_CHANNELS`
tuple) is what the frontend uses to decide aspect ratio + grid layout.

Quota: ~106 units per query (same as Shorts: search=100 + videos=1 +
commentThreads=5). With 10K daily quota and both YT discoverers running,
budget ~45 queries/day per hypothesis if both fire on every query.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import requests

from ..models import ShortVideoLink, TimeWindow, TypedQuery
from ..temporal import to_youtube_params
from .base import ShortVideoDiscoverer
from .youtube_shorts import (
    _GEO_TO_REGION,
    _engagement_score,
    _extract_hashtags,
    _parse_iso_duration,
    _parse_published_at,
    _safe_int,
)

log = logging.getLogger(__name__)

_YT_API = "https://www.googleapis.com/youtube/v3"
_WATCH_URL = "https://www.youtube.com/watch?v={video_id}"

# Long-form lower bound — anything shorter is Shorts territory and already
# covered. Upper bound left open (educational/expert content can be 30+ min).
MIN_LONG_DURATION_SEC = 182


def _region_from_query(query: TypedQuery) -> Optional[str]:
    for g in query.geo_proxies:
        code = _GEO_TO_REGION.get((g or "").strip().lower())
        if code:
            return code
    return None


class YouTubeDiscoverer(ShortVideoDiscoverer):
    """Dual-mode YouTube long-form discoverer (Phase 1.6).

    Mirrors `youtube_shorts.YouTubeShortsDiscoverer` — defaults to free
    search-based discovery via headless/DDG, with the YT Data API path
    reserved as an enrichment step for top-N triage candidates.

    Modes:
    - ``search`` (default) — `search_with_fallback(vertical="videos")` →
      filter to `youtube.com/watch?v=` URLs, build `ShortVideoLink`.
      Zero YT API quota burn.
    - ``api`` — legacy `videos.list` + `commentThreads.list` path.
    """

    channel_id = "youtube"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        top_n_enrichment: int = 5,
        comments_per_video: int = 5,
        request_timeout: int = 15,
        enable_transcripts: bool = True,
        transcript_languages: tuple[str, ...] = ("en", "en-US", "en-IN", "hi", "hi-IN"),
        # YT API's classes: short<4min, medium=4-20min, long=>20min, any=all.
        # We default to `any` and use our Python-side MIN_LONG_DURATION_SEC
        # filter — `medium` excludes 3-4 min reviews that are genuinely
        # long-form (above the Shorts band) but below YT's 4-min cutoff.
        video_duration: str = "any",
        prefer_api_mode: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY", "").strip()
        # Phase 1.6 — `available` covers both modes: API key OR any web-search
        # backend with a videos vertical (headless / DDG / Brave).
        from ..backends.registry import get_brave, get_ddg, get_headless
        self.has_api_key = bool(self.api_key)
        self.available = (
            self.has_api_key
            or get_brave().available
            or get_ddg().available
            or get_headless().available
        )
        self.prefer_api_mode = prefer_api_mode
        try:
            from .. import backend_health
            if self.has_api_key:
                backend_health.report(
                    "youtube", "ok",
                    message="YOUTUBE_API_KEY configured",
                )
            else:
                backend_health.report(
                    "youtube", "missing_key",
                    message="YOUTUBE_API_KEY not set — search-mode only",
                )
        except Exception:
            pass
        self.top_n_enrichment = top_n_enrichment
        self.comments_per_video = comments_per_video
        self.request_timeout = request_timeout
        self.enable_transcripts = enable_transcripts
        self.transcript_languages = transcript_languages
        self.video_duration = video_duration

    def _select_mode(self) -> str:
        if self.prefer_api_mode and self.has_api_key:
            return "api"
        if not self.has_api_key:
            return "search"
        try:
            from .. import backend_health
            if backend_health.is_blocked("youtube"):
                return "search"
        except Exception:
            pass
        return "search"

    async def discover(
        self,
        query: TypedQuery,
        window: TimeWindow,
        count: int = 10,
    ) -> List[ShortVideoLink]:
        if not self.available:
            return []
        mode = self._select_mode()
        if mode == "search":
            from ._youtube_search import youtube_via_search
            return await youtube_via_search(
                query, window, count,
                long_only=True,
                channel_id=self.channel_id,
            )
        return await asyncio.to_thread(self._discover_sync, query, window, count)

    # Phase 1.6 — Enrichment hook. Mirror of YouTubeShortsDiscoverer; orchestrator
    # calls this on top-N triage candidates only, saving ~10x quota vs the
    # legacy "API for all discovery" path. See youtube_shorts.enrich_via_api
    # for the canonical doc.
    async def enrich_via_api(
        self,
        links: List[ShortVideoLink],
        *,
        fetch_comments: bool = True,
        fetch_transcript: bool = True,
    ) -> List[ShortVideoLink]:
        if not self.has_api_key or not links:
            return links
        from ._youtube_search import extract_video_id
        vid_to_link: Dict[str, ShortVideoLink] = {}
        for lk in links:
            vid = extract_video_id(lk.url) or extract_video_id(lk.canonical_url or "")
            if vid:
                vid_to_link[vid] = lk
        if not vid_to_link:
            return links
        try:
            details = await asyncio.to_thread(
                self._fetch_video_details, list(vid_to_link.keys()),
            )
        except Exception as e:
            log.warning("YT (long) enrich videos.list failed: %s", e)
            return links
        for vid, d in details.items():
            lk = vid_to_link.get(vid)
            if not lk:
                continue
            lk.view_count = d.get("view_count")
            lk.like_count = d.get("like_count")
            lk.comment_count = d.get("comment_count")
            if d.get("duration_sec") is not None:
                lk.duration_sec = d["duration_sec"]
            if d.get("creator"):
                lk.creator = d["creator"]
            if d.get("thumbnail_url"):
                lk.thumbnail_url = d["thumbnail_url"]
            if d.get("caption"):
                lk.caption = d["caption"]
            if d.get("observed_at"):
                lk.observed_at = d["observed_at"]
            lk.signal_tags = [
                t for t in (lk.signal_tags or []) if t != "yt_search_mode"
            ] + ["yt_api_enriched"]
        if fetch_comments:
            top_n = min(self.top_n_enrichment, len(vid_to_link))
            for vid in list(vid_to_link.keys())[:top_n]:
                lk = vid_to_link[vid]
                try:
                    lk.top_comments = await asyncio.to_thread(
                        self._fetch_top_comments, vid,
                    )
                except Exception as e:
                    log.info("YT (long) comments for %s failed: %s", vid, e)
        if fetch_transcript and self.enable_transcripts:
            top_n = min(self.top_n_enrichment, len(vid_to_link))
            for vid in list(vid_to_link.keys())[:top_n]:
                lk = vid_to_link[vid]
                try:
                    lk.transcript = await asyncio.to_thread(
                        self._fetch_transcript, vid,
                    )
                except Exception:
                    pass
        return links

    # ── Sync pipeline ──────────────────────────────────────────────────────

    def _discover_sync(
        self,
        query: TypedQuery,
        window: TimeWindow,
        count: int,
    ) -> List[ShortVideoLink]:
        try:
            video_ids = self._search_video_ids(query, window, count)
        except Exception as e:
            log.warning("YT long-form search failed for %r: %s", query.text[:60], e)
            return []
        if not video_ids:
            return []

        try:
            details = self._fetch_video_details(video_ids)
        except Exception as e:
            log.warning("YT long-form videos.list failed: %s", e)
            return []

        kept: List[Dict[str, Any]] = []
        for vid in video_ids:
            d = details.get(vid)
            if not d:
                continue
            dur = d.get("duration_sec")
            # Drop anything in the Shorts band (≤181s) — YT Shorts discoverer
            # handles those. Duration unknown? Keep it (could be either).
            if dur is not None and dur < MIN_LONG_DURATION_SEC:
                continue
            kept.append(d)

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

    # ── search.list ───────────────────────────────────────────────────────

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
            "videoDuration": self.video_duration,
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
            f"{_YT_API}/search", params=params, timeout=self.request_timeout
        )
        # Cost meter — charge before raise_for_status; skip on 429.
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

    # ── videos.list (batched) ─────────────────────────────────────────────

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
                f"{_YT_API}/videos", params=params, timeout=self.request_timeout
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
                    "url": _WATCH_URL.format(video_id=vid),
                    "title": title,
                    "snippet": description[:280],
                    "caption": description,
                    "hashtags": _extract_hashtags(f"{title}\n{description}"),
                    "duration_sec": _parse_iso_duration(content.get("duration", "")),
                    "creator": snippet.get("channelTitle") or "",
                    "thumbnail_url": thumb_url,
                    "view_count": _safe_int(stats.get("viewCount")),
                    "like_count": _safe_int(stats.get("likeCount")),
                    "comment_count": _safe_int(stats.get("commentCount")),
                    "observed_at": _parse_published_at(snippet.get("publishedAt", "")),
                }
        return out

    # ── commentThreads.list ───────────────────────────────────────────────

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
            f"{_YT_API}/commentThreads", params=params, timeout=self.request_timeout
        )
        if r.status_code == 403:
            return []  # comments disabled — common, not an error
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

    # ── youtube-transcript-api ────────────────────────────────────────────

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
        text = " ".join(
            (seg.get("text") or "").strip()
            for seg in segments if seg.get("text")
        ).strip()
        if not text:
            return None
        # Long-form transcripts are often 5-20K chars; cap higher than Shorts
        # so the LLM sees meaningful context but still bounded.
        return text[:15000]

    # ── ShortVideoLink assembly ───────────────────────────────────────────

    def _build_link(self, d: Dict[str, Any], query: TypedQuery) -> ShortVideoLink:
        return ShortVideoLink(
            url=d["url"],
            canonical_url=d["url"],
            title=d["title"],
            snippet=d["snippet"],
            channel="youtube",
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


_singleton: Optional[YouTubeDiscoverer] = None


def get_youtube() -> YouTubeDiscoverer:
    global _singleton
    if _singleton is None:
        _singleton = YouTubeDiscoverer()
    return _singleton


def reset_youtube() -> None:
    global _singleton
    _singleton = None
