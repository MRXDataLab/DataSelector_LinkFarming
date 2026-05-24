"""Pydantic models for the Data Selection Module.

Names line up with the build plan §C and the host's hypothesis_engine enums.
ChannelId / Archetype / Verdict are Literal types so they validate at parse time.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# ─── Channel taxonomy ────────────────────────────────────────────────────────

ChannelId = Literal[
    "google_web",
    "google_paa",
    "google_related",
    "reddit",
    "quora",
    "youtube",
    "youtube_shorts",
    "tiktok",
    "instagram_reels",
    "news",
    "substack",
    "trends",
    "marketplace",
]

CHANNEL_IDS: tuple[str, ...] = (
    "google_web",
    "google_paa",
    "google_related",
    "reddit",
    "quora",
    "youtube",
    "youtube_shorts",
    "tiktok",
    "instagram_reels",
    "news",
    "substack",
    "trends",
    "marketplace",
)

SHORT_VIDEO_CHANNELS: tuple[str, ...] = ("youtube_shorts", "tiktok", "instagram_reels")

# ─── Query archetypes (see build plan §C) ────────────────────────────────────

Archetype = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9]

ARCHETYPE_NAMES: Dict[int, str] = {
    1: "entity_pain",
    2: "switching_narrative",
    3: "comparison",
    4: "identity_aspiration",
    5: "question_analysis",
    6: "counter_evidence",
    7: "expert_authority",
    8: "crisis_event",
    9: "hashtag_trend",
}

QueryVertical = Literal["web", "news", "forums", "videos", "paa", "related", "shopping"]
WindowLabel = Literal["7d", "30d", "90d", "1y", "5y", "custom"]
Verdict = Literal["supports", "refutes", "tangential"]

# ─── TimeWindow ──────────────────────────────────────────────────────────────


class TimeWindow(BaseModel):
    start: datetime
    end: datetime
    label: WindowLabel

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v: datetime, info: Any) -> datetime:
        start = info.data.get("start") if hasattr(info, "data") else None
        if start is not None and v <= start:
            raise ValueError("end must be after start")
        return v

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    @classmethod
    def from_label(
        cls,
        label: WindowLabel,
        end: Optional[datetime] = None,
    ) -> "TimeWindow":
        """Build a window by label, anchored to `end` (default: now UTC)."""
        end = end or datetime.now(timezone.utc)
        spans = {
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "90d": timedelta(days=90),
            "1y": timedelta(days=365),
            "5y": timedelta(days=365 * 5),
        }
        if label == "custom":
            raise ValueError("custom requires explicit start/end via TimeWindow(...)")
        return cls(start=end - spans[label], end=end, label=label)


# ─── ChannelFit (L1 — channel scorer output) ─────────────────────────────────


class ChannelFit(BaseModel):
    channel: ChannelId
    fit_score: int = Field(..., ge=0, le=100)
    rationale: str = ""
    expected_signal: str = ""
    sub_scores: Dict[str, float] = Field(default_factory=dict)


# ─── TypedQuery (L2 — query synthesizer output) ──────────────────────────────


class TypedQuery(BaseModel):
    text: str
    channel: ChannelId
    archetype: Archetype
    archetype_name: str
    target_signal: str = ""
    hypothesis_id: str
    falsifier: bool = False
    geo_proxies: List[str] = Field(default_factory=list)


# ─── RawResult (search backend output, format-agnostic) ──────────────────────


class RawResult(BaseModel):
    """Output of every `SearchBackend.search(...)`. Backend-agnostic shape."""

    url: str
    title: str = ""
    snippet: str = ""
    backend: str  # "brave" | "duckduckgo" | "headless_google" | "serpapi"
    vertical: QueryVertical = "web"
    observed_at: Optional[datetime] = None
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── DiscoveredLink (L3 — discoverer output) ─────────────────────────────────


class DiscoveredLink(BaseModel):
    url: str
    canonical_url: str = ""
    title: str = ""
    snippet: str = ""
    channel: ChannelId
    hypothesis_id: str
    query: TypedQuery
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    observed_at: Optional[datetime] = None
    backend_used: Optional[str] = None
    fit_score: Optional[int] = None
    supports_or_refutes: Optional[Verdict] = None
    signal_tags: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    # Populated by L5 dedup (Step 13): when this link's canonical URL or
    # content hash was ALSO discovered via other channels, those channels
    # are listed here. The representative link carries the cross-platform
    # roster; cluster members keep their original (single) `channel`.
    also_found_on: List[ChannelId] = Field(default_factory=list)


class ShortVideoLink(DiscoveredLink):
    """Adds the metadata short-video discoverers carry per the build plan §A.

    `engagement_score` is computed at discovery time as a channel-specific blend
    of (likes, comments, shares) ÷ views. See discoverers/short_video_base.py
    (Step 7) for the formula.
    """

    duration_sec: Optional[int] = None
    caption: str = ""
    hashtags: List[str] = Field(default_factory=list)
    sound_id: Optional[str] = None
    sound_name: Optional[str] = None
    creator: str = ""
    creator_followers: Optional[int] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    comment_count: Optional[int] = None
    share_count: Optional[int] = None
    thumbnail_url: str = ""
    top_comments: List[str] = Field(default_factory=list)
    transcript: Optional[str] = None
    engagement_score: Optional[float] = None
