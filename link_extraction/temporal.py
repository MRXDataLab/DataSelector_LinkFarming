"""TimeWindow → per-channel API parameter translation.

Every channel speaks a different time dialect. This module is the single
translation point so `TimeWindow` propagates cleanly from UI → L3 discoverers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from .models import ChannelId, TimeWindow


# ─── Brave Search API ────────────────────────────────────────────────────────


def to_brave_params(window: TimeWindow) -> Dict[str, str]:
    """Brave `freshness` param: pd | pw | pm | py. >1y is unsupported (omit)."""
    days = window.days
    if days <= 1:
        return {"freshness": "pd"}
    if days <= 7:
        return {"freshness": "pw"}
    if days <= 31:
        return {"freshness": "pm"}
    if days <= 365:
        return {"freshness": "py"}
    return {}


# ─── DuckDuckGo (duckduckgo-search package) ──────────────────────────────────


def to_ddg_timelimit(window: TimeWindow) -> Optional[str]:
    """DDG `timelimit`: d | w | m | y. >1y returns None (no filter)."""
    days = window.days
    if days <= 1:
        return "d"
    if days <= 7:
        return "w"
    if days <= 31:
        return "m"
    if days <= 365:
        return "y"
    return None


# ─── Google (headless) ───────────────────────────────────────────────────────


def to_google_tbs(window: TimeWindow) -> str:
    """Google `tbs=qdr:*` or `cdr:1,cd_min:...,cd_max:...` for custom ranges."""
    days = window.days
    if days <= 1:
        return "qdr:d"
    if days <= 7:
        return "qdr:w"
    if days <= 31:
        return "qdr:m"
    if days <= 90:
        return "qdr:m3"
    if days <= 365:
        return "qdr:y"
    return (
        f"cdr:1,cd_min:{window.start.strftime('%m/%d/%Y')}"
        f",cd_max:{window.end.strftime('%m/%d/%Y')}"
    )


# ─── Reddit (PRAW search) ────────────────────────────────────────────────────


def to_reddit_t(window: TimeWindow) -> str:
    """Reddit `t` param for search: hour | day | week | month | year | all."""
    days = window.days
    if days <= 1:
        return "day"
    if days <= 7:
        return "week"
    if days <= 31:
        return "month"
    if days <= 365:
        return "year"
    return "all"


# ─── YouTube Data API v3 ─────────────────────────────────────────────────────


def to_youtube_params(window: TimeWindow) -> Dict[str, str]:
    """`publishedAfter` / `publishedBefore` in RFC 3339."""

    def _iso(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "publishedAfter": _iso(window.start),
        "publishedBefore": _iso(window.end),
    }


# ─── pytrends ────────────────────────────────────────────────────────────────


def to_pytrends_timeframe(window: TimeWindow) -> str:
    """pytrends timeframe string. Custom range uses 'YYYY-MM-DD YYYY-MM-DD'."""
    days = window.days
    if days <= 7:
        return "now 7-d"
    if days <= 30:
        return "today 1-m"
    if days <= 90:
        return "today 3-m"
    if days <= 365:
        return "today 12-m"
    if days <= 365 * 5:
        return "today 5-y"
    return f"{window.start.strftime('%Y-%m-%d')} {window.end.strftime('%Y-%m-%d')}"


# ─── GDELT 2.0 ───────────────────────────────────────────────────────────────


def to_gdelt_params(window: TimeWindow) -> Dict[str, str]:
    """GDELT 2.0 STARTDATETIME / ENDDATETIME: YYYYMMDDHHMMSS."""
    fmt = "%Y%m%d%H%M%S"
    return {
        "STARTDATETIME": window.start.strftime(fmt),
        "ENDDATETIME": window.end.strftime(fmt),
    }


# ─── Capability matrix ───────────────────────────────────────────────────────

# True  → channel API supports temporal filter natively
# False → discoverer must apply window client-side using observed_at
CHANNEL_TIME_FILTER_SUPPORT: Dict[ChannelId, bool] = {
    "google_web": True,
    "google_paa": True,
    "google_related": True,
    "reddit": True,
    "quora": True,
    "youtube": True,
    "youtube_shorts": True,
    "tiktok": False,
    "instagram_reels": False,
    "news": True,
    "substack": False,
    "trends": True,
    "marketplace": False,
}
