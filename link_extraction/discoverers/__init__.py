"""Channel-native discoverers (L3 — Multi-Channel Discovery).

Each discoverer speaks one platform's native related-graph and emits
`DiscoveredLink` (or `ShortVideoLink` for short-video channels).

Public surface as steps land:
- Step 6 (✅): YouTubeShortsDiscoverer  ← priority #1
- Step 10: RedditDiscoverer, GooglePAADiscoverer, NewsDiscoverer
- Step 11: TikTokDiscoverer, (InstagramReelsDiscoverer punted to v1.1)
- Step 12: QuoraDiscoverer, YouTubeDiscoverer (long-form), SubstackDiscoverer,
           MarketplaceDiscoverer
"""
from __future__ import annotations

from .base import Discoverer, ShortVideoDiscoverer
from .google_paa import GooglePAADiscoverer, get_google_paa
from .marketplace import MarketplaceDiscoverer, get_marketplace
from .news import NewsDiscoverer, get_news
from .quora import QuoraDiscoverer, get_quora
from .reddit import RedditDiscoverer, get_reddit
from .substack import SubstackDiscoverer, get_substack
from .tiktok import TikTokDiscoverer, get_tiktok
from .youtube import YouTubeDiscoverer, get_youtube
from .youtube_shorts import YouTubeShortsDiscoverer, get_youtube_shorts

__all__ = [
    "Discoverer",
    "GooglePAADiscoverer",
    "MarketplaceDiscoverer",
    "NewsDiscoverer",
    "QuoraDiscoverer",
    "RedditDiscoverer",
    "ShortVideoDiscoverer",
    "SubstackDiscoverer",
    "TikTokDiscoverer",
    "YouTubeDiscoverer",
    "YouTubeShortsDiscoverer",
    "get_google_paa",
    "get_marketplace",
    "get_news",
    "get_quora",
    "get_reddit",
    "get_substack",
    "get_tiktok",
    "get_youtube",
    "get_youtube_shorts",
]
