"""Search backends + fallback registry.

Preference order (per user decision, locked in build plan):
    1. Brave             (primary; fast, 2k/mo free)
    2. DuckDuckGo        (fallback; off if USE_DUCKDUCKGO=0)
    3. Headless Google   (last resort; PAA/Related-only paths bypass tiers 1-2)
    4. SerpAPI           (stub; flip ENABLE_SERPAPI=1 to activate when implemented)
"""
from __future__ import annotations

from .base import SearchBackend
from .brave import BraveBackend
from .duckduckgo import DuckDuckGoBackend
from .headless_google import HeadlessGoogleBackend
from .registry import (
    get_brave,
    get_ddg,
    get_headless,
    get_serpapi,
    search_with_fallback,
)
from .serpapi_stub import SerpApiStubBackend

__all__ = [
    "SearchBackend",
    "BraveBackend",
    "DuckDuckGoBackend",
    "HeadlessGoogleBackend",
    "SerpApiStubBackend",
    "get_brave",
    "get_ddg",
    "get_headless",
    "get_serpapi",
    "search_with_fallback",
]
