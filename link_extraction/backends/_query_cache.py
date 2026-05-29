"""In-memory LRU+TTL cache for search backend results.

Why this exists
---------------
Within a single batch, the L2 query synthesizer + the ~9 channel
discoverers produce a lot of overlap. For a typical 10-hypothesis
manifest run we see hundreds of `search_with_fallback()` calls and a
high fraction of them are duplicates from different discoverers
("Brigade Group reviews" gets asked by quora, marketplace, AND substack;
each of those compiles to a `site:host …` query string that often
collides with another discoverer's). Brave's free tier is 2,000 calls /
month — without caching, a single batch can chew through 5-15% of it.

What it caches
--------------
The OUTPUT of `search_with_fallback(query, vertical, count, window)`,
keyed by those four inputs plus the active `BackendPreferences` priority
(so a Safe-mode run doesn't return a cached Full-mode result and vice
versa). Cache lives entirely in process memory — no Redis, no disk
persistence, intentionally simple.

TTL is bounded so stale results don't haunt subsequent runs:
  • default: 3600 sec (1 hour) — long enough for one batch + a few retries
  • override: `OUTTLYR_SEARCH_CACHE_TTL_SEC`

Max size guards against runaway memory (~1KB per entry × 1000 = ~1MB):
  • default: 1000 entries
  • override: `OUTTLYR_SEARCH_CACHE_MAX_SIZE`

Disable entirely with: `OUTTLYR_SEARCH_CACHE_DISABLE=1`

Failures are NOT cached — only non-empty result lists. This is on
purpose so a transient Brave 503 doesn't poison the cache and prevent
retries from succeeding. Empty results from a healthy provider ARE
cached (preventing repeat zero-result calls).

Observability
-------------
`stats()` returns a dict with hits / misses / evictions / size. Surfaced
in the `/backend-health` JSON so the UI can show "saved N calls" to the
analyst.
"""
from __future__ import annotations

import logging
import os
import re
import time
from collections import OrderedDict
from typing import List, Optional, Tuple

from ..models import RawResult, TimeWindow

log = logging.getLogger(__name__)


# ── Config (env-driven) ──────────────────────────────────────────────────────


def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, str(default)))
        return max(0, v)
    except (TypeError, ValueError):
        return default


CACHE_TTL_SEC: int = _env_int("OUTTLYR_SEARCH_CACHE_TTL_SEC", 3600)
CACHE_MAX_SIZE: int = _env_int("OUTTLYR_SEARCH_CACHE_MAX_SIZE", 1000)
CACHE_DISABLED: bool = os.getenv("OUTTLYR_SEARCH_CACHE_DISABLE", "0").lower() in (
    "1", "true", "yes", "on",
)


# ── Cache key normalization ──────────────────────────────────────────────────


# Collapse whitespace + lowercase so "Brigade Group" and "brigade   group" hit
# the same cache entry. We intentionally do NOT remove stopwords or stem —
# that's the L2 synthesizer's job and changing semantics here would hide bugs.
_WS_RE = re.compile(r"\s+")


def _normalize_query(q: str) -> str:
    return _WS_RE.sub(" ", (q or "").strip().lower())


def _window_label(window: Optional[TimeWindow]) -> str:
    """Stable string for the window (None ⇒ 'any')."""
    if window is None:
        return "any"
    # TimeWindow is a dataclass with a `label` field in our codebase.
    return getattr(window, "label", "any")


def make_key(
    query: str,
    vertical: str,
    count: int,
    window: Optional[TimeWindow],
    priority_chain: Tuple[str, ...],
) -> Tuple[str, str, int, str, Tuple[str, ...]]:
    """Build the cache key. Stable across process lifetime."""
    return (
        _normalize_query(query),
        vertical,
        count,
        _window_label(window),
        priority_chain,
    )


# ── LRU + TTL storage ────────────────────────────────────────────────────────


# Each entry stores (expires_at_monotonic, result_list).
_STORE: "OrderedDict[tuple, Tuple[float, List[RawResult]]]" = OrderedDict()

# Stats
_HITS: int = 0
_MISSES: int = 0
_EVICTIONS: int = 0
_STORES: int = 0


def get(key: tuple) -> Optional[List[RawResult]]:
    """Return cached results, or None on miss/expired/disabled.

    Side effects: bumps `_HITS` or `_MISSES`; marks the entry as
    most-recently-used (LRU); evicts expired entries lazily on access.
    """
    global _HITS, _MISSES
    if CACHE_DISABLED:
        return None
    entry = _STORE.get(key)
    if entry is None:
        _MISSES += 1
        return None
    expires_at, results = entry
    if time.monotonic() >= expires_at:
        # Lazy expiry — drop it.
        _STORE.pop(key, None)
        _MISSES += 1
        return None
    # Re-insert at the tail to mark as recently-used.
    _STORE.move_to_end(key)
    _HITS += 1
    # Return a SHALLOW copy of the list so callers' mutations don't poison
    # the cache (the RawResult dataclasses themselves are treated as
    # immutable by the pipeline, so we don't deep-copy each one).
    return list(results)


def put(key: tuple, results: List[RawResult]) -> None:
    """Store results. No-op if disabled, results is empty AND we
    aren't caching empty-from-healthy (we are — see module docstring).
    Enforces `CACHE_MAX_SIZE` via LRU eviction.
    """
    global _EVICTIONS, _STORES
    if CACHE_DISABLED:
        return
    if CACHE_MAX_SIZE <= 0:
        return
    # Store a copy so the caller's later mutation can't leak into the cache.
    expires_at = time.monotonic() + max(1, CACHE_TTL_SEC)
    _STORE[key] = (expires_at, list(results))
    _STORE.move_to_end(key)
    _STORES += 1
    # Evict oldest entries until we're at the cap.
    while len(_STORE) > CACHE_MAX_SIZE:
        _STORE.popitem(last=False)
        _EVICTIONS += 1


def clear() -> None:
    """Drop all cached entries. Used by tests + manual reset endpoints."""
    global _HITS, _MISSES, _EVICTIONS, _STORES
    _STORE.clear()
    _HITS = _MISSES = _EVICTIONS = _STORES = 0


def stats() -> dict:
    """Snapshot of cache metrics. Cheap; safe to call on every request."""
    total = _HITS + _MISSES
    hit_rate = (_HITS / total) if total > 0 else 0.0
    return {
        "enabled": not CACHE_DISABLED,
        "ttl_sec": CACHE_TTL_SEC,
        "max_size": CACHE_MAX_SIZE,
        "size": len(_STORE),
        "hits": _HITS,
        "misses": _MISSES,
        "stores": _STORES,
        "evictions": _EVICTIONS,
        "hit_rate": round(hit_rate, 4),
        # Rough cost-savings estimate — every hit is a Brave/DDG/Google
        # call we didn't make. Useful as the UI's "saved N calls" line.
        "calls_saved": _HITS,
    }
