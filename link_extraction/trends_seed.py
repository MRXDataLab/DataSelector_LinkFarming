"""Step 4 — Trends seed helpers (pytrends wrapper).

Feeds the L2 query synthesizer with the *live consumer phrasings* people
are typing into Google Search within the user-selected TimeWindow. Three
public functions:

    fetch_rising_queries(entity, window, geo="", top_k=5) -> List[str]
    fetch_related_topics(entity, window, geo="")          -> List[str]
    interest_over_time(entity, window, geo="")            -> Dict[str, int]

Design choices:
- Every call wraps pytrends in try/except. Trends is rate-limited (~10
  req/min) and frequently returns empty payloads — we swallow errors and
  return empty results rather than crashing the pipeline.
- Results memoized via `_cached(...)` keyed on `(entity, window.label, geo)`
  so the same hypothesis-entity doesn't hit Trends twice within one job.
- Geo input is normalized: hypothesis decomposer emits demonyms like
  "indian" or "india"; we translate to ISO-3166 alpha-2 (e.g. "IN").
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from .models import TimeWindow
from .temporal import to_pytrends_timeframe

log = logging.getLogger(__name__)


# ─── Geo normalization ───────────────────────────────────────────────────────

# Decomposer.geo_hints emits demonyms + country names lowercased. We map the
# most common ones into ISO 3166-1 alpha-2 codes that pytrends expects.
# Empty string = global (pytrends default).
_GEO_ALIASES: Dict[str, str] = {
    "india": "IN", "indian": "IN", "bharat": "IN",
    "us": "US", "usa": "US", "america": "US", "american": "US",
    "uk": "GB", "britain": "GB", "british": "GB", "england": "GB", "english": "GB",
    "canada": "CA", "canadian": "CA",
    "australia": "AU", "australian": "AU",
    "germany": "DE", "german": "DE",
    "france": "FR", "french": "FR",
    "japan": "JP", "japanese": "JP",
    "china": "CN", "chinese": "CN",
    "brazil": "BR", "brazilian": "BR",
    "mexico": "MX", "mexican": "MX",
    "south africa": "ZA",
    "indonesia": "ID", "indonesian": "ID",
    "philippines": "PH", "filipino": "PH",
    "uae": "AE", "emirates": "AE",
    "singapore": "SG", "singaporean": "SG",
}


def normalize_geo(geo_hint: str) -> str:
    """Translate a decomposer geo_hint (e.g. 'indian', 'india') → 'IN'.

    Empty string when no match. Already-ISO codes pass through unchanged.
    """
    if not geo_hint:
        return ""
    g = geo_hint.strip().lower()
    if len(g) == 2 and g.isalpha():
        return g.upper()
    return _GEO_ALIASES.get(g, "")


def geo_from_hints(geo_hints: List[str]) -> str:
    """Pick the first resolvable ISO code from a list of decomposer hints."""
    for h in geo_hints or []:
        code = normalize_geo(h)
        if code:
            return code
    return ""


# ─── pytrends client (singleton; lazy) ───────────────────────────────────────

_client = None
_client_unavailable_reason: Optional[str] = None


def _get_client():
    """Lazy TrendReq init. Returns None if pytrends isn't usable."""
    global _client, _client_unavailable_reason
    if _client is not None:
        return _client
    if _client_unavailable_reason is not None:
        return None
    try:
        from pytrends.request import TrendReq  # type: ignore
        _client = TrendReq(hl="en-US", tz=0, timeout=(5, 15))
        return _client
    except Exception as e:
        _client_unavailable_reason = f"{type(e).__name__}: {e}"
        log.warning("pytrends unavailable: %s", _client_unavailable_reason)
        return None


def reset_client() -> None:
    """Drop the cached TrendReq (used after rate-limit cooldowns / tests)."""
    global _client, _client_unavailable_reason
    _client = None
    _client_unavailable_reason = None


# ─── In-process cache ────────────────────────────────────────────────────────

# Cache key: (entity_lower, window.label, geo_code, fn_tag).
# Stores raw list/dict; small to avoid stale data on long-running processes.
_cache: Dict[Tuple[str, str, str, str], Any] = {}


def _cache_key(entity: str, window: TimeWindow, geo: str, tag: str) -> Tuple[str, str, str, str]:
    return (entity.strip().lower(), window.label, geo or "", tag)


def clear_cache() -> None:
    _cache.clear()


# ─── Build payload (shared) ──────────────────────────────────────────────────

def _build_payload(entity: str, window: TimeWindow, geo: str) -> bool:
    """Call `build_payload` on the shared client; return True on success.

    pytrends.build_payload mutates client state — every subsequent call
    (related_queries, interest_over_time) reads from this payload.
    """
    c = _get_client()
    if c is None:
        return False
    timeframe = to_pytrends_timeframe(window)
    try:
        c.build_payload(kw_list=[entity], timeframe=timeframe, geo=geo or "")
        return True
    except Exception as e:
        log.info("pytrends.build_payload failed for %r geo=%r: %s", entity, geo, e)
        return False


# ─── Public API ──────────────────────────────────────────────────────────────

def fetch_rising_queries(
    entity: str,
    window: TimeWindow,
    geo: str = "",
    top_k: int = 5,
) -> List[str]:
    """Live rising query phrasings for `entity` in `window` (and optional geo).

    Returns up to `top_k` query strings, deduped, in pytrends' ranked order.
    Empty list on any pytrends error / rate limit / missing data.

    These are *the* consumer phrasings to inject into L2 archetype slots —
    far more truthful than marketer language.
    """
    if not entity or not entity.strip():
        return []
    geo = (geo or "").strip().upper()
    key = _cache_key(entity, window, geo, "rising")
    if key in _cache:
        return _cache[key][:top_k]

    if not _build_payload(entity, window, geo):
        _cache[key] = []
        return []

    c = _get_client()
    try:
        related = c.related_queries()  # type: ignore[union-attr]
    except Exception as e:
        log.info("pytrends.related_queries failed for %r: %s", entity, e)
        _cache[key] = []
        return []

    if not related or entity not in related:
        _cache[key] = []
        return []

    rising_df = related[entity].get("rising") if isinstance(related[entity], dict) else None
    if rising_df is None or getattr(rising_df, "empty", True):
        _cache[key] = []
        return []

    out: List[str] = []
    seen: set[str] = set()
    try:
        for q in rising_df["query"].tolist():
            s = str(q).strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
    except Exception as e:
        log.info("pytrends rising_df parse failed for %r: %s", entity, e)
        _cache[key] = []
        return []

    _cache[key] = out
    return out[:top_k]


def fetch_related_topics(
    entity: str,
    window: TimeWindow,
    geo: str = "",
    top_k: int = 10,
) -> List[str]:
    """Adjacent topics for `entity` (Trends `related_topics → rising`).

    Used in L1 rationale enrichment ("Trends shows muesli, oats, granola
    co-rising with cornflakes…") rather than L2 query synthesis.
    """
    if not entity or not entity.strip():
        return []
    geo = (geo or "").strip().upper()
    key = _cache_key(entity, window, geo, "topics")
    if key in _cache:
        return _cache[key][:top_k]

    if not _build_payload(entity, window, geo):
        _cache[key] = []
        return []

    c = _get_client()
    try:
        topics = c.related_topics()  # type: ignore[union-attr]
    except Exception as e:
        log.info("pytrends.related_topics failed for %r: %s", entity, e)
        _cache[key] = []
        return []

    if not topics or entity not in topics:
        _cache[key] = []
        return []

    rising_df = topics[entity].get("rising") if isinstance(topics[entity], dict) else None
    if rising_df is None or getattr(rising_df, "empty", True):
        _cache[key] = []
        return []

    out: List[str] = []
    seen: set[str] = set()
    try:
        for name in rising_df["topic_title"].tolist():
            s = str(name).strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
    except Exception as e:
        log.info("pytrends related_topics parse failed for %r: %s", entity, e)
        _cache[key] = []
        return []

    _cache[key] = out
    return out[:top_k]


def interest_over_time(
    entity: str,
    window: TimeWindow,
    geo: str = "",
) -> Dict[str, int]:
    """Daily/weekly interest series for `entity`.

    Returns a `{iso_date: int}` map (0–100 popularity). Empty dict on error.
    Surfaced in the UI Trends panel; not consumed by L2.
    """
    if not entity or not entity.strip():
        return {}
    geo = (geo or "").strip().upper()
    key = _cache_key(entity, window, geo, "iot")
    if key in _cache:
        return _cache[key]

    if not _build_payload(entity, window, geo):
        _cache[key] = {}
        return {}

    c = _get_client()
    try:
        df = c.interest_over_time()  # type: ignore[union-attr]
    except Exception as e:
        log.info("pytrends.interest_over_time failed for %r: %s", entity, e)
        _cache[key] = {}
        return {}

    if df is None or getattr(df, "empty", True) or entity not in df.columns:
        _cache[key] = {}
        return {}

    out: Dict[str, int] = {}
    try:
        for ts, val in df[entity].items():
            out[ts.strftime("%Y-%m-%d")] = int(val)
    except Exception as e:
        log.info("pytrends iot parse failed for %r: %s", entity, e)
        _cache[key] = {}
        return {}

    _cache[key] = out
    return out


# ─── Batch helper for orchestrator ───────────────────────────────────────────

def batch_rising_queries(
    entities: List[str],
    window: TimeWindow,
    geo: str = "",
    top_k: int = 5,
    sleep_between: float = 1.0,
) -> Dict[str, List[str]]:
    """Fetch rising queries for many entities, respecting Trends rate limits.

    Sleeps `sleep_between` seconds between *uncached* calls. Cached entities
    don't trigger sleeps. Returns `{entity: [phrases...]}` (empty list per
    entity on failure).
    """
    out: Dict[str, List[str]] = {}
    first = True
    for ent in entities:
        ent = (ent or "").strip()
        if not ent:
            continue
        cache_hit = _cache_key(ent, window, geo, "rising") in _cache
        if not first and not cache_hit:
            time.sleep(sleep_between)
        out[ent] = fetch_rising_queries(ent, window, geo=geo, top_k=top_k)
        first = False
    return out
