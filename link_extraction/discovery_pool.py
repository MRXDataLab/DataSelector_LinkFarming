"""Per-hypothesis discovery pool — Phase 2 query unification.

The L3 discovery layer has 9 channel discoverers; 5–6 of them route through
`search_with_fallback(vertical="web")` with `site:<host>` queries that
share the same underlying topic terms. For a single hypothesis we end up
making the same general Brave web query 5+ times with only the `site:`
prefix differing — each instance returns nearly the same SERP, just
filtered to a different subset of hosts.

This module fixes that. The orchestrator runs **one or two consolidated
prequery searches** at the top of L3 and parks the RawResults in a
`DiscoveryPool` (a per-hypothesis ContextVar). Discoverers that opt into
the pool check it first: if the pool already has ≥ `count` results
matching the discoverer's host pattern, the discoverer skips its
own `search_with_fallback()` call entirely. If the pool has < `count`
matches, the discoverer can either use what's there + supplement with
a smaller fallback call, or fall through to its original behavior.

What this saves
---------------
For a 10-hypothesis Brigade-style manifest run with all 9 channels
enabled, we measured ~270–450 backend calls. Phase 1 (LRU+TTL query
cache) cut ~30–50% of those. This module cuts another 30–40% by
collapsing site-scoped queries into a shared general search.

What this does NOT change
-------------------------
  • `search_with_fallback()` semantics — discoverers that don't opt in
    keep working as before. The pool is optional context, not a global
    rewrite.
  • Vertical-specific paths (Brave news, Brave videos, YouTube Data
    API, headless-only verticals) — those don't share data shape with
    a general web search, so they stay independent.
  • Post-processing in each discoverer (URL canonicalization, host
    filtering, snippet parsing) — the discoverer still owns its
    `RawResult` → `DiscoveredLink` conversion. We just pre-stage the
    `RawResult`s.

Pool lifecycle
--------------
  1. Orchestrator creates `DiscoveryPool()` per hypothesis at L3 start.
  2. Runs 1–2 prequeries via `search_with_fallback()` and calls
     `pool.add_results(raw, source_label="prequery")`.
  3. Binds the pool to the `_current_pool_cv` ContextVar.
  4. Channel discoverers run as usual; opted-in ones call
     `pool.pick_for_host_patterns(...)` before their own search.
  5. After L3 finishes, the ContextVar is reset.

Thread-safety
-------------
Pool itself is plain Python — single-event-loop discovery is the
norm; we don't claim asyncio safety beyond that. The ContextVar is
asyncio-friendly (each task sees the pool bound at creation time).
"""
from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

from .models import RawResult

log = logging.getLogger(__name__)


# ── Host matching utilities ──────────────────────────────────────────────────


def _host_of(url: str) -> str:
    """Return lowercased hostname for `url`, or empty string on failure."""
    try:
        return (urlparse(url).hostname or "").lower()
    except (ValueError, AttributeError):
        return ""


def host_matches(url: str, patterns: Sequence[str]) -> bool:
    """Match `url`'s host against a list of patterns.

    Pattern grammar:
      • exact host:           "quora.com"           → matches host == "quora.com"
      • subdomain wildcard:   "*.substack.com"      → matches "*.substack.com"
                                                       AND "substack.com"
      • plain substring:      "trustpilot"          → matches "trustpilot.com",
                                                       "uk.trustpilot.com", etc.
    """
    host = _host_of(url)
    if not host:
        return False
    for p in patterns:
        p = p.lower().strip()
        if not p:
            continue
        if p.startswith("*."):
            tail = p[2:]  # drop the "*."
            if host == tail or host.endswith("." + tail):
                return True
        elif "." in p and not p.startswith("."):
            # Treat as exact/eTLD+1 — match host == p OR host ends with ".p"
            # (so "reddit.com" matches "old.reddit.com" too).
            if host == p or host.endswith("." + p):
                return True
        else:
            # Plain substring search.
            if p in host:
                return True
    return False


# ── Pool ─────────────────────────────────────────────────────────────────────


@dataclass
class DiscoveryPool:
    """Per-hypothesis bag of RawResults shared across L3 discoverers.

    Stores results in insertion order (preserves the relevance ranking
    Brave/Google returned). De-duplicates by URL across multiple
    `add_results()` calls so a result that appears in both the web and
    news prequery doesn't double-up.
    """

    # Insertion-ordered list (Python dict preserves order; we use it
    # both for O(1) URL dedup and ordered iteration on pick).
    _by_url: dict = field(default_factory=dict)
    # Stats for observability — surfaced via `stats()` and rolled up
    # into the global registry stats for the UI pill.
    n_added: int = 0
    n_duplicate: int = 0
    n_lookups: int = 0
    n_lookup_hits: int = 0
    n_picks: int = 0  # total RawResults handed to discoverers via pick_*

    def add_results(self, results: Iterable[RawResult],
                    source_label: str = "") -> int:
        """Merge `results` into the pool. Returns count of NEW additions."""
        new_count = 0
        for r in results:
            if not r.url:
                continue
            self.n_added += 1
            if r.url in self._by_url:
                self.n_duplicate += 1
                continue
            self._by_url[r.url] = r
            new_count += 1
        if source_label:
            log.debug(
                "pool.add[%s]: +%d (dupes=%d, total=%d)",
                source_label, new_count, self.n_duplicate, len(self._by_url),
            )
        return new_count

    def size(self) -> int:
        return len(self._by_url)

    def pick_for_host_patterns(
        self,
        patterns: Sequence[str],
        count: int,
    ) -> List[RawResult]:
        """Return up to `count` results whose URL host matches any
        pattern. Order preserved (insertion = relevance).

        Discoverers should use this first; if the result count is below
        threshold they fall back to a per-channel search.
        """
        self.n_lookups += 1
        out: List[RawResult] = []
        for r in self._by_url.values():
            if host_matches(r.url, patterns):
                out.append(r)
                if len(out) >= count:
                    break
        if out:
            self.n_lookup_hits += 1
            self.n_picks += len(out)
        return out

    def pick_by_predicate(
        self,
        predicate: Callable[[RawResult], bool],
        count: int,
    ) -> List[RawResult]:
        """Generic pick — for discoverers needing more than host matching
        (e.g. marketplace + review-site host list, or news + a long set
        of trusted publishers).
        """
        self.n_lookups += 1
        out: List[RawResult] = []
        for r in self._by_url.values():
            if predicate(r):
                out.append(r)
                if len(out) >= count:
                    break
        if out:
            self.n_lookup_hits += 1
            self.n_picks += len(out)
        return out

    def stats(self) -> dict:
        return {
            "size": len(self._by_url),
            "added": self.n_added,
            "duplicates": self.n_duplicate,
            "lookups": self.n_lookups,
            "lookup_hits": self.n_lookup_hits,
            "picks": self.n_picks,
        }


# ── ContextVar plumbing ──────────────────────────────────────────────────────


_current_pool_cv: contextvars.ContextVar[Optional[DiscoveryPool]] = \
    contextvars.ContextVar("outtlyr_discovery_pool", default=None)


def set_current_pool(pool: Optional[DiscoveryPool]) -> contextvars.Token:
    """Bind a pool to the active async context. Pair with `reset_current_pool`.

    The orchestrator calls this once at the start of L3 for each
    hypothesis, after running the prequery. Discoverers that opt-in
    read it via `current_pool()`.
    """
    return _current_pool_cv.set(pool)


def reset_current_pool(token: contextvars.Token) -> None:
    """Unbind the pool from the active context (post-L3 cleanup)."""
    _current_pool_cv.reset(token)


def current_pool() -> Optional[DiscoveryPool]:
    """Return the active pool, or None when discovery isn't running.

    Discoverers should treat `None` as "pool disabled — fall through to
    your normal search path." That makes pool participation strictly
    additive — turning the pool off (or running discoverers outside
    the orchestrator) doesn't break anything.
    """
    return _current_pool_cv.get()


# ── Aggregate stats for the UI / observability surface ──────────────────────


# Running totals across the process lifetime — surface alongside the cache
# stats so the "saved N calls" line on the UI reflects pool savings too.
_PROCESS_TOTALS = {
    "pools_built": 0,
    "results_added": 0,
    "lookups": 0,
    "lookup_hits": 0,
    "picks": 0,
    "calls_saved_estimate": 0,  # crude: count successful host-pattern picks
}


def record_pool_close(pool: DiscoveryPool) -> None:
    """Roll up a finishing pool's stats into the process-wide totals."""
    s = pool.stats()
    _PROCESS_TOTALS["pools_built"] += 1
    _PROCESS_TOTALS["results_added"] += s["added"]
    _PROCESS_TOTALS["lookups"] += s["lookups"]
    _PROCESS_TOTALS["lookup_hits"] += s["lookup_hits"]
    _PROCESS_TOTALS["picks"] += s["picks"]
    # Each successful pick replaces what would have been one
    # `search_with_fallback()` call. Conservative estimate.
    _PROCESS_TOTALS["calls_saved_estimate"] += s["lookup_hits"]


def process_stats() -> dict:
    return dict(_PROCESS_TOTALS)
