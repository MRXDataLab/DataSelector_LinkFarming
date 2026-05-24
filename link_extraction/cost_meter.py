"""Per-hypothesis cost meter — tracks Gemini $$, YT API quota units, and Brave calls.

Why it exists: Step 14 + locked recommendation #4 — every pipeline run has
a hard $0.50 ceiling so a runaway hypothesis can't burn the daily quota.

The meter is a lightweight in-memory ledger keyed by `job_id`. Three call
sites report into it:

  1. `_llm.py`            → `charge_llm(job_id, model, input_tokens, output_tokens)`
  2. YT discoverers       → `charge_yt_quota(job_id, op, units)`
                            op ∈ {"search.list" (100), "videos.list" (1),
                                  "commentThreads.list" (1)}
  3. Brave-backed paths   → `charge_brave_call(job_id)` (free tier, count only)

Before each expensive call, the caller asks `would_exceed(job_id, est_cost)`.
If True, it raises `CostLimitExceeded`. The orchestrator catches that and
emits `pipeline_error` with the breakdown — so the link grid still renders
whatever made it through, and the user sees exactly why the run stopped.

This module is the SoT for pricing. Update `_PRICING` here when Gemini
publishes new tariffs.
"""
from __future__ import annotations

import contextvars
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ─── Pricing (USD) ───────────────────────────────────────────────────────────
# Per Google's published rates for Gemini API. Update as Google revises them.
# Keys = model name as it appears in `_llm.DEFAULT_MODEL`. Values per
# 1 MILLION tokens (multiply token count × price ÷ 1e6).

@dataclass(frozen=True)
class _ModelPrice:
    input_per_million: float    # USD per 1M input tokens
    output_per_million: float   # USD per 1M output tokens


_PRICING: Dict[str, _ModelPrice] = {
    # Defaults reflect published Gemini API rates as of 2026-05.
    "gemini-2.5-flash":    _ModelPrice(0.30, 2.50),
    "gemini-2.5-flash-lite": _ModelPrice(0.10, 0.40),
    "gemini-2.5-pro":      _ModelPrice(1.25, 10.00),
    "gemini-2.0-flash":    _ModelPrice(0.10, 0.40),
}
# Fallback when an unrecognised model name appears (defensive — should match
# the most-likely default).
_FALLBACK_PRICE = _PRICING["gemini-2.5-flash"]


# ─── Default budget ──────────────────────────────────────────────────────────
# Per-hypothesis $ ceiling. Configurable via env so analysts can tune
# without code changes. The orchestrator reads this at job-create time.

def default_cost_cap_usd() -> float:
    raw = os.getenv("OUTTLYR_COST_CAP_USD", "0.50").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.50


# YT Data API quota units per operation. The API itself caps at 10K/day on
# the free tier — we track per-hypothesis so the meter can also flag
# "you spent X% of today's YT quota on this one hypothesis".
YT_QUOTA_COSTS: Dict[str, int] = {
    "search.list":          100,
    "videos.list":          1,
    "commentThreads.list":  1,
}


# ─── Ledger ──────────────────────────────────────────────────────────────────


@dataclass
class JobCostEntry:
    job_id: str
    llm_usd: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_calls: int = 0
    yt_quota_units: int = 0
    yt_calls_by_op: Dict[str, int] = field(default_factory=dict)
    brave_calls: int = 0
    cap_usd: float = field(default_factory=default_cost_cap_usd)
    hit_cap: bool = False
    hit_cap_reason: Optional[str] = None
    breakdown_by_call: List[Dict] = field(default_factory=list)

    def total_usd(self) -> float:
        # YT quota is free; Brave free tier; only LLM costs $.
        # (Cost meter is forward-compatible — when paid YT / Brave tiers come
        # online, add their per-unit prices here.)
        return self.llm_usd

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "total_usd": round(self.total_usd(), 6),
            "cap_usd": self.cap_usd,
            "hit_cap": self.hit_cap,
            "hit_cap_reason": self.hit_cap_reason,
            "llm": {
                "calls": self.llm_calls,
                "input_tokens": self.llm_input_tokens,
                "output_tokens": self.llm_output_tokens,
                "usd": round(self.llm_usd, 6),
            },
            "yt": {
                "quota_units": self.yt_quota_units,
                "calls_by_op": dict(self.yt_calls_by_op),
            },
            "brave": {"calls": self.brave_calls},
        }


class CostMeter:
    """Thread-safe per-job cost ledger.

    Singleton (`get_meter()`); reset only in tests.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[str, JobCostEntry] = {}

    def ledger(self, job_id: str) -> JobCostEntry:
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                entry = JobCostEntry(job_id=job_id)
                self._entries[job_id] = entry
            return entry

    def get(self, job_id: str) -> Optional[JobCostEntry]:
        return self._entries.get(job_id)

    def set_cap(self, job_id: str, cap_usd: float) -> None:
        e = self.ledger(job_id)
        with self._lock:
            e.cap_usd = max(0.0, float(cap_usd))

    def reset(self, job_id: Optional[str] = None) -> None:
        with self._lock:
            if job_id is None:
                self._entries.clear()
            else:
                self._entries.pop(job_id, None)

    # ── Charges ──────────────────────────────────────────────────────────

    def charge_llm(
        self,
        job_id: Optional[str],
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record a Gemini call. Returns the USD cost charged."""
        price = _PRICING.get(model, _FALLBACK_PRICE)
        cost = (input_tokens / 1_000_000.0) * price.input_per_million \
             + (output_tokens / 1_000_000.0) * price.output_per_million
        if not job_id:
            return cost
        e = self.ledger(job_id)
        with self._lock:
            e.llm_usd += cost
            e.llm_input_tokens += int(input_tokens or 0)
            e.llm_output_tokens += int(output_tokens or 0)
            e.llm_calls += 1
            e.breakdown_by_call.append({
                "kind": "llm",
                "model": model,
                "in_tokens": int(input_tokens or 0),
                "out_tokens": int(output_tokens or 0),
                "usd": round(cost, 6),
            })
        return cost

    def charge_yt_quota(
        self, job_id: Optional[str], op: str, units: Optional[int] = None,
    ) -> int:
        """Record a YouTube Data API call. Returns units charged."""
        u = units if units is not None else YT_QUOTA_COSTS.get(op, 1)
        if not job_id:
            return u
        e = self.ledger(job_id)
        with self._lock:
            e.yt_quota_units += u
            e.yt_calls_by_op[op] = e.yt_calls_by_op.get(op, 0) + 1
        return u

    def charge_brave_call(self, job_id: Optional[str]) -> None:
        if not job_id:
            return
        e = self.ledger(job_id)
        with self._lock:
            e.brave_calls += 1

    # ── Caps ─────────────────────────────────────────────────────────────

    def would_exceed(self, job_id: Optional[str], extra_usd: float) -> bool:
        """True if charging an additional `extra_usd` would breach the cap."""
        if not job_id:
            return False
        e = self.ledger(job_id)
        if e.cap_usd <= 0:
            return False  # cap disabled
        return (e.total_usd() + extra_usd) > e.cap_usd

    def mark_capped(self, job_id: Optional[str], reason: str) -> None:
        if not job_id:
            return
        e = self.ledger(job_id)
        with self._lock:
            e.hit_cap = True
            e.hit_cap_reason = reason


class CostLimitExceeded(Exception):
    """Raised by call sites when proceeding would breach the per-job cap."""

    def __init__(self, job_id: str, reason: str, breakdown: Dict):
        super().__init__(f"cost cap exceeded for {job_id}: {reason}")
        self.job_id = job_id
        self.reason = reason
        self.breakdown = breakdown


# ─── Module singleton ────────────────────────────────────────────────────────

_meter_singleton: Optional[CostMeter] = None
_meter_lock = threading.Lock()


def get_meter() -> CostMeter:
    global _meter_singleton
    if _meter_singleton is None:
        with _meter_lock:
            if _meter_singleton is None:
                _meter_singleton = CostMeter()
    return _meter_singleton


def reset_meter() -> None:
    """Test-only: drop the singleton."""
    global _meter_singleton
    _meter_singleton = None


# ─── Job-id ambient context ──────────────────────────────────────────────────
# Orchestrator sets this before kicking off a pipeline; LLM client + YT
# discoverers read it implicitly so we don't have to plumb job_id through
# every function signature. asyncio.to_thread() propagates ContextVars
# automatically since Python 3.9.

_current_job_id_cv: contextvars.ContextVar[Optional[str]] = \
    contextvars.ContextVar("outtlyr_current_job_id", default=None)


def set_current_job(job_id: Optional[str]) -> contextvars.Token:
    """Set the ambient job_id for the current async context. Returns a Token
    that should be passed to `reset_current_job()` when the pipeline exits.
    """
    return _current_job_id_cv.set(job_id)


def reset_current_job(token: contextvars.Token) -> None:
    _current_job_id_cv.reset(token)


def current_job_id() -> Optional[str]:
    return _current_job_id_cv.get()
