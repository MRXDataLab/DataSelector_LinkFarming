"""Verbatim quote extractor — Phase 3 stage L6.5.

Given a (hypothesis, link body text) pair, asks Gemini Flash to pull the
1-2 sentences from the body that most strongly support OR refute the
hypothesis. The result is a structured ``ExtractedQuote`` that:

  • Carries the verbatim sentence(s) — no paraphrasing
  • Tags the stance (supports / refutes / neutral)
  • Carries a 0-1 relevance score the synthesizer ranks on
  • Optionally tags WHICH manifest cohort/trigger/attribute the quote
    evidences (so the per-hypothesis synthesis can say "supports the HNI
    prior because…")

Cost shape (per call, Gemini 2.0 Flash):
  • Input: ~600 token system prompt + capped body (8K chars ≈ 2K tokens)
  • Output: ~120 token JSON
  • Cost: ~$0.0008 per link  →  ~$0.05 per hypothesis at 60 links each
  • Cap: `OUTTLYR_QUOTE_MAX_BODY_CHARS` env (default 8000)

The extractor is failure-soft: any error returns a sentinel
``ExtractedQuote.from_failure()`` so the orchestrator can keep rolling
even when one body fetch is blocked / one LLM call hits a quota wall.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from . import _llm

log = logging.getLogger(__name__)


MAX_BODY_CHARS_FOR_LLM = int(os.getenv("OUTTLYR_QUOTE_MAX_BODY_CHARS", "8000"))
MAX_OUTPUT_TOKENS = int(os.getenv("OUTTLYR_QUOTE_MAX_OUTPUT_TOKENS", "300"))


# ─── Output type ─────────────────────────────────────────────────────────────


@dataclass
class ExtractedQuote:
    """One quote extracted from a single (hypothesis, link body) pair."""
    quote: str = ""
    stance: Literal["supports", "refutes", "neutral", "unknown"] = "unknown"
    relevance_score: float = 0.0   # 0-1, LLM self-rated
    # Which manifest entity the quote evidences (free-form short label)
    cohort_evidenced: str = ""     # e.g. "HNI", "salaried 35-50"
    trigger_evidenced: str = ""    # e.g. "marriage", "retirement"
    attribute_evidenced: str = ""  # e.g. "trust", "premium positioning"
    extractor_note: str = ""       # 1-line "what this evidences" gloss
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.quote) and not self.error

    @classmethod
    def from_failure(cls, msg: str) -> "ExtractedQuote":
        return cls(error=msg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quote": self.quote,
            "stance": self.stance,
            "relevance_score": self.relevance_score,
            "cohort_evidenced": self.cohort_evidenced,
            "trigger_evidenced": self.trigger_evidenced,
            "attribute_evidenced": self.attribute_evidenced,
            "extractor_note": self.extractor_note,
            "error": self.error,
        }


# ─── Prompt ──────────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """You are a research-evidence extractor for a market-research analyst.

Given:
  • One hypothesis statement (a falsifiable claim about market behaviour)
  • The full text body of one web page (article / forum thread / review)
  • Optional manifest context — the brand under study + its target cohorts,
    life triggers (e.g. marriage), and brand attributes

Your job: pull the SINGLE most-evidential sentence (or at most 2 consecutive
sentences) from the body that DIRECTLY supports OR refutes the hypothesis.

CRITICAL RULES:
  • The quote MUST be VERBATIM — copied character-for-character from the body.
    Do NOT paraphrase. Do NOT translate. Preserve original capitalisation
    and punctuation.
  • If no sentence in the body materially evidences the hypothesis, return
    quote="" and stance="neutral".
  • When the quote evidences a specific manifest cohort/trigger/attribute,
    name that in the corresponding field. Otherwise leave it empty.
  • relevance_score is your confidence (0-1) that the quote actually
    evidences the hypothesis. Strong direct evidence ≥ 0.8; weak adjacent
    mention ≤ 0.4.
  • extractor_note is a ≤ 12-word gloss of what the quote evidences
    (e.g. "buyer dismissed Brigade Avalon for poor layout").

Return JSON exactly matching this schema:
{
  "quote": "<verbatim sentence(s), or empty string>",
  "stance": "supports" | "refutes" | "neutral",
  "relevance_score": <0.0..1.0>,
  "cohort_evidenced": "<empty or one cohort name from manifest>",
  "trigger_evidenced": "<empty or one life-trigger keyword>",
  "attribute_evidenced": "<empty or one brand attribute>",
  "extractor_note": "<≤12-word gloss>"
}
"""


def _build_user_prompt(
    *,
    hypothesis_statement: str,
    body_text: str,
    link_title: str,
    link_url: str,
    manifest_cohorts: List[str],
    manifest_triggers: List[str],
    manifest_attributes: List[str],
    brand_name: str,
) -> str:
    """Compose the per-link extraction prompt."""
    body = (body_text or "").strip()
    if len(body) > MAX_BODY_CHARS_FOR_LLM:
        body = body[:MAX_BODY_CHARS_FOR_LLM] + "\n…[truncated]"
    context_block = ""
    if brand_name or manifest_cohorts or manifest_triggers or manifest_attributes:
        context_block = "MANIFEST CONTEXT:\n"
        if brand_name:
            context_block += f"  brand: {brand_name}\n"
        if manifest_cohorts:
            context_block += f"  cohorts: {', '.join(manifest_cohorts[:6])}\n"
        if manifest_triggers:
            context_block += f"  triggers: {', '.join(manifest_triggers[:8])}\n"
        if manifest_attributes:
            context_block += f"  attributes: {', '.join(manifest_attributes[:6])}\n"

    return (
        f"HYPOTHESIS:\n{hypothesis_statement.strip()}\n\n"
        f"{context_block}\n"
        f"LINK URL: {link_url}\n"
        f"LINK TITLE: {link_title or '(no title)'}\n\n"
        f"BODY TEXT:\n{body}\n"
    )


# ─── Public API ──────────────────────────────────────────────────────────────


def extract_quote(
    *,
    hypothesis_statement: str,
    body_text: str,
    link_title: str = "",
    link_url: str = "",
    manifest_cohorts: Optional[List[str]] = None,
    manifest_triggers: Optional[List[str]] = None,
    manifest_attributes: Optional[List[str]] = None,
    brand_name: str = "",
    job_id: Optional[str] = None,
    timeout_sec: int = 30,
) -> ExtractedQuote:
    """One-shot quote extraction for one (hypothesis, link) pair.

    Never raises. Returns `ExtractedQuote.from_failure(...)` on any error
    so the orchestrator can keep rolling.
    """
    if not (hypothesis_statement and body_text):
        return ExtractedQuote.from_failure("missing hypothesis or body text")
    if not _llm.is_available():
        return ExtractedQuote.from_failure("LLM (Vertex Gemini) not configured")

    user_prompt = _build_user_prompt(
        hypothesis_statement=hypothesis_statement,
        body_text=body_text,
        link_title=link_title,
        link_url=link_url,
        manifest_cohorts=manifest_cohorts or [],
        manifest_triggers=manifest_triggers or [],
        manifest_attributes=manifest_attributes or [],
        brand_name=brand_name,
    )

    parsed = _llm.call_llm(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        expect_json=True,
        temperature=0.1,
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=timeout_sec,
        job_id=job_id,
    )
    if parsed is None or not isinstance(parsed, dict):
        return ExtractedQuote.from_failure("LLM returned None or non-dict")

    # Sanity-trim and coerce types
    try:
        relevance = float(parsed.get("relevance_score", 0.0))
        if relevance < 0: relevance = 0.0
        if relevance > 1: relevance = 1.0
    except Exception:
        relevance = 0.0

    quote = (parsed.get("quote") or "").strip()
    stance_raw = (parsed.get("stance") or "neutral").strip().lower()
    if stance_raw not in ("supports", "refutes", "neutral"):
        stance_raw = "neutral"

    return ExtractedQuote(
        quote=quote,
        stance=stance_raw,  # type: ignore[arg-type]
        relevance_score=relevance,
        cohort_evidenced=(parsed.get("cohort_evidenced") or "").strip()[:80],
        trigger_evidenced=(parsed.get("trigger_evidenced") or "").strip()[:40],
        attribute_evidenced=(parsed.get("attribute_evidenced") or "").strip()[:40],
        extractor_note=(parsed.get("extractor_note") or "").strip()[:200],
    )
