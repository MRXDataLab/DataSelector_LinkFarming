"""Per-hypothesis synthesis generator — Phase 3 stage L7.

Given a hypothesis, its triaged links (with extracted quotes from L6.5),
and the active ``ResearchContext``, this module produces a structured
``SynthesisOutput`` that an analyst can paste straight into a research
report. The shape:

  • ``verdict_summary``      one-line "supports / refutes / mixed" call
  • ``synthesis_paragraph``  2-3 sentence evidence rationale citing the
                             strongest supports + refutes quotes
  • ``strongest_support``    the single highest-relevance support quote
                             (with source URL) — the headline citation
  • ``strongest_refute``     same for the refutation side
  • ``cohort_coverage``      Dict[cohort_label → support_count, refute_count]
                             — surfaces "HNI coverage is thin" gaps
  • ``trigger_coverage``     same for life-triggers
  • ``evidence_gaps``        list of cohort / trigger / attribute labels
                             from the manifest that have ZERO quotes
  • ``llm_used``             tracks whether Gemini ran or the
                             deterministic fallback was used

The synthesizer respects ``cost_meter`` — one call per hypothesis at
~$0.001 each, so 32-hyp Brigade runs add ~$0.03 over the triage spend.

Failure modes are all soft: missing LLM → deterministic fallback that
stitches together the strongest quotes verbatim with a stat-sheet front.
"""
from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import _llm
from .quote_extractor import ExtractedQuote

log = logging.getLogger(__name__)


MAX_OUTPUT_TOKENS = int(os.getenv("OUTTLYR_SYNTH_MAX_OUTPUT_TOKENS", "600"))
MAX_QUOTES_FED_TO_LLM = int(os.getenv("OUTTLYR_SYNTH_MAX_QUOTES_IN", "20"))


# ─── Output type ─────────────────────────────────────────────────────────────


@dataclass
class CitedQuote:
    """One quote + its source URL for the synthesis output."""
    quote: str
    stance: str
    relevance_score: float
    source_url: str
    source_title: str = ""
    extractor_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quote": self.quote,
            "stance": self.stance,
            "relevance_score": self.relevance_score,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "extractor_note": self.extractor_note,
        }


@dataclass
class SynthesisOutput:
    hypothesis_id: str
    verdict_summary: str = ""          # one-line headline call
    synthesis_paragraph: str = ""      # 2-3 sentence rationale
    strongest_support: Optional[CitedQuote] = None
    strongest_refute: Optional[CitedQuote] = None
    supports_count: int = 0
    refutes_count: int = 0
    tangential_count: int = 0
    cohort_coverage: Dict[str, Dict[str, int]] = field(default_factory=dict)
    trigger_coverage: Dict[str, Dict[str, int]] = field(default_factory=dict)
    evidence_gaps: List[str] = field(default_factory=list)
    llm_used: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "verdict_summary": self.verdict_summary,
            "synthesis_paragraph": self.synthesis_paragraph,
            "strongest_support": (
                self.strongest_support.to_dict() if self.strongest_support else None
            ),
            "strongest_refute": (
                self.strongest_refute.to_dict() if self.strongest_refute else None
            ),
            "supports_count": self.supports_count,
            "refutes_count": self.refutes_count,
            "tangential_count": self.tangential_count,
            "cohort_coverage": self.cohort_coverage,
            "trigger_coverage": self.trigger_coverage,
            "evidence_gaps": self.evidence_gaps,
            "llm_used": self.llm_used,
            "error": self.error,
        }


# ─── Coverage analysis (deterministic) ───────────────────────────────────────


def _build_coverage(
    quotes_with_meta: List[Dict[str, Any]],
    manifest_axis: List[str],
    quote_field: str,
) -> Dict[str, Dict[str, int]]:
    """Build {axis_value → {supports: n, refutes: n}} coverage map.

    Counts only quotes whose ``quote_field`` (cohort_evidenced / trigger_
    evidenced) matches one of the manifest axis values.
    """
    out: Dict[str, Dict[str, int]] = {
        a: {"supports": 0, "refutes": 0} for a in manifest_axis
    }
    if not manifest_axis:
        return out
    axis_l = {a.lower(): a for a in manifest_axis}
    for q in quotes_with_meta:
        eq: ExtractedQuote = q["extracted"]
        if not eq.ok:
            continue
        val = (eq.to_dict().get(quote_field) or "").strip().lower()
        if not val:
            continue
        # Match by substring — manifest labels like "salaried 35-50" may
        # be partially captured by the LLM ("salaried"). Pick the first
        # axis value whose lowercase form is contained in either direction.
        matched = None
        for axl, original in axis_l.items():
            if val in axl or axl in val:
                matched = original
                break
        if matched is None:
            continue
        if eq.stance == "supports":
            out[matched]["supports"] += 1
        elif eq.stance == "refutes":
            out[matched]["refutes"] += 1
    return out


def _identify_gaps(
    coverage: Dict[str, Dict[str, int]],
    label: str,
    threshold: int = 1,
) -> List[str]:
    """Manifest axis values that received <= threshold quotes total."""
    gaps: List[str] = []
    for axis_value, counts in coverage.items():
        total = counts["supports"] + counts["refutes"]
        if total < threshold:
            gaps.append(f"{label}:{axis_value}")
    return gaps


def _verdict_summary_from_counts(s: int, r: int) -> str:
    """Deterministic one-liner from supports/refutes counts."""
    if s == 0 and r == 0:
        return "Inconclusive — no direct evidence found"
    if r == 0:
        return f"Supports — {s} corroborating sources, 0 refutations"
    if s == 0:
        return f"Refutes — {r} disconfirming sources, 0 corroborations"
    if s >= 3 * r:
        return f"Largely supports — {s} corroborating vs {r} refuting"
    if r >= 3 * s:
        return f"Largely refutes — {r} disconfirming vs {s} corroborating"
    return f"Mixed evidence — {s} supports / {r} refutes"


# ─── LLM prompt ──────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """You are a market-research analyst writing a one-paragraph evidence summary.

Given:
  • One hypothesis statement (a falsifiable claim)
  • A list of verbatim quotes extracted from web sources, each tagged
    with stance (supports/refutes), relevance, and source URL
  • Manifest context — brand under study + target cohorts + life triggers

Your job: write a 2-3 sentence ``synthesis_paragraph`` that:
  • Names the overall verdict (supports / refutes / mixed / inconclusive)
  • Cites the strongest corroborating quote (use ≤ 15 verbatim words in
    double-quotes, inline; do NOT paraphrase) AND the strongest refuting
    quote when both exist
  • Calls out evidence gaps if a manifest cohort or life-trigger has
    zero/thin coverage (e.g. "HNI cohort coverage is thin — only 2 sources")
  • Stays under 90 words total
  • Uses neutral, analyst-grade language. No marketing puffery.

Also produce a ``verdict_summary`` of ≤ 12 words that lead-lines the
paragraph (e.g. "Mixed evidence — 8 supports / 3 refutes; HNI gap").

Return JSON exactly matching this schema:
{
  "verdict_summary": "<≤12-word headline call>",
  "synthesis_paragraph": "<2-3 sentence evidence rationale, ≤ 90 words>"
}
"""


def _build_user_prompt(
    *,
    hypothesis_statement: str,
    brand_name: str,
    manifest_cohorts: List[str],
    manifest_triggers: List[str],
    manifest_attributes: List[str],
    supports_quotes: List[Dict[str, Any]],
    refutes_quotes: List[Dict[str, Any]],
    coverage_gloss: str,
) -> str:
    """Compose the synthesis prompt."""
    s_block = ""
    for i, q in enumerate(supports_quotes[:8], 1):
        s_block += (
            f"  S{i} [score={q['relevance_score']:.2f}]: \"{q['quote'][:240]}\""
            f"  — {q['source_url'][:80]}\n"
        )
    r_block = ""
    for i, q in enumerate(refutes_quotes[:6], 1):
        r_block += (
            f"  R{i} [score={q['relevance_score']:.2f}]: \"{q['quote'][:240]}\""
            f"  — {q['source_url'][:80]}\n"
        )
    return (
        f"HYPOTHESIS:\n{hypothesis_statement.strip()}\n\n"
        f"MANIFEST CONTEXT:\n"
        f"  brand: {brand_name or '(unspecified)'}\n"
        f"  cohorts: {', '.join(manifest_cohorts[:6]) or '(none)'}\n"
        f"  triggers: {', '.join(manifest_triggers[:8]) or '(none)'}\n"
        f"  attributes: {', '.join(manifest_attributes[:6]) or '(none)'}\n\n"
        f"COVERAGE BY MANIFEST AXIS:\n  {coverage_gloss}\n\n"
        f"SUPPORTS QUOTES ({len(supports_quotes)} total):\n{s_block or '  (none)\n'}\n"
        f"REFUTES QUOTES ({len(refutes_quotes)} total):\n{r_block or '  (none)\n'}\n"
    )


# ─── Public API ──────────────────────────────────────────────────────────────


def synthesize(
    *,
    hypothesis_id: str,
    hypothesis_statement: str,
    triaged_quotes: List[Dict[str, Any]],
    brand_name: str = "",
    manifest_cohorts: Optional[List[str]] = None,
    manifest_triggers: Optional[List[str]] = None,
    manifest_attributes: Optional[List[str]] = None,
    use_llm: bool = True,
    job_id: Optional[str] = None,
) -> SynthesisOutput:
    """Produce a SynthesisOutput for one hypothesis.

    Args:
      hypothesis_id, hypothesis_statement: the hypothesis being scored.
      triaged_quotes: list of dicts each carrying:
        {extracted: ExtractedQuote, source_url: str, source_title: str}
        — typically produced by the orchestrator's L6.5 quote stage.
      brand_name + manifest_*: from the ambient ResearchContext.
      use_llm: when False, returns a deterministic stitched paragraph
        (no Gemini call, $0 cost).
      job_id: cost-meter ledger key.

    Never raises. Errors become ``.error`` on the returned object.
    """
    out = SynthesisOutput(hypothesis_id=hypothesis_id)

    # Bucket quotes by stance
    supports: List[Dict[str, Any]] = []
    refutes: List[Dict[str, Any]] = []
    tangential: List[Dict[str, Any]] = []
    for q in triaged_quotes:
        eq: ExtractedQuote = q["extracted"]
        if not eq.ok:
            continue
        rec = {
            "quote": eq.quote,
            "stance": eq.stance,
            "relevance_score": eq.relevance_score,
            "source_url": q.get("source_url", ""),
            "source_title": q.get("source_title", ""),
            "extractor_note": eq.extractor_note,
            "extracted": eq,
        }
        if eq.stance == "supports":
            supports.append(rec)
        elif eq.stance == "refutes":
            refutes.append(rec)
        else:
            tangential.append(rec)
    supports.sort(key=lambda r: -r["relevance_score"])
    refutes.sort(key=lambda r: -r["relevance_score"])

    out.supports_count = len(supports)
    out.refutes_count = len(refutes)
    out.tangential_count = len(tangential)

    if supports:
        s0 = supports[0]
        out.strongest_support = CitedQuote(
            quote=s0["quote"], stance="supports",
            relevance_score=s0["relevance_score"],
            source_url=s0["source_url"],
            source_title=s0["source_title"],
            extractor_note=s0["extractor_note"],
        )
    if refutes:
        r0 = refutes[0]
        out.strongest_refute = CitedQuote(
            quote=r0["quote"], stance="refutes",
            relevance_score=r0["relevance_score"],
            source_url=r0["source_url"],
            source_title=r0["source_title"],
            extractor_note=r0["extractor_note"],
        )

    # Coverage analysis (deterministic)
    cohort_axis = list(manifest_cohorts or [])
    trigger_axis = list(manifest_triggers or [])
    quotes_for_coverage = supports + refutes
    out.cohort_coverage = _build_coverage(
        quotes_for_coverage, cohort_axis, "cohort_evidenced",
    )
    out.trigger_coverage = _build_coverage(
        quotes_for_coverage, trigger_axis, "trigger_evidenced",
    )
    out.evidence_gaps = (
        _identify_gaps(out.cohort_coverage, "cohort", threshold=1)
        + _identify_gaps(out.trigger_coverage, "trigger", threshold=1)
    )

    # Deterministic fallback ALWAYS computed — used when LLM unavailable
    # or use_llm=False, or as a safety net if the LLM response is bad.
    deterministic_summary = _verdict_summary_from_counts(
        out.supports_count, out.refutes_count,
    )
    deterministic_paragraph_parts: List[str] = [deterministic_summary + "."]
    if out.strongest_support:
        deterministic_paragraph_parts.append(
            f'Strongest support: "{out.strongest_support.quote[:220]}" '
            f'(source: {out.strongest_support.source_url[:80]}).'
        )
    if out.strongest_refute:
        deterministic_paragraph_parts.append(
            f'Strongest refutation: "{out.strongest_refute.quote[:220]}" '
            f'(source: {out.strongest_refute.source_url[:80]}).'
        )
    if out.evidence_gaps:
        deterministic_paragraph_parts.append(
            f"Coverage gaps: {', '.join(out.evidence_gaps[:4])}."
        )
    out.verdict_summary = deterministic_summary
    out.synthesis_paragraph = " ".join(deterministic_paragraph_parts)

    # Upgrade with LLM if requested + available.
    # Always call — even when 0 quotes were extracted — so the LLM can
    # at least write a clean coverage-gap summary from the count headlines.
    if use_llm and _llm.is_available():
        # Coverage gloss for the prompt
        coverage_lines = []
        for cohort, counts in out.cohort_coverage.items():
            total = counts["supports"] + counts["refutes"]
            if total > 0:
                coverage_lines.append(
                    f"cohort:{cohort} +{counts['supports']}/-{counts['refutes']}"
                )
        for trigger, counts in out.trigger_coverage.items():
            total = counts["supports"] + counts["refutes"]
            if total > 0:
                coverage_lines.append(
                    f"trigger:{trigger} +{counts['supports']}/-{counts['refutes']}"
                )
        coverage_gloss = "; ".join(coverage_lines) or "(no axis hits)"

        user_prompt = _build_user_prompt(
            hypothesis_statement=hypothesis_statement,
            brand_name=brand_name,
            manifest_cohorts=cohort_axis,
            manifest_triggers=trigger_axis,
            manifest_attributes=list(manifest_attributes or []),
            supports_quotes=supports[:MAX_QUOTES_FED_TO_LLM],
            refutes_quotes=refutes[:MAX_QUOTES_FED_TO_LLM],
            coverage_gloss=coverage_gloss,
        )
        parsed = _llm.call_llm(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            expect_json=True,
            temperature=0.2,
            max_tokens=MAX_OUTPUT_TOKENS,
            timeout=30,
            job_id=job_id,
        )
        if isinstance(parsed, dict):
            llm_summary = (parsed.get("verdict_summary") or "").strip()
            llm_paragraph = (parsed.get("synthesis_paragraph") or "").strip()
            if llm_summary:
                out.verdict_summary = llm_summary[:160]
            if llm_paragraph:
                out.synthesis_paragraph = llm_paragraph[:1200]
            out.llm_used = True
        else:
            out.error = "LLM returned None or non-dict; using deterministic fallback"

    return out
