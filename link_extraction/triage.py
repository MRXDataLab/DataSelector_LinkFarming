"""Step 7 — L6 Measured Triage.

Turns raw `DiscoveredLink` lists into ranked, classified evidence:
each link gets `supports_or_refutes` (verdict), `confidence`, and
`signal_tags` populated. Two paths sharing one classifier:

  • Short-video path  — `ShortVideoLink` carries caption + top_comments +
    transcript already. No body fetch needed. This is the YT Shorts hero
    path: text-rich payload arrives pre-extracted from Step 6.

  • Long-form text path — fetch URL body, strip HTML, take first
    `body_char_budget` characters as the evidence snippet.

Pipeline per hypothesis:

  1. Build per-hypothesis triage context (regex from expected_signals +
     expected_counter_signals + decomposition pains/aspirations).
  2. Pre-rank candidates and cap at `max_triage` (locked = 30 per §6.2).
     Short-video: by engagement_score. Long-form: by initial order.
  3. Extract evidence text per link.
  4. Compute per-link `signal_hit_rate` + `counter_hit_rate` via regex —
     this becomes both a triage feature AND the deterministic fallback
     verdict source when the LLM is unavailable.
  5. ONE Gemini batched call per `batch_size` links, JSON-mode, returns
     verdict + confidence + extra signal_tags.
  6. Mutate links with the verdict and return final ranked list (supports
     first by confidence desc, then refutes, then tangential).

Hard contract — `triage(...)` never raises. Empty input → empty output.
LLM failure → deterministic fallback. Body fetch failure → triage uses
title+snippet only.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Pattern, Sequence, Tuple

import requests

from . import _llm
from .decomposer import Decomposition
from .models import DiscoveredLink, ShortVideoLink, Verdict

log = logging.getLogger(__name__)


# ─── Defaults (locked per handoff §6.2) ──────────────────────────────────────

MAX_TRIAGE_PER_HYPOTHESIS = 30   # top-30 fetch budget (recommendation #2)
BODY_CHAR_BUDGET = 1500           # per the build plan §L6
BATCH_SIZE = 6                    # links per LLM call (headroom for verdict JSON)
BODY_FETCH_TIMEOUT_SEC = 5
# Each verdict object is ~120-180 tokens of JSON; budget ~250 per item plus
# overhead so 6-item batches comfortably finish under cap.
LLM_MAX_TOKENS_BASE = 800
LLM_MAX_TOKENS_PER_ITEM = 260

# Words that don't contribute to signal regex (don't widen the net).
_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "rate", "rates", "ratio", "the", "to", "with",
}

# Domains that almost always block bot fetches; skip them rather than waste
# the 5 s budget. Triage proceeds with title+snippet only.
_FETCH_BLOCKLIST = {
    "instagram.com", "www.instagram.com",
    "tiktok.com", "www.tiktok.com",
    "x.com", "twitter.com",
    "facebook.com", "www.facebook.com",
}


# ─── Triage context ──────────────────────────────────────────────────────────


@dataclass
class TriageContext:
    hypothesis_id: str
    hypothesis_statement: str
    expected_signals: List[str]
    expected_counter_signals: List[str]
    signal_terms: List[str]          # expanded terms used in regex
    counter_terms: List[str]
    signal_regex: Optional[Pattern]
    counter_regex: Optional[Pattern]
    decomp: Decomposition


def _terms_from_signals(signals: Iterable[str], decomp_extras: Iterable[str]) -> List[str]:
    """Expand snake_case signal names + decomposer hits into regex terms.

    'taste_complaints' → ['taste', 'complaints']; merges in pain/aspiration
    words from decomposition for richer matching. Drops _STOPWORDS.
    """
    seen: set[str] = set()
    out: List[str] = []
    for sig in signals:
        if not isinstance(sig, str):
            continue
        for piece in re.split(r"[_\s/]+", sig.strip().lower()):
            piece = piece.strip()
            if not piece or piece in _STOPWORDS or piece in seen:
                continue
            seen.add(piece)
            out.append(piece)
    for term in decomp_extras:
        t = (term or "").strip().lower()
        if not t or t in _STOPWORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _compile_terms(terms: List[str]) -> Optional[Pattern]:
    """Compile a case-insensitive regex that allows common English suffixes.

    `\\bterm\\b` is too strict — 'product' wouldn't match 'products',
    'complaint' wouldn't match 'complaints'/'complaining'. We allow up to
    a 3-char suffix so plurals/conjugations match while keeping false
    positives bounded.
    """
    if not terms:
        return None
    escaped = [re.escape(t) for t in terms if t]
    if not escaped:
        return None
    return re.compile(
        r"\b(" + "|".join(escaped) + r")(?:s|es|ed|ing|er|ers|ly)?\b",
        re.IGNORECASE,
    )


def build_context(
    hypothesis: Dict[str, Any], decomp: Decomposition
) -> TriageContext:
    """Build triage context once per hypothesis. Cheap; safe to recompute."""
    expected = list(hypothesis.get("expected_signals") or [])
    counters = list(hypothesis.get("expected_counter_signals") or [])
    # Decomposer-observed pains reinforce "refutes/supports decline" detection.
    signal_terms = _terms_from_signals(expected, list(decomp.pains))
    counter_terms = _terms_from_signals(counters, list(decomp.aspirations))
    return TriageContext(
        hypothesis_id=decomp.hypothesis_id,
        hypothesis_statement=hypothesis.get("statement", ""),
        expected_signals=expected,
        expected_counter_signals=counters,
        signal_terms=signal_terms,
        counter_terms=counter_terms,
        signal_regex=_compile_terms(signal_terms),
        counter_regex=_compile_terms(counter_terms),
        decomp=decomp,
    )


# ─── Text extraction ─────────────────────────────────────────────────────────


def _short_video_text(link: ShortVideoLink, char_budget: int) -> str:
    """Concat caption + top_comments + transcript into one triage blob."""
    chunks: List[str] = []
    if link.title:
        chunks.append(link.title)
    if link.caption and link.caption != link.title:
        chunks.append(link.caption)
    if link.top_comments:
        chunks.append("Top comments: " + " // ".join(link.top_comments))
    if link.transcript:
        chunks.append("Transcript: " + link.transcript)
    text = "\n".join(chunks).strip()
    return text[:char_budget]


def _strip_html(html: str) -> str:
    """Conservative HTML → plain text. Drops scripts/styles, decodes basics."""
    if not html:
        return ""
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = (
        text.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
    )
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_body(
    url: str, char_budget: int, timeout: float = BODY_FETCH_TIMEOUT_SEC
) -> str:
    """Best-effort body fetch with HTML strip. Empty string on any failure."""
    try:
        host = re.sub(r"^https?://", "", url).split("/")[0].lower()
        if host in _FETCH_BLOCKLIST:
            return ""
        r = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; OuttlyrTriage/1.0; "
                    "+https://github.com/MRXDataLab/IntentTerminal_V2.1)"
                ),
                "Accept": "text/html,application/xhtml+xml",
            },
            allow_redirects=True,
        )
        ctype = r.headers.get("Content-Type", "").lower()
        if r.status_code >= 400 or "html" not in ctype:
            return ""
        return _strip_html(r.text)[:char_budget]
    except Exception as e:
        log.debug("body fetch failed for %s: %s", url, e)
        return ""


def extract_text(
    link: DiscoveredLink, char_budget: int = BODY_CHAR_BUDGET
) -> str:
    """Pull triage-relevant text from any DiscoveredLink. Sync; safe to thread."""
    # Always include title + snippet as anchor (they're cheap and reliable).
    anchor_parts: List[str] = []
    if link.title:
        anchor_parts.append(link.title)
    if link.snippet:
        anchor_parts.append(link.snippet)
    anchor = " — ".join(anchor_parts)

    if isinstance(link, ShortVideoLink):
        sv_text = _short_video_text(link, char_budget)
        if anchor and anchor not in sv_text:
            sv_text = (anchor + "\n" + sv_text).strip()
        return sv_text[:char_budget]

    body = _fetch_body(link.url, char_budget=max(0, char_budget - len(anchor) - 8))
    if anchor and body:
        return f"{anchor}\n{body}"[:char_budget]
    return (body or anchor)[:char_budget]


# ─── Signal hit rate ─────────────────────────────────────────────────────────


def _hit_rate(text: str, regex: Optional[Pattern]) -> Tuple[float, List[str]]:
    """Return (normalized hit count, matched-term list).

    Rate = matches / max(words, 1) so longer texts don't dominate. Bounded
    at 1.0. Matched terms preserved in first-seen order for signal_tags.
    """
    if not text or regex is None:
        return 0.0, []
    matches = regex.findall(text)
    if not matches:
        return 0.0, []
    word_count = max(len(re.findall(r"\w+", text)), 1)
    rate = min(1.0, len(matches) / word_count)
    seen: set[str] = set()
    tags: List[str] = []
    for m in matches:
        ml = m.lower()
        if ml in seen:
            continue
        seen.add(ml)
        tags.append(ml)
    return rate, tags


# ─── Deterministic fallback verdict ──────────────────────────────────────────


def _deterministic_verdict(
    text: str, sig_rate: float, ctr_rate: float,
    strictness: str = "liberal",   # value-equal to DEFAULT_STRICTNESS, kept inline to avoid forward-ref
) -> Tuple[Verdict, float]:
    """Rule-based fallback when the LLM is unavailable.

    - If no text and no hits → tangential (low confidence).
    - If counter hit rate dominates → refutes.
    - If signal hit rate dominates → supports.
    - Otherwise tangential.

    `strictness=liberal` widens the supports/refutes net to also catch
    weak-but-directional hits (any signal match at all), and bumps the
    base confidence so tangential is rarely the safer call.
    """
    if not text:
        return "tangential", 0.2 if strictness != "liberal" else 0.3
    if ctr_rate > sig_rate and ctr_rate > 0:
        base = 0.5 if strictness == "liberal" else 0.4
        return "refutes", min(0.85, base + ctr_rate * 25)
    if sig_rate > 0:
        base = 0.5 if strictness == "liberal" else 0.4
        return "supports", min(0.85, base + sig_rate * 25)
    return "tangential", 0.3


# ─── LLM batch verdict ───────────────────────────────────────────────────────


# ─── Triage strictness ───────────────────────────────────────────────────────
# Three prompt variants. "liberal" is the default per locked decision so the
# analyst sees more decisive verdicts; "strict" is the original cautious mode.

TriageStrictness = str  # Literal["strict", "balanced", "liberal"]

_TRIAGE_PROMPT_COMMON = (
    "You are a measured-triage classifier for market-research evidence. "
    "Given one hypothesis and a numbered batch of evidence snippets, "
    "classify each snippet as 'supports', 'refutes', or 'tangential' with "
    "a calibrated confidence in [0.0, 1.0]. "
    "DEFINITIONS: "
    "'supports' = the evidence is consistent with what the hypothesis predicts; "
    "'refutes' = the evidence contradicts the hypothesis; "
    "'tangential' = on-topic but neither supports nor refutes. "
)

_TRIAGE_PROMPT_OUTPUT = (
    "Return ONLY a JSON object with this exact shape: "
    '{"verdicts": [{"id": <int>, "verdict": "supports|refutes|tangential", '
    '"confidence": <float 0..1>, "signal_tags": [<short keyword>, ...]}]}. '
    "Output one entry per input id. No prose, no markdown."
)

_STANCE_STRICT = (
    "Be conservative with 'supports'/'refutes' — prefer 'tangential' when "
    "the evidence is thin, promotional, or unrelated to the hypothesis's "
    "specific claim. Set high confidence (>0.7) only on unambiguous matches. "
)
_STANCE_BALANCED = (
    "Use your best judgement: call 'supports'/'refutes' when the evidence "
    "clearly points one way, 'tangential' when the link is on-topic but "
    "doesn't actually answer the hypothesis. Calibrate confidence honestly. "
)
_STANCE_LIBERAL = (
    "Lean toward 'supports' or 'refutes' whenever the evidence has a "
    "discernible directional signal — even partial / indirect / single-data-"
    "point evidence still counts. Reserve 'tangential' for content that is "
    "genuinely off-topic, promotional fluff, or so generic it could apply to "
    "any brand in the category. When in doubt between tangential and a weak "
    "support/refute, PREFER the directional verdict at confidence 0.4-0.6. "
)

_TRIAGE_PROMPTS: Dict[str, str] = {
    "strict":   _TRIAGE_PROMPT_COMMON + _STANCE_STRICT   + _TRIAGE_PROMPT_OUTPUT,
    "balanced": _TRIAGE_PROMPT_COMMON + _STANCE_BALANCED + _TRIAGE_PROMPT_OUTPUT,
    "liberal":  _TRIAGE_PROMPT_COMMON + _STANCE_LIBERAL  + _TRIAGE_PROMPT_OUTPUT,
}

DEFAULT_STRICTNESS: TriageStrictness = "liberal"


def get_triage_prompt(strictness: TriageStrictness) -> str:
    """Return the LLM system prompt for the requested strictness; falls back
    to 'liberal' if the value is unknown."""
    return _TRIAGE_PROMPTS.get(strictness, _TRIAGE_PROMPTS[DEFAULT_STRICTNESS])


# Kept as a name for backward compat with any external imports.
_TRIAGE_SYSTEM_PROMPT = _TRIAGE_PROMPTS[DEFAULT_STRICTNESS]


def _build_batch_prompt(
    ctx: TriageContext, batch: List[Tuple[int, DiscoveredLink, str]]
) -> str:
    """Compose the user message for one LLM batch call."""
    items = []
    for i, link, text in batch:
        snippet = text[:1000] if text else (link.title or "")[:200]
        items.append({
            "id": i,
            "channel": link.channel,
            "url": link.url,
            "snippet": snippet,
        })
    payload = {
        "hypothesis_id": ctx.hypothesis_id,
        "hypothesis_statement": ctx.hypothesis_statement,
        "expected_signals": ctx.expected_signals,
        "expected_counter_signals": ctx.expected_counter_signals,
        "evidence": items,
    }
    return json.dumps(payload, ensure_ascii=False)


def _llm_verdict_batch(
    ctx: TriageContext,
    batch: List[Tuple[int, DiscoveredLink, str]],
    strictness: TriageStrictness = DEFAULT_STRICTNESS,
) -> Dict[int, Dict[str, Any]]:
    """One Gemini call per batch. Returns {id: {verdict, confidence, signal_tags}}.

    Empty dict on any failure path; caller falls back to deterministic.
    """
    if not batch:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    user_msg = _build_batch_prompt(ctx, batch)
    max_tokens = LLM_MAX_TOKENS_BASE + LLM_MAX_TOKENS_PER_ITEM * len(batch)
    parsed = _llm.call_llm(
        get_triage_prompt(strictness),
        user_msg,
        expect_json=True,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    if not isinstance(parsed, dict):
        log.info("triage LLM batch (%d items) returned non-dict; falling back", len(batch))
        return {}
    verdicts = parsed.get("verdicts")
    if not isinstance(verdicts, list):
        return {}
    valid_verdicts = {"supports", "refutes", "tangential"}
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        try:
            i = int(v.get("id"))
        except (TypeError, ValueError):
            continue
        verdict = v.get("verdict")
        if verdict not in valid_verdicts:
            continue
        try:
            conf = float(v.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        tags = v.get("signal_tags") or []
        if isinstance(tags, list):
            tags = [str(t).strip().lower() for t in tags if isinstance(t, str) and t.strip()]
        else:
            tags = []
        out[i] = {"verdict": verdict, "confidence": conf, "signal_tags": tags}
    return out


# ─── Pre-ranking ─────────────────────────────────────────────────────────────


def _within_channel_score(link: DiscoveredLink) -> float:
    """Order links *within a single channel* before they enter the round-robin.

    - Short-video: by engagement_score (None → 0.0).
    - Everything else: by discovery order (so the FIFO from the backend
      response is preserved). Returns 0.0 here, and we rely on Python's
      stable sort to keep insertion order.
    """
    if isinstance(link, ShortVideoLink):
        return link.engagement_score or 0.0
    return 0.0


def _channel_balanced_cut(
    links: Sequence[DiscoveredLink], max_triage: int,
) -> List[DiscoveredLink]:
    """Pick top `max_triage` links via channel-balanced round-robin.

    Why: the old pure-engagement sort buried text channels (Reddit, Quora,
    News, Marketplace, etc.) because only ShortVideoLink has engagement_score.
    With 60 candidates split 32 YT Shorts / 21 Reddit / 7 YT long-form and
    a budget of 30, the pure sort would take ~30 YT links and drop EVERY
    Reddit thread. The triage CSV would be YT-only.

    The round-robin algorithm:
      1. Group links by channel
      2. Sort each group internally by `_within_channel_score` (engagement
         for shorts, FIFO for text)
      3. Rotate across channels — take 1 link from each in turn — until
         we've collected `max_triage` items or all groups are empty

    Result: diverse channel coverage in the triage cut. Channels with
    fewer links empty out earlier and the remaining slots go to richer
    channels, but no channel is shut out entirely.
    """
    if not links:
        return []

    by_channel: Dict[str, List[DiscoveredLink]] = {}
    for lk in links:
        by_channel.setdefault(lk.channel, []).append(lk)

    # Sort within each channel by best-signal (descending)
    for ch in by_channel:
        by_channel[ch].sort(key=_within_channel_score, reverse=True)

    # Round-robin pop. Channels are visited in their first-seen order
    # from `links` (stable). Within each channel, best link first.
    out: List[DiscoveredLink] = []
    queues = {ch: list(group) for ch, group in by_channel.items()}
    while len(out) < max_triage:
        progress = False
        for ch in list(queues.keys()):
            if not queues[ch]:
                continue
            out.append(queues[ch].pop(0))
            progress = True
            if len(out) >= max_triage:
                break
        if not progress:
            break  # every channel exhausted

    return out


# Kept for backward compat with any external callers
def _pre_rank_score(link: DiscoveredLink) -> float:
    return _within_channel_score(link)


# ─── Public API ──────────────────────────────────────────────────────────────


async def triage(
    hypothesis: Dict[str, Any],
    decomp: Decomposition,
    links: Sequence[DiscoveredLink],
    *,
    max_triage: int = MAX_TRIAGE_PER_HYPOTHESIS,
    body_char_budget: int = BODY_CHAR_BUDGET,
    batch_size: int = BATCH_SIZE,
    use_llm: bool = True,
    strictness: TriageStrictness = DEFAULT_STRICTNESS,
) -> List[DiscoveredLink]:
    """Triage all `links` for one hypothesis. Returns ranked list (supports→refutes→tangential).

    Mutates each link's `supports_or_refutes`, `confidence`, `signal_tags`
    in place and also returns them sorted by (verdict bucket, confidence desc).
    Never raises.
    """
    if not links:
        return []
    ctx = build_context(hypothesis, decomp)

    # Channel-balanced round-robin cut — see `_channel_balanced_cut` docstring
    # for why this replaces the old pure-engagement sort.
    capped = _channel_balanced_cut(links, max_triage)

    # Extract evidence text per link (off the event loop for body fetches)
    evidence: List[Tuple[int, DiscoveredLink, str, float, float, List[str]]] = []
    extract_calls = [
        asyncio.to_thread(extract_text, link, body_char_budget)
        for link in capped
    ]
    texts = await asyncio.gather(*extract_calls, return_exceptions=True)
    for i, (link, text_or_exc) in enumerate(zip(capped, texts)):
        if isinstance(text_or_exc, Exception):
            log.info("extract_text raised for %s: %s", link.url, text_or_exc)
            text = link.title or ""
        else:
            text = text_or_exc or ""
        sig_rate, sig_tags = _hit_rate(text, ctx.signal_regex)
        ctr_rate, ctr_tags = _hit_rate(text, ctx.counter_regex)
        evidence.append((i, link, text, sig_rate, ctr_rate, sig_tags + ctr_tags))

    # LLM batch verdicts (best-effort, deterministic fallback per link)
    llm_results: Dict[int, Dict[str, Any]] = {}
    if use_llm and _llm.is_available() and evidence:
        for start in range(0, len(evidence), batch_size):
            chunk = evidence[start: start + batch_size]
            batch_input = [(i, link, text) for (i, link, text, *_rest) in chunk]
            llm_results.update(
                await asyncio.to_thread(_llm_verdict_batch, ctx, batch_input, strictness)
            )

    # Apply verdicts
    out: List[DiscoveredLink] = []
    for (i, link, text, sig_rate, ctr_rate, hit_tags) in evidence:
        llm_v = llm_results.get(i)
        if llm_v:
            link.supports_or_refutes = llm_v["verdict"]
            link.confidence = llm_v["confidence"]
            merged_tags = list({*hit_tags, *llm_v.get("signal_tags", [])})
        else:
            verdict, conf = _deterministic_verdict(text, sig_rate, ctr_rate, strictness)
            link.supports_or_refutes = verdict
            link.confidence = conf
            merged_tags = list({*hit_tags})
        # Preserve any prior signal_tags from earlier stages, plus what we found
        link.signal_tags = sorted(set(link.signal_tags) | set(merged_tags))
        out.append(link)

    # Final ranking: supports first (by confidence desc), then refutes,
    # then tangential. Within bucket, higher confidence wins.
    bucket_order: Dict[str, int] = {"supports": 0, "refutes": 1, "tangential": 2}
    out.sort(
        key=lambda lk: (
            bucket_order.get(lk.supports_or_refutes or "tangential", 3),
            -(lk.confidence or 0.0),
        )
    )
    return out


def group_by_verdict(
    links: Sequence[DiscoveredLink],
) -> Dict[str, List[DiscoveredLink]]:
    """Convenience for UI/CSV — group triaged links by their verdict."""
    out: Dict[str, List[DiscoveredLink]] = {
        "supports": [], "refutes": [], "tangential": []
    }
    for link in links:
        bucket = link.supports_or_refutes or "tangential"
        out.setdefault(bucket, []).append(link)
    return out
