"""Step 5 — L2 Query Synthesizer (9 archetypes + Trends amplifier).

Builds per-channel `TypedQuery` lists for one hypothesis. Three input sources:
  1. `Decomposition`         — entities, pains, aspirations, identity, geo
  2. Trends rising_queries   — live consumer phrasings (Step 4)
  3. One LLM slot-filler call — picks alt products, dim terms, expert roles,
     category terms, year anchors that aren't in the hypothesis text

The LLM **fills slots only** — it does not freewrite queries. Every final
query string is assembled by deterministic Python from a fixed template
table. The archetype mix per channel is curated, not LLM-driven.

Hard validations (RAISE if violated):
  • Each channel returned must include ≥1 counter_evidence (falsifier) query
  • Each channel returned must include ≥1 pain-bearing query OR ≥1 aspiration
    query (when the decomposition supplies either)

If the LLM is unavailable, falls back to a deterministic-only slot set
derived from `Decomposition.competitor_anchors` + sensible defaults.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import _llm
from .decomposer import Decomposition
from .models import (
    ARCHETYPE_NAMES,
    SHORT_VIDEO_CHANNELS,
    Archetype,
    ChannelFit,
    ChannelId,
    TimeWindow,
    TypedQuery,
)
from .trends_seed import fetch_rising_queries, geo_from_hints

log = logging.getLogger(__name__)


# ─── Channel × archetype mix ─────────────────────────────────────────────────
#
# Curated, not LLM-driven. The mix reflects each channel's native content
# patterns (PAA = questions; news = crises; marketplace = product comparisons;
# short-video = hashtags + identity).
#
# Archetype IDs map to ARCHETYPE_NAMES:
#   1 entity_pain     2 switching_narrative   3 comparison
#   4 identity_aspiration  5 question_analysis  6 counter_evidence (FALSIFIER)
#   7 expert_authority     8 crisis_event       9 hashtag_trend

CHANNEL_ARCHETYPES: Dict[ChannelId, List[Archetype]] = {
    # Short-video — full short-video template set, hashtag mandatory.
    "youtube_shorts": [1, 2, 3, 4, 6, 7, 9],
    "tiktok": [1, 2, 3, 4, 6, 8, 9],
    "instagram_reels": [1, 2, 4, 6, 9],
    # Long-form text channels
    "reddit": [1, 2, 3, 5, 6, 8],
    "quora": [3, 4, 5, 6],
    "google_web": [1, 2, 3, 5, 6, 7],
    "google_paa": [3, 5, 6],            # PAA biases hard toward questions
    "google_related": [1, 3, 5],
    "youtube": [1, 3, 4, 6, 7, 8],      # long-form video
    "news": [1, 6, 8],                  # crisis-heavy
    "substack": [3, 4, 7, 8],
    "marketplace": [1, 3, 4, 6],        # review-leaning
    "trends": [],                       # amplifier only, never a discovery target
}

# Hashtag archetype (9) only applies to short-video channels by spec.
# Question archetype (5) is "rarely used" in short-video — excluded above.

MAX_QUERIES_PER_CHANNEL = 8

# Defaults when the LLM slot-fill is unavailable or returns blanks.
_FALLBACK_DIM_TERMS = ["health", "taste", "price"]
_FALLBACK_EXPERT_ROLES = ["expert", "nutritionist", "reviewer"]
_FALLBACK_ALTERNATIVES: List[str] = []  # forces template skip when truly empty


# ─── Slot bundle ─────────────────────────────────────────────────────────────


class _Slots:
    """All values needed to fill templates. Plain attribute container."""

    def __init__(
        self,
        *,
        brand: str,
        brand_aliases: List[str],
        alternatives: List[str],
        pains: List[str],
        aspirations: List[str],
        identity: List[str],
        geo: List[str],
        category_terms: List[str],
        dimension_terms: List[str],
        expert_roles: List[str],
        year_anchors: List[str],
        rising_phrases: List[str],
    ) -> None:
        self.brand = brand
        self.brand_aliases = brand_aliases
        self.alternatives = alternatives
        self.pains = pains
        self.aspirations = aspirations
        self.identity = identity
        self.geo = geo
        self.category_terms = category_terms
        self.dimension_terms = dimension_terms
        self.expert_roles = expert_roles
        self.year_anchors = year_anchors
        self.rising_phrases = rising_phrases


# ─── LLM slot-filler ─────────────────────────────────────────────────────────


_SLOT_SYSTEM_PROMPT = (
    "You are a precise slot-filler for a market-research query generator. "
    "Given one hypothesis about a brand/product, output ONLY a JSON object "
    "with these exact keys: "
    "category_terms (2-3 short category words, e.g. ['breakfast cereal', 'cereal']), "
    "dimension_terms (2-4 comparison axes consumers care about, "
    "e.g. ['health','taste','nutrition','price']), "
    "expert_roles (2-3 expert nouns, e.g. ['nutritionist','dietitian','food blogger']), "
    "alternative_products (3-5 plausible substitute brand/product names), "
    "year_anchors (1-2 year strings like ['2024','2025']). "
    "Use lowercase, plural where natural, no explanations, no markdown."
)


def _slot_fill_via_llm(
    hypothesis: Dict[str, Any], decomp: Decomposition
) -> Optional[Dict[str, List[str]]]:
    """Single LLM call to fill the reasoning slots. None on any failure."""
    if not _llm.is_available():
        return None

    user_msg = json.dumps(
        {
            "hypothesis_id": decomp.hypothesis_id,
            "statement": hypothesis.get("statement", ""),
            "rationale": hypothesis.get("rationale", ""),
            "dimension": hypothesis.get("dimension", ""),
            "primary_entity": decomp.primary_entity,
            "competitor_anchors": decomp.competitor_anchors,
            "pains": decomp.pains,
            "aspirations": decomp.aspirations,
            "geo_hints": decomp.geo_hints,
        },
        ensure_ascii=False,
    )

    raw = _llm.call_llm(_SLOT_SYSTEM_PROMPT, user_msg, expect_json=True)
    if not isinstance(raw, dict):
        return None

    def _str_list(key: str, cap: int) -> List[str]:
        val = raw.get(key, [])
        if not isinstance(val, list):
            return []
        out: List[str] = []
        seen: set[str] = set()
        for x in val:
            if not isinstance(x, str):
                continue
            s = x.strip().lower()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) >= cap:
                break
        return out

    return {
        "category_terms": _str_list("category_terms", 3),
        "dimension_terms": _str_list("dimension_terms", 4),
        "expert_roles": _str_list("expert_roles", 3),
        "alternative_products": _str_list("alternative_products", 5),
        "year_anchors": _str_list("year_anchors", 2),
    }


# ─── Slot assembly ───────────────────────────────────────────────────────────


def _assemble_slots(
    hypothesis: Dict[str, Any],
    decomp: Decomposition,
    window: TimeWindow,
    *,
    use_llm: bool = True,
    rising_phrases_override: Optional[List[str]] = None,
) -> _Slots:
    """Build the _Slots bundle. LLM call is best-effort; Trends call is too."""
    brand = decomp.primary_entity or (decomp.entities[0] if decomp.entities else "")
    aliases = [e for e in decomp.entities if e != brand][:3]

    llm_slots = _slot_fill_via_llm(hypothesis, decomp) if use_llm else None

    if llm_slots:
        category_terms = llm_slots["category_terms"]
        dimension_terms = llm_slots["dimension_terms"] or _FALLBACK_DIM_TERMS[:]
        expert_roles = llm_slots["expert_roles"] or _FALLBACK_EXPERT_ROLES[:]
        # Pool LLM alternatives + decomposer-observed competitor anchors.
        alt_pool = list(llm_slots["alternative_products"])
        for a in decomp.competitor_anchors:
            a_low = a.strip().lower()
            if a_low and a_low != (brand or "").lower() and a_low not in alt_pool:
                alt_pool.append(a_low)
        alternatives = alt_pool[:5]
        year_anchors = llm_slots["year_anchors"] or ["2024", "2025"]
    else:
        # Deterministic-only fallback path
        category_terms = []
        dimension_terms = _FALLBACK_DIM_TERMS[:]
        expert_roles = _FALLBACK_EXPERT_ROLES[:]
        alternatives = [a.lower() for a in decomp.competitor_anchors][:5]
        year_anchors = ["2024", "2025"]

    # Trends amplifier — best-effort live pull, scoped to geo + window.
    geo_code = geo_from_hints(decomp.geo_hints)
    if rising_phrases_override is not None:
        rising = list(rising_phrases_override)
    elif brand:
        rising = fetch_rising_queries(brand, window, geo=geo_code, top_k=5)
    else:
        rising = []

    return _Slots(
        brand=brand,
        brand_aliases=aliases,
        alternatives=alternatives,
        pains=list(decomp.pains),
        aspirations=list(decomp.aspirations),
        identity=list(decomp.identity_claims),
        geo=list(decomp.geo_hints),
        category_terms=category_terms,
        dimension_terms=dimension_terms,
        expert_roles=expert_roles,
        year_anchors=year_anchors,
        rising_phrases=rising,
    )


# ─── Template generators (one per archetype) ─────────────────────────────────


def _gen_archetype(
    archetype: Archetype, slots: _Slots, *, short_video: bool
) -> List[Tuple[str, str]]:
    """Return list of (query_text, target_signal) for one archetype.

    Empty list if required slots are missing for this archetype.
    """
    b = slots.brand
    if not b:
        return []

    if archetype == 1:  # entity_pain
        if short_video:
            return [
                (f"{b} cringe", "user_complaint_clip"),
                (f"{b} fail", "user_complaint_clip"),
            ]
        out = []
        for p in slots.pains[:3]:
            out.append((f"{b} {p}", "user_complaint_text"))
        if not out:
            out.append((f"{b} review negative", "user_complaint_text"))
        return out

    if archetype == 2:  # switching_narrative
        if not slots.alternatives:
            return []
        if short_video:
            return [
                (f"pov i stopped buying {b}", "switching_narrative"),
                (f"why i quit {b}", "switching_narrative"),
            ]
        return [
            (f"why i switched from {b} to {alt}", "switching_narrative")
            for alt in slots.alternatives[:3]
        ]

    if archetype == 3:  # comparison
        if not slots.alternatives:
            return []
        if short_video:
            return [
                (f"{b} vs {alt}", "comparison_video")
                for alt in slots.alternatives[:3]
            ]
        out = []
        dims = slots.dimension_terms or ["review"]
        for alt in slots.alternatives[:3]:
            dim = dims[0]
            out.append((f"{b} vs {alt} {dim}", "comparison_text"))
        return out

    if archetype == 4:  # identity_aspiration
        if not slots.aspirations:
            return []
        cat = slots.category_terms[0] if slots.category_terms else (b or "product")
        if short_video:
            return [
                (f"{slots.aspirations[0]} {cat} routine", "aspiration_clip"),
            ]
        out = []
        for asp in slots.aspirations[:2]:
            out.append((f"best {cat} for {asp}", "aspiration_post"))
        return out

    if archetype == 5:  # question_analysis (long-form only by spec)
        if short_video:
            return []
        if slots.rising_phrases:
            return [
                (f"why is {b} {phrase.lower()}", "question_thread")
                for phrase in slots.rising_phrases[:2]
                if phrase.lower() != b.lower()
            ] or [(f"why is {b} losing customers", "question_thread")]
        if slots.pains:
            return [(f"why is {b} {slots.pains[0]}", "question_thread")]
        return [(f"why is {b} declining", "question_thread")]

    if archetype == 6:  # counter_evidence — MANDATORY FALSIFIER
        if short_video:
            return [
                (f"defending {b}", "counter_evidence"),
                (f"why i love {b}", "counter_evidence"),
            ]
        return [
            (f"why i love {b}", "counter_evidence"),
            (f"{b} is actually good", "counter_evidence"),
        ]

    if archetype == 7:  # expert_authority
        if not slots.expert_roles:
            return []
        cat = slots.category_terms[0] if slots.category_terms else b
        if short_video:
            return [
                (f"{role} reacts {b}", "expert_review_clip")
                for role in slots.expert_roles[:2]
            ]
        return [
            (f"{cat} {role}", "expert_authority_text")
            for role in slots.expert_roles[:2]
        ]

    if archetype == 8:  # crisis_event
        if short_video:
            return [
                (f"{b} scandal", "crisis_clip"),
                (f"{b} controversy", "crisis_clip"),
            ]
        return [
            (f"{b} {y}", "crisis_event") for y in slots.year_anchors[:2]
        ] + [(f"{b} controversy", "crisis_event")]

    if archetype == 9:  # hashtag_trend — short-video only by spec
        if not short_video:
            return []
        b_tag = b.replace(" ", "").replace("'", "").lower()
        out = [(f"#{b_tag}", "hashtag_trend")]
        if slots.category_terms and slots.aspirations:
            cat_tag = slots.category_terms[0].replace(" ", "")
            asp_tag = slots.aspirations[0].replace(" ", "")
            out.append((f"#{cat_tag}{asp_tag}", "hashtag_trend"))
        return out

    return []


# ─── Public API ──────────────────────────────────────────────────────────────


def synthesize_queries(
    hypothesis: Dict[str, Any],
    decomp: Decomposition,
    channel_fits: Sequence[ChannelFit],
    window: TimeWindow,
    *,
    use_llm: bool = True,
    rising_phrases_override: Optional[List[str]] = None,
    max_per_channel: int = MAX_QUERIES_PER_CHANNEL,
) -> Dict[ChannelId, List[TypedQuery]]:
    """Synthesize typed queries for every channel above L1's threshold.

    Args:
        hypothesis: raw hypothesis dict (statement / rationale / dimension)
        decomp: L0 output for the same hypothesis
        channel_fits: L1 output — only channels in this list get queries
        window: user-selected TimeWindow (used for Trends amplifier)
        use_llm: set False to force deterministic fallback (tests / offline)
        rising_phrases_override: inject Trends results from elsewhere (tests)
        max_per_channel: hard cap to keep per-hypothesis query budget bounded

    Returns:
        `{channel_id: [TypedQuery, ...]}` — never includes a channel that
        couldn't generate ≥1 falsifier.
    """
    slots = _assemble_slots(
        hypothesis,
        decomp,
        window,
        use_llm=use_llm,
        rising_phrases_override=rising_phrases_override,
    )

    if not slots.brand:
        log.warning("synthesize_queries: no brand extracted from %s", decomp.hypothesis_id)
        return {}

    geo_proxies = [g for g in decomp.geo_hints]

    out: Dict[ChannelId, List[TypedQuery]] = {}
    for cf in channel_fits:
        channel: ChannelId = cf.channel
        archetypes = CHANNEL_ARCHETYPES.get(channel, [])
        if not archetypes:
            continue

        short_video = channel in SHORT_VIDEO_CHANNELS
        produced: List[TypedQuery] = []
        seen_text: set[str] = set()

        for arch in archetypes:
            for text, target_signal in _gen_archetype(arch, slots, short_video=short_video):
                key = text.lower().strip()
                if not key or key in seen_text:
                    continue
                seen_text.add(key)
                produced.append(
                    TypedQuery(
                        text=text,
                        channel=channel,
                        archetype=arch,
                        archetype_name=ARCHETYPE_NAMES[arch],
                        target_signal=target_signal,
                        hypothesis_id=decomp.hypothesis_id,
                        falsifier=(arch == 6),
                        geo_proxies=geo_proxies,
                    )
                )
                if len(produced) >= max_per_channel:
                    break
            if len(produced) >= max_per_channel:
                break

        produced = _ensure_falsifier(produced, channel, decomp.hypothesis_id, slots, geo_proxies)
        if not _validate_channel_coverage(produced, slots):
            log.info(
                "channel %s for %s failed coverage; dropping",
                channel, decomp.hypothesis_id,
            )
            continue

        out[channel] = produced

    return out


# ─── Validation helpers ──────────────────────────────────────────────────────


def _ensure_falsifier(
    queries: List[TypedQuery],
    channel: ChannelId,
    hypothesis_id: str,
    slots: _Slots,
    geo_proxies: List[str],
) -> List[TypedQuery]:
    """Inject a counter_evidence query if none present. Falsifier is MANDATORY."""
    if any(q.falsifier for q in queries):
        return queries
    short_video = channel in SHORT_VIDEO_CHANNELS
    if not slots.brand:
        return queries
    text = f"defending {slots.brand}" if short_video else f"why i love {slots.brand}"
    queries.append(
        TypedQuery(
            text=text,
            channel=channel,
            archetype=6,
            archetype_name=ARCHETYPE_NAMES[6],
            target_signal="counter_evidence",
            hypothesis_id=hypothesis_id,
            falsifier=True,
            geo_proxies=geo_proxies,
        )
    )
    return queries


def _validate_channel_coverage(queries: List[TypedQuery], slots: _Slots) -> bool:
    """Per-channel validation — at least one falsifier + (pain OR aspiration cover).

    If the decomposition surfaced neither pains nor aspirations, only the
    falsifier requirement is enforced.
    """
    if not queries:
        return False
    if not any(q.falsifier for q in queries):
        return False
    has_pain_or_aspiration = any(
        q.archetype in (1, 4) for q in queries
    )
    decomp_has_either = bool(slots.pains or slots.aspirations)
    if decomp_has_either and not has_pain_or_aspiration:
        return False
    return True


# ─── MECE pair pooling (per locked decision §6.3) ────────────────────────────


def synthesize_pair_pooled(
    primary_hypothesis: Dict[str, Any],
    primary_decomp: Decomposition,
    contrarian_hypothesis: Dict[str, Any],
    contrarian_decomp: Decomposition,
    channel_fits: Sequence[ChannelFit],
    window: TimeWindow,
    **kwargs: Any,
) -> Dict[ChannelId, List[TypedQuery]]:
    """Pool queries from a contrarian pair into one cluster.

    Per locked decision §6.3: when a hypothesis has `contrarian_pair_id`,
    synthesize queries for both sides and merge by channel. Each query
    keeps its originating `hypothesis_id` so triage attributes correctly.
    """
    primary = synthesize_queries(
        primary_hypothesis, primary_decomp, channel_fits, window, **kwargs
    )
    contrarian = synthesize_queries(
        contrarian_hypothesis, contrarian_decomp, channel_fits, window, **kwargs
    )

    merged: Dict[ChannelId, List[TypedQuery]] = {}
    for channel in set(primary) | set(contrarian):
        seen: set[str] = set()
        combined: List[TypedQuery] = []
        for q in primary.get(channel, []) + contrarian.get(channel, []):
            key = q.text.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            combined.append(q)
        merged[channel] = combined
    return merged
