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
    # Archetype 10 = proxy_search (geo+category+competitor combos).
    # PLACED EARLY in each channel's list so proxy queries get priority
    # budget allocation — they're the highest-yield queries for
    # branded hypotheses with thin direct coverage.
    #
    # Phase 1.5 fix — drop nonsense archetype×channel combos that produced
    # zero-result queries against host-restricted backends:
    #   * marketplace dropped archetype 6 (counter_evidence: "why i love X")
    #     — Amazon / Zepto / MakeMyTrip don't host opinion content.
    #   * news dropped archetype 6 — news sites don't host first-person
    #     "why i love" stories either.
    #   * marketplace dropped archetype 4 (identity_aspiration: "best X for
    #     gen-z") — also rare on marketplace surface; reviews-leaning
    #     pages are surfaced by archetypes 1+3.
    #
    # Short-video — full short-video template set, hashtag mandatory.
    "youtube_shorts": [1, 10, 2, 3, 4, 6, 7, 9],
    "tiktok": [1, 10, 2, 3, 4, 6, 8, 9],
    "instagram_reels": [1, 10, 2, 4, 6, 9],
    # Long-form text channels
    "reddit": [10, 1, 2, 3, 5, 6, 8],
    "quora": [10, 3, 4, 5, 6],
    "google_web": [10, 1, 2, 3, 5, 6, 7],
    "google_paa": [10, 3, 5, 6],            # PAA biases hard toward questions
    "google_related": [10, 1, 3, 5],
    "youtube": [10, 1, 3, 4, 6, 7, 8],      # long-form video
    "news": [10, 1, 8],                     # crisis-heavy (dropped 6)
    "substack": [10, 3, 4, 7, 8],
    "marketplace": [10, 1, 3],              # review-leaning (dropped 4, 6)
    "trends": [],                       # amplifier only, never a discovery target
    # Phase 1.6 — new channels
    "scholar": [10, 5, 7, 1],           # academic: questions + expert + pain
    "google_maps": [10, 1, 3],          # local-place reviews: pain + comparison
}

# Hashtag archetype (9) only applies to short-video channels by spec.
# Question archetype (5) is "rarely used" in short-video — excluded above.

# Per-channel query budget. Phase 2 bumped to 18 so manifest-aware
# proxies (life-triggers × brand, cohort × brand, competitor full
# crossproduct) all fit alongside the brand-bound templates. At 18 ×
# ~7 active channels ≈ 126 backend calls per hypothesis — still inside
# Brave's 2k/mo allowance for ~15 hypotheses/month. When triage is off
# (skip_triage=True), there's no LLM cost so this is mostly headless +
# DDG load which scales fine.
MAX_QUERIES_PER_CHANNEL = 18

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
        category_topics: Optional[List[str]] = None,
        no_brand_mode: bool = False,
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
        # No-brand fallback: when the hypothesis has no real brand named,
        # the synthesizer uses these category topics (drawn from
        # core_problem_statement + rationale) as the search subject.
        self.category_topics = category_topics or []
        self.no_brand_mode = no_brand_mode


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

    # Category topics extracted by the decomposer from core_problem +
    # rationale + statement. Always merged into category_terms /
    # dimension_terms so the proxy_search archetype (10) has rich
    # geo × category material — whether or not a brand was found.
    decomp_topics = list(getattr(decomp, "category_topics", []) or [])
    if decomp_topics:
        for t in decomp_topics:
            if t not in category_terms and len(category_terms) < 6:
                category_terms.append(t)

    # No-brand mode: hypothesis had no real entity to anchor queries on
    # (e.g. "the developer lacks…" with no specific firm named). Use the
    # category topics extracted by the decomposer as the search subject.
    no_brand = not bool(brand)
    if no_brand and decomp_topics:
        # Pick a synthetic "brand" — prefer a domain phrase (multi-word
        # category term like "real estate", "site visit") over a single
        # word, then fall back to the first topic. The decomposer puts
        # domain terms BEFORE signal-derived terms in the list, so the
        # first multi-word phrase tends to be the most natural anchor.
        multi_word = [t for t in decomp_topics if " " in t]
        if multi_word:
            synthetic = multi_word[0]
        else:
            synthetic = decomp_topics[0]
        brand = synthetic
        # Aliases — additional domain terms the templates can blend in.
        aliases = [t for t in decomp_topics
                   if t != synthetic and len(t.split()) <= 2][:3]
        # Also merge the topics into the dimension/category buckets so
        # archetype 3 (comparison) and 5 (question_analysis) have richer
        # input. Avoid duplicates while preserving order.
        for t in decomp_topics:
            if t not in category_terms and len(category_terms) < 5:
                category_terms.append(t)
            if t not in dimension_terms and len(dimension_terms) < 8:
                dimension_terms.append(t)

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
        category_topics=decomp_topics,
        no_brand_mode=no_brand,
    )


# ─── Template generators (one per archetype) ─────────────────────────────────


def _gen_archetype(
    archetype: Archetype, slots: _Slots, *, short_video: bool
) -> List[Tuple[str, str]]:
    """Return list of (query_text, target_signal) for one archetype.

    Empty list if required slots are missing for this archetype.

    In ``no_brand_mode`` (no specific brand named in the hypothesis) the
    templates degrade gracefully to category-driven queries. e.g. an
    "entity_pain" query becomes a buyer-complaint search ("property
    review negative") rather than a literal brand-name search.
    """
    b = slots.brand
    if not b:
        return []

    # In no-brand mode, falsifier/love templates ("why i love real estate")
    # read oddly, so we route those archetypes through category-aware
    # phrasings rather than the literal brand-name template.
    nb = slots.no_brand_mode

    if archetype == 1:  # entity_pain
        if short_video:
            if nb:
                geo_prefix = slots.geo[0] if slots.geo else ""
                return [
                    (f"{geo_prefix} {b} buyer complaints".strip(), "user_complaint_clip"),
                    (f"{b} disappointment", "user_complaint_clip"),
                ]
            return [
                (f"{b} cringe", "user_complaint_clip"),
                (f"{b} fail", "user_complaint_clip"),
            ]
        out = []
        for p in slots.pains[:3]:
            out.append((f"{b} {p}", "user_complaint_text"))
        if not out:
            geo_prefix = slots.geo[0] if slots.geo else ""
            out.append(
                ((f"{geo_prefix} {b} buyer complaints".strip()
                  if nb else f"{b} review negative"),
                 "user_complaint_text")
            )
        return out

    if archetype == 2:  # switching_narrative
        if not slots.alternatives:
            # No alternatives — but in no-brand mode we still want some
            # switching-like signal from the broader category.
            if nb and slots.pains:
                return [
                    (f"abandoned {b} because of {p}", "switching_narrative")
                    for p in slots.pains[:2]
                ]
            return []
        if short_video:
            if nb:
                return [
                    (f"why i chose {alt} instead of {b}", "switching_narrative")
                    for alt in slots.alternatives[:2]
                ]
            return [
                (f"pov i stopped buying {b}", "switching_narrative"),
                (f"why i quit {b}", "switching_narrative"),
            ]
        if nb:
            return [
                (f"chose {alt} over {b} reasons", "switching_narrative")
                for alt in slots.alternatives[:3]
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
        if nb:
            # Category-mode falsifier: search for positive sentiment ABOUT
            # the category instead of awkward "why i love real estate".
            if short_video:
                return [
                    (f"happy {b} buyer story", "counter_evidence"),
                    (f"{b} positive review", "counter_evidence"),
                ]
            return [
                (f"{b} positive review", "counter_evidence"),
                (f"why I'm happy with my {b}", "counter_evidence"),
                (f"satisfied {b} customer experience", "counter_evidence"),
            ]
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

    if archetype == 10:  # proxy_search — close-proxy expansion
        # Fan out into geo+category+competitor combinations so a branded
        # hypothesis (e.g. "Brigade Builders") also surfaces adjacent
        # audience signal ("Bengaluru rental", "Whitefield property").
        # Pure deterministic generator — no LLM call. Caps at ~8 queries
        # to stay inside the per-channel budget.
        proxies: List[Tuple[str, str]] = []
        # Pick at most 2 geo anchors (city/country) — single cities are
        # more specific than country-level; prefer the first hit.
        geo_anchors: List[str] = []
        for g in slots.geo[:3]:
            gl = (g or "").strip().lower()
            if gl and gl not in [a.lower() for a in geo_anchors]:
                geo_anchors.append(g.title())
        # Pick up to 3 category terms — these are the "topic" anchors.
        cats = [c for c in slots.category_terms[:4] if c]
        # No geo? Still produce some category proxies (no-geo path).
        if not geo_anchors and not cats:
            return []

        if geo_anchors and cats:
            # Geo × category — the canonical "Bengaluru real estate"
            # pattern. Up to 3 combinations.
            for ga in geo_anchors[:2]:
                for ct in cats[:2]:
                    proxies.append(
                        (f"{ga} {ct}".strip(), "proxy_geo_category")
                    )
                    if len(proxies) >= 4:
                        break
                if len(proxies) >= 4:
                    break

        # Brand × geo (when both exist) — surfaces local coverage of
        # the brand. "Brigade Builders Bengaluru reviews"
        if geo_anchors and not slots.no_brand_mode:
            proxies.append(
                (f"{b} {geo_anchors[0]} reviews", "proxy_brand_geo")
            )
            if len(proxies) < 6 and slots.pains:
                proxies.append(
                    (f"{b} {geo_anchors[0]} {slots.pains[0]}", "proxy_brand_geo_pain")
                )

        # Competitor × geo — flushes out competitor-side signal that's
        # often richer than the focal-brand search.
        if geo_anchors and slots.alternatives:
            proxies.append(
                (f"{slots.alternatives[0]} {geo_anchors[0]} review",
                 "proxy_competitor_geo")
            )

        # Category-only proxies (no geo) — useful when geo wasn't extracted.
        if not geo_anchors and cats:
            for ct in cats[:3]:
                proxies.append((f"{ct} buyer review", "proxy_category"))
                if slots.pains:
                    proxies.append(
                        (f"{ct} {slots.pains[0]}", "proxy_category_pain"))
                if len(proxies) >= 4:
                    break

        # ── Phase 2-D — manifest-aware proxies ─────────────────────────
        # When a ResearchContext is active (came from a Full Manifest JSON),
        # add high-value proxy variants that exploit the manifest's
        # structured enrichment: life-triggers, cohorts, brand attributes,
        # pain verbatims.
        try:
            from .research_context import current_research_context
            rc = current_research_context()
        except Exception:
            rc = None

        if rc is not None and rc.is_active():
            geo_first = geo_anchors[0] if geo_anchors else ""
            brand = b if not slots.no_brand_mode else ""

            # Brand × life-trigger — the canonical "property purchase
            # after marriage" / "property purchase after retirement"
            # pattern. Gold for situational_occasion hypotheses.
            for trig in list(rc.life_triggers)[:3]:
                if not trig:
                    continue
                if brand:
                    if geo_first:
                        proxies.append((
                            f"{brand} {geo_first} after {trig}".strip(),
                            "proxy_brand_lifetrigger",
                        ))
                    else:
                        proxies.append((
                            f"{brand} after {trig}", "proxy_brand_lifetrigger",
                        ))
                # Even no-brand: "real estate after marriage" surfaces
                # situational-context content from forums/news.
                if cats:
                    proxies.append((
                        f"{cats[0]} after {trig}", "proxy_category_lifetrigger",
                    ))

            # Brand × cohort — "Brigade Group for HNI",
            # "Brigade Group for salaried 35-50". These often hit
            # broker / consultancy pages with cohort-specific reviews.
            for cohort in list(rc.target_cohorts)[:3]:
                if not cohort or not brand:
                    continue
                proxies.append((
                    f"{brand} for {cohort}", "proxy_brand_cohort",
                ))

            # Brand × brand-attribute (counter-evidence variants for
            # supports-side recall). "Brigade Group trust" pulls positive
            # press; "Brigade Group quality" pulls testimonials.
            for attr in list(rc.brand_attributes)[:2]:
                if not attr or not brand:
                    continue
                proxies.append((
                    f"{brand} {attr}", "proxy_brand_attribute",
                ))

            # Pain-verbatim queries — the client's own words. "low site
            # visit conversion real estate India" surfaces the same
            # complaint discussed by other brokers / agents on forums.
            # Conservative: only when the verbatim is a clean phrase
            # (not a "Yes, it is more or less the same" filler).
            for verbatim in list(rc.pain_verbatims)[:2]:
                v = (verbatim or "").strip().lower()
                # Skip filler answers
                if not v or len(v) < 12 or v.startswith(("yes", "no", "maybe", "i think")):
                    continue
                if cats:
                    proxies.append((
                        f"{cats[0]} {v[:50]}".strip(),
                        "proxy_pain_verbatim",
                    ))

            # Competitor full crossproduct — beyond the brand-vs-top-1
            # competitor that the comparison archetype already produces,
            # generate brand-vs-each-additional-competitor explicitly.
            for comp in list(rc.competitors)[1:4]:  # skip [0] (already covered)
                if not comp or not brand:
                    continue
                proxies.append((
                    f"{brand} vs {comp.lower()} comparison",
                    "proxy_competitor_pair",
                ))

        # Short-video adjustment — drop the "reviews" suffix (cringe on
        # TikTok/Shorts); use punchier search phrases.
        if short_video:
            sv_proxies: List[Tuple[str, str]] = []
            for txt, sig in proxies[:8]:
                sv = txt.replace(" reviews", "").replace(" review", "")
                sv_proxies.append((sv, sig))
            proxies = sv_proxies

        # Cap raised from 8 → 14 because the manifest-driven proxies are
        # the highest-value queries — we don't want to drop them just to
        # keep budget tight.
        return proxies[:14]

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
    """Inject a counter_evidence query if none present. Falsifier is MANDATORY.

    Phase 1.5 — channel-aware falsifier templates. The literal "why i love X"
    phrase only makes sense on opinion-rich channels (Reddit, Quora, YouTube,
    Substack). On marketplace / news / google_paa, that phrase returns zero
    results and burns query budget. Use channel-appropriate counter-evidence
    phrasings instead.
    """
    if any(q.falsifier for q in queries):
        return queries
    short_video = channel in SHORT_VIDEO_CHANNELS
    if not slots.brand:
        return queries

    # Channel-tailored counter-evidence phrasing
    b = slots.brand
    if short_video:
        text = f"defending {b}"
    elif channel == "marketplace":
        # Marketplace surfaces reviews + product pages. The counter-evidence
        # signal here is positive reviews / endorsements, not "I love it"
        # blog posts.
        text = f"{b} positive review"
    elif channel == "news":
        # News surfaces journalist-authored counter-narratives — accolades,
        # awards, defenses against bad press.
        text = f"{b} award OR endorsement"
    elif channel == "google_paa":
        # PAA returns question forms — phrase the falsifier as a question.
        text = f"why is {b} good"
    elif channel == "google_related":
        text = f"{b} review positive"
    else:
        # Reddit / Quora / Substack / YouTube / Google web — opinion-rich,
        # original first-person template still works.
        text = f"why i love {b}"

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
