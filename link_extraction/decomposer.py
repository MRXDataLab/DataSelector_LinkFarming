"""L0 — Hypothesis decomposer.

Pure-Python, no LLM. Takes one approved hypothesis and emits structured slot
fillers the L2 query synthesizer will plug into archetype templates:

    entities             — brands/products/orgs (spaCy NER + cap-noun regex)
    primary_entity       — first entity by appearance in `statement`
    competitor_anchors   — non-primary entities (capped at 5)
    pains                — lexicon hits from a curated negative-descriptor set
    aspirations          — lexicon hits from a positive identity/lifestyle set
    identity_claims      — audience-demographic tokens (gen-z, mom, athlete…)
    geo_hints            — country + demonym matches (India, Indian, US…)
    signal_archetypes    — per expected_signal → behavior/sentiment/comparison/…

No network calls. Loads spaCy lazily; falls back to regex-only entity
extraction if spaCy or the en_core_web_sm model isn't installed.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ─── Curated lexicons ─────────────────────────────────────────────────────────

PAIN_WORDS: set[str] = {
    "bad", "bland", "boring", "broken", "complaint", "complaints",
    "complain", "complaining", "decline", "declining", "disappointed",
    "dropped", "expensive", "fail", "failing", "fake", "frustrating",
    "frustration", "garbage", "gap", "gaps", "hate", "hated", "horrible",
    "ignored", "irrelevant", "lacking", "lose", "losing", "lost",
    "mismatch", "misaligned", "missed", "outdated", "overpriced",
    "poor", "problem", "ripoff", "scam", "shrinking", "stale", "stopped",
    "stuck", "terrible", "tired", "trash", "ugly", "useless", "weak",
    "worse", "worst", "worsening",
}

ASPIRATION_WORDS: set[str] = {
    "active", "affordable", "aspirational", "authentic", "balance", "calm",
    "clean", "conscious", "convenient", "cool", "craft", "creative",
    "delicious", "eco", "elegant", "empowered", "energizing", "energy",
    "ethical", "exclusive", "fashionable", "fit", "fitness", "fresh",
    "fun", "gourmet", "happy", "healthy", "indulgent", "luxe", "luxury",
    "mindful", "modern", "natural", "nutritious", "organic", "pleasure",
    "premium", "pride", "productive", "refined", "satisfying", "savory",
    "self-care", "selfcare", "sleek", "stylish", "successful",
    "sustainable", "tasty", "traditional", "trendy", "trusted", "wellness",
}

IDENTITY_TOKENS: List[str] = [
    "athletes", "athlete", "boomers", "boomer", "dads", "dad",
    "diy", "eco-conscious", "ecoconscious", "gamers", "gamer",
    "gen-z", "gen z", "genz", "millennials", "millennial",
    "moms", "mom", "muslim", "parents", "parent",
    "professionals", "professional", "seniors", "senior",
    "students", "student", "teens", "teen", "vegan", "vegetarian", "youth",
]

# Place names + demonyms — both routed to `geo_hints`, never to `entities`.
GEO_DEMONYMS: set[str] = {
    "indian", "american", "british", "european", "chinese", "japanese",
    "korean", "german", "french", "italian", "spanish", "mexican",
    "canadian", "australian", "russian", "brazilian", "south asian",
    "latin", "african", "middle eastern", "desi",
}

GEO_PLACES: set[str] = {
    "india", "usa", "uk", "u.s.", "u.s.a.", "us", "china", "japan",
    "germany", "france", "brazil", "russia", "canada", "australia",
    "italy", "spain", "mexico", "korea", "north america", "south america",
    "europe", "asia", "africa",
    # Major Indian cities — mentioning any of these implies geo=india for
    # query proxy expansion. Listed lower-case for the case-folded matcher.
    "bengaluru", "bangalore", "mumbai", "bombay", "delhi", "new delhi",
    "ncr", "gurugram", "gurgaon", "noida", "chennai", "madras",
    "hyderabad", "kolkata", "calcutta", "pune", "ahmedabad", "kochi",
    "cochin", "jaipur", "lucknow", "indore", "chandigarh", "surat",
    "nagpur", "thiruvananthapuram", "trivandrum", "vizag", "visakhapatnam",
    "coimbatore", "vadodara",
}

# Indian-city → state/region map. Used by proxy-query generation to
# surface "Bangalore rental" → "Karnataka real estate" style adjacents.
INDIA_CITIES: frozenset[str] = frozenset({
    "bengaluru", "bangalore", "mumbai", "bombay", "delhi", "gurugram",
    "gurgaon", "noida", "chennai", "hyderabad", "kolkata", "pune",
    "ahmedabad", "kochi", "jaipur", "lucknow", "indore",
})

GEO_BLOCKLIST: set[str] = GEO_DEMONYMS | GEO_PLACES

# Suffixes that mark a hyphenated compound as a descriptor, not an entity.
# "identity-driven", "data-driven", "value-based", "vegan-friendly" → drop.
DESCRIPTOR_SUFFIXES: set[str] = {
    "aware", "based", "centric", "driven", "first", "focused", "free",
    "friendly", "heavy", "ish", "led", "light", "like", "oriented",
    "powered", "ready", "rich",
}

# Sentence-initial words / connectives / quantifiers to drop from the
# regex entity candidates. Capitalized sentence-starts like "Given that…",
# "Proving X requires…", "No meaningful opposite…" routinely get picked
# up by the cap-noun regex and pollute primary_entity selection.
CAP_STOPWORDS: set[str] = {
    "the", "a", "an", "this", "that", "these", "those",
    "why", "what", "how", "where", "when", "who", "which",
    "is", "are", "was", "were", "be", "been", "in", "on", "at",
    "and", "or", "but", "for", "to", "of", "by", "with",
    # Sentence-initial connectives + participles often capitalised in
    # analyst-written hypothesis text. (Long list — false positives here
    # are cheaper than letting them pollute primary_entity.)
    "given", "proving", "directly", "building", "adding", "noting",
    "considering", "leveraging", "according", "regarding", "based",
    "however", "moreover", "furthermore", "therefore", "thus", "hence",
    "indeed", "still", "yet", "also", "additionally", "specifically",
    "such", "many", "most", "some", "any", "all", "every", "each",
    "no", "yes", "not", "none", "neither", "either", "both",
    "his", "her", "their", "its", "our", "your", "my", "we", "they",
    "it", "he", "she", "us", "them", "him", "i",
    "do", "does", "did", "will", "would", "could", "should", "shall",
    "may", "might", "must", "can", "has", "have", "had",
    "more", "less", "much", "very", "too", "so", "just", "only",
    "even", "ever", "never", "always", "often", "sometimes",
    "if", "unless", "while", "though", "although", "because", "since",
}

# Generic role/audience nouns that LOOK like proper nouns when capitalised
# at sentence start ("Prospects perceive…", "Customers report…") but are
# never the actual brand/product being investigated. Filtered out of the
# entity list entirely so they never become primary_entity.
GENERIC_NOUN_STOPWORDS: set[str] = {
    "prospects", "prospect", "customers", "customer",
    "users", "user", "buyers", "buyer", "clients", "client",
    "consumers", "consumer", "shoppers", "shopper",
    "people", "person", "individuals", "individual",
    "audiences", "audience", "members", "member",
    "subscribers", "subscriber", "viewers", "viewer",
    "readers", "reader", "listeners", "listener",
    "fans", "fan", "followers", "follower",
    "respondents", "respondent", "participants", "participant",
    "brands", "brand", "companies", "company",
    "competitors", "competitor", "competition",
    "developers", "developer", "developments", "development",
    "agents", "agent", "vendors", "vendor", "suppliers", "supplier",
    "retailers", "retailer", "manufacturers", "manufacturer",
    "providers", "provider", "operators", "operator",
    "households", "household", "families", "family",
    "stakeholders", "stakeholder", "decision", "decisions",
    "purchase", "purchases", "purchasing", "purchaser",
    "fgds", "fgd",  # focus group discussion abbreviation
    "pestle", "swot", "tam", "sam", "som",  # analyst acronyms
}

# Map signal keywords → archetype labels (build plan §C).
SIGNAL_ARCHETYPE_MAP: Dict[str, str] = {
    # sentiment
    "complaint": "sentiment", "complaints": "sentiment",
    "feedback": "sentiment", "satisfaction": "sentiment",
    "ratings": "sentiment", "rating": "sentiment",
    "review": "sentiment", "reviews": "sentiment",
    "sentiment": "sentiment", "feelings": "sentiment",
    # comparison
    "substitution": "comparison", "alternative": "comparison",
    "alternatives": "comparison", "vs": "comparison",
    "preference_for_alternatives": "comparison",
    "preferences": "comparison", "comparison": "comparison",
    "cross_category": "comparison",
    # behavior
    "switching": "behavior", "repeat_purchase": "behavior",
    "purchase": "behavior", "purchases": "behavior",
    "subscription": "behavior", "rate": "behavior", "rates": "behavior",
    "stickiness": "behavior", "loyalty": "behavior",
    "usage": "behavior", "adoption": "behavior",
    # narrative
    "narrative": "narrative", "narratives": "narrative",
    "stories": "narrative", "framing": "narrative",
    "gap": "narrative", "gaps": "narrative",
    "feature_gaps": "narrative", "consideration_set": "narrative",
    # demographic
    "demographic": "demographic", "demographics": "demographic",
    "cohort": "demographic", "generation": "demographic",
    "age": "demographic", "audience": "demographic",
    # price
    "price": "price", "prices": "price", "pricing": "price",
    "elasticity": "price", "cost": "price",
}

ARCHETYPE_FALLBACK = "other"

# Multi-word lexicons are matched by substring; single words by \b boundary.
CAP_NOUN_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9'’&-]*(?:\s+[A-Z][A-Za-z0-9'’&-]*)*)\b"
)

# ─── Decomposition model ──────────────────────────────────────────────────────


class Decomposition(BaseModel):
    hypothesis_id: str
    entities: List[str] = Field(default_factory=list)
    primary_entity: Optional[str] = None
    competitor_anchors: List[str] = Field(default_factory=list)
    pains: List[str] = Field(default_factory=list)
    aspirations: List[str] = Field(default_factory=list)
    identity_claims: List[str] = Field(default_factory=list)
    geo_hints: List[str] = Field(default_factory=list)
    signal_archetypes: Dict[str, str] = Field(default_factory=dict)
    raw_signals: List[str] = Field(default_factory=list)
    raw_counter_signals: List[str] = Field(default_factory=list)
    # Category-level topic keywords extracted from core_problem_statement +
    # rationale. Used by the query synthesizer as a fallback when no real
    # brand is named in the hypothesis (e.g. "property", "developer",
    # "amenities", "pricing"). Populated by `_extract_category_topics()`.
    category_topics: List[str] = Field(default_factory=list)


# ─── spaCy loader ─────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_spacy():
    """Returns a loaded spaCy nlp object, or None if spaCy/model isn't available."""
    try:
        import spacy
    except ImportError:
        log.info("spaCy not installed — regex-only entity extraction")
        return None
    try:
        return spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])
    except OSError:
        log.warning(
            "spaCy 'en_core_web_sm' model not found — regex-only extraction. "
            "Install: python -m spacy download en_core_web_sm"
        )
        return None


# ─── Public API ───────────────────────────────────────────────────────────────


def decompose(hypothesis: Dict[str, Any]) -> Decomposition:
    """Decompose one hypothesis dict into the Decomposition model.

    Accepts both the host's hypothesis_engine output and the Kellogg's CSV
    row shape. Required: `hypothesis_id` or `id`, `statement`.
    Optional: `expected_signals`, `expected_counter_signals`, `rationale`,
    `core_problem_statement`.

    Phase 2 — when a ``ResearchContext`` is set in the ambient ContextVar
    (i.e. this hypothesis came from a Full Manifest JSON, not a CSV), the
    manifest's structured enrichment OVERRIDES the regex heuristics:

      • ``primary_entity`` = manifest.client_brand_name
      • ``competitor_anchors`` = manifest.competitors (merged with extracted)
      • ``geo_hints``      = manifest.geo_hints + extracted
      • ``identity_claims`` += manifest.target_cohorts
      • ``category_topics`` += manifest.life_triggers + brand_attributes

    When no manifest context is active, behaviour is unchanged.
    """
    hyp_id = hypothesis.get("hypothesis_id") or hypothesis.get("id") or ""
    statement = hypothesis.get("statement", "")
    rationale = hypothesis.get("rationale", "")
    core_problem = hypothesis.get("core_problem_statement", "")
    raw_signals = list(hypothesis.get("expected_signals", []) or [])
    raw_counter = list(hypothesis.get("expected_counter_signals", []) or [])

    text = " ".join(filter(None, [statement, rationale, core_problem]))

    entities = _extract_entities(text)
    competitor_anchors, primary_entity = _split_primary_and_competitors(
        entities, statement
    )

    pains = _scan_lexicon(text, PAIN_WORDS)
    aspirations = _scan_lexicon(text, ASPIRATION_WORDS)
    identity_claims = _scan_identity(text)
    geo_hints = _scan_geo(text)

    signal_archetypes = {sig: _classify_signal(sig) for sig in raw_signals}

    category_topics = _extract_category_topics(
        statement, core_problem, rationale,
        signals=raw_signals + raw_counter,
    )

    # ── Phase 2 — ResearchContext overrides + merges ───────────────────
    try:
        from .research_context import current_research_context
        rc = current_research_context()
    except Exception:
        rc = None

    if rc is not None and rc.is_active():
        # 1. Brand override — manifest names the actual client brand which
        #    the hypothesis text rarely contains explicitly ("the developer").
        if rc.client_brand_name:
            primary_entity = rc.client_brand_name

        # 2. Competitor merge — manifest competitors take priority but we
        #    keep any new ones the regex found (some hypotheses name a
        #    secondary competitor the manifest didn't).
        if rc.competitors:
            merged = list(rc.competitors)
            for c in competitor_anchors:
                if c not in merged and c.lower() != (rc.client_brand_name or "").lower():
                    merged.append(c)
            competitor_anchors = merged[:8]   # cap to keep query gen bounded

        # 3. Geo merge — manifest geo first (since it's authoritative),
        #    appended with anything `_scan_geo` found that's not duplicate.
        if rc.geo_hints:
            merged_geo = list(rc.geo_hints)
            for g in geo_hints:
                if g not in merged_geo:
                    merged_geo.append(g)
            geo_hints = merged_geo

        # 4. Identity claims merge — cohorts like "salaried 35-50" or
        #    "HNI" become identity_claims so archetype 4 queries pick
        #    them up ("best property for salaried 35-50").
        if rc.target_cohorts:
            merged_id = list(identity_claims)
            for c in rc.target_cohorts:
                if c not in merged_id:
                    merged_id.append(c)
            identity_claims = merged_id

        # 5. Category-topic enrichment — life triggers + brand attributes
        #    become extra topic anchors for archetype 10 proxy_search.
        for term in list(rc.life_triggers) + list(rc.brand_attributes):
            if term and term not in category_topics:
                category_topics.append(term)
        # Cap so the synthesizer's per-channel budget isn't blown.
        category_topics = category_topics[:24]

    return Decomposition(
        hypothesis_id=hyp_id,
        entities=entities,
        primary_entity=primary_entity,
        competitor_anchors=competitor_anchors,
        pains=pains,
        aspirations=aspirations,
        identity_claims=identity_claims,
        geo_hints=geo_hints,
        signal_archetypes=signal_archetypes,
        raw_signals=raw_signals,
        raw_counter_signals=raw_counter,
        category_topics=category_topics,
    )


# ─── Entity extraction ────────────────────────────────────────────────────────


def _extract_entities(text: str) -> List[str]:
    """Return deduped, order-preserving list of brand/product/org spans.

    spaCy NER handles `ORG`, `PRODUCT`, `WORK_OF_ART`, `FAC`.
    Capitalized-noun-phrase regex catches new brands spaCy doesn't know
    (e.g. "MTR Upma"). Geo terms and stopwords are filtered out.
    """
    if not text:
        return []

    nlp = _load_spacy()
    candidates: List[str] = []
    handled: set[str] = set()  # text spans already accepted by spaCy

    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            etext = ent.text.strip()
            if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART", "FAC"):
                candidates.append(etext)
                handled.add(etext.lower())
            # NORP/GPE/LOC routed to geo/identity, not entities
            elif ent.label_ in ("NORP", "GPE", "LOC"):
                handled.add(etext.lower())

    # Regex supplement
    for match in CAP_NOUN_PHRASE_RE.findall(text):
        s = match.strip()
        sl = s.lower()
        if sl in handled:
            continue
        if sl in CAP_STOPWORDS:
            continue
        if sl in GEO_BLOCKLIST:
            continue
        # Multi-word phrase starting with a stopword? Drop the stopword head.
        parts = s.split()
        if parts and parts[0].lower() in CAP_STOPWORDS:
            s2 = " ".join(parts[1:])
            if s2 and s2.lower() not in CAP_STOPWORDS and s2.lower() not in GEO_BLOCKLIST:
                candidates.append(s2)
            continue
        candidates.append(s)

    # Dedup, preserve order, case-insensitive
    out: List[str] = []
    seen: set[str] = set()
    for c in candidates:
        cl = c.lower()
        if cl in seen or len(c) < 2:
            continue
        seen.add(cl)
        out.append(c)

    return _post_filter_entities(out)


def _post_filter_entities(entities: List[str]) -> List[str]:
    """Drop entities that are really identity/geo/lexicon tokens, and collapse
    possessive variants ("Kellogg" + "Kellogg's" → keep "Kellogg's")."""
    # Pass 1: drop tokens that belong in identity/geo/aspiration/pain bucket
    lex_blocklist = (
        {t.lower() for t in IDENTITY_TOKENS}
        | {t.lower() for t in GEO_BLOCKLIST}
        | {t.lower() for t in ASPIRATION_WORDS}
        | {t.lower() for t in PAIN_WORDS}
        | GENERIC_NOUN_STOPWORDS  # role-nouns: Prospects, Customers, Developers
        | CAP_STOPWORDS           # sentence-initial: Given, Directly, No
    )
    survivors: List[str] = []
    for e in entities:
        el = e.lower()
        # Whole-string match against lexicons
        if el in lex_blocklist:
            continue
        # Single-token short uppercase abbreviations < 3 chars or sentence
        # connectives that slipped through ("No", "It", "We"). Anything
        # this short is almost never a real brand name.
        if len(e) < 3 and " " not in e:
            continue
        # Hyphenated compound containing a lexicon term (e.g. "Self-care",
        # "Eco-friendly") — drop.
        if "-" in el and any(
            part in lex_blocklist or part in IDENTITY_TOKENS for part in el.split("-")
        ):
            continue
        # Hyphenated descriptor like "Identity-driven", "Data-based" — drop.
        if "-" in el:
            parts = el.split("-")
            if len(parts) == 2 and parts[1] in DESCRIPTOR_SUFFIXES:
                continue
        # Multi-word entity whose FIRST token is a generic-noun stopword
        # ("Prospects in Bengaluru" — keep "Bengaluru" only). For now,
        # drop the whole thing rather than try to recover; the user
        # rarely loses anything important.
        first_word = e.split()[0].lower() if e else ""
        if first_word in GENERIC_NOUN_STOPWORDS or first_word in CAP_STOPWORDS:
            continue
        survivors.append(e)

    # Pass 2: possessive collapse — drop "Kellogg" if "Kellogg's" or
    # "Kellogg's Corn Flakes" also present.
    def _root(s: str) -> str:
        return re.sub(r"['’]s?$", "", s.lower()).strip()

    pass2: List[str] = []
    for e in survivors:
        e_root = _root(e)
        # Is there a longer survivor whose first word's root equals this entity's root?
        redundant = False
        for other in survivors:
            if other == e:
                continue
            other_first_root = _root(other.split()[0]) if other else ""
            if other_first_root == e_root and len(other) > len(e):
                redundant = True
                break
        if not redundant:
            pass2.append(e)

    # Pass 3: substring merge — drop "Corn Flakes" if "Kellogg's Corn Flakes"
    # is also present. Normalizes case + punctuation, requires a word-boundary
    # match so "Corn" doesn't get swallowed by "Popcorn".
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()

    normed = [(e, _norm(e)) for e in pass2]
    out: List[str] = []
    for i, (e, ne) in enumerate(normed):
        if not ne:
            out.append(e)
            continue
        redundant = False
        for j, (_other, no) in enumerate(normed):
            if i == j or not no:
                continue
            if len(no) > len(ne) and re.search(rf"\b{re.escape(ne)}\b", no):
                redundant = True
                break
            # Exact normalized tie — keep the first occurrence.
            if no == ne and j < i:
                redundant = True
                break
        if not redundant:
            out.append(e)
    return out


def _split_primary_and_competitors(
    entities: List[str], statement: str
) -> Tuple[List[str], Optional[str]]:
    """First entity by appearance in `statement` → primary; rest → competitors."""
    if not entities:
        return [], None
    statement_l = statement.lower()
    ordered = sorted(
        entities,
        key=lambda e: statement_l.find(e.lower()) if e.lower() in statement_l else 9999,
    )
    primary = ordered[0]
    competitors = [e for e in entities if e != primary][:5]
    return competitors, primary


# ─── Lexicon scans ────────────────────────────────────────────────────────────


def _scan_lexicon(text: str, lexicon: set[str]) -> List[str]:
    if not text:
        return []
    lowered = text.lower()
    found: List[str] = []
    seen: set[str] = set()
    for word in lexicon:
        if word in seen:
            continue
        if " " in word or "-" in word:
            # Compound terms: substring match
            if word in lowered:
                found.append(word)
                seen.add(word)
        else:
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                found.append(word)
                seen.add(word)
    return sorted(found)


def _scan_identity(text: str) -> List[str]:
    if not text:
        return []
    lowered = text.lower()
    found: List[str] = []
    for token in IDENTITY_TOKENS:
        # Allow optional hyphen/space variants for "gen-z" / "gen z" / "genz"
        pat = re.escape(token).replace(r"\ ", r"[\s-]?").replace(r"\-", r"[\s-]?")
        if re.search(rf"\b{pat}\b", lowered) and token not in found:
            found.append(token)
    return found


def _scan_geo(text: str) -> List[str]:
    if not text:
        return []
    lowered = text.lower()
    found: List[str] = []
    for term in sorted(GEO_BLOCKLIST, key=len, reverse=True):
        # Longest-first to avoid double-matching ("south asian" before "asian")
        if re.search(rf"\b{re.escape(term)}\b", lowered) and term not in found:
            found.append(term)
    return found


# ─── Category topic extractor ─────────────────────────────────────────────────


# Common domain terms found in market-research hypotheses. Conservative —
# we want high precision (false positives confuse query synthesis).
_CATEGORY_TOPIC_TERMS: List[str] = [
    # Real estate / property
    "real estate", "property", "properties", "apartment", "apartments",
    "flat", "flats", "villa", "villas", "house", "houses", "home", "homes",
    "amenities", "amenity", "micro-market", "micromarket", "developer",
    "luxury", "premium", "affordable housing", "residential", "commercial",
    "site visit", "site-visit", "floor plan", "carpet area", "rera",
    "possession", "booking", "down payment", "emi",
    # Travel
    "hotel", "hotels", "flight", "flights", "vacation", "trip", "travel",
    "booking", "homestay", "resort", "package tour",
    # FMCG / food
    "breakfast", "cereal", "snack", "snacks", "beverage", "beverages",
    "packaged food", "instant", "ready-to-eat",
    # Generic
    "pricing", "positioning", "product", "feature", "features", "design",
    "purchase", "buyer", "shopper", "review", "reviews", "experience",
    "service", "support", "warranty", "delivery",
    "investment", "value", "quality", "trust", "satisfaction",
    "complaint", "complaints", "feedback", "decision",
]


def _extract_category_topics(
    statement: str,
    core_problem: str,
    rationale: str,
    *,
    signals: Optional[List[str]] = None,
) -> List[str]:
    """Pull domain/category topic phrases from the hypothesis context.

    Order matters: longer phrases first so "real estate" wins over "estate".
    Returns deduped, lowercase, ordered list capped at 12 to keep query
    synthesis bounded.

    These topics feed the no-brand fallback path in the query synthesizer
    when `primary_entity` would otherwise be empty/junk. For the real-estate
    hypotheses in the reported bug, this surfaces ["property", "developer",
    "amenities", "pricing", "site visit", ...] from core_problem.
    """
    text = " ".join(s for s in (statement, core_problem, rationale) if s).lower()
    if not text:
        return []
    found: List[str] = []
    seen: set[str] = set()
    # Sort longest-first so "real estate" matches before "estate".
    for term in sorted(_CATEGORY_TOPIC_TERMS, key=len, reverse=True):
        if term in seen:
            continue
        if " " in term or "-" in term:
            if term in text:
                found.append(term)
                seen.add(term)
        else:
            if re.search(rf"\b{re.escape(term)}\b", text):
                found.append(term)
                seen.add(term)
        if len(found) >= 12:
            break

    # Pull additional context from signals themselves: "site_visit",
    # "price_perception", "exit_narratives" all carry topic information.
    for sig in (signals or []):
        s = (sig or "").lower().replace("_", " ").strip()
        if not s or s in seen or len(s) > 40:
            continue
        # Skip pure-archetype labels like "behavior", "sentiment".
        if s in {"sentiment", "behavior", "comparison", "narrative",
                 "demographic", "price"}:
            continue
        found.append(s)
        seen.add(s)
        if len(found) >= 16:
            break
    return found


# ─── Signal archetype classifier ──────────────────────────────────────────────


def _classify_signal(signal: str) -> str:
    """Map a snake_case or free-text signal label to its archetype name."""
    if not signal:
        return ARCHETYPE_FALLBACK
    lowered = signal.lower().replace("_", " ")
    # Sort keywords longest-first so "feature_gaps" beats "gap"
    for keyword in sorted(SIGNAL_ARCHETYPE_MAP, key=len, reverse=True):
        if keyword.replace("_", " ") in lowered:
            return SIGNAL_ARCHETYPE_MAP[keyword]
    return ARCHETYPE_FALLBACK
