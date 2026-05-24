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
}

GEO_BLOCKLIST: set[str] = GEO_DEMONYMS | GEO_PLACES

# Suffixes that mark a hyphenated compound as a descriptor, not an entity.
# "identity-driven", "data-driven", "value-based", "vegan-friendly" → drop.
DESCRIPTOR_SUFFIXES: set[str] = {
    "aware", "based", "centric", "driven", "first", "focused", "free",
    "friendly", "heavy", "ish", "led", "light", "like", "oriented",
    "powered", "ready", "rich",
}

# Sentence-initial words to drop from the regex entity candidates.
CAP_STOPWORDS: set[str] = {
    "the", "a", "an", "this", "that", "these", "those",
    "why", "what", "how", "where", "when", "who", "which",
    "is", "are", "was", "were", "be", "been", "in", "on", "at",
    "and", "or", "but", "for", "to", "of", "by", "with",
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
    )
    survivors: List[str] = []
    for e in entities:
        el = e.lower()
        # Whole-string match against lexicons
        if el in lex_blocklist:
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
