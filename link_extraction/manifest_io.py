"""Full Manifest JSON parser — Phase 2 ingestion path.

The host's intent-discovery flow emits a `Full_Manifest_*.json` carrying
WAY more context than the CSV path: client conversation transcript,
dimensional priors, volunteered facts, competitor names, target cohorts,
life-event triggers. This module extracts that context into a
``ParsedManifest`` so downstream stages (decomposer, query synthesizer)
can build richer queries than they could from the hypothesis statement
alone.

Manifest top-level shape (schema v1.x — observed in production):

    {
      "bundle_id": "fm_20260527_172858",
      "schema_version": "1.1",
      "frozen_at": "<iso>",
      "research_intent": {
        "north_star": "<one-line research question>",
        "enriched_essence": "<paragraph problem statement>"
      },
      "conversation_summary": {
        "client_pain_points":   [{"point": str, "verbatim_phrasing": str, ...}],
        "volunteered_context":  [str, ...],
        "pivotal_moments":      [{"turn_index": int, "what_happened": str}],
        "full_transcript":      [{"role": "agent"|"client", "content": str}],
      },
      "dimensional_priors": {
        "price_economics":     {"label": str, "current_state": str,
                                "shift_signal": str, ...},
        "product_offering":    {...},
        "distribution_channel":{...},
        "brand_meaning":       {...},
        "identity_culture":    {...},
        "situational_occasion":{...},
        "cohort_dynamics":     {...},
        "regulatory_structural":{...},
      },
      "hypotheses": {
        "core_problems": [
          {
            "id": "cp_001", "statement": "...", "priority": "high",
            "hypotheses": [
              { "id": "h_001", "statement": "...", "dimension": str,
                "force_assignment": str, "mece_cluster_id": str,
                "expected_signals": [...], "expected_counter_signals": [...],
                "investigation_priority": str, "rationale": str,
                "contrarian_pair_id": str|null },
              ...
            ]
          },
          ...
        ]
      }
    }

The parser never crashes — schema mismatches become entries on
``ParsedManifest.warnings``. Bring-your-own-brand: the manifest does NOT
name the client's brand (it's THE client talking to OUR agent), so the
caller must inject ``client_brand_name`` via the API request.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ─── Output types ────────────────────────────────────────────────────────────


@dataclass
class ParsedManifestHypothesis:
    """One hypothesis from the manifest, normalised to the dict shape
    the existing batch runner expects (same fields as csv_io output)."""
    row_index: int                       # 1-based across the flattened list
    hypothesis: Dict[str, Any]           # the dict consumed by decompose()
    core_problem_id: str
    window_label_override: Optional[str] = None
    max_triage_override: Optional[int] = None


@dataclass
class ParsedManifest:
    """Structured view of a Full Manifest JSON.

    The fields below map directly onto inputs the L0/L2 stages can use to
    enrich query construction beyond what the hypothesis statement alone
    provides.
    """

    # Source metadata
    bundle_id: str = ""
    schema_version: str = ""
    frozen_at: str = ""

    # Research intent (one-line north-star + paragraph problem statement)
    north_star: str = ""
    enriched_essence: str = ""

    # Conversation-derived facts
    client_pain_points: List[Dict[str, Any]] = field(default_factory=list)
    volunteered_context: List[str] = field(default_factory=list)
    pivotal_moments: List[Dict[str, Any]] = field(default_factory=list)

    # Extracted structured context (the actual high-value enrichment)
    competitors: List[str] = field(default_factory=list)
    geo_hints: List[str] = field(default_factory=list)
    target_cohorts: List[str] = field(default_factory=list)
    life_triggers: List[str] = field(default_factory=list)
    brand_attributes: List[str] = field(default_factory=list)
    price_anchors: List[str] = field(default_factory=list)
    regulatory_anchors: List[str] = field(default_factory=list)

    # The full dimensional priors (8 dims) — kept verbatim so the
    # synthesizer can pick which shift_signal to use per hypothesis.
    dimensional_priors: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Flattened hypotheses (core_problem_id stamped on each)
    hypotheses: List[ParsedManifestHypothesis] = field(default_factory=list)
    core_problems: Dict[str, str] = field(default_factory=dict)  # cp_id → statement

    # Non-fatal warnings (missing keys, malformed entries) — caller decides
    # whether to surface in the UI.
    warnings: List[str] = field(default_factory=list)

    # ── Convenience views ────────────────────────────────────────────

    @property
    def hypothesis_count(self) -> int:
        return len(self.hypotheses)

    @property
    def core_problem_count(self) -> int:
        return len(self.core_problems)

    def to_research_context(
        self, client_brand_name: str,
    ) -> Dict[str, Any]:
        """JSON-safe dict for the ResearchContext ContextVar.

        Caller supplies ``client_brand_name`` because the manifest never
        names the brand (the client IS the brand owner).
        """
        return {
            "client_brand_name": client_brand_name.strip(),
            "north_star": self.north_star,
            "enriched_essence": self.enriched_essence,
            "competitors": list(self.competitors),
            "geo_hints": list(self.geo_hints),
            "target_cohorts": list(self.target_cohorts),
            "life_triggers": list(self.life_triggers),
            "brand_attributes": list(self.brand_attributes),
            "price_anchors": list(self.price_anchors),
            "regulatory_anchors": list(self.regulatory_anchors),
            "pain_verbatims": [
                p.get("verbatim_phrasing", "") for p in self.client_pain_points
                if p.get("verbatim_phrasing")
            ],
            "dimensional_priors": dict(self.dimensional_priors),
        }


# ─── Extraction regexes ──────────────────────────────────────────────────────

# Common brand-list patterns in volunteered_context strings.
#   "Key competitors are Prestige, Sobha, DLF, Godrej, and Lodha."
#   "Direct competitors include Acme, Beta and Gamma."
#   "compete with X and Y"
_COMPETITOR_PHRASE_RE = re.compile(
    r"\b(?:key\s+)?competitors?\s+(?:are|is|include[ds]?|comprise[ds]?)\s+([^.;]+)",
    re.IGNORECASE,
)
_COMPETE_WITH_RE = re.compile(
    r"\b(?:compete[ds]?\s+(?:with|against)|going\s+up\s+against)\s+([^.;]+)",
    re.IGNORECASE,
)
_RIVAL_RE = re.compile(
    r"\b(?:rivals?|key\s+rivals?)\s+(?:are|include[ds]?)\s+([^.;]+)",
    re.IGNORECASE,
)

# Currency / price markers (extracted as price anchors; INR/USD also hint geo).
_PRICE_RE = re.compile(
    r"\b(INR|Rs\.?|₹|\$|USD|GBP|£|EUR|€)\s*[\d.,]+(?:\s*(crore|lakh|million|billion|k))?",
    re.IGNORECASE,
)

# Regulatory anchors — RERA / FDA / GDPR / HIPAA etc.
_REGULATORY_ACRONYMS_RE = re.compile(
    r"\b(RERA|FDA|FCC|FTC|SEBI|RBI|GDPR|HIPAA|CCPA|FSSAI|TRAI|IRDAI|IPO|MCA)\b"
)

# Indian / global cohort phrases — pattern: "<adjective> aged <range>" or
# "<segment> of <demographic>" — we cap-noun-extract the candidate phrase.
_AGE_RANGE_RE = re.compile(
    r"(?:aged?|ages?)\s+(\d{2})\s*[-–to]+\s*(\d{2})\b",
    re.IGNORECASE,
)

# Life-trigger keywords — surfaced from situational_occasion + volunteered_context.
_LIFE_TRIGGER_KEYWORDS = [
    "marriage", "wedding", "engaged",
    "birth", "newborn", "baby",
    "moving", "relocat", "transfer",
    "retirement", "retire", "pension",
    "promotion", "salary hike", "salary increase", "raise",
    "graduation", "first job", "starting career",
    "divorce", "separation",
    "death", "bereavement", "inheritance",
    "diagnosis", "illness",
    "school admission", "college admission",
    "going abroad", "emigration", "immigration",
    "buying first home", "first home",
    "empty nest", "kids leaving",
]

# Geo hint keywords used to mark India/US/UK from manifest content
# without explicit city names. (Currency + regulator are strongest signals.)
_GEO_HINT_FROM_REGULATOR: Dict[str, str] = {
    "RERA": "india",   "SEBI": "india", "RBI": "india", "FSSAI": "india",
    "TRAI": "india",   "IRDAI": "india",
    "FDA": "us",       "FCC": "us",   "FTC": "us",
    "GDPR": "europe",  "CCPA": "us",  "HIPAA": "us",
}
_GEO_HINT_FROM_CURRENCY: Dict[str, str] = {
    "INR": "india", "RS": "india", "₹": "india", "RUPEE": "india",
    "USD": "us",    "$": "us",     "DOLLAR": "us",
    "GBP": "uk",    "£": "uk",     "POUND": "uk",
    "EUR": "europe", "€": "europe",
}


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _extract_competitors(volunteered: List[str]) -> List[str]:
    """Pull competitor brand names out of volunteered_context bullets.

    Looks for "<intro> X, Y, Z, and W." patterns. Splits on commas /
    `and` / `&`. Keeps capitalised tokens of >=3 chars; drops verbs and
    role-nouns (we reuse the decomposer's GENERIC_NOUN_STOPWORDS for that).
    """
    try:
        from .decomposer import GENERIC_NOUN_STOPWORDS, CAP_STOPWORDS
    except Exception:
        GENERIC_NOUN_STOPWORDS, CAP_STOPWORDS = set(), set()

    found: List[str] = []
    seen: set[str] = set()
    for v in volunteered:
        if not v or not isinstance(v, str):
            continue
        # Try each regex; first match wins per bullet.
        m = (_COMPETITOR_PHRASE_RE.search(v)
             or _COMPETE_WITH_RE.search(v)
             or _RIVAL_RE.search(v))
        if not m:
            continue
        tail = m.group(1)
        # Strip trailing words like "in the market", "this year"
        tail = re.split(r"\b(?:in|on|this|that|over|since|across)\b", tail)[0]
        # Split on commas / and / & — be permissive about whitespace so the
        # post-comma "and Lodha" pattern still splits "and" off properly.
        parts = re.split(r"\s*,\s*|\s*\band\b\s*|\s*&\s*", tail)
        for part in parts:
            name = part.strip().strip(".;:")
            # Some patterns ("Prestige, Sobha, DLF, Godrej, and Lodha")
            # leave a vestigial "and" at the boundary — drop it.
            name = re.sub(r"^(?:and|or)\s+", "", name, flags=re.IGNORECASE)
            if not name or len(name) < 3:
                continue
            # Must start with capital letter (proper noun heuristic)
            if not name[0].isupper():
                continue
            nl = name.lower()
            if nl in GENERIC_NOUN_STOPWORDS or nl in CAP_STOPWORDS:
                continue
            if nl in seen:
                continue
            seen.add(nl)
            found.append(name)
    return found


def _extract_geo_hints(text_blob: str, dim_priors: Dict[str, Any]) -> List[str]:
    """Infer geo hints from currency, regulator, and city mentions."""
    out: List[str] = []
    seen: set[str] = set()

    def add(hint: str) -> None:
        h = hint.lower().strip()
        if h and h not in seen:
            seen.add(h)
            out.append(h)

    # Currency markers (strong signal — INR ⇒ india)
    for m in _PRICE_RE.finditer(text_blob or ""):
        sym = (m.group(1) or "").strip().upper().lstrip("$").lstrip("£")
        h = _GEO_HINT_FROM_CURRENCY.get(sym)
        if h:
            add(h)

    # Regulator acronyms (also strong)
    for m in _REGULATORY_ACRONYMS_RE.finditer(text_blob or ""):
        h = _GEO_HINT_FROM_REGULATOR.get(m.group(1).upper())
        if h:
            add(h)

    # Indian cities from the decomposer's vocabulary
    try:
        from .decomposer import INDIA_CITIES
    except Exception:
        INDIA_CITIES = frozenset()
    blob_l = (text_blob or "").lower()
    for city in INDIA_CITIES:
        if city in blob_l:
            add(city)
            # India also implied when ANY India city named
            add("india")
    return out


def _extract_life_triggers(text_blob: str) -> List[str]:
    """Pick keyword matches from a text blob. Order-preserved, deduped."""
    out: List[str] = []
    seen: set[str] = set()
    blob_l = (text_blob or "").lower()
    # Sort longest-first so 'salary hike' matches before 'salary'.
    for kw in sorted(_LIFE_TRIGGER_KEYWORDS, key=len, reverse=True):
        if kw in blob_l and kw not in seen:
            seen.add(kw)
            out.append(kw)
    return out


def _extract_cohorts(dim_priors: Dict[str, Any]) -> List[str]:
    """Cohort phrases from cohort_dynamics + identity_culture states.

    Captures age-range phrases (e.g. 'salaried 35-50') and explicit segment
    names (e.g. 'HNI', 'Gen Z'). De-duped.
    """
    out: List[str] = []
    seen: set[str] = set()

    def add(item: str) -> None:
        k = item.lower().strip()
        if k and k not in seen and len(k) >= 3:
            seen.add(k)
            out.append(item.strip())

    blob_pieces = []
    for key in ("cohort_dynamics", "identity_culture"):
        v = (dim_priors.get(key) or {})
        blob_pieces.append(v.get("current_state") or "")
        blob_pieces.append(v.get("shift_signal") or "")
    blob = " ".join(blob_pieces)
    if not blob.strip():
        return out

    # Age-range capture — "salaried individuals aged 35-50", "Gen Z aged 18-26"
    # Skip leading prepositions / conjunctions / filler ("are", "for", "of"…)
    # so the resulting cohort phrase reads naturally as a search term.
    _AGE_FILLER = {
        "are", "is", "for", "of", "and", "or", "to", "from",
        "with", "without", "the", "a", "an",
    }
    for m in _AGE_RANGE_RE.finditer(blob):
        start = m.start()
        prefix = blob[max(0, start - 60):start].rstrip()
        # Pull tokens before the age phrase, trim filler from the front.
        words = prefix.split()[-3:]
        # Drop leading filler tokens
        while words and words[0].lower() in _AGE_FILLER:
            words.pop(0)
        label_words = [w for w in words if w[:1].isalpha()]
        if label_words:
            add(f"{' '.join(label_words)} aged {m.group(1)}-{m.group(2)}")
        else:
            add(f"aged {m.group(1)}-{m.group(2)}")

    # Explicit segment terms — HNI, Gen Z, millennials, etc.
    for term in ("HNI", "HNIs", "High Net-worth Individuals", "high net worth",
                 "Gen Z", "Gen-Z", "millennial", "millennials", "boomer",
                 "salaried", "self-employed", "student", "retiree",
                 "first-time buyer", "first-time homebuyer"):
        if term.lower() in blob.lower():
            add(term)

    return out


def _extract_brand_attributes(dim_priors: Dict[str, Any]) -> List[str]:
    """Adjective list from brand_meaning.current_state.

    Treats commas / 'and' as splitters; drops stopwords; keeps adjectives
    and short noun-adjuncts ('premium positioning' → 'premium').
    """
    brand = (dim_priors.get("brand_meaning") or {}).get("current_state") or ""
    if not brand:
        return []
    # Look for the pattern "is one of <X, Y, Z, and W>" or "perceived as
    # <X, Y, Z>" then split. We accept any noun-phrase that doesn't end in
    # a period.
    m = re.search(
        r"(?:is\s+one\s+of|perceived\s+as|stands\s+for|represents?)\s+([^.]+)",
        brand, re.IGNORECASE,
    )
    if not m:
        return []
    parts = re.split(r"\s*,\s*|\s+and\s+|\s*&\s*", m.group(1))
    out: List[str] = []
    seen: set[str] = set()
    for p in parts:
        a = p.strip().strip(".;:").lower()
        # Strip filler words at front ("of premium positioning" → "premium
        # positioning", "and premium positioning" → "premium positioning")
        a = re.sub(r"^(?:and|or|the|a|an|of|in)\s+", "", a)
        if len(a) >= 3 and a not in seen:
            seen.add(a)
            out.append(a)
    return out[:6]


def _extract_price_anchors(text_blob: str) -> List[str]:
    """Currency+amount phrases as raw price anchors."""
    out: List[str] = []
    seen: set[str] = set()
    for m in _PRICE_RE.finditer(text_blob or ""):
        s = m.group(0).strip()
        if s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out[:8]


def _extract_regulatory_anchors(text_blob: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for m in _REGULATORY_ACRONYMS_RE.finditer(text_blob or ""):
        s = m.group(1)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ─── Public API ──────────────────────────────────────────────────────────────


def parse_manifest_json(blob: Any) -> ParsedManifest:
    """Parse a Full Manifest JSON dict (or raw text) into a ``ParsedManifest``.

    Never raises — schema mismatches collect as warnings.
    """
    if isinstance(blob, str):
        try:
            blob = json.loads(blob)
        except Exception as e:
            out = ParsedManifest()
            out.warnings.append(f"JSON parse failed: {type(e).__name__}: {e}")
            return out
    if not isinstance(blob, dict):
        out = ParsedManifest()
        out.warnings.append("Manifest root is not a JSON object")
        return out

    m = ParsedManifest()
    m.bundle_id = str(blob.get("bundle_id") or "")
    m.schema_version = str(blob.get("schema_version") or "")
    m.frozen_at = str(blob.get("frozen_at") or "")

    # research_intent
    ri = blob.get("research_intent") or {}
    m.north_star = str(ri.get("north_star") or "")
    m.enriched_essence = str(ri.get("enriched_essence") or "")

    # conversation_summary
    cs = blob.get("conversation_summary") or {}
    m.client_pain_points = [
        p for p in (cs.get("client_pain_points") or [])
        if isinstance(p, dict)
    ]
    m.volunteered_context = [
        v for v in (cs.get("volunteered_context") or [])
        if isinstance(v, str)
    ]
    m.pivotal_moments = [
        p for p in (cs.get("pivotal_moments") or [])
        if isinstance(p, dict)
    ]

    # dimensional_priors — keep verbatim
    m.dimensional_priors = blob.get("dimensional_priors") or {}

    # Build the giant blob we run regex extraction against. Includes:
    # north_star + enriched_essence + every volunteered_context bullet +
    # every dimensional prior's current_state + shift_signal +
    # every pain verbatim. This is the universe of "manifest text" we
    # know about — extractors mine it.
    big_blob_parts: List[str] = [m.north_star, m.enriched_essence]
    big_blob_parts.extend(m.volunteered_context)
    for dim_body in m.dimensional_priors.values():
        if isinstance(dim_body, dict):
            big_blob_parts.append(dim_body.get("current_state") or "")
            big_blob_parts.append(dim_body.get("shift_signal") or "")
    for p in m.client_pain_points:
        big_blob_parts.append(p.get("point") or "")
        big_blob_parts.append(p.get("verbatim_phrasing") or "")
    big_blob = " ".join(s for s in big_blob_parts if s)

    # Structured extractions
    m.competitors = _extract_competitors(m.volunteered_context)
    m.geo_hints = _extract_geo_hints(big_blob, m.dimensional_priors)
    m.target_cohorts = _extract_cohorts(m.dimensional_priors)
    m.life_triggers = _extract_life_triggers(big_blob)
    m.brand_attributes = _extract_brand_attributes(m.dimensional_priors)
    m.price_anchors = _extract_price_anchors(big_blob)
    m.regulatory_anchors = _extract_regulatory_anchors(big_blob)

    # Hypotheses — flatten core_problems[].hypotheses[]
    hyp_section = blob.get("hypotheses") or {}
    cps = hyp_section.get("core_problems") or []
    row_idx = 0
    seen_hyp_ids: set[str] = set()
    for cp in cps:
        if not isinstance(cp, dict):
            continue
        cp_id = str(cp.get("id") or "").strip() or "_uncategorized"
        cp_statement = str(cp.get("statement") or "")
        m.core_problems[cp_id] = cp_statement
        sub_hyps = cp.get("hypotheses") or []
        for h in sub_hyps:
            if not isinstance(h, dict):
                continue
            row_idx += 1
            hyp_id = str(h.get("id") or "").strip() or f"h_auto_{row_idx:03d}"
            if hyp_id in seen_hyp_ids:
                m.warnings.append(f"duplicate hypothesis id {hyp_id!r}; skipped")
                continue
            seen_hyp_ids.add(hyp_id)

            # Normalise into the dict shape `decompose()` expects (the same
            # one ParsedBatch produces, so the existing batch_runner works).
            statement = str(h.get("statement") or "").strip()
            if not statement:
                m.warnings.append(
                    f"hypothesis {hyp_id} missing statement; skipped"
                )
                continue

            hypothesis_dict: Dict[str, Any] = {
                "hypothesis_id": hyp_id,
                "statement": statement,
                "core_problem_id": cp_id,
                "core_problem_statement": cp_statement,
            }
            # Copy through optional fields if present
            for k in (
                "dimension", "force_assignment", "investigation_priority",
                "contrarian_pair_id", "rationale", "mece_cluster_id",
            ):
                v = h.get(k)
                if v is not None and v != "":
                    hypothesis_dict[k] = v
            for k in ("expected_signals", "expected_counter_signals"):
                v = h.get(k)
                if isinstance(v, list) and v:
                    hypothesis_dict[k] = list(v)

            m.hypotheses.append(ParsedManifestHypothesis(
                row_index=row_idx,
                hypothesis=hypothesis_dict,
                core_problem_id=cp_id,
            ))

    if not m.hypotheses:
        m.warnings.append("manifest contains no parseable hypotheses")

    return m


def preview_summary(manifest: ParsedManifest) -> Dict[str, Any]:
    """JSON-safe summary for the `/manifest/preview` endpoint."""
    return {
        "bundle_id": manifest.bundle_id,
        "schema_version": manifest.schema_version,
        "frozen_at": manifest.frozen_at,
        "research_intent": {
            "north_star": manifest.north_star,
            "enriched_essence": manifest.enriched_essence,
        },
        "extracted_context": {
            "competitors": manifest.competitors,
            "geo_hints": manifest.geo_hints,
            "target_cohorts": manifest.target_cohorts,
            "life_triggers": manifest.life_triggers,
            "brand_attributes": manifest.brand_attributes,
            "price_anchors": manifest.price_anchors,
            "regulatory_anchors": manifest.regulatory_anchors,
        },
        "dimensional_priors_summary": {
            dim: {
                "label": (body or {}).get("label", ""),
                "shift_signal": (body or {}).get("shift_signal", "")[:200]
                                if isinstance(body, dict) else "",
            }
            for dim, body in manifest.dimensional_priors.items()
        },
        "core_problem_count": manifest.core_problem_count,
        "hypothesis_count": manifest.hypothesis_count,
        "core_problems": [
            {
                "core_problem_id": cp_id,
                "statement": stmt,
                "hypothesis_ids": [
                    h.hypothesis["hypothesis_id"] for h in manifest.hypotheses
                    if h.core_problem_id == cp_id
                ],
            }
            for cp_id, stmt in manifest.core_problems.items()
        ],
        "warnings": manifest.warnings,
    }
