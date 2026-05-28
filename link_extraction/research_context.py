"""ResearchContext — manifest-derived enrichment carried through the pipeline.

A ContextVar so the orchestrator can set it once per pipeline run and
downstream L0/L2 stages can read it without explicit threading through
function signatures (same pattern as `BackendPreferences` and the
`cost_meter` job id).

When a job is started from a Full Manifest JSON, the orchestrator builds a
``ResearchContext`` from the parsed manifest + the user-supplied
``client_brand_name`` and pushes it into ``current_research_context()``.
The decomposer and query synthesizer then check that context and:

  • use ``client_brand_name`` as the canonical primary_entity (overriding
    the regex heuristic that produced bogus "Prospects" / "FGDs" / "No"
    extractions in earlier rounds)
  • merge ``competitors`` into the alternatives list (full N×M brand-vs-
    competitor query coverage)
  • merge ``geo_hints`` into the decomposer's hints (so India shows up
    even when the hypothesis statement omits the country/city)
  • use ``target_cohorts`` as identity_claims for archetype 4 queries
  • use ``life_triggers`` to spawn proxy-search queries ("property
    purchase after marriage", "after parents retirement")
  • use ``brand_attributes`` to drive the counter-evidence falsifier
    template ("trust", "quality" become "X positive review" subjects)
  • use ``pain_verbatims`` to seed verbatim query templates

When NO ResearchContext is set (CSV path or single-hypothesis quick
runs), the decomposer/synthesizer fall back to their normal heuristic
extractions — there's no breaking change.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ResearchContext:
    """Frozen snapshot of manifest enrichment for one pipeline run."""

    # Brand the user is researching (always supplied — manifest doesn't
    # name it because the client IS the brand owner).
    client_brand_name: str = ""

    # Free-form research intent strings — used to enrich category_topics
    # in the decomposer's fallback path.
    north_star: str = ""
    enriched_essence: str = ""

    # Lists extracted by manifest_io
    competitors: tuple = ()
    geo_hints: tuple = ()
    target_cohorts: tuple = ()
    life_triggers: tuple = ()
    brand_attributes: tuple = ()
    price_anchors: tuple = ()
    regulatory_anchors: tuple = ()
    pain_verbatims: tuple = ()

    # Full dimensional priors (8 dims) — the synthesizer can pick a
    # specific dim's shift_signal as a query seed.
    dimensional_priors: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResearchContext":
        """Build from the dict shape `ParsedManifest.to_research_context()` emits."""
        return cls(
            client_brand_name=(d.get("client_brand_name") or "").strip(),
            north_star=d.get("north_star", "") or "",
            enriched_essence=d.get("enriched_essence", "") or "",
            competitors=tuple(d.get("competitors") or ()),
            geo_hints=tuple(d.get("geo_hints") or ()),
            target_cohorts=tuple(d.get("target_cohorts") or ()),
            life_triggers=tuple(d.get("life_triggers") or ()),
            brand_attributes=tuple(d.get("brand_attributes") or ()),
            price_anchors=tuple(d.get("price_anchors") or ()),
            regulatory_anchors=tuple(d.get("regulatory_anchors") or ()),
            pain_verbatims=tuple(d.get("pain_verbatims") or ()),
            dimensional_priors=dict(d.get("dimensional_priors") or {}),
        )

    def is_active(self) -> bool:
        """True when the context carries SOMETHING useful (not just defaults)."""
        return bool(
            self.client_brand_name
            or self.competitors
            or self.target_cohorts
            or self.life_triggers
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe view (for SSE events / memory persistence)."""
        return {
            "client_brand_name": self.client_brand_name,
            "north_star": self.north_star,
            "enriched_essence": self.enriched_essence,
            "competitors": list(self.competitors),
            "geo_hints": list(self.geo_hints),
            "target_cohorts": list(self.target_cohorts),
            "life_triggers": list(self.life_triggers),
            "brand_attributes": list(self.brand_attributes),
            "price_anchors": list(self.price_anchors),
            "regulatory_anchors": list(self.regulatory_anchors),
            "pain_verbatims": list(self.pain_verbatims),
            "dimensional_priors": dict(self.dimensional_priors),
        }


# ─── ContextVar ──────────────────────────────────────────────────────────────


_NULL = ResearchContext()  # the "no-manifest" sentinel — is_active() = False

_research_context_var: ContextVar[ResearchContext] = ContextVar(
    "research_context", default=_NULL,
)


def current_research_context() -> ResearchContext:
    """Return the ambient ResearchContext (or the null sentinel)."""
    return _research_context_var.get()


def set_current_research_context(ctx: ResearchContext) -> Token:
    """Push a new context onto the stack. Returns a token for reset."""
    return _research_context_var.set(ctx)


def reset_current_research_context(token: Token) -> None:
    _research_context_var.reset(token)
