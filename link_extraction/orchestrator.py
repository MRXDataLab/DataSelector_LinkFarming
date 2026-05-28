"""Step 8 — Pipeline orchestrator.

Runs the full L0 → L7 pipeline for **one hypothesis** and emits structured
events along the way so a frontend SSE stream (Step 9) can stitch a live
progress view. The orchestrator is intentionally synchronous in shape
(one hypothesis in → one ranked link list out); per-hypothesis parallelism
lives in `job_runner.py`.

Pipeline order (channels with no registered discoverer are gracefully
skipped — Day-10 demo ships YouTube Shorts only):

  L0  decompose                — pure-Python
  L1  score_channels           — deterministic
  L2  synthesize_queries       — Gemini slot-fill + Trends amplifier
  L3  discover (parallel/chan) — per-channel native discoverer
  L4  temporal filter          — already applied at L3 (API-side window)
  L6  triage                   — top-30 ranked + Gemini batch verdict
  L7  group + emit             — supports/refutes/tangential buckets

Event types emitted via the injected `emit` callback (Step 9 wires this
to an `asyncio.Queue` per subscriber for SSE):

  pipeline_start
  stage_start / stage_done             (one pair per L0/L1/L2/L3/L6/L7)
  channel_discovered                   (one per channel after its discovery)
  pipeline_complete
  pipeline_error
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from .backends.preferences import (
    BackendPreferences,
    DEFAULT_PREFERENCES,
    reset_current_preferences,
    set_current_preferences,
)
from .cost_meter import (
    current_job_id as _current_job_id,
    default_cost_cap_usd,
    get_meter,
    reset_current_job,
    set_current_job,
)
from .decomposer import Decomposition, decompose
from .dedup import (
    LinkCluster,
    cluster_links,
    cluster_stats,
    propagate_verdicts_to_members,
    representatives,
)
from .geo import (
    reset_current_geo_hints,
    set_current_geo_hints,
    tag_links as _tag_geo_scopes,
)
from .discoverers.base import Discoverer
from .models import (
    ChannelFit,
    ChannelId,
    DiscoveredLink,
    TimeWindow,
    TypedQuery,
)
from .query_synthesizer import synthesize_queries
from .source_selector import score_channels
from .triage import DEFAULT_STRICTNESS, TriageStrictness, group_by_verdict, triage

log = logging.getLogger(__name__)


# ─── Event type ──────────────────────────────────────────────────────────────


@dataclass
class PipelineEvent:
    """Single structured event from the orchestrator.

    `kind` is the discriminator (matches the SSE event name in Step 9).
    `data` is a JSON-serialisable dict — keep payloads small; large blobs
    (full link lists) belong on the JobState, not on the wire per-event.
    """

    kind: str
    hypothesis_id: str
    stage: Optional[str] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "hypothesis_id": self.hypothesis_id,
            "stage": self.stage,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


Emit = Callable[[PipelineEvent], None]


def _noop_emit(_ev: PipelineEvent) -> None:
    """Default emit when the caller doesn't care about events."""


# ─── Discoverer registry ─────────────────────────────────────────────────────


class DiscovererRegistry:
    """Maps channel_id → Discoverer instance.

    Hosts can wire a richer registry later (Reddit, PAA, News in Step 10;
    TikTok in Step 11; etc.). For Day-10 demo we ship YouTube Shorts only.
    """

    def __init__(self) -> None:
        self._registry: Dict[ChannelId, Discoverer] = {}

    def register(self, discoverer: Discoverer) -> None:
        if not discoverer.channel_id:
            raise ValueError("Discoverer.channel_id must be set")
        self._registry[discoverer.channel_id] = discoverer

    def get(self, channel: ChannelId) -> Optional[Discoverer]:
        return self._registry.get(channel)

    def available_channels(self) -> List[ChannelId]:
        return [cid for cid, d in self._registry.items() if d.available]

    def __contains__(self, channel: ChannelId) -> bool:
        return channel in self._registry


def default_registry() -> DiscovererRegistry:
    """Convenience: build a registry pre-populated with the live discoverers.

    Order matters for parallel scheduling — short-video first (priority #1
    hero channel for the Day-10 demo), then long-form text channels.

    9 channels register here. The two unregistered surfaces:
      • `trends` — pure L2 amplifier, never a discovery target
      • `instagram_reels` — deferred to v1.1 per locked decision §6.5
        (lowest-yield short-video surface + most CAPTCHA-prone)
    """
    from .discoverers.google_maps import get_google_maps
    from .discoverers.google_paa import get_google_paa
    from .discoverers.google_related import get_google_related
    from .discoverers.google_web import get_google_web
    from .discoverers.marketplace import get_marketplace
    from .discoverers.news import get_news
    from .discoverers.quora import get_quora
    from .discoverers.reddit import get_reddit
    from .discoverers.scholar import get_scholar
    from .discoverers.substack import get_substack
    from .discoverers.tiktok import get_tiktok
    from .discoverers.youtube import get_youtube
    from .discoverers.youtube_shorts import get_youtube_shorts

    reg = DiscovererRegistry()
    # Short-video (priority #1 first per locked decision)
    reg.register(get_youtube_shorts())   # Step 6  — hero short-video (dual-mode: search-first, YT API as fallback)
    reg.register(get_tiktok())           # Step 11 — secondary short-video
    # Long-form video
    reg.register(get_youtube())          # Step 12 — long-form video (dual-mode: search-first)
    # Discussion / Q&A
    reg.register(get_reddit())           # Step 10 — PRAW or Brave fallback
    reg.register(get_quora())            # Step 12 — Brave site:quora.com
    # General web + search-graph + news
    reg.register(get_google_web())       # Step 14b — general web articles (NEW)
    reg.register(get_google_paa())       # Step 10  — headless Chromium PAA
    reg.register(get_google_related())   # Step 14d — headless Chromium related-searches (NEW)
    reg.register(get_news())             # Step 10  — Brave news (Phase 1.6: + DDG + headless fallback)
    # Long-form essays + reviews + commerce
    reg.register(get_substack())         # Step 12 — Brave site:substack.com
    reg.register(get_marketplace())      # Step 12 — ecom + reviews + quick commerce
    # Phase 1.6 — free-tier maximisation channels
    reg.register(get_scholar())          # Google Scholar (academic citations)
    reg.register(get_google_maps())      # Google Maps local-pack (place reviews)
    return reg


# ─── Pipeline result ─────────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """Full output of one hypothesis pipeline run."""

    hypothesis_id: str
    decomposition: Decomposition
    channel_fits: List[ChannelFit]
    queries_by_channel: Dict[ChannelId, List[TypedQuery]]
    links_by_channel: Dict[ChannelId, List[DiscoveredLink]]
    triaged_links: List[DiscoveredLink]
    grouped: Dict[str, List[DiscoveredLink]]  # supports / refutes / tangential
    elapsed_sec: float
    channels_skipped: List[ChannelId]
    # L5 dedup output — what triage actually saw vs what discovery returned.
    # `clusters` carries the full collapse graph (representatives + members)
    # so callers can show "this link was also found on X, Y" in the UI.
    clusters: List[LinkCluster] = field(default_factory=list)
    # Step 14 — cost meter snapshot at pipeline end. `None` when the meter
    # had nothing to report (e.g. use_llm=False + no YT calls).
    cost: Optional[Dict[str, Any]] = None
    # Phase 3 — per-link extracted quotes + the per-hypothesis synthesis
    # paragraph. Both `None` when `skip_synthesis=True` (the default).
    # `quotes_by_url`: maps a triaged link's URL → its ExtractedQuote dict.
    # `synthesis`: SynthesisOutput.to_dict() for this hypothesis.
    quotes_by_url: Optional[Dict[str, Dict[str, Any]]] = None
    synthesis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ─── Helpers ─────────────────────────────────────────────────────────────────


async def _run_quote_synthesis(
    *,
    hyp_id: str,
    hypothesis: Dict[str, Any],
    grouped: Dict[str, List[DiscoveredLink]],
    emit: Any,
    research_context: Optional[Any],
    max_synthesis_links: int,
    job_id: Optional[str],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """L6.5 (quote extraction) + L7 (synthesis) — Phase 3.

    Fetches the body for each supports + refutes link (top-N per bucket
    by triage confidence), extracts a verbatim evidence quote via Gemini,
    then rolls the per-link quotes up into one synthesis paragraph for
    the hypothesis.

    Returns (quotes_by_url, synthesis_dict). Both empty/None on full
    failure but the orchestrator always wraps this in a try/except so
    one bad link never breaks the run.
    """
    from . import body_fetcher
    from . import quote_extractor
    from . import synthesizer

    supports_links = list(grouped.get("supports", []))
    refutes_links = list(grouped.get("refutes", []))
    tangential_links = list(grouped.get("tangential", []))
    # Take the top-N by confidence per stance.
    supports_links.sort(key=lambda l: -(l.confidence or 0))
    refutes_links.sort(key=lambda l: -(l.confidence or 0))
    tangential_links.sort(key=lambda l: -(l.confidence or 0))
    # Phase 3 — primary candidates are supports + refutes (the verdicts we
    # most want quoted). If those are empty (all tangential), fall back to
    # the top tangential links so the quote extractor + synthesizer still
    # have something concrete to work with. Better an "inconclusive but
    # here's what was found" paragraph than a silent empty result.
    candidates = (
        supports_links[:max_synthesis_links]
        + refutes_links[:max_synthesis_links]
    )
    if not candidates:
        candidates = tangential_links[:max_synthesis_links]
    # If still empty (no triaged links at all — extremely unusual), bail.
    if not candidates:
        emit(PipelineEvent(
            kind="stage_done", hypothesis_id=hyp_id, stage="L7_synthesis",
            data={"skipped": True, "reason": "no triaged candidates"},
        ))
        return {}, {}

    statement = hypothesis.get("statement", "") or ""
    brand_name = ""
    cohorts: List[str] = []
    triggers: List[str] = []
    attributes: List[str] = []
    if research_context is not None:
        try:
            brand_name = research_context.client_brand_name or ""
            cohorts = list(research_context.target_cohorts or ())
            triggers = list(research_context.life_triggers or ())
            attributes = list(research_context.brand_attributes or ())
        except Exception:
            pass

    # ── L6.5 — bulk-fetch bodies then extract quotes ──────────────────
    emit(PipelineEvent(
        kind="stage_start", hypothesis_id=hyp_id, stage="L6.5_quote_extract",
        data={"n_candidates": len(candidates),
              "n_supports": len(supports_links[:max_synthesis_links]),
              "n_refutes": len(refutes_links[:max_synthesis_links])},
    ))
    urls = [(l.canonical_url or l.url) for l in candidates if l.url]
    body_map = await body_fetcher.fetch_many(urls)

    quotes_by_url: Dict[str, Dict[str, Any]] = {}
    triaged_quotes_for_synth: List[Dict[str, Any]] = []
    # Quote extraction is sync (LLM call wrapped in to_thread for parallelism).
    async def _one_quote(link: DiscoveredLink) -> None:
        url = link.canonical_url or link.url
        fb = body_map.get(url)
        # Pick the best available text: full body > snippet > title.
        # When the live HTTP fetch failed (JS-rendered pages, anti-bot
        # 403s, paywalls), we still have the search-result snippet which
        # is usually 1-3 sentences — enough for the LLM to identify a
        # verbatim quote IF the snippet itself contains evidence.
        body_text = ""
        if fb is not None and fb.ok and fb.text:
            body_text = fb.text
        elif link.snippet:
            # Synthetic mini-body: snippet + title (search-result level)
            body_text = (
                f"{link.title or ''}\n\n{link.snippet}"
            ).strip()
        if not body_text:
            return
        eq = await asyncio.to_thread(
            quote_extractor.extract_quote,
            hypothesis_statement=statement,
            body_text=body_text,
            link_title=link.title or "",
            link_url=url,
            manifest_cohorts=cohorts,
            manifest_triggers=triggers,
            manifest_attributes=attributes,
            brand_name=brand_name,
            job_id=job_id,
        )
        if eq is None or not eq.ok:
            return
        quotes_by_url[url] = eq.to_dict()
        triaged_quotes_for_synth.append({
            "extracted": eq,
            "source_url": url,
            "source_title": link.title or "",
        })

    await asyncio.gather(
        *(_one_quote(l) for l in candidates),
        return_exceptions=True,
    )
    emit(PipelineEvent(
        kind="stage_done", hypothesis_id=hyp_id, stage="L6.5_quote_extract",
        data={"n_quotes": len(quotes_by_url),
              "n_fetched": sum(1 for fb in body_map.values() if fb.ok)},
    ))

    # ── L7 — per-hypothesis synthesis ─────────────────────────────────
    emit(PipelineEvent(
        kind="stage_start", hypothesis_id=hyp_id, stage="L7_synthesis",
    ))
    synth = await asyncio.to_thread(
        synthesizer.synthesize,
        hypothesis_id=hyp_id,
        hypothesis_statement=statement,
        triaged_quotes=triaged_quotes_for_synth,
        brand_name=brand_name,
        manifest_cohorts=cohorts,
        manifest_triggers=triggers,
        manifest_attributes=attributes,
        use_llm=True,
        job_id=job_id,
    )
    emit(PipelineEvent(
        kind="stage_done", hypothesis_id=hyp_id, stage="L7_synthesis",
        data={"llm_used": synth.llm_used,
              "verdict_summary": synth.verdict_summary,
              "evidence_gaps": synth.evidence_gaps},
    ))
    return quotes_by_url, synth.to_dict()


def _filter_by_window(
    links: Sequence[DiscoveredLink], window: TimeWindow
) -> List[DiscoveredLink]:
    """Drop links whose `observed_at` falls outside the window.

    Most discoverers apply the API-side time filter already (per the L4
    locked decision § temporal). This is a belt-and-suspenders client-side
    pass for channels without native temporal support (TikTok, IG Reels,
    Substack, Marketplace).
    """
    kept: List[DiscoveredLink] = []
    for lk in links:
        oa = lk.observed_at
        if oa is None:
            kept.append(lk)
            continue
        if oa.tzinfo is None:
            oa = oa.replace(tzinfo=timezone.utc)
        if window.start <= oa <= window.end:
            kept.append(lk)
    return kept


# ─── Main pipeline ───────────────────────────────────────────────────────────


async def run_pipeline(
    hypothesis: Dict[str, Any],
    window: TimeWindow,
    *,
    registry: Optional[DiscovererRegistry] = None,
    emit: Emit = _noop_emit,
    use_llm: bool = True,
    max_triage: int = 30,
    discover_count_per_query: int = 10,
    max_links_per_channel: int = 50,
    job_id: Optional[str] = None,
    cost_cap_usd: Optional[float] = None,
    triage_strictness: TriageStrictness = DEFAULT_STRICTNESS,
    backend_preferences: Optional[BackendPreferences] = None,
    skip_triage: bool = False,
    research_context: Optional[Any] = None,  # ResearchContext from manifest
    skip_synthesis: bool = True,             # Phase 3 — default off (cheap)
    max_synthesis_links: int = 12,           # supports/refutes links to quote per hyp
) -> PipelineResult:
    """Run L0–L7 for one hypothesis and emit progress events.

    When ``skip_triage`` is True the pipeline stops after L5 (dedup) and
    returns every discovered link with ``verdict="unclassified"``.  No LLM
    triage calls are made and YouTube enrichment (comments / transcripts)
    is skipped, preserving API credits.  Useful for testing link diversity
    across all channels.

    Never raises — catastrophic failures populate `PipelineResult.error`
    and emit a `pipeline_error` event. Channel discoveries are run in
    parallel; failures in one channel don't take down the others.
    """
    started_at = time.monotonic()
    hyp_id = (
        hypothesis.get("hypothesis_id")
        or hypothesis.get("id")
        or "unknown"
    )
    reg = registry or default_registry()
    channels_skipped: List[ChannelId] = []

    # Step 14: bind the ambient job_id + per-job cost cap. The ContextVar
    # propagates into asyncio.to_thread() calls so YT discoverers + _llm
    # can charge the meter without explicit threading. We use `job_id or
    # hyp_id` — when running outside the job_runner (e.g. direct
    # run_pipeline calls in tests) the hypothesis_id is a reasonable proxy.
    meter_job_id = job_id or hyp_id
    cv_token = set_current_job(meter_job_id)
    effective_cap = cost_cap_usd if cost_cap_usd is not None else default_cost_cap_usd()
    get_meter().set_cap(meter_job_id, effective_cap)

    # Step #1 (this change): bind backend preferences for this pipeline run.
    # search_with_fallback() reads them via ContextVar.
    effective_prefs = backend_preferences or DEFAULT_PREFERENCES
    prefs_token = set_current_preferences(effective_prefs)

    # Phase 2 — manifest research context. When set, the decomposer +
    # synthesizer pick up brand, competitors, geo, cohorts, life triggers.
    research_token = None
    rc_summary: Optional[Dict[str, Any]] = None
    if research_context is not None:
        try:
            from .research_context import set_current_research_context
            research_token = set_current_research_context(research_context)
            rc_summary = research_context.to_dict()
        except Exception as e:
            log.warning("failed to set research context: %s", e)

    emit(PipelineEvent(
        kind="pipeline_start",
        hypothesis_id=hyp_id,
        data={
            "window_label": window.label,
            "use_llm": use_llm,
            "available_channels": list(reg.available_channels()),
            "cost_cap_usd": effective_cap,
            "meter_job_id": meter_job_id,
            "triage_strictness": triage_strictness,
            "max_triage": max_triage,
            "backend_preferences": effective_prefs.to_dict(),
            "skip_triage": skip_triage,
            "research_context": rc_summary,  # None when CSV path
        },
    ))

    # Backend health snapshot — emitted right after pipeline_start so the
    # UI can show which web-search backends are dark before any query
    # fires. Lets analysts diagnose silent failures (Brave 402, headless
    # CAPTCHA cooldown, missing API keys) the moment a run begins.
    try:
        from .backend_health import snapshot as _backend_snapshot
        emit(PipelineEvent(
            kind="backend_health",
            hypothesis_id=hyp_id,
            data=_backend_snapshot(),
        ))
    except Exception as e:
        log.warning("backend_health snapshot failed: %s", e)

    # ── L0 decompose ──────────────────────────────────────────────────────
    emit(PipelineEvent(kind="stage_start", hypothesis_id=hyp_id, stage="L0_decompose"))
    decomp = decompose(hypothesis)
    emit(PipelineEvent(
        kind="stage_done", hypothesis_id=hyp_id, stage="L0_decompose",
        data={
            "primary_entity": decomp.primary_entity,
            "n_entities": len(decomp.entities),
            "n_pains": len(decomp.pains),
            "n_aspirations": len(decomp.aspirations),
            "geo_hints": decomp.geo_hints,
        },
    ))
    # Bind geo_hints to the ambient context so backends (Brave) can bias
    # their country parameter to surface region-specific results.
    geo_token = set_current_geo_hints(decomp.geo_hints)

    # ── L1 score channels ─────────────────────────────────────────────────
    emit(PipelineEvent(kind="stage_start", hypothesis_id=hyp_id, stage="L1_source_select"))
    fits = score_channels(hypothesis, decomp)
    emit(PipelineEvent(
        kind="stage_done", hypothesis_id=hyp_id, stage="L1_source_select",
        data={
            "top_channels": [
                {"channel": f.channel, "fit_score": f.fit_score}
                for f in fits
            ],
        },
    ))

    if not fits:
        result = PipelineResult(
            hypothesis_id=hyp_id, decomposition=decomp, channel_fits=[],
            queries_by_channel={}, links_by_channel={}, triaged_links=[],
            grouped={"supports": [], "refutes": [], "tangential": []},
            elapsed_sec=time.monotonic() - started_at,
            channels_skipped=channels_skipped,
            error="L1 produced no channels above threshold",
        )
        emit(PipelineEvent(
            kind="pipeline_error", hypothesis_id=hyp_id,
            data={"error": result.error},
        ))
        return result

    # ── L2 synthesize queries ─────────────────────────────────────────────
    emit(PipelineEvent(kind="stage_start", hypothesis_id=hyp_id, stage="L2_query_synth"))
    queries_by_channel = synthesize_queries(
        hypothesis, decomp, fits, window, use_llm=use_llm
    )
    emit(PipelineEvent(
        kind="stage_done", hypothesis_id=hyp_id, stage="L2_query_synth",
        data={
            "queries_per_channel": {
                ch: len(qs) for ch, qs in queries_by_channel.items()
            },
            "total_queries": sum(len(qs) for qs in queries_by_channel.values()),
        },
    ))

    # ── L3 discovery (parallel per channel) ───────────────────────────────
    emit(PipelineEvent(kind="stage_start", hypothesis_id=hyp_id, stage="L3_discover"))
    links_by_channel: Dict[ChannelId, List[DiscoveredLink]] = {}

    async def _run_channel(channel: ChannelId, queries: List[TypedQuery]) -> None:
        discoverer = reg.get(channel)
        if discoverer is None or not discoverer.available:
            channels_skipped.append(channel)
            emit(PipelineEvent(
                kind="channel_discovered", hypothesis_id=hyp_id, stage="L3_discover",
                data={
                    "channel": channel,
                    "n_links": 0,
                    "skipped": True,
                    "reason": "no_discoverer" if discoverer is None else "unavailable",
                },
            ))
            return
        try:
            links = await discoverer.batch_discover(
                queries,
                window,
                count_per_query=discover_count_per_query,
                max_total=max_links_per_channel,
            )
        except Exception as e:
            log.warning("discoverer %s raised: %s", channel, e)
            links = []
        # L4 client-side window guard (no-op when discoverer applied it natively)
        links = _filter_by_window(links, window)
        links_by_channel[channel] = links
        emit(PipelineEvent(
            kind="channel_discovered", hypothesis_id=hyp_id, stage="L3_discover",
            data={
                "channel": channel,
                "n_links": len(links),
                "n_queries_run": len(queries),
                "skipped": False,
            },
        ))

    await asyncio.gather(*(
        _run_channel(ch, qs) for ch, qs in queries_by_channel.items()
    ))
    total_discovered = sum(len(v) for v in links_by_channel.values())
    emit(PipelineEvent(
        kind="stage_done", hypothesis_id=hyp_id, stage="L3_discover",
        data={
            "total_links": total_discovered,
            "links_per_channel": {ch: len(v) for ch, v in links_by_channel.items()},
            "channels_skipped": channels_skipped,
        },
    ))

    # ── L4 (inline) geo tagging — India vs ROW ────────────────────────────
    # Pure-fn host-pattern classifier; runs in microseconds for any link
    # count. Sets `geo_scope` on every link in place; dedup + triage
    # downstream inherit the tag.
    all_links: List[DiscoveredLink] = [
        link for links in links_by_channel.values() for link in links
    ]
    geo_counts = _tag_geo_scopes(all_links)
    emit(PipelineEvent(
        kind="stage_done", hypothesis_id=hyp_id, stage="L4_geo_tag",
        data={"total_links": len(all_links), **geo_counts},
    ))

    # ── L5 dedup + cross-platform clustering ──────────────────────────────
    emit(PipelineEvent(
        kind="stage_start", hypothesis_id=hyp_id, stage="L5_dedup",
        data={"input_links": len(all_links)},
    ))
    clusters = cluster_links(all_links)
    triage_input = representatives(clusters)
    dedup_stats = cluster_stats(clusters)
    emit(PipelineEvent(
        kind="stage_done", hypothesis_id=hyp_id, stage="L5_dedup",
        data=dedup_stats,
    ))

    # ── L5.5 YT API enrichment (Phase 1.6) ────────────────────────────────
    # When triage is on, enrich the top-N YT-host candidates with the YT
    # Data API. This burns ~106 quota units PER batch (not per discover
    # call), saving ~10x vs the old "API for all discovery" path. Skipped
    # entirely when `skip_triage=True` (discovery-only runs use search
    # data + no API spend).
    if not skip_triage and triage_input:
        try:
            from .discoverers._youtube_search import extract_video_id
            yt_links_pending = [
                lk for lk in triage_input[:max_triage]
                if extract_video_id(lk.url) is not None
                and "yt_api_enriched" not in (lk.signal_tags or [])
            ]
            if yt_links_pending:
                emit(PipelineEvent(
                    kind="stage_start", hypothesis_id=hyp_id,
                    stage="L5.5_yt_enrichment",
                    data={"n_candidates": len(yt_links_pending),
                          "max_triage": max_triage},
                ))
                # Split between shorts + long discoverers based on URL shape
                from .discoverers.youtube_shorts import get_youtube_shorts
                from .discoverers.youtube import get_youtube
                shorts_d = get_youtube_shorts()
                long_d = get_youtube()
                shorts_pending = [
                    lk for lk in yt_links_pending if "/shorts/" in (lk.url or "")
                ]
                long_pending = [
                    lk for lk in yt_links_pending if "/shorts/" not in (lk.url or "")
                ]
                # Run both enrichments concurrently
                enriched_shorts, enriched_long = await asyncio.gather(
                    shorts_d.enrich_via_api(shorts_pending),
                    long_d.enrich_via_api(long_pending),
                    return_exceptions=True,
                )
                emit(PipelineEvent(
                    kind="stage_done", hypothesis_id=hyp_id,
                    stage="L5.5_yt_enrichment",
                    data={
                        "n_enriched": len([
                            lk for lk in yt_links_pending
                            if "yt_api_enriched" in (lk.signal_tags or [])
                        ]),
                    },
                ))
        except Exception as e:
            log.warning("YT enrichment hook failed (non-fatal): %s", e)

    # ── L6 triage (on representatives only) ───────────────────────────────
    if skip_triage:
        # Discovery-only mode: skip all LLM triage. Every link gets
        # verdict="unclassified" so the results view can group by channel.
        triaged = list(triage_input)
        for lk in triaged:
            lk.supports_or_refutes = "unclassified"
            lk.confidence = 0.0
        grouped = {"supports": [], "refutes": [], "tangential": [],
                   "unclassified": triaged}
        # Build channel health report for diagnostics
        channel_health: Dict[str, Dict[str, Any]] = {}
        for ch, ch_links in links_by_channel.items():
            channel_health[ch] = {
                "status": "ok" if ch_links else "no_results",
                "links_found": len(ch_links),
            }
        for ch in channels_skipped:
            channel_health[ch] = {
                "status": "skipped",
                "links_found": 0,
                "reason": "no_discoverer_or_unavailable",
            }
        emit(PipelineEvent(
            kind="stage_done", hypothesis_id=hyp_id, stage="L6_triage",
            data={
                "skipped": True,
                "verdict_counts": {"unclassified": len(triaged)},
                "n_triaged": len(triaged),
                "channel_health": channel_health,
            },
        ))
    else:
        emit(PipelineEvent(
            kind="stage_start", hypothesis_id=hyp_id, stage="L6_triage",
            data={"n_candidates": len(triage_input), "max_triage": max_triage},
        ))
        if triage_input:
            triaged = await triage(
                hypothesis, decomp, triage_input,
                max_triage=max_triage,
                use_llm=use_llm,
                strictness=triage_strictness,
            )
        else:
            triaged = []
        # Push verdicts down to non-representative cluster members so any
        # per-member CSV export carries the same supports/refutes/confidence.
        propagate_verdicts_to_members(clusters)
        grouped = group_by_verdict(triaged)
        emit(PipelineEvent(
            kind="stage_done", hypothesis_id=hyp_id, stage="L6_triage",
            data={
                "verdict_counts": {k: len(v) for k, v in grouped.items()},
                "n_triaged": len(triaged),
            },
        ))

    # ── L6.5 quote extraction + L7 synthesis (Phase 3) ────────────────────
    # Skipped entirely when skip_synthesis=True OR skip_triage=True (no
    # supports/refutes verdicts to quote-extract). Cost-aware: respects
    # OUTTLYR_COST_CAP_USD via the same cost_meter the LLM uses for triage.
    quotes_by_url: Optional[Dict[str, Dict[str, Any]]] = None
    synthesis_payload: Optional[Dict[str, Any]] = None

    if not skip_triage and not skip_synthesis:
        try:
            quotes_by_url, synthesis_payload = await _run_quote_synthesis(
                hyp_id=hyp_id,
                hypothesis=hypothesis,
                grouped=grouped,
                emit=emit,
                research_context=research_context,
                max_synthesis_links=max_synthesis_links,
                job_id=meter_job_id,
            )
        except Exception as e:
            log.warning("L6.5/L7 synthesis stage failed (non-fatal): %s", e)

    # ── L7 emit complete ──────────────────────────────────────────────────
    elapsed = time.monotonic() - started_at

    # Step 14: snapshot the cost ledger before the ContextVar resets.
    cost_entry = get_meter().get(meter_job_id)
    cost_payload = cost_entry.to_dict() if cost_entry is not None else None
    if cost_payload is not None:
        emit(PipelineEvent(
            kind="cost_summary", hypothesis_id=hyp_id,
            data=cost_payload,
        ))

    result = PipelineResult(
        hypothesis_id=hyp_id,
        decomposition=decomp,
        channel_fits=fits,
        queries_by_channel=queries_by_channel,
        links_by_channel=links_by_channel,
        triaged_links=triaged,
        grouped=grouped,
        elapsed_sec=elapsed,
        channels_skipped=channels_skipped,
        clusters=clusters,
        cost=cost_payload,
        quotes_by_url=quotes_by_url,
        synthesis=synthesis_payload,
    )
    emit(PipelineEvent(
        kind="pipeline_complete", hypothesis_id=hyp_id,
        data={
            "elapsed_sec": round(elapsed, 3),
            "verdict_counts": {k: len(v) for k, v in grouped.items()},
            "channels_used": list(links_by_channel.keys()),
            "channels_skipped": channels_skipped,
            "total_usd": cost_payload["total_usd"] if cost_payload else 0.0,
            "skip_triage": skip_triage,
        },
    ))
    reset_current_job(cv_token)
    reset_current_preferences(prefs_token)
    reset_current_geo_hints(geo_token)
    if research_token is not None:
        try:
            from .research_context import reset_current_research_context
            reset_current_research_context(research_token)
        except Exception:
            pass
    return result
