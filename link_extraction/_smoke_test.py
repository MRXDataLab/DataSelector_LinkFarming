"""Smoke test — validates models, temporal mapping, decomposer, channel
scorer, and search backends.

Run from MRX_DataSelector using the host venv:

    cd /Users/vdogg/Documents/mrxdatalabs_Station/MRX_DataSelector
    "../MRX_Module_1 Claude/backend/venv/bin/python" -m link_extraction._smoke_test

Exit code 0 if all checks pass. Live network sections are skipped gracefully
when API keys are missing.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Load .env from a few candidate locations (MRX_DataSelector first, then host).
_HERE = Path(__file__).resolve()
_ENV_CANDIDATES = [
    _HERE.parent.parent / ".env",                                                    # MRX_DataSelector/.env
    _HERE.parent.parent.parent / "MRX_Module_1 Claude" / "backend" / ".env",         # host
]
for _env in _ENV_CANDIDATES:
    if _env.exists():
        load_dotenv(_env)
        break

from link_extraction import (  # noqa: E402
    CHANNEL_ARCHETYPES,
    SHORT_VIDEO_CHANNELS,
    ChannelFit,
    DiscoveredLink,
    PipelineEvent,
    ShortVideoLink,
    TimeWindow,
    TypedQuery,
    await_completion,
    batch_rising_queries,
    build_context,
    clear_jobs,
    create_job,
    decompose,
    default_registry,
    fetch_related_topics,
    fetch_rising_queries,
    geo_from_hints,
    get_all_channel_scores,
    get_google_paa,
    get_job,
    get_marketplace,
    get_news,
    get_quora,
    get_reddit,
    get_substack,
    get_tiktok,
    get_youtube,
    get_youtube_shorts,
    group_by_verdict,
    interest_over_time,
    normalize_geo,
    run_pipeline,
    score_channels,
    subscribe,
    synthesize_queries,
    triage,
)
from link_extraction import _llm  # noqa: E402
from link_extraction import temporal as T  # noqa: E402
from link_extraction.discoverers.youtube_shorts import (  # noqa: E402
    _engagement_score,
    _extract_hashtags,
    _parse_iso_duration,
)
from link_extraction.backends import (  # noqa: E402
    get_brave,
    get_ddg,
    get_headless,
    get_serpapi,
    search_with_fallback,
)


# Sample hypothesis from the Kellogg's CSV (Platform_Source_Selector sample)
H_006 = {
    "hypothesis_id": "h_006",
    "statement": (
        "The core Corn Flakes product is perceived as bland and misaligned "
        "with Indian taste preferences, which are shifting back towards "
        "traditional, savory breakfast options."
    ),
    "dimension": "product",
    "force_assignment": "Demand Gravity",
    "expected_signals": [
        "taste_complaints",
        "product_feature_gaps",
        "preference_for_alternatives",
    ],
    "expected_counter_signals": [
        "positive_product_feedback",
        "taste_satisfaction",
        "repeat_purchase_rate",
    ],
    "rationale": (
        "A foreign product's fit with local tastes is a fundamental question. "
        "The portfolio's decline could be rooted in a basic mismatch with the "
        "Indian palate."
    ),
    "core_problem_statement": (
        "Why is the core Kellogg's Corn Flakes portfolio experiencing a "
        "consistent year-over-year decline in India?"
    ),
}

# A short-video / Gen-Z identity hypothesis to stress identity_claims + aspirations
H_GENZ = {
    "hypothesis_id": "h_test_genz",
    "statement": (
        "Gen-Z consumers are rejecting Kellogg's cornflakes because muesli "
        "signals self-care identity that cornflakes cannot match."
    ),
    "expected_signals": [
        "switching_narratives",
        "identity_expression",
        "cross_category_substitution",
    ],
    "expected_counter_signals": ["repeat_purchase_rate"],
    "rationale": (
        "Identity-driven food choices are central to Gen-Z wellness "
        "behaviour; cornflakes carry a childish association incompatible "
        "with aspirational self-care."
    ),
}


# ── 1. models ─────────────────────────────────────────────────────────────────

def test_models() -> None:
    print("\n=== MODELS ===")
    w = TimeWindow.from_label("90d")
    assert w.days == 90, f"expected 90 days, got {w.days}"
    print(f"  TimeWindow(90d): {w.start.date()} → {w.end.date()}  days={w.days}")

    q = TypedQuery(
        text="kelloggs cornflakes boring",
        channel="reddit",
        archetype=1,
        archetype_name="entity_pain",
        hypothesis_id="h_006",
        falsifier=False,
    )
    print(f"  TypedQuery: archetype {q.archetype} ({q.archetype_name})  falsifier={q.falsifier}")

    fit = ChannelFit(channel="youtube_shorts", fit_score=85, rationale="strong cultural fit")
    print(f"  ChannelFit: {fit.channel} = {fit.fit_score}")

    link = DiscoveredLink(
        url="https://www.reddit.com/r/india/comments/abc",
        channel="reddit",
        hypothesis_id="h_006",
        query=q,
        title="Why I stopped eating cornflakes",
    )
    print(f"  DiscoveredLink: {link.channel}  {link.url[:60]}")

    sv = ShortVideoLink(
        url="https://youtube.com/shorts/abc123",
        channel="youtube_shorts",
        hypothesis_id="h_006",
        query=q,
        duration_sec=45,
        caption="why cornflakes are boring #breakfast",
        hashtags=["breakfast"],
        creator="@foodtok",
        view_count=12500,
        like_count=890,
        thumbnail_url="https://example.com/thumb.jpg",
    )
    print(
        f"  ShortVideoLink: {sv.duration_sec}s  views={sv.view_count:,}  "
        f"tags={sv.hashtags}"
    )

    # Validator check — end must be after start
    try:
        TimeWindow(start=w.end, end=w.start, label="custom")
    except Exception:
        print("  ✓ TimeWindow validator rejected end<start")
    else:
        raise AssertionError("TimeWindow validator did not reject end<start")


# ── 2. temporal ───────────────────────────────────────────────────────────────

def test_temporal() -> None:
    print("\n=== TEMPORAL ===")
    headers = (
        f"  {'label':>6}  {'brave':>12}  {'ddg':>4}  {'google_tbs':>22}  "
        f"{'reddit':>6}  {'pytrends':>14}"
    )
    print(headers)
    print("  " + "─" * (len(headers) - 2))
    for label in ("7d", "30d", "90d", "1y", "5y"):
        w = TimeWindow.from_label(label)  # type: ignore[arg-type]
        brave = T.to_brave_params(w).get("freshness", "—")
        ddg = T.to_ddg_timelimit(w) or "—"
        tbs = T.to_google_tbs(w)
        rt = T.to_reddit_t(w)
        pyt = T.to_pytrends_timeframe(w)
        print(
            f"  {label:>6}  {brave:>12}  {ddg:>4}  {tbs:>22}  {rt:>6}  {pyt:>14}"
        )

    # YouTube + GDELT spot-checks
    w90 = TimeWindow.from_label("90d")
    yt = T.to_youtube_params(w90)
    assert "publishedAfter" in yt and yt["publishedAfter"].endswith("Z")
    gd = T.to_gdelt_params(w90)
    assert len(gd["STARTDATETIME"]) == 14
    print(f"  YouTube  publishedAfter={yt['publishedAfter']}")
    print(f"  GDELT    STARTDATETIME={gd['STARTDATETIME']}")


# ── 3. backends ───────────────────────────────────────────────────────────────

async def test_backends() -> None:
    print("\n=== BACKENDS ===")
    print(f"  Brave             available: {get_brave().available}")
    print(f"  DuckDuckGo        available: {get_ddg().available}")
    print(f"  Headless Google   available: {get_headless().available}")
    print(f"  SerpAPI (stub)    available: {get_serpapi().available}  (expected False)")

    if not get_brave().available and not get_ddg().available:
        print("  ⚠ no usable search backend in this env — skipping live queries")
        return

    print("\n--- Direct Brave query ---")
    w = TimeWindow.from_label("90d")
    q = "kelloggs cornflakes india review"
    if get_brave().available:
        r = await get_brave().search(q, vertical="web", count=5, window=w)
        print(f"  Brave web: {len(r)} results")
        for it in r[:3]:
            print(f"    - {it.title[:80]}")
            print(f"      {it.url[:90]}")

    print("\n--- Direct DDG query ---")
    if get_ddg().available:
        r = await get_ddg().search(q, vertical="web", count=5, window=w)
        print(f"  DDG web: {len(r)} results")
        for it in r[:3]:
            print(f"    - {it.title[:80]}")
            print(f"      {it.url[:90]}")
    else:
        print("  (DDG not available — install duckduckgo-search or set USE_DUCKDUCKGO=1)")

    print("\n--- Fallback chain (Brave → DDG → Headless) ---")
    r = await search_with_fallback(q, vertical="web", count=8, window=w, min_results=3)
    print(f"  search_with_fallback: {len(r)} unique results")
    backends_used = {it.backend for it in r}
    print(f"  backends used: {backends_used}")

    # SerpAPI stub should always be empty
    r_stub = await get_serpapi().search(q, vertical="web", count=5, window=w)
    assert r_stub == [], "SerpAPI stub must return []"
    print("  ✓ SerpAPI stub returned [] as expected")


# ── main ─────────────────────────────────────────────────────────────────────


# ── 4. decomposer ─────────────────────────────────────────────────────────────


def test_decomposer() -> None:
    print("\n=== DECOMPOSER (L0) ===")

    d6 = decompose(H_006)
    print(f"\n  --- h_006 (cornflakes, product dimension) ---")
    print(f"  entities:           {d6.entities}")
    print(f"  primary_entity:     {d6.primary_entity!r}")
    print(f"  competitor_anchors: {d6.competitor_anchors}")
    print(f"  pains:              {d6.pains}")
    print(f"  aspirations:        {d6.aspirations}")
    print(f"  identity_claims:    {d6.identity_claims}")
    print(f"  geo_hints:          {d6.geo_hints}")
    print(f"  signal_archetypes:  {d6.signal_archetypes}")

    assert d6.hypothesis_id == "h_006", "hypothesis_id passthrough"
    assert any("kellogg" in e.lower() or "corn flakes" in e.lower()
               for e in d6.entities), f"missed Kellogg's/Corn Flakes in {d6.entities}"
    assert any(p in {"bland", "misaligned", "decline", "mismatch"} for p in d6.pains), \
        f"expected pain hits in {d6.pains}"
    assert any(g in {"india", "indian"} for g in d6.geo_hints), \
        f"expected India in geo_hints, got {d6.geo_hints}"
    assert d6.signal_archetypes["taste_complaints"] == "sentiment"
    assert d6.signal_archetypes["product_feature_gaps"] == "narrative"
    assert d6.signal_archetypes["preference_for_alternatives"] == "comparison"
    assert d6.raw_counter_signals == H_006["expected_counter_signals"]

    # Substring-merge: "Corn Flakes" and "Kellogg's Corn Flakes" name the same
    # brand; only the longer/more specific form should survive. Self-comparison
    # queries ("Corn Flakes vs Kellogg's Corn Flakes") arise when both leak through.
    competitors_l = {c.lower() for c in d6.competitor_anchors}
    assert not ("corn flakes" in competitors_l and "kellogg's corn flakes" in competitors_l), \
        f"substring-merge failed: both forms in competitor_anchors: {d6.competitor_anchors}"
    primary_l = (d6.primary_entity or "").lower()
    assert not any(
        c.lower() != primary_l and (c.lower() in primary_l or primary_l in c.lower())
        for c in d6.competitor_anchors
    ), f"competitor anchor is a substring of primary {d6.primary_entity!r}: {d6.competitor_anchors}"
    print("  ✓ h_006 assertions pass")

    d_genz = decompose(H_GENZ)
    print(f"\n  --- h_test_genz (identity_expression) ---")
    print(f"  entities:           {d_genz.entities}")
    print(f"  primary_entity:     {d_genz.primary_entity!r}")
    print(f"  pains:              {d_genz.pains}")
    print(f"  aspirations:        {d_genz.aspirations}")
    print(f"  identity_claims:    {d_genz.identity_claims}")
    print(f"  signal_archetypes:  {d_genz.signal_archetypes}")

    assert "gen-z" in d_genz.identity_claims or "gen z" in d_genz.identity_claims, \
        f"expected gen-z identity, got {d_genz.identity_claims}"
    assert any(a in {"self-care", "selfcare", "wellness", "aspirational"}
               for a in d_genz.aspirations), \
        f"expected aspirational hits, got {d_genz.aspirations}"
    print("  ✓ h_test_genz assertions pass")


# ── 5. channel scorer (L1) ────────────────────────────────────────────────────


def _print_fits(fits, indent="  "):
    print(
        f"{indent}{'channel':<18} {'score':>5}  "
        f"{'dim':>3} {'frc':>3} {'sig':>5} {'aud':>3} {'cnf':>3} {'acc':>3}"
    )
    for f in fits:
        s = f.sub_scores
        print(
            f"{indent}{f.channel:<18} {f.fit_score:>5}  "
            f"{int(s['dimension_affinity']):>3} "
            f"{int(s['force_alignment']):>3} "
            f"{s['signal_detectability']:>5.1f} "
            f"{int(s['audience_match']):>3} "
            f"{int(s['confirmation_balance']):>3} "
            f"{int(s['data_accessibility']):>3}"
        )


def test_channel_scorer() -> None:
    print("\n=== L1 CHANNEL SCORER ===")

    # ── h_006: product / Demand Gravity / high
    h006 = {**H_006, "investigation_priority": "high"}
    d006 = decompose(h006)
    fits6 = score_channels(h006, d006)
    print(f"\n  --- h_006 (product / Demand Gravity / high) — top {len(fits6)} ---")
    _print_fits(fits6)

    top_ids_6 = [f.channel for f in fits6]
    assert len(fits6) <= 5, f"top_n=5 violated: {len(fits6)} returned"
    assert fits6 == sorted(fits6, key=lambda f: f.fit_score, reverse=True), \
        "fits must be sorted descending by fit_score"
    assert all(f.fit_score >= 50 for f in fits6), \
        "all returned fits must clear fit_threshold=50"
    # Product dimension should surface review-rich + discussion channels
    assert any(c in {"reddit", "marketplace", "youtube", "quora", "google_web"}
               for c in top_ids_6), \
        f"product hypothesis should surface review/discussion channels, got {top_ids_6}"
    print("  ✓ h_006: sorted, thresholded, capped at top 5, product-relevant")

    # ── h_test_genz: identity_expression / Reinforcement Stability / high
    # Reinforcement Stability declares anti_platforms=[news] — verify exclusion.
    h_genz = {
        **H_GENZ,
        "dimension": "identity_expression",
        "force_assignment": "Reinforcement Stability",
        "investigation_priority": "high",
    }
    d_genz = decompose(h_genz)
    fits_genz = score_channels(h_genz, d_genz)
    print(f"\n  --- h_test_genz (identity_expression / Reinforcement Stability / high) ---")
    _print_fits(fits_genz)

    top_genz_ids = [f.channel for f in fits_genz]
    # Anti-platform exclusion is the critical assertion
    all_scores = get_all_channel_scores(h_genz, d_genz)
    all_channel_set = {f.channel for f in all_scores}
    assert "news" not in all_channel_set, \
        f"news must be excluded by Reinforcement Stability anti_platforms; "\
        f"got {all_channel_set}"
    # Identity-expression hypotheses should surface at least one short-video channel
    short_video_present = any(
        c in {"tiktok", "instagram_reels", "youtube_shorts"} for c in top_genz_ids
    )
    assert short_video_present, \
        f"identity_expression must surface ≥1 short-video channel, got {top_genz_ids}"
    print("  ✓ news correctly excluded (anti_platform for Reinforcement Stability)")
    print(f"  ✓ short-video channel(s) in top: "
          f"{[c for c in top_genz_ids if c in {'tiktok', 'instagram_reels', 'youtube_shorts'}]}")

    # ── Below-threshold visibility via get_all_channel_scores
    all6 = get_all_channel_scores(h006, d006)
    below = [f for f in all6 if f.fit_score < 50]
    print(f"\n  Below-threshold channels for h_006 ({len(below)}): "
          f"{[(f.channel, f.fit_score) for f in below[:5]]}")




# ── 6. trends seed (Step 4) ───────────────────────────────────────────────────


def test_trends() -> None:
    print("\n=== TRENDS SEED (L2 amplifier) ===")

    # Geo normalization — pure-function checks, no network
    assert normalize_geo("indian") == "IN"
    assert normalize_geo("India") == "IN"
    assert normalize_geo("us") == "US"
    assert normalize_geo("IN") == "IN"  # already ISO
    assert normalize_geo("") == ""
    assert normalize_geo("atlantis") == ""
    assert geo_from_hints(["indian", "india"]) == "IN"
    assert geo_from_hints(["unknown", "us"]) == "US"
    assert geo_from_hints([]) == ""
    print("  ✓ geo normalization: indian→IN, us→US, atlantis→'', empty hints→''")

    # Empty-input guards — no network
    assert fetch_rising_queries("", TimeWindow.from_label("90d")) == []
    assert fetch_rising_queries("   ", TimeWindow.from_label("90d")) == []
    assert fetch_related_topics("", TimeWindow.from_label("90d")) == []
    assert interest_over_time("", TimeWindow.from_label("90d")) == {}
    print("  ✓ empty-entity guards return [] / {}")

    # Live call — Trends rate-limits aggressively; tolerate empty
    w = TimeWindow.from_label("90d")
    rising = fetch_rising_queries("kelloggs", w, geo="IN", top_k=5)
    assert isinstance(rising, list), f"expected list, got {type(rising)}"
    assert all(isinstance(s, str) and s for s in rising), \
        f"all rising entries must be non-empty strings, got {rising}"
    print(f"  rising_queries('kelloggs', 90d, IN): {len(rising)} result(s)")
    for q in rising[:5]:
        print(f"    - {q}")
    if not rising:
        print("  (empty — likely Trends rate-limit; tolerated)")

    # Cache check — second call must hit cache (no extra Trends request)
    rising2 = fetch_rising_queries("kelloggs", w, geo="IN", top_k=5)
    assert rising2 == rising, "cached call must return identical list"
    print("  ✓ in-process cache returns identical list on repeat call")

    # related_topics + interest_over_time — same tolerance
    topics = fetch_related_topics("kelloggs", w, geo="IN", top_k=5)
    assert isinstance(topics, list)
    print(f"  related_topics('kelloggs', 90d, IN): {len(topics)} result(s)")
    for t in topics[:3]:
        print(f"    - {t}")

    iot = interest_over_time("kelloggs", w, geo="IN")
    assert isinstance(iot, dict)
    assert all(isinstance(v, int) for v in iot.values())
    print(f"  interest_over_time('kelloggs', 90d, IN): {len(iot)} data point(s)")
    if iot:
        sample = list(iot.items())[:2]
        print(f"    sample: {sample}")

    # Batch — confirms it doesn't blow up with multiple entities
    batch = batch_rising_queries(
        ["kelloggs", "muesli"], w, geo="IN", top_k=3, sleep_between=0.5,
    )
    assert set(batch.keys()) == {"kelloggs", "muesli"}
    assert all(isinstance(v, list) for v in batch.values())
    print(f"  batch_rising_queries({{kelloggs, muesli}}): "
          f"{ {k: len(v) for k, v in batch.items()} }")




# ── 7. query synthesizer (Step 5) ─────────────────────────────────────────────


def _show_queries(by_channel, indent="    "):
    for ch, qs in by_channel.items():
        print(f"{indent}{ch:<18} ({len(qs)} queries)")
        for q in qs:
            tag = "F " if q.falsifier else "  "
            print(f"{indent}  {tag}[{q.archetype}:{q.archetype_name:<22}] {q.text}")


def test_llm() -> None:
    print("\n=== LLM SHIM (Gemini) ===")
    diag = _llm.diagnostics()
    for k, v in diag.items():
        print(f"  {k:>22}: {v}")

    if not _llm.is_available():
        print("  ⚠ no Gemini key set — skipping live ping")
        return

    out = _llm.call_llm(
        "Return ONLY a JSON object with key 'ok' = true and key 'echo' = the user's word.",
        "ping",
        expect_json=True,
        max_tokens=100,
    )
    print(f"  live ping → {out!r}")
    assert isinstance(out, dict), f"expected dict, got {type(out).__name__}"
    assert out.get("ok") is True, f"expected ok=true, got {out}"
    print("  ✓ Gemini round-trip OK (JSON mode)")


def test_query_synth() -> None:
    print("\n=== QUERY SYNTHESIZER (L2) ===")
    w = TimeWindow.from_label("90d")

    # ── h_006 (product / Demand Gravity) — long-form heavy
    h006 = {**H_006, "investigation_priority": "high"}
    d006 = decompose(h006)
    fits6 = score_channels(h006, d006)
    print(f"\n  --- h_006 — channels: {[f.channel for f in fits6]} ---")

    # Force deterministic path first (use_llm=False) — must succeed without network
    rising_seed = ["kelloggs muesli 1 kg", "kelloggs cornflakes"]
    by_channel = synthesize_queries(
        h006, d006, fits6, w,
        use_llm=False,
        rising_phrases_override=rising_seed,
    )
    _show_queries(by_channel)

    # Every returned channel must include ≥1 falsifier
    for ch, qs in by_channel.items():
        falsifiers = [q for q in qs if q.falsifier]
        assert falsifiers, f"channel {ch} missing falsifier"
        assert all(q.hypothesis_id == "h_006" for q in qs), \
            f"channel {ch}: hypothesis_id passthrough broken"
    print(f"\n  ✓ all {len(by_channel)} channels have ≥1 falsifier; hypothesis_id passthrough OK")

    # MAX_QUERIES_PER_CHANNEL cap respected
    for ch, qs in by_channel.items():
        assert len(qs) <= 8, f"channel {ch} exceeded cap: {len(qs)}"
    print("  ✓ MAX_QUERIES_PER_CHANNEL=8 cap respected")

    # ── h_test_genz (identity_expression / Reinforcement Stability) — short-video heavy
    h_genz = {
        **H_GENZ,
        "dimension": "identity_expression",
        "force_assignment": "Reinforcement Stability",
        "investigation_priority": "high",
    }
    d_genz = decompose(h_genz)
    fits_genz = score_channels(h_genz, d_genz)
    print(f"\n  --- h_test_genz — channels: {[f.channel for f in fits_genz]} ---")

    by_channel_genz = synthesize_queries(
        h_genz, d_genz, fits_genz, w,
        use_llm=False,
        rising_phrases_override=[],
    )
    _show_queries(by_channel_genz)

    # Short-video channel(s) should include archetype 9 (hashtag)
    sv_channels = [c for c in by_channel_genz if c in SHORT_VIDEO_CHANNELS]
    assert sv_channels, f"identity_expression should surface short-video channels in synth output"
    for ch in sv_channels:
        archetypes_used = {q.archetype for q in by_channel_genz[ch]}
        assert 9 in archetypes_used, \
            f"short-video channel {ch} missing hashtag archetype (9): {archetypes_used}"
    print(f"  ✓ short-video channels {sv_channels} all include hashtag archetype #9")

    # No long-form channel should leak archetype 9
    for ch, qs in by_channel_genz.items():
        if ch in SHORT_VIDEO_CHANNELS:
            continue
        assert all(q.archetype != 9 for q in qs), \
            f"long-form channel {ch} leaked hashtag archetype #9"
    print("  ✓ long-form channels never receive hashtag archetype #9")

    # YouTube Shorts hero check — confirm we generate rich query mix for the priority #1 channel
    yt_shorts_qs = by_channel_genz.get("youtube_shorts") or by_channel.get("youtube_shorts") or []
    if yt_shorts_qs:
        archs = sorted({q.archetype for q in yt_shorts_qs})
        print(f"  ✓ youtube_shorts query mix: archetypes={archs}  count={len(yt_shorts_qs)}")
        assert 6 in archs, "youtube_shorts must include falsifier (archetype 6)"
        assert any(q.archetype == 9 for q in yt_shorts_qs), \
            "youtube_shorts must include hashtag (archetype 9)"

    # ── LLM path (best-effort, tolerates missing key)
    print("\n  --- LLM slot-fill path (h_006) ---")
    by_channel_llm = synthesize_queries(
        h006, d006, fits6, w,
        use_llm=True,
        rising_phrases_override=rising_seed,
    )
    delta = sum(len(qs) for qs in by_channel_llm.values()) - \
            sum(len(qs) for qs in by_channel.values())
    print(f"  LLM path delta vs deterministic: {delta:+d} queries")
    # If LLM was available, expect richer alternative coverage — log a sample
    sample_ch = next(iter(by_channel_llm), None)
    if sample_ch:
        sample = by_channel_llm[sample_ch][:3]
        for q in sample:
            print(f"    {sample_ch}: [{q.archetype_name}] {q.text}")




# ── 8. YouTube Shorts discoverer (Step 6) ─────────────────────────────────────


async def test_yt_shorts(_state: Dict[str, Any] | None = None) -> None:
    print("\n=== YOUTUBE SHORTS DISCOVERER (L3, priority #1) ===")

    # Pure-fn checks — no network
    assert _parse_iso_duration("PT15S") == 15
    assert _parse_iso_duration("PT1M") == 60
    assert _parse_iso_duration("PT1M30S") == 90
    assert _parse_iso_duration("PT2H3M4S") == 7384
    assert _parse_iso_duration("") is None
    assert _parse_iso_duration("garbage") is None
    print("  ✓ ISO 8601 duration parser: PT15S/PT1M/PT1M30S/PT2H3M4S OK")

    assert _extract_hashtags("cornflakes are #boring #breakfast #BREAKFAST") == \
        ["boring", "breakfast"]
    assert _extract_hashtags("no tags here") == []
    print("  ✓ hashtag extractor dedups case-insensitively")

    # Engagement score
    assert _engagement_score(views=1000, likes=100, comments=10) == round(
        (100 + 10 * 3) / 1000, 6
    )
    assert _engagement_score(views=0, likes=10, comments=1) is None
    assert _engagement_score(views=None, likes=10, comments=1) is None
    assert _engagement_score(views=100, likes=None, comments=None) == 0.0
    print("  ✓ engagement_score: comment-weighted, None on zero/missing views")

    yt = get_youtube_shorts()
    print(f"  YOUTUBE_API_KEY available: {yt.available}")
    if not yt.available:
        print("  ⚠ YOUTUBE_API_KEY not set — skipping live discovery")
        return

    # Live query — drives the priority-#1 hero path end-to-end
    w = TimeWindow.from_label("1y")  # broaden window so Indian Shorts have inventory
    q = TypedQuery(
        text="kelloggs cornflakes india",
        channel="youtube_shorts",
        archetype=1,
        archetype_name="entity_pain",
        target_signal="user_complaint_clip",
        hypothesis_id="h_006",
        falsifier=False,
        geo_proxies=["india"],
    )
    results = await yt.discover(q, w, count=8)
    print(f"  discover() returned {len(results)} Shorts")

    for r in results[:5]:
        dur = f"{r.duration_sec}s" if r.duration_sec is not None else "?"
        views = f"{r.view_count:,}" if r.view_count else "?"
        likes = f"{r.like_count:,}" if r.like_count else "?"
        eng = f"{r.engagement_score:.4f}" if r.engagement_score is not None else "—"
        title = (r.title or "")[:60]
        print(f"    [{dur:>5}  views={views:>8}  likes={likes:>6}  eng={eng}]  {title}")
        print(f"      url:       {r.url}")
        print(f"      creator:   {r.creator}")
        print(f"      hashtags:  {r.hashtags[:6]}")
        print(f"      comments:  {len(r.top_comments)} fetched")
        if r.top_comments:
            print(f"        ↳ {r.top_comments[0][:100]}")
        if r.transcript:
            print(f"      transcript: {len(r.transcript)} chars — {r.transcript[:80]}…")
        else:
            print(f"      transcript: (none)")

    # Hard invariants
    for r in results:
        assert r.channel == "youtube_shorts", f"wrong channel: {r.channel}"
        assert r.hypothesis_id == "h_006", "hypothesis_id passthrough broken"
        assert r.query.archetype == 1, "query passthrough broken"
        assert r.url.startswith("https://www.youtube.com/shorts/"), \
            f"non-shorts URL leaked: {r.url}"
        if r.duration_sec is not None:
            assert r.duration_sec <= 181, \
                f"long-form video leaked (duration={r.duration_sec}s): {r.url}"
        assert r.backend_used == "youtube_data_api_v3"
        # Engagement score must be well-formed when computable
        if r.view_count and r.view_count > 0:
            assert r.engagement_score is not None
            assert 0.0 <= r.engagement_score <= 1.0

    if results:
        n_enriched = sum(1 for r in results if r.top_comments or r.transcript)
        print(f"  ✓ {n_enriched}/{len(results)} results enriched with comments/transcripts")
        # At least the top-N should have been enriched (limited by top_n_enrichment=5)
        assert n_enriched >= 1 or all(not r.view_count for r in results[:5]), \
            "expected top-N enrichment to populate at least one comment/transcript"
        print(f"  ✓ all {len(results)} ShortVideoLinks pass shape invariants")

    # Stash for downstream triage test (avoids re-spending quota)
    if _state is not None:
        _state["yt_shorts_results"] = results




# ── 9. L6 measured triage (Step 7) ────────────────────────────────────────────


async def test_triage(state: Dict[str, Any]) -> None:
    print("\n=== L6 MEASURED TRIAGE ===")

    # ── Pure-fn coverage on signal expansion + regex
    from link_extraction.triage import (
        _compile_terms,
        _deterministic_verdict,
        _hit_rate,
        _strip_html,
        _terms_from_signals,
    )

    terms = _terms_from_signals(
        ["taste_complaints", "switching_narratives"],
        ["bland", "boring"],
    )
    assert {"taste", "complaints", "switching", "narratives", "bland", "boring"} <= set(terms)
    print(f"  ✓ signal expansion: {terms[:6]}…")

    rx = _compile_terms(["bland", "boring", "stop"])
    rate, tags = _hit_rate(
        "the cornflakes were bland and boring; I stopped eating them.", rx
    )
    # Suffix-aware regex: "stop" matches "stopped" via the suffix group; tag
    # surfaces as "stop" (the canonical term, suffix dropped during match).
    assert rate > 0
    assert {"bland", "boring"} <= set(tags), f"unexpected tags: {tags}"
    print(f"  ✓ hit_rate (suffix-aware): rate={rate:.3f}, tags={tags}")

    assert _strip_html("<p>hello <b>world</b><script>x</script></p>") == "hello world"
    print("  ✓ _strip_html removes script + tags + collapses whitespace")

    v, c = _deterministic_verdict("text with bland and boring", 0.05, 0.0)
    assert v == "supports" and 0.0 < c <= 0.85
    v, c = _deterministic_verdict("text with bland and boring", 0.0, 0.04)
    assert v == "refutes"
    v, c = _deterministic_verdict("", 0.0, 0.0)
    assert v == "tangential"
    print("  ✓ deterministic verdict rules behave as specified")

    # ── Live triage on the YT Shorts harvested in Step 6
    shorts: List[ShortVideoLink] = state.get("yt_shorts_results") or []
    if not shorts:
        print("  ⚠ no upstream YT Shorts available (Step 6 was skipped) — skipping live triage")
        return

    h006 = {**H_006, "investigation_priority": "high"}
    decomp = decompose(h006)
    ctx = build_context(h006, decomp)
    print(f"  triage context built: {len(ctx.signal_terms)} signal terms, "
          f"{len(ctx.counter_terms)} counter terms")
    print(f"    signal_terms[:6]: {ctx.signal_terms[:6]}")
    print(f"    counter_terms[:6]: {ctx.counter_terms[:6]}")

    print(f"\n  Triaging {len(shorts)} YT Shorts for {h006['hypothesis_id']} (LLM enabled)…")
    triaged = await triage(h006, decomp, shorts, max_triage=10, batch_size=8, use_llm=True)
    assert len(triaged) == len(shorts), \
        f"triage returned {len(triaged)} but input had {len(shorts)}"

    for r in triaged:
        v = r.supports_or_refutes
        assert v in {"supports", "refutes", "tangential"}, f"bad verdict: {v}"
        assert r.confidence is not None and 0.0 <= r.confidence <= 1.0, \
            f"bad confidence: {r.confidence}"
        assert r.hypothesis_id == "h_006"

    buckets = group_by_verdict(triaged)
    print(f"\n  Verdict distribution: "
          f"supports={len(buckets['supports'])}  "
          f"refutes={len(buckets['refutes'])}  "
          f"tangential={len(buckets['tangential'])}")

    for bucket_name in ("supports", "refutes", "tangential"):
        if not buckets[bucket_name]:
            continue
        print(f"\n  --- {bucket_name.upper()} (top 3) ---")
        for r in buckets[bucket_name][:3]:
            conf = f"{r.confidence:.2f}" if r.confidence is not None else "—"
            tags = (r.signal_tags or [])[:4]
            print(f"    [{conf}]  {(r.title or '')[:64]}")
            print(f"          signal_tags={tags}")

    # Hard expectation: triage must have RUN the LLM (not just deterministic
    # fallback). Signature of LLM-derived verdicts vs deterministic ones:
    # confidence ≠ exactly 0.3 (the fallback's tangential constant) AND/OR
    # signal_tags populated by the LLM. We check that at least one link has
    # either non-trivial confidence or LLM-supplied tags.
    llm_evidence = [
        lk for lk in triaged
        if (lk.signal_tags and any(len(t) > 1 for t in lk.signal_tags))
        or (lk.confidence is not None and abs((lk.confidence or 0) - 0.3) > 0.01)
    ]
    assert llm_evidence, (
        "no LLM-derived verdict signature found — every link returned "
        "deterministic-fallback confidence=0.30 with no signal tags. "
        "Likely LLM batch parse failure."
    )
    decided = len(buckets["supports"]) + len(buckets["refutes"])
    print(f"\n  ✓ {len(llm_evidence)}/{len(triaged)} links carry LLM-derived verdict signature")
    print(f"  ℹ {decided}/{len(triaged)} links classified supports/refutes "
          f"(tangential is a valid outcome when evidence is on-topic but not specific)")

    # Falsifier-bearing links (counter_evidence archetype) should mostly land
    # in supports/refutes given evidence quality — but at minimum we should
    # not see them silently dropped or stripped of fields.
    falsifier_links = [lk for lk in triaged if lk.query.falsifier]
    if falsifier_links:
        print(f"  ✓ {len(falsifier_links)} falsifier-query links survived triage")

    # Deterministic-path probe (use_llm=False) — must succeed offline
    triaged_det = await triage(h006, decomp, shorts[:3], max_triage=3, use_llm=False)
    for r in triaged_det:
        assert r.supports_or_refutes in {"supports", "refutes", "tangential"}
        assert r.confidence is not None
    print(f"  ✓ deterministic fallback verdict OK on {len(triaged_det)} links")




# ── 10. Orchestrator + job runner (Step 8) ────────────────────────────────────


async def test_orchestrator(state: Dict[str, Any]) -> None:
    print("\n=== ORCHESTRATOR + JOB RUNNER ===")

    # Registry sanity — YouTube Shorts should be registered + available
    reg = default_registry()
    available = reg.available_channels()
    assert "youtube_shorts" in available, \
        f"expected youtube_shorts in default registry, got {available}"
    print(f"  registered & available: {available}")

    # ── Direct run_pipeline (no job_runner) for h_006 ─────────────────────
    h006 = {**H_006, "investigation_priority": "high"}
    w = TimeWindow.from_label("1y")
    events: List[PipelineEvent] = []
    print(f"\n  Running pipeline for {h006['hypothesis_id']} (window=1y, use_llm=True)…")
    result = await run_pipeline(
        h006, w,
        registry=reg,
        emit=events.append,
        use_llm=True,
        max_triage=10,
    )

    print(f"  pipeline returned in {result.elapsed_sec:.2f}s with "
          f"{len(events)} events")
    print(f"  channels used:    {list(result.links_by_channel.keys())}")
    print(f"  channels skipped: {result.channels_skipped}")

    # Event-flow invariants
    event_kinds = [e.kind for e in events]
    print(f"  event kinds: {event_kinds[:3]}…{event_kinds[-2:]}")
    assert events[0].kind == "pipeline_start", \
        f"expected pipeline_start first, got {events[0].kind}"
    assert events[-1].kind in ("pipeline_complete", "pipeline_error"), \
        f"expected terminal event last, got {events[-1].kind}"

    stage_kinds = {e.stage for e in events if e.stage}
    expected_stages = {"L0_decompose", "L1_source_select", "L2_query_synth",
                       "L3_discover", "L6_triage"}
    assert expected_stages <= stage_kinds, \
        f"missing pipeline stages: {expected_stages - stage_kinds}"
    print(f"  ✓ all 5 pipeline stages emitted: {sorted(stage_kinds)}")

    # Every stage_start should have a matching stage_done
    starts = [e.stage for e in events if e.kind == "stage_start"]
    dones = [e.stage for e in events if e.kind == "stage_done"]
    assert set(starts) == set(dones), \
        f"unpaired stage events: starts={starts} dones={dones}"
    print(f"  ✓ every stage_start matched by a stage_done")

    # Channel discovery events
    chan_events = [e for e in events if e.kind == "channel_discovered"]
    assert chan_events, "expected at least one channel_discovered event"
    yt_event = next((e for e in chan_events if e.data.get("channel") == "youtube_shorts"),
                    None)
    assert yt_event is not None, "youtube_shorts channel event missing"
    assert not yt_event.data.get("skipped"), "youtube_shorts should not be skipped"
    print(f"  ✓ youtube_shorts discovery emitted with "
          f"{yt_event.data.get('n_links')} links across "
          f"{yt_event.data.get('n_queries_run')} queries")

    # Verdict distribution
    print(f"  verdict counts:    "
          f"supports={len(result.grouped['supports'])}  "
          f"refutes={len(result.grouped['refutes'])}  "
          f"tangential={len(result.grouped['tangential'])}")
    assert len(result.triaged_links) > 0, "pipeline produced zero triaged links"
    assert sum(len(v) for v in result.grouped.values()) == len(result.triaged_links)

    # Top supports — the demo headline
    if result.grouped["supports"]:
        print(f"\n  --- Top SUPPORTS (Day-10 demo material) ---")
        for r in result.grouped["supports"][:3]:
            conf = f"{r.confidence:.2f}" if r.confidence is not None else "—"
            tags = (r.signal_tags or [])[:3]
            print(f"    [{conf}]  {(r.title or '')[:60]}")
            print(f"          {tags}")

    # ── job_runner round-trip with live event subscription ────────────────
    print(f"\n  --- JobRunner + SSE-style subscription ---")
    clear_jobs()

    # Use a tighter h_006-like hypothesis so this is fast (smaller window).
    # Subscribe BEFORE awaiting, to verify history-replay-then-stream behaviour.
    job_id = create_job(
        h006,
        TimeWindow.from_label("1y"),
        registry=reg,
        use_llm=True,
        max_triage=5,
    )
    print(f"  created job_id={job_id}")
    job = get_job(job_id)
    assert job is not None
    assert job.status in ("pending", "running")

    streamed: List[str] = []
    async def _drain():
        async for ev in subscribe(job_id):
            streamed.append(ev.kind)

    drain_task = asyncio.create_task(_drain())
    final = await await_completion(job_id, timeout=120)
    # Give the subscriber a moment to drain the terminal events + sentinel.
    await asyncio.wait_for(drain_task, timeout=10)

    print(f"  job final status: {final.status}")
    print(f"  events streamed via subscribe(): {len(streamed)}")
    assert final.status == "done", f"job ended in {final.status}: {final.error}"
    assert streamed[0] == "pipeline_start", \
        f"subscriber missed pipeline_start: {streamed[:3]}"
    assert streamed[-1] in ("pipeline_complete", "pipeline_error"), \
        f"subscriber missed terminal event: {streamed[-3:]}"
    assert len(streamed) == len(final.history), \
        f"subscriber stream ({len(streamed)}) ≠ job history ({len(final.history)})"
    print(f"  ✓ subscriber stream matched job history exactly")

    # Summary payload for /jobs/{id}
    summary = final.to_summary()
    assert summary["status"] == "done"
    assert "verdict_counts" in summary
    print(f"  job summary: status={summary['status']}  "
          f"verdicts={summary['verdict_counts']}  "
          f"elapsed={summary['elapsed_sec']}s")




# ── 11. Long-form discoverers (Step 10) ───────────────────────────────────────


async def test_long_form_discoverers() -> None:
    print("\n=== LONG-FORM DISCOVERERS (Reddit / Google PAA / News) ===")

    w = TimeWindow.from_label("90d")
    q = TypedQuery(
        text="kelloggs cornflakes india review",
        channel="reddit",
        archetype=1, archetype_name="entity_pain",
        target_signal="user_complaint_text",
        hypothesis_id="h_006", falsifier=False,
        geo_proxies=["india"],
    )

    # ── Reddit ────────────────────────────────────────────────────────────
    r = get_reddit()
    print(f"\n  Reddit (mode={r.mode}, available={r.available})")
    if r.available:
        results = await r.discover(q, w, count=5)
        print(f"    discovered {len(results)} reddit threads")
        for lk in results[:3]:
            print(f"    - {lk.title[:70]}")
            print(f"      {lk.url}")
            print(f"      backend={lk.backend_used}  tags={lk.signal_tags[:2]}")
        for lk in results:
            assert lk.channel == "reddit"
            assert lk.hypothesis_id == "h_006"
            assert "reddit.com" in lk.url, f"non-reddit URL leaked: {lk.url}"

    # ── Google PAA ────────────────────────────────────────────────────────
    paa = get_google_paa()
    q_paa = TypedQuery(
        **{**q.model_dump(), "channel": "google_paa", "archetype": 5,
           "archetype_name": "question_analysis"}
    )
    print(f"\n  Google PAA (available={paa.available})")
    if paa.available:
        try:
            results = await paa.discover(q_paa, w, count=4)
            print(f"    discovered {len(results)} PAA entries")
            for lk in results[:3]:
                print(f"    - {lk.title[:70]}")
                print(f"      {lk.url[:80]}")
            for lk in results:
                assert lk.channel == "google_paa"
                assert lk.hypothesis_id == "h_006"
        except Exception as e:
            # Headless can fail in sandboxes / CAPTCHA — don't fail the smoke test
            print(f"    headless PAA failed (tolerated): {type(e).__name__}: {e}")

    # ── News ──────────────────────────────────────────────────────────────
    news = get_news()
    q_news = TypedQuery(
        **{**q.model_dump(), "channel": "news", "archetype": 8,
           "archetype_name": "crisis_event"}
    )
    print(f"\n  News (available={news.available})")
    if news.available:
        results = await news.discover(q_news, w, count=5)
        print(f"    discovered {len(results)} news articles")
        for lk in results[:3]:
            print(f"    - {lk.title[:70]}")
            print(f"      {lk.url[:80]}")
        for lk in results:
            assert lk.channel == "news"
            assert lk.hypothesis_id == "h_006"

    # ── Registry sanity — Step 12 adds 4 more channels (9 total) ─────────
    reg = default_registry()
    available = set(reg.available_channels())
    print(f"\n  default_registry().available_channels(): {sorted(available)}")
    expected = {
        "youtube_shorts", "tiktok",                              # short-video
        "youtube",                                               # long-form video
        "reddit", "quora",                                       # discussion / Q&A
        "google_paa", "news",                                    # search-graph + news
        "substack", "marketplace",                               # essays + reviews
    }
    missing = expected - available
    assert not missing, f"missing discoverers in default_registry: {missing}"
    print(f"  ✓ all 9 Step-6+Step-10+Step-11+Step-12 channels registered & available")
    # IG Reels is deliberately NOT registered (per locked decision §6.5 — v1.1)
    assert "instagram_reels" not in available, \
        "instagram_reels should NOT be in registry — deferred to v1.1 by locked decision"
    print(f"  ✓ instagram_reels correctly absent (deferred to v1.1)")




# ── 12. TikTok discoverer (Step 11) ───────────────────────────────────────────


async def test_tiktok() -> None:
    print("\n=== TIKTOK DISCOVERER (Step 11) ===")

    # ── Pure-fn URL parsing (no network) ─────────────────────────────────
    from link_extraction.discoverers.tiktok import (
        TikTokURLKind,
        _canonical,
        _extract_hashtags,
        _parse_tiktok_url,
    )

    # Video URLs
    assert _parse_tiktok_url("https://www.tiktok.com/@foodtok/video/12345") == \
        (TikTokURLKind.VIDEO, "foodtok", "12345")
    assert _parse_tiktok_url("https://tiktok.com/@a_b.c/video/9999") == \
        (TikTokURLKind.VIDEO, "a_b.c", "9999")
    assert _parse_tiktok_url("https://m.tiktok.com/@user/video/7777?lang=en") == \
        (TikTokURLKind.VIDEO, "user", "7777")
    # Discover pages (Brave-dominant)
    assert _parse_tiktok_url("https://www.tiktok.com/discover/kelloggs-cornflakes-edeka") == \
        (TikTokURLKind.DISCOVER, "kelloggs-cornflakes-edeka", "")
    # Tag pages
    assert _parse_tiktok_url("https://www.tiktok.com/tag/cornflakes") == \
        (TikTokURLKind.TAG, "cornflakes", "")
    # Rejected forms
    assert _parse_tiktok_url("https://www.tiktok.com/@user") is None  # profile
    assert _parse_tiktok_url("https://vm.tiktok.com/ABC123") is None  # shortlink
    assert _parse_tiktok_url("https://youtube.com/shorts/abc") is None  # wrong host
    assert _parse_tiktok_url("") is None
    print("  ✓ URL parser: classifies VIDEO/DISCOVER/TAG; rejects profiles/shortlinks/foreign hosts")

    assert _canonical(TikTokURLKind.VIDEO, "user", "123") == \
        "https://www.tiktok.com/@user/video/123"
    assert _canonical(TikTokURLKind.DISCOVER, "k-c", "") == \
        "https://www.tiktok.com/discover/k-c"
    assert _canonical(TikTokURLKind.TAG, "cornflakes", "") == \
        "https://www.tiktok.com/tag/cornflakes"
    print("  ✓ _canonical() round-trips for all 3 URL kinds")

    assert _extract_hashtags("#GenZ #fyp #cornflakesreview #fyp") == \
        ["genz", "fyp", "cornflakesreview"]
    print("  ✓ hashtag extractor: case-fold + dedup")

    # ── Live discovery ───────────────────────────────────────────────────
    tt = get_tiktok()
    print(f"\n  available={tt.available}")
    if not tt.available:
        print("  ⚠ Brave unavailable — skipping live TikTok discovery")
        return

    # Use a Gen-Z identity query (h_test_genz territory — short-video native)
    w = TimeWindow.from_label("1y")
    q = TypedQuery(
        text="kelloggs cornflakes",
        channel="tiktok",
        archetype=1, archetype_name="entity_pain",
        target_signal="user_complaint_clip",
        hypothesis_id="h_006", falsifier=False,
        geo_proxies=["india"],
    )
    results = await tt.discover(q, w, count=5)
    print(f"  discover() returned {len(results)} TikTok videos")

    for r in results[:4]:
        print(f"    - {r.title[:70]}")
        print(f"      url:      {r.url}")
        print(f"      creator:  {r.creator}  hashtags: {r.hashtags[:4]}")
        print(f"      backend:  {r.backend_used}")

    # Hard invariants — every link must be a canonical TikTok URL (video,
    # discover, or tag form), have hypothesis_id passthrough, be a
    # ShortVideoLink (so triage takes the short-video path), and have
    # basic-mode counts as None (not 0) so the LLM can distinguish
    # "unknown" from "actually zero".
    accepted_prefixes = (
        "https://www.tiktok.com/@",         # video
        "https://www.tiktok.com/discover/", # aggregate topic
        "https://www.tiktok.com/tag/",      # aggregate hashtag
    )
    for r in results:
        assert r.channel == "tiktok", f"wrong channel: {r.channel}"
        assert r.hypothesis_id == "h_006", "hypothesis_id passthrough broken"
        assert isinstance(r, ShortVideoLink), \
            "TikTok must emit ShortVideoLink (triage's short-video path)"
        assert any(r.url.startswith(p) for p in accepted_prefixes), \
            f"non-canonical URL leaked: {r.url}"
        assert r.creator and r.creator.startswith("@"), \
            f"creator missing or unformatted: {r.creator!r}"
        # v1 basic mode: counts should be None (not 0). Triage handles missing.
        assert r.view_count is None and r.like_count is None, \
            "v1 basic mode should NOT fabricate view/like counts"

    # Demand non-zero results when Brave is live — Brave's TikTok index is
    # rich enough that "kelloggs cornflakes" should always surface ≥1 page.
    assert len(results) > 0, \
        "TikTok discovery returned 0 with Brave live — index parse regression?"

    kinds_seen = {r.backend_used.rsplit(":", 1)[-1] for r in results}
    print(f"  ✓ all {len(results)} links pass shape invariants " +
          f"(URL kinds seen: {sorted(kinds_seen)})")




# ── 13. Step 12 discoverers (Quora / YT long-form / Substack / Marketplace) ───


async def test_step12_discoverers() -> None:
    print("\n=== STEP 12 DISCOVERERS (Quora / YT long-form / Substack / Marketplace) ===")

    w = TimeWindow.from_label("1y")

    # ── Quora ─────────────────────────────────────────────────────────────
    q_quora = TypedQuery(
        text="kelloggs cornflakes india",
        channel="quora",
        archetype=5, archetype_name="question_analysis",
        target_signal="question_thread",
        hypothesis_id="h_006", falsifier=False,
        geo_proxies=["india"],
    )
    quora = get_quora()
    print(f"\n  Quora (available={quora.available})")
    if quora.available:
        results = await quora.discover(q_quora, w, count=5)
        print(f"    discovered {len(results)} Quora threads")
        if not results:
            print("    (Brave's Quora index is sparse for narrow queries — "
                  "0 is common and tolerated; v1.1 will add headless scrape)")
        for r in results[:3]:
            print(f"    - {(r.title or '')[:70]}")
            print(f"      {r.url}")
        for r in results:
            assert r.channel == "quora"
            assert r.hypothesis_id == "h_006"
            assert "quora.com" in r.url, f"non-Quora URL leaked: {r.url}"
            # Canonical form: no /answer/ suffix
            assert "/answer/" not in r.url, f"answer suffix not stripped: {r.url}"

    # ── YouTube long-form ─────────────────────────────────────────────────
    q_yt = TypedQuery(
        text="kelloggs cornflakes india review",
        channel="youtube",
        archetype=7, archetype_name="expert_authority",
        target_signal="expert_authority_text",
        hypothesis_id="h_006", falsifier=False,
        geo_proxies=["india"],
    )
    yt = get_youtube()
    print(f"\n  YouTube long-form (available={yt.available}, video_duration={yt.video_duration})")
    if yt.available:
        results = await yt.discover(q_yt, w, count=3)
        print(f"    discovered {len(results)} long-form YT videos")
        if not results:
            print("    (no ≥182s videos for this query — tolerated, narrow topics often "
                  "lack long-form coverage)")
        for r in results[:3]:
            dur = f"{r.duration_sec}s" if r.duration_sec else "?"
            views = f"{r.view_count:,}" if r.view_count else "?"
            print(f"    - [{dur}, views={views}] {(r.title or '')[:55]}")
            print(f"      {r.url}")
        for r in results:
            assert r.channel == "youtube", f"wrong channel: {r.channel}"
            assert r.hypothesis_id == "h_006"
            assert r.url.startswith("https://www.youtube.com/watch?v="), \
                f"non-canonical watch URL: {r.url}"
            assert isinstance(r, ShortVideoLink), \
                "YT long-form must emit ShortVideoLink (carries duration/views/etc)"
            # Long-form filter: ≥ 182s, otherwise it's Shorts territory
            if r.duration_sec is not None:
                assert r.duration_sec >= 182, \
                    f"Shorts-band video leaked into long-form: {r.duration_sec}s"

    # ── Substack ──────────────────────────────────────────────────────────
    q_sub = TypedQuery(
        text="breakfast cereal industry decline",
        channel="substack",
        archetype=7, archetype_name="expert_authority",
        target_signal="expert_essay",
        hypothesis_id="h_006", falsifier=False,
        geo_proxies=["india"],
    )
    sub = get_substack()
    print(f"\n  Substack (available={sub.available})")
    if sub.available:
        results = await sub.discover(q_sub, w, count=5)
        print(f"    discovered {len(results)} Substack essays")
        for r in results[:3]:
            print(f"    - {(r.title or '')[:70]}")
            print(f"      {r.url}")
        for r in results:
            assert r.channel == "substack"
            assert r.hypothesis_id == "h_006"
            assert "substack.com" in r.url, f"non-substack URL leaked: {r.url}"
            assert "/p/" in r.url, f"non-essay URL leaked (homepage?): {r.url}"

    # ── Marketplace ───────────────────────────────────────────────────────
    q_mkt = TypedQuery(
        text="kelloggs cornflakes review",
        channel="marketplace",
        archetype=1, archetype_name="entity_pain",
        target_signal="user_complaint_text",
        hypothesis_id="h_006", falsifier=False,
        geo_proxies=["india"],
    )
    mkt = get_marketplace()
    print(f"\n  Marketplace (available={mkt.available})")
    if mkt.available:
        results = await mkt.discover(q_mkt, w, count=5)
        print(f"    discovered {len(results)} marketplace reviews")
        for r in results[:3]:
            print(f"    - {(r.title or '')[:70]}")
            print(f"      {r.url}")
            print(f"      tags={r.signal_tags[:2]}")
        for r in results:
            assert r.channel == "marketplace"
            assert r.hypothesis_id == "h_006"
            # Should carry a `host:` signal tag
            assert any(t.startswith("host:") for t in r.signal_tags), \
                f"marketplace link missing host tag: {r.signal_tags}"


async def main() -> int:
    state: Dict[str, Any] = {}
    try:
        test_models()
        test_temporal()
        test_decomposer()
        test_channel_scorer()
        test_trends()
        test_llm()
        test_query_synth()
        await test_backends()
        await test_yt_shorts(state)
        await test_triage(state)
        await test_orchestrator(state)
        await test_long_form_discoverers()
        await test_tiktok()
        await test_step12_discoverers()
    except AssertionError as e:
        print(f"\n✗ ASSERTION FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return 1
    print("\n✓ Smoke test complete (Steps 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 10 + 11 + 12)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
