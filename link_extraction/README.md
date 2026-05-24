# Data Selection Module — `link_extraction`

**Status:** under active build (Steps 1–3 of 14 shipped)
**Location:** `MRX_DataSelector/link_extraction/` (sibling to the host repo at `MRX_Module_1 Claude/`)
**Will eventually replace, in the host repo:** `services/link_farming.py` + `services/discovery_engine.py` + `api/manifest.py` + `api/discovery.py` (those stay live until parity is validated end-to-end)
**Planning doc:** `MRX_DataSelector/DATA_SELECTION_MODULE_CONTEXT.md`. This README is the as-built overview.

---

## 1. What this module does

Given a set of **approved hypotheses** from the upstream Hypothesis Engine plus a user-selected **time window**, the Data Selection Module discovers and ranks the best links on the internet that evidence (or refute) each hypothesis. Output is a per-hypothesis link set grouped by verdict (`supports` / `refutes` / `tangential`), downloadable as CSV.

**Three rules the design follows everywhere:**

1. **Sources serve hypotheses, not topics.** Every link carries `hypothesis_id`. Discovery is scoped per hypothesis, never per brand.
2. **Channel choice is deterministic; the LLM only writes rationales.** Cuts cost ~60% and makes outputs reproducible. The (dimension × force × channel) weight grid in [`channel_weights.yaml`](channel_weights.yaml) is hand-curated, not hallucinated.
3. **Search queries are composed from archetypes, not freewritten.** Nine archetypes (entity-pain, switching, comparison, aspiration, question, counter, expert, crisis, hashtag) parameterised by `{entity, geo, time, rising_phrase}`. Falsifier queries are mandatory, not optional.

---

## 2. Where this fits in Outtlyr

```
INTAKE                →  HYPOTHESIS ENGINE             →  DATA SELECTION (this module)
(Teal agent)             (Decomposer / Generator /        (this module)
ChatGPT-style            MECE Auditor + Ecosystem)        L0 decompose → L7 emit
conversation +
brief upload
```

Upstream contract: every hypothesis carries
`{hypothesis_id, statement, dimension, force_assignment, expected_signals, expected_counter_signals, contrarian_pair_id, investigation_priority, rationale, core_problem_*}`.
Dimension enum has 10 values, force enum has 5 (the host's actual five: *Demand Gravity, Choice Architecture Pressure, Value Elasticity Field, Reinforcement Stability, Competitive Energy Field*), priority enum has 3 (`high | medium | low`).

---

## 3. Pipeline architecture (7 layers)

```
APPROVED HYPOTHESES + USER-SELECTED TimeWindow
   │
   ▼
[L0] Hypothesis Decomposition           ── pure-Python, no LLM
     entities / pains / aspirations / identity_claims / geo_hints
     + per-signal archetype classification
   │
   ▼
[L1] Source Selector                    ── deterministic table + ONE LLM call
     channel_weights.yaml → fit_score per channel
     anti-platform exclusion / drop fit_score<50 / cap top 5
     LLM writes per-channel rationale + signal_fingerprint
   │
   ▼
[L2] Query Synthesizer                  ── 9 archetypes + Trends amplifier
     pytrends rising_queries → consumer phrasings injected into slots
     Hard validate: ≥1 falsifier, ≥1 pain, ≥1 aspiration per channel
   │
   ▼
[L3] Channel-Native Discovery           ── parallel per (hypothesis × channel)
     YT Data API + Reddit PRAW + headless PAA + Brave News + …
     Walks each channel's native related-graph (PAA BFS, relatedToVideoId,
     crossposts, Quora related-questions sidebar)
   │
   ▼
[L4] Temporal Filtering                 ── three places: query-time, result-time, trends-time
   │
   ▼
[L5] Dedup + Cross-Platform Clustering
     Canonical URL (strip utm_*/gclid/fbclid, sort params)
     Content hash (title + body[:200]) — kills syndicated dupes
     Cross-platform: same URL discovered via N channels → one node
   │
   ▼
[L6] Measured Triage                    ── fetch + regex + LLM verdict
     Top 30 per hypothesis: fetch body[:1500B] (or caption+comments for shorts)
     signal_keyword_hit_rate (regex from expected_signals) + ONE LLM batch call
     Verdict: supports | refutes | tangential + confidence
   │
   ▼
[L7] Group + Rank + Emit                ── grouped by hypothesis_id → verdict bucket
     SSE-streamed progress; CSV + JSON exports
```

---

## 4. Search backends (Brave → DDG → Headless Google)

Three-tier fallback chain. Headless is the **last resort** (slow, CAPTCHA-prone), not a peer.

| Tier | Backend | Role | Activation |
|---|---|---|---|
| 1 | **Brave Search API** | primary web/news/forums/videos | `BRAVE_API_KEY` set |
| 2 | **DuckDuckGo** | fallback when Brave returns <3 results | `USE_DUCKDUCKGO=1` + `pip install duckduckgo-search` |
| 3 | **Headless Google** (Playwright Chromium) | final fallback **+ the only path for PAA and Related Searches** | `pip install playwright && playwright install chromium` |
| — | **SerpAPI** | placeholder; stub returns `[]` | flip `ENABLE_SERPAPI=1` and replace [`serpapi_stub.py`](backends/serpapi_stub.py) when re-enabling |

The single entry point is [`backends.search_with_fallback(query, vertical, count, window, min_results)`](backends/registry.py). Verticals `paa` and `related` skip tiers 1–2 and go straight to headless because no other backend exposes those Google SERP features.

**Headless concurrency** is capped at `HEADLESS_CONCURRENCY=3` via a global asyncio Semaphore (~450 MB RAM at most). On CAPTCHA detection, the backend self-disables for `HEADLESS_CAPTCHA_COOLDOWN_SEC=30`s and short-circuits to `[]` so the caller can fall back.

---

## 5. Channel inventory

13 channels across 4 format families. Short-video channels (🎬) get their own discoverer track with `ShortVideoLink` metadata (duration, caption, hashtags, view/like counts, top comments).

| Channel | Format | Backend | Time-filter support | Cap per hyp |
|---|---|---|---|---|
| `google_web` | text | Brave→DDG→Headless | API native | 30 |
| `google_paa` | text (questions) | Headless **only** | API native | 50 |
| `google_related` | text | Headless **only** | API native | 20 |
| `reddit` | discussion | PRAW | API native | 50 |
| `quora` | Q&A | Brave→DDG→Headless + Playwright sidebar | API native | 30 |
| `youtube` | long video | YT Data API | API native | 30 |
| 🎬 `youtube_shorts` | short video | YT Data API (`videoDuration=short`) | API native | 30 |
| 🎬 `tiktok` | short video | Brave→DDG→Headless | client-side only | 30 |
| 🎬 `instagram_reels` | short video | Brave→DDG→Headless | client-side only | 20 |
| `news` | text | Brave News + GDELT 2.0 (for >1y) | API native | 20 |
| `substack` | long text | Substack search + Brave | client-side only | 15 |
| `trends` | rising queries (no links) | pytrends | API native | feeds L2 only |
| `marketplace` | reviews | Brave `site:` + Trustpilot scrape | client-side only | 20 |

---

## 6. Configuration

Env vars are loaded from either `MRX_DataSelector/.env` (preferred) or the host repo's `MRX_Module_1 Claude/backend/.env` (fallback). Both gitignored. See [`../.env.example`](../.env.example) for the complete template.

| Variable | Default | Purpose |
|---|---|---|
| `BRAVE_API_KEY` | — | required for primary search tier |
| `USE_DUCKDUCKGO` | `1` | `0` to disable DDG fallback |
| `ENABLE_SERPAPI` | `0` | flip to `1` after replacing the stub |
| `SERPAPI_KEY` | — | only read when stub is replaced |
| `HEADLESS_CONCURRENCY` | `3` | max parallel Chromium contexts (≈150 MB each) |
| `HEADLESS_CAPTCHA_COOLDOWN_SEC` | `30` | backend self-disable window on CAPTCHA hit |
| `YOUTUBE_API_KEY` | — | wired in Step 6 (YT Shorts) |
| `REDDIT_CLIENT_ID` / `_SECRET` / `_USER_AGENT` | — | wired in Step 10 (Reddit) |
| `VERTEX_AI_API_KEY` | — | Gemini on Vertex AI (primary when `HYPOTHESIS_PROVIDER=gcp_gemini`) — used by `_llm.py` |
| `GEMINI_API_KEY` | — | Gemini in Google AI Studio (fallback path) — used by `_llm.py` |
| `HYPOTHESIS_PROVIDER` | — | set to `gcp_gemini` to prefer Vertex; otherwise Studio first |

---

## 7. Current build status

| # | Step | Status | Validation |
|---|---|---|---|
| 1 | Models, temporal, channel_weights.yaml, 4 search backends | ✅ done | smoke test: live Brave query returns 5+ Indian Kellogg's URLs; all temporal translations correct across 7d/30d/90d/1y/5y |
| 2 | L0 decomposer (entities/pains/aspirations/identity/geo + signal archetypes) | ✅ done | smoke test against real Kellogg's `h_006` from sample CSV + synthetic Gen-Z case |
| 3 | L1 channel scorer (deterministic) | ✅ done | h_006 top: `marketplace(82) reddit(81) youtube(76) google_paa(70) youtube_shorts(69)`; identity_expression top surfaces all three short-video channels; `news` excluded by `Reinforcement Stability` anti_platforms |
| 4 | Trends seed (`trends_seed.py` — pytrends wrapper feeding L2) | ✅ done | live call: `rising_queries("kelloggs", 90d, IN)` → 4 phrasings incl. `kelloggs muesli 1 kg`; geo `indian→IN`, `us→US`; in-process cache verified; pytrends errors → `[]` |
| 5 | L2 query synthesizer (9 archetypes + Trends amplifier + LLM slot-fill via Gemini + falsifier validation) | ✅ done | h_006: 4 channels × 8 queries each, every channel has ≥1 falsifier; identity_expression: short-video channels include archetype #9 (hashtag), long-form never does; YT Shorts mix = {1,4,6,7,9}; LLM path adds +4 queries (alt-product enrichment); MECE pair pooling via `synthesize_pair_pooled()`. **LLM = native Gemini** (Vertex AI primary when `HYPOTHESIS_PROVIDER=gcp_gemini`, Gemini Studio fallback) — no OpenRouter dependency |
| **6** | 🎬 **YouTube Shorts discoverer (priority #1 channel)** | ✅ done | live `kelloggs cornflakes india` (1y) → 7 Shorts, all ≤181s, full `ShortVideoLink` payload (duration/views/likes/comments/hashtags/engagement_score); top-N enrichment fetches up to 5 top comments per video via commentThreads.list; native transcripts via `youtube-transcript-api` (EN→HI fallback). Demo content includes: brand-defending official Kellogg's India Shorts (falsifier), competitor "Bagrry Cornflakes" comparison, Dermatologist breakfast critique, and "Better Breakfast Options" switching-narrative comments |
| 7 | L6 measured triage (`triage.py` — short-video + long-form paths, signal regex, Gemini batch verdict, top-30 budget) | ✅ done | live triage of 7 YT Shorts for h_006 yielded 3 SUPPORTS (Bagrry comparison @ 0.90, dermatologist critique @ 0.90, Kellogg's Multigrain pivot @ 0.85 — Gemini correctly read brand pivot as evidence of original product failing) + 4 TANGENTIAL; suffix-aware regex matches plurals/conjugations; batch_size=6 + dynamic max_tokens keeps JSON responses uncut; deterministic fallback verified |
| 8 | Orchestrator (`orchestrator.py`) + Job runner (`job_runner.py` — in-memory store, per-subscriber asyncio.Queue fanout, SSE-ready event stream) | ✅ done | live h_006 pipeline run in 78.75s with 16 events covering all 5 stages (L0→L1→L2→L3→L6); 9 queries × YT Shorts → 49 candidates → 10 triaged → 2 SUPPORTS (`Bagrry Cornflakes` @ 0.90, Hindi `corn flakes chivda` Short @ 0.75 — Indian consumers re-purposing cornflakes into traditional savory poha, direct evidence of h_006's taste-misalignment claim); JobRunner round-trip verified — subscriber stream matched job history exactly (16/16 events, no drops) |
| 9 | API + minimal frontend (ShortVideoGrid) | ✅ **DAY-10 DEMO SHIPPED** | Standalone FastAPI app at [`../api/app.py`](../api/app.py); single-page demo at [`../api/static/index.html`](../api/static/index.html) (teal/dark, hypothesis form, live SSE log, grouped Shorts grid). Verified end-to-end: `POST /start` → SSE events → `GET /results` w/ thumbnails + engagement bars + verdicts |
| 10 | Long-form discoverers (Reddit + Google PAA + News) | ✅ done | All 4 channels in `default_registry()` now: `youtube_shorts`, `reddit`, `google_paa`, `news`. Reddit dual-mode (PRAW if creds present, Brave `site:reddit.com` fallback always works) — live discovery found r/CasualIreland "Change of flavour in Bran Flakes" (taste complaint), r/AskIndia "Need some breakfast recommendations" (substitution), r/cereal direct product complaint. News via Brave returned "Is Kellogg's Corn Flakes Actually Healthy?". PAA wired through headless Chromium. Orchestrator on h_006 now reports `channels_used: ['reddit', 'youtube_shorts']` |
| 11 | TikTok discoverer (Instagram Reels deferred to v1.1 per locked decision §6.5) | ✅ done | `TikTokDiscoverer(ShortVideoDiscoverer)` via Brave `site:tiktok.com`; URL parser classifies VIDEO (`/@user/video/id`) + DISCOVER (`/discover/topic-slug`) + TAG (`/tag/name`); aggregate pages emit `@tiktok/discover` / `@tiktok/tag` synthetic creators with topic slugs as hashtags. Live "kelloggs cornflakes" returned 5 Discover pages (Brave's TikTok index is Discover-dominated; canonical video URLs are rare). Registered in `default_registry()`; **5 channels live**: youtube_shorts, tiktok, reddit, google_paa, news |
| 12 | Quora + YouTube long-form + Substack + Marketplace discoverers | ✅ done | 4 new discoverers + `_common.raw_result_to_link` helper. **Substack** (Brave `site:substack.com`, essay-path filter `/p/`) — 5 essays incl. "Breakfast Cereals: A Buying Guide". **Marketplace** (Brave OR-of-`site:` across Trustpilot/Amazon/Flipkart/etc, `host:` signal tags) — 5 Trustpilot Kellogg's review pages. **YouTube long-form** (YT Data API w/ `videoDuration="any"` + Python 182s floor) — 1 video at 234s "KELLOGG'S CORN FLAKES LAB TESTED". **Quora** (Brave `site:quora.com`, `/answer/` suffix stripped for dedup) — Brave's Quora index is sparse; 0-result tolerated, headless v1.1. **`default_registry()` now ships 9 channels** — all non-Trends + non-IG-Reels channels live |
| 13 | L5 cross-platform dedup + clustering (`dedup.py`) | ✅ done | Two-pass: canonical URL normalization (strips utm_*/gclid/fbclid/16+ other trackers, sorts params, drops fragment, lowercases host) + content-hash dedup (title + body[:200]) for syndicated copies. Returns `LinkCluster` list; representative goes to triage; verdict propagates to cluster members. New `also_found_on: List[ChannelId]` field on `DiscoveredLink` carries cross-platform roster. Orchestrator wires L5 between L3 and L6 with new `stage_start`/`stage_done` events. API + CSV (column #20) + UI card chip all surface `also_found_on` |
| 14 | Source memory store + cost meter + frontend polish + integration plan | ✅ done | **Memory store** (`memory_store.py`) — atomic JSON snapshots to `.memory/{jobs,batches}/`; FastAPI `on_event("startup")` restores past runs (verified: kill server → restart → `/memory/jobs` returns the prior run, CSV download still works on restored state). **Cost meter** (`cost_meter.py`) — per-job ledger tracking Gemini tokens × pricing → USD, YT API quota units, Brave call count; ContextVar `current_job_id` propagates across `asyncio.to_thread` so YT discoverers + `_llm.py` charge implicitly; default $0.50 cap (locked rec #4); `cost` field on PipelineResult + JSON `/results` + `cost_summary` event before pipeline_complete. **Frontend polish** — header cost pill, collapsible "Past runs" memory browser with hypothesis_id filter + click-to-load. **Integration plan** ([`INTEGRATION.md`](../INTEGRATION.md)) — file-by-file migration into host repo with rollback checklist |

Total estimate: ~20 working days. **All 14 steps shipped** ✅. Standalone demo runs at `MRX_DataSelector/api/`. Integration into host repo is documented in [`INTEGRATION.md`](../INTEGRATION.md).

---

## 8. File layout

```
MRX_DataSelector/
├── DATA_SELECTION_MODULE_CONTEXT.md   ← planning context
├── .env.example                       ← env template (copy to .env)
└── link_extraction/                   ← the package
    ├── README.md                      ← this file
    ├── __init__.py                    ← public exports
    ├── models.py                      ← Pydantic v2 models (TimeWindow / TypedQuery /
    │                                    ChannelFit / RawResult / DiscoveredLink /
    │                                    ShortVideoLink)
    ├── temporal.py                    ← TimeWindow → per-channel API param maps
    ├── channel_weights.yaml           ← 10 dims × 13 channels + 5 forces (with
    │                                    anti_platforms) + priority_weights +
    │                                    signal_detectability_base + confirmation_balance
    ├── decomposer.py                  ← L0: spaCy NER + curated lexicons
    ├── source_selector.py             ← L1: deterministic channel scoring   ✅
    ├── query_synthesizer.py           ← L2: archetypes + Trends amplifier   (Step 5)
    ├── trends_seed.py                 ← Trends amplifier helpers            (Step 4)
    ├── triage.py                      ← L6: fetch + regex + LLM verdict     (Step 7)
    ├── orchestrator.py                ← Stage 3 parallel runner             (Step 8)
    ├── job_runner.py                  ← in-memory job store + SSE plumbing  (Step 8)
    ├── discoverers/
    │   ├── base.py                    ← Discoverer ABC + ShortVideoDiscoverer
    │   ├── youtube_shorts.py          ← 🎬 PRIORITY #1                       (Step 6)
    │   ├── reddit.py
    │   ├── google_paa.py
    │   ├── news.py
    │   ├── tiktok.py / instagram_reels.py
    │   ├── quora.py / youtube.py / substack.py / marketplace.py
    │   └── trends.py
    ├── backends/                                                            ← ✅ Step 1
    │   ├── base.py                    ← SearchBackend ABC
    │   ├── brave.py                   ← PRIMARY
    │   ├── duckduckgo.py              ← FALLBACK (off if not installed)
    │   ├── headless_google.py         ← FINAL FALLBACK + PAA/Related-only path
    │   ├── serpapi_stub.py            ← placeholder
    │   └── registry.py                ← Brave → DDG → Headless fallback chain
    └── _smoke_test.py                 ← live validation across all shipped steps
```

The Python venv lives at `MRX_Module_1 Claude/backend/venv/` (host repo). It already has spaCy + en_core_web_sm, pydantic, pytrends, etc. installed. New deps for this module (`duckduckgo-search`, `playwright`, `spacy`) are recorded in the host's `backend/requirements.txt`.

---

## 9. Running the smoke test

From `MRX_DataSelector/`, with `BRAVE_API_KEY` populated in either `MRX_DataSelector/.env` or the host's `backend/.env`:

```bash
cd /Users/vdogg/Documents/mrxdatalabs_Station/MRX_DataSelector
"../MRX_Module_1 Claude/backend/venv/bin/python" -m link_extraction._smoke_test
```

Exit code `0` if all assertions pass. Live Brave queries are skipped gracefully when keys are missing. The smoke test grows with each step:

- **Step 1:** model parsing, temporal translation table, live Brave query, fallback-chain probe
- **Step 2:** decomposer assertions on real `h_006` (Kellogg's CSV) + synthetic Gen-Z case
- **Step 3:** channel scorer ranks for h_006 + identity_expression (verifies anti-platform exclusion + short-video presence)
- **Step 4:** trends_seed geo normalization, empty-input guards, live `rising_queries("kelloggs", 90d, IN)`, in-process cache hit, `related_topics`, `interest_over_time`, `batch_rising_queries`
- **Step 5:** query_synthesizer deterministic path (use_llm=False) on h_006 + h_test_genz; per-channel falsifier guarantee; hashtag-archetype confined to short-video; MAX_QUERIES_PER_CHANNEL cap; live Gemini round-trip (Vertex → Studio fallback chain) — JSON-mode `{ok:true, echo:'ping'}` verified
- **Step 6:** YT Shorts discoverer pure-fn parsers (ISO 8601 duration, hashtag extractor, engagement_score); live YouTube Data API v3 pull for "kelloggs cornflakes india" returning ≥1 enriched ShortVideoLink with comments + full payload; shape invariants (channel id, hypothesis_id passthrough, canonical `youtube.com/shorts/` URL, duration cap 181s)
- **Step 7:** triage pure-fns (signal expansion, suffix-aware regex, HTML strip, deterministic verdict rules); live LLM-driven triage of the Step-6 YT Shorts set against h_006 — every link receives an LLM verdict + confidence + signal_tags, decisive (supports/refutes) verdicts surface where evidence is strong, tangential verdicts where evidence is on-topic but not specific to the claim
- **Step 8:** default discoverer registry; live end-to-end `run_pipeline(h_006, 1y)` emits 16 structured events across L0/L1/L2/L3/L6, every `stage_start` paired with a `stage_done`, channels skipped (marketplace/reddit/youtube — no discoverer registered yet) reported via `channel_discovered` events; `create_job()` → `subscribe()` → `await_completion()` round-trip verified, subscriber stream length exactly matches job history (no event drops, no duplicates)
- **Step 9 (Day-10 demo):** `uvicorn api.app:app --port 8087` end-to-end — `POST /start` returns job_id, `GET /jobs/{id}/events` streams SSE events with proper `event:` / `data:` framing + `stream_end` bookend (17 events on h_006 1y/triage=3, 78.5s elapsed), `GET /jobs/{id}/results?wait=true` returns full grouped payload incl. `is_short_video` discrimination + thumbnails + hashtags + engagement_score + signal_tags + top_comments + decomposition + channel_fits + queries_by_channel; static page loads (25KB) with sample_hypotheses.json pre-loaded
- **Step 10:** `RedditDiscoverer` returns 5 reddit threads via Brave-fallback (PRAW creds optional — auto-detected at init); `NewsDiscoverer` returns 5 Brave-news articles; `GooglePAADiscoverer` initialises against headless Chromium (PAA box-empty for narrow queries is tolerated); `default_registry()` now has 4 channels available; orchestrator's `channels_used` includes both `reddit` and `youtube_shorts` on h_006 — no longer skipped
- **Step 11:** `TikTokDiscoverer` URL parser classifies all 3 TikTok URL kinds (VIDEO/DISCOVER/TAG); `_canonical()` round-trips for each; live Brave query returns 5 Discover-page ShortVideoLinks with synthetic creators + topic-slug hashtags; `strict_videos_only=False` default accepts aggregate pages (Brave's TikTok index reality); `default_registry()` now has 5 channels available; IG Reels deliberately absent (deferred to v1.1)
- **Step 12:** `QuoraDiscoverer` (Brave `site:quora.com` w/ `/answer/<author>` dedup); `YouTubeDiscoverer` (long-form, YT Data API, 182s floor instead of `videoDuration=medium` which excludes 3-4 min reviews) — found "KELLOGG'S CORN FLAKES LAB TESTED" (234s, 33K views, expert-authority signal); `SubstackDiscoverer` (Brave `site:substack.com` w/ `/p/<slug>` essay filter) — found "Breakfast Cereals: A Buying Guide"; `MarketplaceDiscoverer` (Brave OR-of-`site:` across Trustpilot/Amazon/Flipkart/Influenster/etc + `host:` signal tags) — 5 Trustpilot Kellogg's review pages; **9 channels registered & available** in `default_registry()`; IG Reels still deliberately absent
- **Step 13:** synthetic L5 tests (6 canonical URL cases incl. idempotency, content_hash collides on syndicated copies but stays distinct on different content, cross-platform same-URL cluster correctly flags `also_found_on=['tiktok']` when a YT URL is rediscovered via TikTok); live h_006 run produced L5 events `100 input → 100 clusters, 0 cross-platform` (no rediscovery for this product hypothesis — expected); API now returns 29-column CSV with `also_found_on` at position 20

---

## 10. Design decisions locked in

| # | Decision | Where it lives |
|---|---|---|
| 1 | Backend preference: **Brave → DuckDuckGo → Headless Google**. SerpAPI is a stub. | [`backends/registry.py`](backends/registry.py) |
| 2 | Headless concurrency = **3** (≈450 MB RAM cap); CAPTCHA cooldown = **30 s**. | [`backends/headless_google.py`](backends/headless_google.py) |
| 3 | **YouTube Shorts is priority #1 channel** for the first end-to-end demo. | build sequence in §7 |
| 4 | Channel scoring is **deterministic** (YAML weights), LLM only writes rationales. | [`channel_weights.yaml`](channel_weights.yaml) + `source_selector.py` |
| 5 | Channel-fit threshold = **50**; top-N cap = **5**. | `source_selector.py` |
| 6 | `TimeWindow` is **propagated three places**: query-time API filter, Trends amplifier timeframe, result-time client-side filter on `observed_at`. | [`temporal.py`](temporal.py) + each discoverer |
| 7 | Every link carries `hypothesis_id` and (for shorts) full engagement metadata; cross-platform dedup keeps one node with N source-references. | `models.py` `ShortVideoLink` + L5 |

---

## 11. Open decisions still pending

1. **LLM dispatch for L1 rationale generation.** Should register stage `source_selection` in `services/stage_models.PRESETS` (12 cells: 4 providers × 3 weights). Deferred until ready.
2. **Triage fetch budget:** top 30 per hypothesis (good triage, slower) or top 15 (faster, may misrank borderlines)? Recommend 30.
3. **MECE pair handling:** pool queries from a contrarian pair (`H_CP1_03` + `H_CP1_04`) as one cluster, or run isolated? Recommend pooled.
4. **Per-hypothesis cost cap:** default $0.50? Comes online with the cost meter in Step 14.
5. **Reels in v1 or punt to v1.1?** Lowest-yield short-video surface; headless pages are flaky.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Hypothesis** | One falsifiable claim from the upstream Hypothesis Engine; carries `dimension`, `force_assignment`, `expected_signals`, `contrarian_pair_id`. |
| **Channel** | A data source the module knows how to discover from (one of the 13 in §5). |
| **Backend** | A search API/scraper the channel discoverer uses underneath (Brave/DDG/headless/SerpAPI). Channels ≠ backends. |
| **Archetype** | One of 9 query templates (entity-pain, switching, comparison, aspiration, question, counter, expert, crisis, hashtag). Slots filled per hypothesis. |
| **Falsifier query** | Archetype 6 — a query designed to surface counter-evidence. At least one per channel is mandatory. |
| **TimeWindow** | User-selected temporal bound; one of `7d/30d/90d/1y/5y/custom`. Propagated through every layer. |
| **PAA** | People Also Ask — Google's question rabbit-hole. Only accessible via headless. |
| **ShortVideoLink** | Pydantic subclass of `DiscoveredLink` with duration, caption, hashtags, view/like counts, top comments, transcript. |
| **Verdict** | Triage output: `supports | refutes | tangential` per link per hypothesis. |
| **Decomposition** | L0 output: `{entities, pains, aspirations, identity_claims, geo_hints, signal_archetypes}` for a hypothesis. |
