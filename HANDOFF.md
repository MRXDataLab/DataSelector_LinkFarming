# Data Selection Module — Session Handoff

**Purpose:** Paste this as the first message of a fresh Claude Code session. It tells the new session exactly where the work stands, what's running, what's locked, and what comes next — so it can pick up at Step 4 without re-exploring.

**Date of handoff:** 2026-05-23
**Companion doc (planning context):** [`MRX_DataSelector/DATA_SELECTION_MODULE_CONTEXT.md`](./DATA_SELECTION_MODULE_CONTEXT.md) — read this if you need full background on what Outtlyr is and the upstream contract from the Hypothesis Engine.
**As-built doc (module README):** [`MRX_DataSelector/link_extraction/README.md`](./link_extraction/README.md) — engineering-level overview, design decisions, channel inventory.

---

## 1. Where everything lives

```
/Users/vdogg/Documents/mrxdatalabs_Station/
│
├── MRX_DataSelector/                            ← module dev (you are here)
│   ├── DATA_SELECTION_MODULE_CONTEXT.md         ← planning doc (background)
│   ├── HANDOFF.md                               ← this file
│   ├── .env.example
│   └── link_extraction/                         ← the Python package
│       ├── README.md
│       ├── __init__.py, models.py, temporal.py,
│       │   channel_weights.yaml, decomposer.py, source_selector.py,
│       │   _smoke_test.py
│       └── backends/
│           ├── base.py, brave.py, duckduckgo.py,
│           ├── headless_google.py, serpapi_stub.py,
│           └── registry.py
│
└── MRX_Module_1 Claude/                         ← host repo (FastAPI + Next.js)
    └── backend/
        ├── venv/                                ← Python venv (USE THIS)
        ├── .env                                 ← real API keys (gitignored)
        ├── requirements.txt                     ← module deps already added
        └── services/                            ← legacy link_farming.py
                                                   etc. lives here, untouched
```

**The venv lives in the host repo.** It already has spaCy + `en_core_web_sm`, pydantic 2.12, pytrends, requests, dotenv, PyYAML installed. Two deps are listed in `requirements.txt` but **not yet installed**: `duckduckgo-search` and `playwright`. Brave alone is enough for current functionality; install the others when DDG fallback or headless PAA scraping is needed (Step 6+).

---

## 2. Build status — 3 of 14 steps done

| # | Step | Status | Validation |
|---|---|---|---|
| 1 | Foundation: Pydantic models, temporal mapping, channel_weights.yaml, 4 search backends, fallback registry | ✅ | smoke test: 5 Indian Kellogg's URLs returned via live Brave; temporal translations correct across 7d/30d/90d/1y/5y; SerpAPI stub returns `[]` |
| 2 | L0 hypothesis decomposer (spaCy NER + curated lexicons → entities/pains/aspirations/identity/geo + signal archetype classification) | ✅ | h_006 → entities `['Corn Flakes', "Kellogg's Corn Flakes"]`, pains `['bland','decline','misaligned','mismatch']`, geo `['indian','india']`; Gen-Z stress case → identity `['gen-z','gen z']`, aspirations `['aspirational','self-care','wellness']` |
| 3 | L1 channel scorer (deterministic, YAML-driven, no LLM) | ✅ | h_006 (product/Demand Gravity) top 5: `marketplace(82) reddit(81) youtube(76) google_paa(70) youtube_shorts(69)`; identity_expression/Reinforcement Stability top 5 includes all three short-video channels; `news` correctly excluded by anti_platforms |
| **4** | **Trends discoverer** (pytrends → rising_queries; powers Step 5) | **🚧 NEXT** | should return live rising queries for "kelloggs" in India within selected window |
| 5 | L2 query synthesizer (9 archetypes + Trends amplifier + mandatory falsifier validation) | ⏳ | |
| **6** | 🎬 **YouTube Shorts discoverer** (priority #1 channel) | ⏳ | first end-to-end demo lands after Step 9 |
| 7 | L6 measured triage (long-form fetch + short-video caption/comments path) | ⏳ | |
| 8 | Orchestrator + SSE event stream | ⏳ | |
| 9 | API + minimal frontend (ShortVideoGrid for YT Shorts) | ⏳ | **Day-10 demo milestone** |
| 10 | Reddit + Google PAA + News discoverers | ⏳ | |
| 11 | TikTok + Instagram Reels discoverers | ⏳ | |
| 12 | Quora + YouTube long-form discoverers | ⏳ | |
| 13 | L5 dedup + cross-platform clustering | ⏳ | |
| 14 | Full frontend, source memory store, cost meter, legacy cleanup | ⏳ | |

---

## 3. How to run the smoke test (validates everything shipped)

```bash
cd /Users/vdogg/Documents/mrxdatalabs_Station/MRX_DataSelector
"../MRX_Module_1 Claude/backend/venv/bin/python" -m link_extraction._smoke_test
```

Exit code `0` if all assertions pass. Live Brave queries hit the network (~5 sec). The smoke test grows with each step:

- Steps 1+2+3 currently take ~10 seconds end-to-end
- `BRAVE_API_KEY` is read from `MRX_Module_1 Claude/backend/.env` (already populated)
- `DuckDuckGo` reports unavailable (package not installed yet — OK)
- `Headless Google` reports available because `playwright` package is importable, but Chromium binaries probably aren't installed yet. Smoke test doesn't exercise headless, so this is fine until Step 6.

---

## 4. The 7-layer pipeline

```
APPROVED HYPOTHESES + USER-SELECTED TimeWindow
   │
   ▼
[L0] Hypothesis Decomposition          ── ✅ Step 2 — pure-Python, no LLM
[L1] Source Selector                   ── ✅ Step 3 — deterministic YAML;
                                          LLM rationale enrichment deferred
[L2] Query Synthesizer                 ── Step 5 — 9 archetypes; Trends-amplified;
                                          hard-validates ≥1 falsifier per channel
[L3] Channel-Native Discovery          ── Steps 4/6/10/11/12 — one discoverer per channel
[L4] Temporal Filtering                ── propagated through L2/L3
[L5] Dedup + Cross-Platform Clustering ── Step 13
[L6] Measured Triage                   ── Step 7 — fetch + regex + LLM verdict
[L7] Group + Rank + Emit               ── Step 8/9 — SSE-streamed, grouped by hypothesis_id
```

---

## 5. Design decisions LOCKED (do not relitigate without explicit cause)

1. **Backend preference:** Brave → DuckDuckGo → Headless Google (in that order). SerpAPI is a placeholder stub.
2. **Headless concurrency = 3** (≈450 MB RAM cap); **CAPTCHA cooldown = 30s**.
3. **YouTube Shorts is the priority #1 channel** for the first end-to-end demo.
4. **Channel scoring is deterministic** (YAML weights). LLM only writes rationales (deferred).
5. **Channel-fit threshold = 50**; **top-N cap = 5**.
6. `TimeWindow` propagates **three places**: API filter at query time, Trends timeframe at amplifier time, client-side filter on `observed_at` at result time.
7. Every link carries `hypothesis_id`. Short-video links subclass `DiscoveredLink` with duration / caption / hashtags / view counts / top comments / transcript.
8. **9 query archetypes** (entity-pain, switching, comparison, aspiration, question, counter, expert, crisis, hashtag) parameterised by `{entity, geo, time, rising_phrase}`. Counter is mandatory.
9. **Sources serve hypotheses, not topics.** Discovery is hypothesis-scoped, never brand-scoped.

---

## 6. Decisions STILL OPEN (resolve when relevant step lands)

1. **LLM dispatch for L1 rationale generation.** Should the new stage `source_selection` be registered in `services/stage_models.PRESETS` (12 cells: 4 providers × 3 weights)? Decision needed before wiring the optional rationale enrichment. *Affects: end of Step 3 (or first task of Step 5 since Step 5 already needs an LLM).*
2. **Triage fetch budget.** Top 30 per hypothesis (good triage, slower & costlier) or top 15 (faster, may misrank borderlines)? Recommendation: 30. *Affects: Step 7.*
3. **MECE pair handling.** When a hypothesis has `contrarian_pair_id`, pool queries from both into one cluster or run them isolated? Recommendation: pooled. *Affects: Step 5 (query synth) + Step 11 (orchestrator).*
4. **Per-hypothesis cost cap.** Default $0.50? *Affects: Step 14 (cost meter).*
5. **Reels in v1 or punt to v1.1?** Lowest-yield short-video surface; headless pages are flaky. *Affects: Step 11.*

---

## 7. The data model (in [`link_extraction/models.py`](./link_extraction/models.py))

Key Pydantic types the rest of the pipeline produces and consumes:

- **`TimeWindow`** — `{start, end, label}`; `TimeWindow.from_label("90d")` is the standard constructor
- **`Decomposition`** (in `decomposer.py`) — L0 output; `{hypothesis_id, entities, primary_entity, competitor_anchors, pains, aspirations, identity_claims, geo_hints, signal_archetypes}`
- **`ChannelFit`** — L1 output per channel; `{channel, fit_score, rationale, expected_signal, sub_scores}`
- **`TypedQuery`** — L2 output; `{text, channel, archetype (1-9), archetype_name, target_signal, hypothesis_id, falsifier, geo_proxies}`
- **`RawResult`** — backend output, format-agnostic; `{url, title, snippet, backend, vertical, observed_at, raw_metadata}`
- **`DiscoveredLink`** — L3 output; carries `query`, `hypothesis_id`, `discovered_at`, `observed_at`, `backend_used`, `fit_score`, `supports_or_refutes`, `signal_tags`, `confidence`
- **`ShortVideoLink(DiscoveredLink)`** — adds `duration_sec`, `caption`, `hashtags`, `sound_id`, `creator`, `view_count`, `like_count`, `comment_count`, `share_count`, `thumbnail_url`, `top_comments`, `transcript`, `engagement_score`

---

## 8. The 9 query archetypes (build plan §C)

| # | Archetype | Long-form template | Short-video template |
|---|---|---|---|
| 1 | Entity-pain | `{brand} {pain_word}` | `{brand} cringe / fail` |
| 2 | Switching | `why I switched from {brand} to {alt}` | `pov: i stopped buying {brand}` |
| 3 | Comparison | `{brand} vs {alt} {dim_term}` | `{brand} vs {alt}` |
| 4 | Aspiration | `best {cat} for {aspiration}` | `{aspiration} {cat} routine` |
| 5 | Question | `why is {brand} {trend}` | rarely used |
| 6 | **Counter (MANDATORY)** | `why I love {brand}` | `defending {brand}` |
| 7 | Expert | `{cat} {expert_role}` | `nutritionist reacts {brand}` |
| 8 | Crisis | `{brand} {year}` | `{brand} scandal` |
| 9 | Hashtag/trend | n/a | `#{brand}` / `#{cat}{aspiration}` |

Slot values come from `Decomposition` (Step 2) plus the Trends amplifier (Step 4). The LLM call in Step 5 fills slots — it does NOT freewrite queries.

---

## 9. Backend chain (in [`link_extraction/backends/`](./link_extraction/backends))

```
search_with_fallback(query, vertical, count, window, min_results)
    │
    ├─ if vertical in {"paa", "related"} → HeadlessGoogleBackend (only path)
    │
    ├─ try BraveBackend           (primary; merge into accumulated)
    ├─ try DuckDuckGoBackend      (off if not installed; merge)
    └─ try HeadlessGoogleBackend  (final fallback; CAPTCHA-aware singleton)
```

Public function: `link_extraction.backends.search_with_fallback`. Per-backend singletons accessed via `get_brave()`, `get_ddg()`, `get_headless()`, `get_serpapi()`.

**Headless invariants:**
- Global `asyncio.Semaphore(3)` for concurrency cap
- On `"captcha"` or `"unusual traffic"` in page content → enter 30s cooldown; backend short-circuits to `[]` until cooldown expires
- PAA + Related verticals are headless-only (Brave/DDG don't expose them)

---

## 10. Env vars (loaded from `MRX_DataSelector/.env` or host's `backend/.env`)

| Variable | Default | Purpose |
|---|---|---|
| `BRAVE_API_KEY` | — | required; primary search tier (currently set ✅) |
| `USE_DUCKDUCKGO` | `1` | toggle DDG fallback (off if package missing) |
| `ENABLE_SERPAPI` | `0` | stub; flip to 1 and replace `serpapi_stub.py` to activate |
| `HEADLESS_CONCURRENCY` | `3` | parallel Chromium contexts (≈150 MB each) |
| `HEADLESS_CAPTCHA_COOLDOWN_SEC` | `30` | self-disable window on CAPTCHA |
| `YOUTUBE_API_KEY` | — | wired in Step 6 |
| `REDDIT_CLIENT_ID`/`_SECRET`/`_USER_AGENT` | — | wired in Step 10 |

⚠ **Security note:** During this session the user pasted `BRAVE_API_KEY`, `SERPER_API_KEY`, and `SERPAPI_KEY` values into the chat in plaintext on 2026-05-23. The user was told to rotate all three. **Confirm rotation status before the next session writes any code that hits paid APIs.**

---

## 11. Upstream contract (from the host's Hypothesis Engine)

The module consumes hypothesis dicts shaped like:

```python
{
  "hypothesis_id": "h_006",              # or "id"
  "statement": "...",
  "dimension": "product",                # one of 10 (see below)
  "force_assignment": "Demand Gravity",  # one of 5 (see below)
  "investigation_priority": "high",      # high | medium | low
  "expected_signals": [...],
  "expected_counter_signals": [...],
  "contrarian_pair_id": "h_007" | "",
  "rationale": "...",
  "core_problem_id": "cp_001",
  "core_problem_statement": "...",       # optional
  "mece_cluster_id": "...",
}
```

**10 dimensions:** `price, product, brand_perception, distribution, cultural_identity, regulatory, demographic_shift, competitive_shift, situational_context, identity_expression`

**5 forces:** `Demand Gravity, Choice Architecture Pressure, Value Elasticity Field, Reinforcement Stability, Competitive Energy Field`

**3 priorities:** `high, medium, low`

Sample data: there's a 32-hypothesis Kellogg's India CSV at `/Users/vdogg/Documents/mrxdatalabs_Station/Platform_Source_Selector/Hypothesis_Manifest_Indian_Government_is_pushing_Make_in_Ind.csv` from the reference Node/Express spike — use this for end-to-end validation runs.

---

## 12. First actions for the new session

1. **Read this file** (you're doing it).
2. **Read [`link_extraction/README.md`](./link_extraction/README.md)** — engineering-level overview.
3. **Run the smoke test** to confirm the dev environment works:
   ```bash
   cd /Users/vdogg/Documents/mrxdatalabs_Station/MRX_DataSelector
   "../MRX_Module_1 Claude/backend/venv/bin/python" -m link_extraction._smoke_test
   ```
   Expect: exit 0, "Smoke test complete (Steps 1 + 2 + 3)" at the end.
4. **Confirm with the user**: (a) any open decisions in §6 they want resolved before Step 4 starts; (b) whether they want Step 4 (Trends discoverer) next or want to skip ahead.
5. **Start Step 4** — see §13 below.

---

## 13. Step 4 spec — Trends Discoverer (≤0.5 day)

**Goal:** Build `link_extraction/trends_seed.py` that wraps pytrends to extract live rising queries for each hypothesis entity, scoped to the user's TimeWindow. This is the input to Step 5's query synthesizer.

**File to create:** `MRX_DataSelector/link_extraction/trends_seed.py`

**Functions:**

```python
def fetch_rising_queries(
    entity: str,
    window: TimeWindow,
    geo: str = "",  # ISO country code, e.g. "IN" — derived from Decomposition.geo_hints
    top_k: int = 5,
) -> List[str]:
    """Returns rising query phrasings from Google Trends for the given entity.
    Empty list on any pytrends error. Uses temporal.to_pytrends_timeframe(window)."""

def fetch_related_topics(entity: str, window: TimeWindow, geo: str = "") -> List[str]:
    """Adjacent topics — useful for L1 channel rationale enrichment later."""

def interest_over_time(entity: str, window: TimeWindow, geo: str = "") -> Dict[str, int]:
    """Daily/weekly interest series — surfaced in UI Trends panel (not used in L2)."""
```

**Implementation notes:**

- Use `pytrends.request.TrendReq(hl='en-US', tz=0)` — already installed in the host venv
- Geo conversion: `geo_hints=['india']` → `"IN"`; `['us', 'usa']` → `"US"`. Hardcode a small map; default empty string = global
- `pytrends.related_queries(...)` returns `{entity: {'rising': df, 'top': df}}` — guard against `None` or missing keys (pytrends throws on rate limit and on empty results)
- Wrap each call in try/except — return `[]` on any failure. pytrends is rate-limited (~10 req/min); use `time.sleep(1)` between calls in batch usage. Add `@lru_cache` on `(entity, window.label, geo)` so repeat lookups in one job don't hit Trends twice.

**Validation in smoke test:** Add `test_trends()` that calls `fetch_rising_queries("kelloggs", TimeWindow.from_label("90d"), geo="IN")` and asserts:
- Returns a list (not None)
- If non-empty, all entries are non-empty strings
- Either non-empty OR logs the rate-limit reason (Trends sometimes returns nothing for narrow queries)

**Then update:**
- `link_extraction/__init__.py` — export `fetch_rising_queries`, `fetch_related_topics`, `interest_over_time`
- `link_extraction/README.md` §7 — mark Step 4 ✅ when smoke test passes

---

## 14. Reference — full 14-step build sequence

See the current status in §2. Detailed step-by-step is in this conversation history; the high-level recap:

1. ✅ Foundation (models / temporal / weights / backends)
2. ✅ L0 decomposer
3. ✅ L1 channel scorer
4. **Trends discoverer** ← next
5. L2 query synthesizer (LLM call here — first LLM dispatch decision needed)
6. 🎬 YouTube Shorts discoverer (priority #1)
7. L6 measured triage
8. Orchestrator + SSE
9. API + minimal frontend (Day-10 demo)
10. Reddit + Google PAA + News
11. TikTok + Instagram Reels
12. Quora + YouTube long-form
13. L5 dedup + clustering
14. Polish: memory store, cost meter, legacy cleanup

---

**End of handoff.** New session: read this, run the smoke test, confirm §6 decisions you need, then start Step 4.
