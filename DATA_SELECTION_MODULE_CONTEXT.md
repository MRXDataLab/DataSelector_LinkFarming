# Data Selection Module — Session Context

**Purpose of this file:** Drop this into a fresh Claude Code session as the first message. It gives the session everything it needs to understand the host system (Outtlyr / IntentTerminal V2.1) and build the new **Data Selection Module** (a.k.a. Link Extraction v2) without having to re-explore the codebase.

**Date:** 2026-05-23
**Repository:** `https://github.com/MRXDataLab/IntentTerminal_V2.1`
**Local path:** `/Users/vdogg/Documents/mrxdatalabs_Station/MRX_Module_1 Claude/`

---

## 1. System overview

Outtlyr is a market research engine that converts a client brief into a structured research plan, then runs a hypothesis pipeline against it, then crawls the internet for evidence. The product positioning is "Stop Guessing, Start Knowing."

**Three macro-stages:**

```
INTAKE              →  HYPOTHESIS ENGINE          →  DATA SELECTION (this module)
(Teal agent)           (3-stage LLM pipeline)        (link extraction → triage)
ChatGPT-style          Decomposer / Generator /     Multi-channel discovery
conversation +         MECE Auditor +               with hypothesis-aware
brief upload           Ecosystem Graph              source selection
```

The Data Selection Module replaces the current `discovery_engine.py` + `manifest.py` link-farming logic with a hypothesis-driven, channel-aware extractor that gives the user temporal control.

---

## 2. Repository layout (what to know)

```
backend/
├── main.py                       # FastAPI app, mounts all routers
├── .env                          # API keys (see §8)
├── api/                          # Route handlers, one per concern
│   ├── intake.py                 # /api/chat — intake agent loop
│   ├── ecosystem.py              # ecosystem graph generation
│   ├── manifest.py               # ← current Link Farming Manifest (REPLACE)
│   ├── discovery.py              # ← current Discovery Job runner (REPLACE)
│   ├── hypotheses.py             # hypothesis pipeline entry
│   ├── llm_provider.py           # /api/llm-provider — intake provider switch
│   └── brief.py / brief_intake.py
├── services/
│   ├── intent_schema.py          # IntentState v2.0 (Pydantic) + coverage logic
│   ├── hypothesis_engine.py      # decomposer / generator / mece auditor
│   ├── llm_client.py             # call_openrouter() — central LLM dispatch
│   ├── stage_models.py           # 12-preset matrix (provider × weight)
│   ├── discovery_engine.py       # ← current 4-engine search (REPLACE)
│   ├── link_farming.py           # ← consolidated extraction of above (REPLACE)
│   ├── scout.py                  # Serper.dev web/news/shopping
│   ├── scout_subjective.py       # Reddit + YouTube + e-commerce reviews
│   ├── scout_structural.py       # Google Trends (pytrends), PAA tree, OOS sim
│   └── crawler.py                # SourceOracle — niche source discovery
└── kb/agents/                    # Prompt KB — markdown system prompts
    ├── intake_agent.md
    └── manifest_agent.md

frontend/                         # Next.js 14 + React 19 + Tailwind
├── src/components/
│   ├── LLMProviderGate.tsx       # single-page provider configuration
│   ├── StudyConfigPicker.tsx     # 12-preset matrix UI
│   └── ... (intake terminal, hypothesis viewer)

test_briefs/                      # 3 reference briefs that saturate all dimensions
```

**Conventions you'll see:**
- All LLM calls go through `services.llm_client.call_openrouter(system_prompt, user_prompt, expect_json, model, stage)`
- New pipeline stages are added to `stage_models.PRESETS` so each stage's model can be tuned per-preset
- KB prompts live in `backend/kb/agents/*.md` and are loaded by `kb_loader.load_kb()`
- Routers go in `backend/api/<feature>.py` and are included in `main.py`
- Pydantic models for state live in `services/intent_schema.py` and `services/hypothesis_engine.py`

---

## 3. Upstream contract — what Data Selection consumes

The module receives an **approved hypothesis set** from the hypothesis engine. Each hypothesis has this shape (from `services/hypothesis_engine.py`):

```python
{
  "id": "H_CP1_03",                          # stable ID — preserve through pipeline
  "statement": "Gen-Z is rejecting cornflakes because muesli signals self-care identity",
  "dimension": "identity_expression",        # one of 10 enum values (see below)
  "force_assignment": "Reinforcement Stability",  # one of 5 forces
  "expected_signals": [                      # what evidence would confirm/refute
    "Reddit threads comparing cornflakes to muesli as 'boring vs aspirational'",
    "YouTube comments expressing identity-based food rejection",
    "Quora answers framing breakfast as self-care"
  ],
  "contrarian_pair_id": "H_CP1_04" | null,   # MECE pair pointer
  "supporting_rationale": "...",
  "core_problem_id": "CP1"
}
```

**Dimension enum** (10 values, snake_case):
`price`, `product`, `brand_perception`, `distribution`, `cultural_identity`,
`regulatory`, `demographic_shift`, `competitive_shift`, `situational_context`, `identity_expression`

**Force enum** (5 values, title case from LLM):
`Demand Gravity`, `Choice Architecture Pressure`, `Value Elasticity Field`,
`Reinforcement Stability`, `Competitive Energy Field`

**Note on intake dimensions vs hypothesis dimensions:** The intake agent uses 8 dimensions (snake_case, see `kb/agents/intake_agent.md`). The hypothesis engine remaps to 10 dimensions. The data selection module consumes the **hypothesis 10-dimension** set.

---

## 4. Current state — what to replace

### 4.1 `api/manifest.py` (current link-farming manifest)
- Calls one LLM to generate a `boolean_nets / entity_anchors / signal_taxonomy` JSON
- Has a static `DIMENSION_PLATFORMS: Dict[str, List[str]]` lookup
- Translates hypotheses → search_nets but doesn't reason about *which channels fit which hypothesis*

### 4.2 `api/discovery.py` (current job runner)
- Threading-based job store (in-memory `_jobs` dict)
- Endpoints: `POST /discovery/start`, `GET /discovery/status/{id}`, `GET /discovery/results/{id}`, `GET /discovery/csv/{id}`
- LLM triage runs *after* scraping (wasteful)

### 4.3 `services/discovery_engine.py` (current 4 engines)
- `google_direct_search()` — Playwright headless
- `brave_search()` — Brave API
- `serpapi_search()` — SerpAPI
- `duckduckgo_search()` — DDG package
- Pipeline: seeds → 4 verticals × N seeds → PAA BFS → dedup → return
- `paa_depth=3`, Google PAA only

### 4.4 `services/link_farming.py` (consolidated extraction)
A single-module re-export of the above three files. Use this as the **before** reference when designing the new module.

### Weaknesses of current pipeline (all fixed in new design)
| # | Problem | Fix |
|---|---|---|
| 1 | Naive seeding from manifest fields | LLM Source Selector + Query Synthesizer |
| 2 | Same 4 verticals on every seed | Channel-fit filter (drop channels with fit_score<50) |
| 3 | No platform routing per hypothesis | Per-hypothesis channel ranking |
| 4 | No temporal control | `TimeWindow` object propagated through every stage |
| 5 | Triage after scrape | Pre-triage (hypothesis-aware) + post-triage scoring |
| 6 | Only Google has graph traversal (PAA) | Every channel exposes its native related-graph |
| 7 | Hypothesis ID dropped after manifest | Every `DiscoveredLink` carries `hypothesis_id` |

---

## 5. Target architecture — 5 stages

```
APPROVED HYPOTHESES
    │
    ▼
[1] SOURCE SELECTOR        ── LLM ranks channels per hypothesis
    │                         (fit_score, rationale, expected_signal)
    ▼
[2] QUERY SYNTHESIZER      ── Per (hypothesis × channel): 3–5 typed queries
    │                         using channel templates + Trends rising_queries seed
    ▼
[3] MULTI-CHANNEL          ── Parallel discoverers, each speaks one platform's
    DISCOVERY                  native graph (PAA, watch-next, related questions)
    │
    ▼
[4] TEMPORAL FILTERING     ── User-selected window applied at query + result time
    │
    ▼
[5] DEDUP + TRIAGE         ── Canonical-URL dedup; hypothesis-aware
    │                         supports/refutes/tangential classification
    ▼
RANKED LINK SET (grouped by hypothesis_id)
```

### Stage 1 — Source Selector

**Input:** one approved hypothesis
**Output:** ranked list of `ChannelFit` objects

```python
{
  "hypothesis_id": "H_CP1_03",
  "channels": [
    { "id": "reddit",     "fit_score": 92, "rationale": "...", "expected_signal": "user_complaint_text" },
    { "id": "youtube",    "fit_score": 88, ... },
    { "id": "google_paa", "fit_score": 80, ... },
    { "id": "quora",      "fit_score": 76, ... },
    { "id": "trends",     "fit_score": 70, ... },
    { "id": "news",       "fit_score": 40, ... }   # below 50 → dropped
  ]
}
```

One LLM call per hypothesis. Cached. Channels with `fit_score < 50` are skipped entirely.

### Stage 2 — Query Synthesizer

Per `(hypothesis × selected_channel)`, generate 3–5 typed queries.

**Key novelty — Trends-as-seed-amplifier:** Before LLM synthesis, call pytrends `related_queries(seed, timeframe=user_window)['rising']` to get the *live* phrasings consumers are using in the user-selected window. Inject those into the synthesis prompt. The LLM writes queries using actual consumer language, not marketer language.

**Falsifier-first rule:** every channel must include at least one query designed to surface *counter-evidence*. Without this, ranking becomes confirmation-biased.

### Stage 3 — Multi-channel Discovery

Each channel gets a `Discoverer(ABC)` implementation:

```python
class Discoverer(ABC):
    channel_id: str
    def discover(query: TypedQuery, window: TimeWindow) -> List[DiscoveredLink]
    def expand(seed: DiscoveredLink, depth: int) -> List[DiscoveredLink]  # graph BFS
```

Discoverers cap at:
- `max_links_per_query`: 10
- `max_graph_depth`: 2
- `max_total_links_per_channel`: 50

### Stage 4 — Temporal Filtering

Single `TimeWindow` object propagates through every stage:

```python
class TimeWindow:
    start: datetime
    end:   datetime
    label: Literal["7d","30d","90d","1y","5y","custom"]
```

Per-channel param translation lives in `services/link_extraction/temporal.py`.

Every `DiscoveredLink` carries `observed_at` so users can re-slice client-side without re-querying.

### Stage 5 — Dedup + Triage

- Canonical URL normalization (strip `utm_*`, sort query params)
- Content-hash dedup for reposted articles
- Hypothesis-aware LLM triage: `supports | refutes | tangential` (not just relevance)
- Results grouped by `hypothesis_id` so UI can show:
  ```
  Hypothesis H_CP1_03 — Gen-Z taste drift
    ├─ Supporting evidence (42 links)
    ├─ Refuting evidence  (11 links)
    └─ Tangential        (28 links)
  ```

---

## 6. Channel inventory

| Channel | Native related-graph | API/route | Wired today? | Notes |
|---|---|---|---|---|
| `google_web` | PAA recurse | SerpAPI `related_questions` | ✅ in `discovery_engine.py` | |
| `google_paa` | PAA BFS | Same as above, depth>1 | ✅ Google only | |
| `youtube` | `relatedToVideoId` + InnerTube suggestions | YouTube Data API v3 | ⚠️ Partial (`scout_subjective.py`) | Needs `YOUTUBE_API_KEY` |
| `quora` | Related Questions sidebar | SerpAPI `site:quora.com` + headless scrape | ❌ Not built | No official API |
| `reddit` | Subreddit discovery + crosspost graph | PRAW `subreddit.search()` | ⚠️ Partial (`scout_subjective.py`) | Has fallback to LLM sim |
| `trends` | `related_queries` rising/top | pytrends | ✅ `scout_structural.py` | Most valuable — drives Stage 2 |
| `news` | Article similarity | Brave news (✅) or GDELT (❌) | ⚠️ Brave only | GDELT better for >1y horizon |
| `marketplace` | Review date filter | SerpAPI shopping + Trustpilot scrape | ⚠️ Partial | Amazon blacklisted today (`DOMAIN_BLACKLIST`) |

**Action when building:** start with what's wired (`reddit`, `google_paa`, `trends`), then add Quora and YouTube.

---

## 7. Data model (proposed)

```python
# services/link_extraction/models.py

from datetime import datetime
from typing import Literal, List, Optional
from pydantic import BaseModel

ChannelId = Literal["google_web","google_paa","youtube","quora","reddit","trends","news","marketplace"]
QueryType = Literal["entity_pain","comparison","question","trend_seed","review_filter"]
WindowLabel = Literal["7d","30d","90d","1y","5y","custom"]
Verdict = Literal["supports","refutes","tangential"]

class TimeWindow(BaseModel):
    start: datetime
    end:   datetime
    label: WindowLabel

class TypedQuery(BaseModel):
    text:          str
    channel:       ChannelId
    query_type:    QueryType
    target_signal: str
    hypothesis_id: str
    falsifier:     bool          # designed to surface counter-evidence?

class ChannelFit(BaseModel):
    channel:         ChannelId
    fit_score:       int          # 0–100
    rationale:       str
    expected_signal: str

class DiscoveredLink(BaseModel):
    url:                 str
    title:               str
    snippet:             str
    channel:             ChannelId
    hypothesis_id:       str
    query:               TypedQuery
    discovered_at:       datetime
    observed_at:         Optional[datetime] = None
    fit_score:           Optional[int] = None
    supports_or_refutes: Optional[Verdict] = None
    signal_tags:         List[str] = []
```

---

## 8. Proposed file layout

```
backend/services/link_extraction/
├── __init__.py
├── models.py                  # types above
├── source_selector.py         # Stage 1 — LLM channel ranking
├── query_synthesizer.py       # Stage 2 — typed queries + Trends warmup
├── temporal.py                # TimeWindow → per-channel params
├── discoverers/
│   ├── __init__.py
│   ├── base.py                # Discoverer ABC
│   ├── google_web.py
│   ├── google_paa.py
│   ├── youtube.py             # Data API + InnerTube relatedToVideoId
│   ├── quora.py               # SerpAPI site:+ headless related-questions scrape
│   ├── reddit.py              # PRAW with subreddit discovery
│   ├── trends.py              # pytrends rising_queries + interest_over_time
│   ├── news.py                # Brave news + GDELT
│   └── marketplace.py         # Amazon/Trustpilot review date filtering
├── orchestrator.py            # Stage 3 — parallel discoverers per hypothesis
├── triage.py                  # Stage 5 — dedup + supports/refutes scoring
└── job_runner.py              # Per-hypothesis background jobs

backend/api/
└── link_extraction.py         # Endpoints (replaces api/discovery.py + manifest.py)

frontend/src/components/
└── data_selection/
    ├── HypothesisLinkExtractor.tsx     # main UI
    ├── SourceSelectorReview.tsx        # ranked channels + user override
    ├── TimeWindowPicker.tsx            # 7d/30d/90d/1y/5y/custom
    ├── QueryPreview.tsx                # show generated queries before run
    └── ResultsByHypothesis.tsx         # grouped by hyp_id + supports/refutes
```

---

## 9. API surface

```
POST /api/source-selector/run
  body: { hypothesis_ids: [str] }
  → { hypothesis_id: [ChannelFit] }

POST /api/query-synthesizer/run
  body: { hypothesis_id, channels: [ChannelId], time_window: TimeWindow }
  → { hypothesis_id: { channel: [TypedQuery] } }

POST /api/data-selection/start
  body: { hypothesis_id, channels: [ChannelId], time_window, queries: [TypedQuery] }
  → { job_id }

GET  /api/data-selection/status/{job_id}
GET  /api/data-selection/results/{job_id}?hypothesis_id=&verdict=
GET  /api/data-selection/csv/{job_id}
```

**One job per hypothesis.** UI shows per-hypothesis progress. Hypotheses run in parallel; queries within a hypothesis run sequentially per channel to respect rate limits.

---

## 10. Environment / API keys (from `backend/.env`)

```
OPENROUTER_API_KEY     # general LLM dispatch
GEMINI_API_KEY         # intake agent (Gemini Studio)
SERPER_API_KEY         # Serper.dev (used by scout.py)
BRAVE_API_KEY          # Brave Search API
SERPAPI_KEY            # SerpAPI (PAA, related searches)
DEEPSEEK_API_KEY       # DeepSeek reasoner (hybrid hypothesis preset)
VERTEX_AI_API_KEY      # GCP Gemini (default hypothesis provider)
HYPOTHESIS_PROVIDER=gcp_gemini
GCP_GEMINI_RPM=60
```

**Not yet wired (action items):**
- `YOUTUBE_API_KEY` — needed for YouTube discoverer
- `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` — PRAW currently falls back to LLM sim
- GDELT — no key needed, but no client built

---

## 11. Conventions to follow

| Concern | Pattern |
|---|---|
| LLM calls | Always through `call_openrouter(stage=...)` so the stage_models preset routing works |
| New pipeline stage | Add to `services/stage_models.PRESETS` for all 12 (provider × weight) combos |
| Pydantic models | Use `from __future__ import annotations` + `pydantic.BaseModel` |
| API responses | Always Pydantic response_model on FastAPI routes |
| Background jobs | In-memory dict keyed by short UUID (`uuid.uuid4().hex[:8]`), no Redis required |
| LLM prompts | Store as `.md` in `backend/kb/agents/<name>.md`, load via `kb_loader.load_kb()` |
| Frontend | Tailwind + Framer Motion; teal accent (`#14b8a6`); dark `#050505` background |
| Provider colors | gemini_studio=purple, gcp_gemini=blue, deepseek_api=cyan, hybrid=amber |
| Hypothesis ID format | `H_CP{n}_{seq:02d}` — preserve through every downstream artifact |

---

## 12. Open questions for the new session

These came up during planning. Resolve before/during implementation:

1. **Channel cap:** should Source Selector return top-N (e.g. 5) channels always, or all channels with `fit_score >= threshold`?
2. **User override:** should the user be able to manually toggle channels on/off after Source Selector runs, before discovery starts? (Recommend yes — prevents black-box feel.)
3. **Cost ceiling per hypothesis:** what's the max combined LLM + API spend allowed per hypothesis job? Today there's no cap.
4. **Quora extraction:** no official API. Use headless Chromium (slow but reliable) or accept SerpAPI's `site:quora.com` only (faster, misses related-questions sidebar)?
5. **YouTube auth:** YouTube Data API key not set yet. Confirm scope (search + comments + relatedToVideoId) before building.
6. **GDELT vs Brave for news:** GDELT is much better for >1y horizon. Worth building a new client, or stick with Brave?
7. **MECE pair handling:** when a hypothesis has `contrarian_pair_id`, should the discoverer also search for evidence supporting the pair (treating them as one cluster) or keep them isolated?
8. **Marketplace dedup:** Amazon is in `DOMAIN_BLACKLIST` today. Should marketplace discoverer override the blacklist for its own results?

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Outtlyr** | Brand name of the product. Sometimes spelled "Outlly" in older code — canonical is **Outtlyr**. |
| **IntentTerminal** | Codename / repo name (V2.1) |
| **Intake** | Stage 1 — Teal agent conversation that builds IntentState |
| **IntentState** | Pydantic model holding 8-dimension coverage map + decision_meta + research_intent |
| **Saturated / probed_thin / unprobed** | Dimension coverage statuses |
| **Hypothesis Engine** | Stage 2 — 3-stage LLM pipeline: Decomposer → Generator (dual-pass) → MECE Auditor + Ecosystem |
| **MECE** | Mutually Exclusive, Collectively Exhaustive — the audit rule for hypothesis sets |
| **Contrarian pair** | Two hypotheses framed as opposites; one of MECE's structural requirements |
| **Force assignment** | One of 5 strategic forces each hypothesis maps to |
| **Link Farming Manifest** | Current Artifact 3 — JSON of boolean nets + entity anchors. Being replaced. |
| **Data Selection** | New name for the link extraction stage (this module) |
| **Discoverer** | Channel-specific class that implements `discover(query, window)` + `expand(seed)` |
| **PAA** | People Also Ask — Google's question rabbit-hole |
| **Verdict** | Triage output: `supports | refutes | tangential` |
| **TimeWindow** | User-selected temporal bound propagated through the pipeline |

---

## 14. First actions for the new session

When the session opens, do these in order:

1. **Read** `backend/services/link_farming.py` — the consolidated current pipeline (single file).
2. **Read** `backend/services/intent_schema.py` and `backend/services/hypothesis_engine.py` — to understand the upstream contract.
3. **Read** `backend/services/stage_models.py` — to understand the preset matrix you'll need to register new stages in.
4. **Read** `backend/.env` — confirm available API keys.
5. **Confirm** the open questions in §12 with the user before writing code.
6. **Build in this order** (validated end-to-end at each step):
   - `models.py` + `temporal.py` (no LLM cost, foundation)
   - `discoverers/trends.py` (powers Stage 2 — must work first)
   - `source_selector.py` (one LLM call, easy to validate)
   - `query_synthesizer.py` (depends on trends + selector)
   - `discoverers/reddit.py` then `google_paa.py` (already-wired APIs)
   - `discoverers/youtube.py`, `quora.py`, `news.py`, `marketplace.py`
   - `orchestrator.py` + `job_runner.py`
   - `triage.py`
   - `api/link_extraction.py`
   - Frontend components

7. **Do NOT delete** `services/discovery_engine.py`, `services/link_farming.py`, `api/discovery.py`, `api/manifest.py` during build. Keep them parallel until the new module is validated end-to-end. They'll be removed in a separate cleanup PR.

---

## 15. Reference — single example end-to-end

To make this concrete, here's what should happen for one hypothesis:

**Approved hypothesis:**
```json
{
  "id": "H_CP1_03",
  "statement": "Kellogg's is losing Gen-Z because muesli/granola signals self-care identity that cornflakes cannot",
  "dimension": "identity_expression",
  "force_assignment": "Reinforcement Stability",
  "expected_signals": [
    "Reddit/Quora posts framing breakfast as self-care",
    "YouTube comments rejecting cornflakes as 'childish'",
    "Trends data showing muesli > cornflakes among 18–26 cohort"
  ]
}
```

**User selects:** TimeWindow = Past 12 months

**Stage 1 — Source Selector output:**
- reddit (92), youtube (88), quora (80), trends (76), google_paa (70), news (35-dropped)

**Stage 2 — Query Synthesizer output (reddit example):**
```json
[
  { "text": "kellogg's cornflakes boring", "query_type": "entity_pain", "falsifier": false },
  { "text": "why I switched from cornflakes to muesli", "query_type": "question", "falsifier": false },
  { "text": "cornflakes vs muesli health", "query_type": "comparison", "falsifier": false },
  { "text": "why I love cornflakes", "query_type": "entity_pain", "falsifier": true }
]
```

Trends warmup added rising query `"muesli benefits"` to the synthesis seed list.

**Stage 3 — Discovery (reddit only shown):**
- 4 queries × ~10 results = 40 raw → 18 unique after dedup
- Top thread "Why I stopped buying cornflakes" → `expand()` finds 3 crossposts → +3 links

**Stage 4 — Temporal filter:**
All results with `observed_at < 2025-05-23` dropped.

**Stage 5 — Triage:**
- 14 supports, 3 refutes, 4 tangential
- All carry `hypothesis_id: "H_CP1_03"`

**Job result returned to UI:** ranked, grouped, downloadable as CSV.

---

**End of context file.** Paste this as the first message of the new session and start with: *"Read this context file. Confirm understanding, list any open questions, then propose Day-1 deliverables."*
