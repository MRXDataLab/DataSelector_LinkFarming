# Integration Plan — folding MRX_DataSelector into the host repo

This module currently runs as a **standalone FastAPI app** in
`MRX_DataSelector/`. Production integration means folding it into the
host repo at `MRX_Module_1 Claude/backend/` so it lives alongside the
Hypothesis Engine and Intake agents.

This document is the migration checklist. It assumes you've already
validated the module end-to-end against the standalone demo (the
Day-10 acceptance run on h_006 / h_test_genz).

---

## 1. Pre-flight checklist (do these before any code moves)

- [ ] **Smoke test passes** in the standalone module:
      `"../MRX_Module_1 Claude/backend/venv/bin/python" -m link_extraction._smoke_test`
- [ ] **API demo runs**: `uvicorn api.app:app --port 8080` and h_006
      completes inside its $0.50 cost cap
- [ ] **API keys rotated** (if not already): `BRAVE_API_KEY`,
      `SERPER_API_KEY`, `SERPAPI_KEY`, `GEMINI_API_KEY`,
      `VERTEX_AI_API_KEY` — see [`memory/feedback_security.md`](../.claude/projects/-Users-vdogg-Documents-mrxdatalabs-Station-MRX-DataSelector/memory/feedback_security.md)
- [ ] **Host repo is on a clean branch** (e.g. `integration/data-selection-v2`)
- [ ] **`.env` review**: confirm host's `backend/.env` carries
      `YOUTUBE_API_KEY` + `HYPOTHESIS_PROVIDER=gcp_gemini` (the module
      reads from there)

---

## 2. File-by-file moves

### 2.1 Python module

Move the entire `link_extraction/` directory into the host repo as a
sibling of `services/`:

| Source (this repo) | Destination (host repo) |
|---|---|
| `MRX_DataSelector/link_extraction/` | `MRX_Module_1 Claude/backend/services/link_extraction/` |

All internal imports (`from .decomposer import …`) stay unchanged. The
only external dependencies are stdlib + pydantic + requests + spacy +
pytrends + youtube-transcript-api — every one of those is already in
the host venv (see `backend/requirements.txt`).

### 2.2 FastAPI router

The standalone `api/app.py` becomes a router mounted on the host's
`backend/main.py`:

| Source | Destination | Action |
|---|---|---|
| `MRX_DataSelector/api/app.py` | `backend/api/link_extraction.py` | Strip the `FastAPI(...)` app + CORS + static mount; keep only the route handlers. Wrap them in `router = APIRouter(prefix="/api/data-selection", tags=["data-selection"])`. |
| `MRX_DataSelector/api/static/index.html` | Two options — see §3 |
| `MRX_DataSelector/api/static/sample_batch.csv` | `backend/api/static/data_selection/sample_batch.csv` |
| `MRX_DataSelector/api/static/sample_hypotheses.json` | `backend/api/static/data_selection/sample_hypotheses.json` |

Then in `backend/main.py`:

```python
from api.link_extraction import router as data_selection_router
app.include_router(data_selection_router)

# On-startup memory restore (already wraps inside the router; just confirm
# the @app.on_event("startup") hook fires via FastAPI's normal lifecycle).
```

### 2.3 Legacy files — what to delete + replace

Per the handoff §14 + LOCKED rule "keep parallel until validated":

| Legacy file | Action |
|---|---|
| `backend/services/discovery_engine.py` | **Delete** after parity smoke test |
| `backend/services/link_farming.py` | **Delete** after parity smoke test |
| `backend/api/discovery.py` | **Delete** — superseded by `api/link_extraction.py` |
| `backend/api/manifest.py` | **Delete** — superseded by L0 decomposer + L1 source selector |
| `backend/services/scout.py` | **Keep** — used by Hypothesis Engine, not by us |
| `backend/services/scout_subjective.py` | **Keep** — same |
| `backend/services/scout_structural.py` | **Keep** — same |
| `backend/services/crawler.py` | **Keep** — Source Oracle (different feature) |
| `backend/main.py` router mounts for `discovery`/`manifest` | **Remove** the two `include_router(...)` lines |
| Frontend Next.js components `LinkFarmingManifest*`, `DiscoveryJob*` | **Delete** along with `pages/manifest`, `pages/discovery` |

**Migration safety: do these in two PRs.** PR #1 adds the new module
+ router with all legacy files still parallel. PR #2 deletes the
legacy code after a week of dual-running.

### 2.4 LLM dispatch — `services/llm_client.py` vs in-module `_llm.py`

The module ships its own minimal Gemini shim ([`link_extraction/_llm.py`](link_extraction/_llm.py))
because the host's `services/llm_client.py` is OpenRouter-based, and a
locked feedback note ([`feedback_llm_provider.md`](../.claude/projects/-Users-vdogg-Documents-mrxdatalabs-Station-MRX-DataSelector/memory/feedback_llm_provider.md))
says **this module uses Gemini natively, never OpenRouter**.

Three options at integration time:

1. **Keep `_llm.py` as-is.** Simplest. The module retains its own Gemini
   client; the host's `llm_client.py` still serves other stages
   (Hypothesis Engine, Intake) via OpenRouter. Recommended for v1.
2. **Add Gemini support to `services/llm_client.py`**, then swap
   `_llm.call_llm()` calls for `services.llm_client.call_gemini()`.
   Cleaner long-term but requires touching the central LLM client.
3. **Register a new stage in `services/stage_models.PRESETS`** (the
   12-cell provider × weight matrix). Per the handoff's open decision §1,
   register `query_synthesis` and `triage` as stages. Then the module's
   LLM calls go through `call_openrouter(stage=…)` and respect per-preset
   provider routing.

**Recommend Option 1 for the first integration PR.** Revisit Option 3
once you've validated parity with the standalone demo for ≥ 1 week.

### 2.5 Hypothesis Engine handoff contract

The host's `services/hypothesis_engine.py` currently produces hypothesis
dicts with `id` instead of `hypothesis_id`, and the dimension/force values
as title case. The module's L0 decomposer accepts both — see
[`decomposer.py:decompose()`](link_extraction/decomposer.py) — but
double-check that **no downstream code relies on `hypothesis_id` being
absent**.

Add a thin adapter if needed:

```python
# backend/services/link_extraction_adapter.py
from typing import Any, Dict, List
from services.hypothesis_engine import Hypothesis
from services.link_extraction import create_job, TimeWindow

def kick_off_for_hypothesis(h: Hypothesis, window: TimeWindow) -> str:
    return create_job(
        hypothesis={
            "hypothesis_id": h.id,
            "statement": h.statement,
            "dimension": h.dimension,
            "force_assignment": h.force_assignment,
            "investigation_priority": h.investigation_priority,
            "expected_signals": h.expected_signals,
            "expected_counter_signals": h.expected_counter_signals,
            "contrarian_pair_id": h.contrarian_pair_id,
            "rationale": h.supporting_rationale,
            "core_problem_id": h.core_problem_id,
        },
        window=window,
    )
```

### 2.6 Frontend mount

Two options:

**A. Keep the single-page HTML demo** (recommended for v1):
- Move `MRX_DataSelector/api/static/*` into `backend/api/static/data_selection/`
- Serve at `/data-selection/` via FastAPI's existing `StaticFiles` mount
- Link to it from the host's nav as **"Data Selection (legacy UI)"** until the Next.js rebuild lands

**B. Rebuild as Next.js components** (host-repo native):
- New components under `frontend/src/components/data_selection/`:
  - `HypothesisLinkExtractor.tsx` — main page
  - `SourceSelectorReview.tsx` — channel fits + manual override
  - `TimeWindowPicker.tsx` — 7d/30d/90d/1y/5y selector
  - `QueryPreview.tsx` — L2 output
  - `BatchUploader.tsx` — CSV upload + preview
  - `BatchProgressTable.tsx` — per-hyp status
  - `ShortVideoGrid.tsx` — verdict-bucketed card grid (the demo's hero view)
  - `MemoryBrowser.tsx` — past runs
  - `CostMeter.tsx` — per-job spend tracker
- API client in `frontend/src/lib/data-selection.ts` (uses the same
  REST + SSE endpoints — no contract change)

---

## 3. Smoke-test the integrated system

After the merge, the parity test:

```bash
cd "MRX_Module_1 Claude/backend"
source venv/bin/activate

# 1. Run the module's smoke test against the integrated layout
python -m services.link_extraction._smoke_test

# 2. Start the host server
uvicorn main:app --port 8080

# 3. Hit the integrated endpoint (same JSON contract as standalone)
curl -s -X POST http://localhost:8080/api/data-selection/start \
  -H 'Content-Type: application/json' \
  -d @../MRX_DataSelector/api/static/sample_h006.json

# 4. Confirm SSE stream
curl -N http://localhost:8080/api/data-selection/jobs/$JOB_ID/events
```

Verify:
- [ ] All 9 channels register in `default_registry()`
- [ ] h_006 produces ≥ 2 SUPPORTS verdicts
- [ ] L5 dedup events emit
- [ ] `cost.total_usd` under $0.50 cap
- [ ] Memory store persists job snapshot to `.memory/jobs/`
- [ ] After server restart, `/memory/jobs` lists the past run

---

## 4. Configuration changes

`backend/.env` should already have everything, but verify:

| Var | Required? | Owner before/after integration |
|---|---|---|
| `YOUTUBE_API_KEY` | Yes (Step 6) | new — owned by this module |
| `BRAVE_API_KEY` | Yes (Step 1) | already present |
| `GEMINI_API_KEY` | Yes (Step 5+7) | already present |
| `VERTEX_AI_API_KEY` | Yes (Step 5+7) | already present |
| `HYPOTHESIS_PROVIDER` | `gcp_gemini` | already present |
| `REDDIT_CLIENT_ID` / `_SECRET` | Optional — enables PRAW rich mode for Reddit discoverer | new |
| `OUTTLYR_COST_CAP_USD` | Default `0.50` | new — Step 14 cost meter cap |
| `OUTTLYR_MEMORY_DIR` | Default `<repo>/.memory` | new — Step 14 snapshot location |

---

## 5. Rollback plan

If integration breaks in production:

1. **Don't delete legacy code** in PR #1 — they stay parallel
2. Revert just `backend/main.py`'s router mounts:
   ```python
   # Comment out the new mount, re-enable the old ones
   # app.include_router(data_selection_router)
   app.include_router(discovery_router)
   app.include_router(manifest_router)
   ```
3. The Next.js front-end auto-detects which routers are live via the
   discovery probe on app load — no rebuild needed.

---

## 6. Acceptance criteria

PR #1 is mergeable when:

- [ ] Module's smoke test passes against the integrated layout
- [ ] `/api/data-selection/*` endpoints round-trip the h_006 happy path
- [ ] SSE event stream works through the host's middleware
- [ ] `default_registry()` reports all 9 channels available
- [ ] Cost meter tracks LLM + YT usage; demo run under $0.50
- [ ] Memory store survives server restart
- [ ] Existing host endpoints (`/api/chat`, `/api/hypotheses`, etc.) still respond — no regression

PR #2 (legacy cleanup) merges after:

- [ ] ≥ 1 week of dual-running in staging
- [ ] No analyst complaints about parity gaps
- [ ] Cost report for the week shows the new module is cheaper end-to-end
