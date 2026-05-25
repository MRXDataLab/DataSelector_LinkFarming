# Outtlyr — Data Selection Module

Hypothesis-driven multi-channel link discovery and triage. Takes one or more
research hypotheses + a time window, hunts the open web for evidence across
9 channels, and returns a ranked **supports / refutes / tangential** grid
with full provenance — verdicts, signal tags, engagement metadata, cross-platform
clustering, and per-hypothesis cost accounting.

Replaces the host repo's [`services/link_farming.py`](https://github.com/MRXDataLab/IntentTerminal_V2.1)
manifest stack with a hypothesis-aware pipeline. Integration target is
[`MRXDataLab/IntentTerminal_V2.1`](https://github.com/MRXDataLab/IntentTerminal_V2.1) —
see [`INTEGRATION.md`](INTEGRATION.md).

---

## Quick start

```bash
git clone https://github.com/MRXDataLab/DataSelector_LinkFarming.git
cd DataSelector_LinkFarming

# 1. Set up env (point at the host repo's venv if you have one nearby,
#    or build a fresh one with the deps in INTEGRATION.md)
cp .env.example .env
# fill in BRAVE_API_KEY, VERTEX_AI_API_KEY, YOUTUBE_API_KEY
# (the module is Vertex-only — AI Studio key is NOT used)

# 2. Run the smoke test against the full pipeline
python -m link_extraction._smoke_test

# 3. Start the standalone API + single-page UI
uvicorn api.app:app --port 8080
open http://localhost:8080/
```

---

## What it does (the 7-layer pipeline)

```
HYPOTHESIS + TIME WINDOW
       ▼
L0  Decompose         pure-Python — entities / pains / aspirations / geo
       ▼
L1  Source Selector   deterministic — top-5 of 13 channels by (dim × force) grid
       ▼
L2  Query Synthesizer 9 archetypes × Trends amplifier × mandatory counter query
       ▼
L3  Discovery         9 parallel discoverers w/ native related-graph traversal
       ▼
L4  Temporal filter   API-time + client-side fallback for non-time-aware channels
       ▼
L5  Dedup + Clustering canonical URL + content-hash → cross-platform clusters
       ▼
L6  Triage            top-30 → Gemini batch verdict + confidence + signal_tags
       ▼
L7  Emit              SSE-streamed, grouped, CSV/JSON exports
```

Full walkthrough in plain English: [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md).

---

## Channels (9 registered)

| Channel | Discoverer | Status |
|---|---|---|
| `youtube_shorts` ⭐ | YT Data API v3 + transcripts | ✅ priority #1, hero channel |
| `tiktok` | Brave `site:tiktok.com` (video + discover + tag URLs) | ✅ |
| `youtube` | YT Data API v3 long-form (≥182s) | ✅ |
| `reddit` | PRAW (if creds) or Brave fallback | ✅ |
| `quora` | Brave `site:quora.com` | ✅ |
| `google_paa` | Headless Chromium | ✅ |
| `news` | Brave news | ✅ |
| `substack` | Brave `site:substack.com` + essay path filter | ✅ |
| `marketplace` | Brave w/ Trustpilot/Amazon/Flipkart host filter | ✅ |
| `instagram_reels` | — | ⏳ deferred to v1.1 |

---

## Features

- **Live SSE event stream** — pipeline progress per stage rendered live in the UI
- **CSV / JSON download** — 29-column flat CSV (Excel-friendly UTF-8 BOM) or full nested JSON
- **Batch CSV upload** — run many hypotheses across multiple core problems from one CSV; bounded-parallel concurrency
- **Memory store** — atomic JSON snapshots in `.memory/`; jobs survive server restarts; past runs browsable from the UI
- **Cost meter** — per-job Gemini $ + YT API quota + Brave call count; default $0.50 hard cap
- **MECE pair detection** — when contrarian pairs appear in the same batch, both sides are flagged
- **Cross-platform clustering** — `also_found_on` field tracks when the same URL is discovered via N channels

---

## API surface

| Method | Path |
|---|---|
| `POST` | `/api/data-selection/start` |
| `GET` | `/api/data-selection/jobs/{id}` |
| `GET` | `/api/data-selection/jobs/{id}/events` (SSE) |
| `GET` | `/api/data-selection/jobs/{id}/results[?wait=true][&download=1]` |
| `GET` | `/api/data-selection/jobs/{id}/results.csv` |
| `POST` | `/api/data-selection/batch/preview` (multipart CSV) |
| `POST` | `/api/data-selection/batch/start` |
| `GET` | `/api/data-selection/batch/{id}` |
| `GET` | `/api/data-selection/batch/{id}/events` (SSE) |
| `GET` | `/api/data-selection/batch/{id}/results.csv` |
| `GET` | `/api/data-selection/batch/{id}/results.json` |
| `GET` | `/api/data-selection/memory/jobs[?hypothesis_id=...]` |
| `GET` | `/api/data-selection/memory/batches` |
| `GET` | `/api/data-selection/registry` |

Full curl examples + CSV schema: [`api/README.md`](api/README.md).
OpenAPI / Swagger UI: `http://localhost:8080/docs` once running.

---

## Documentation map

| File | What it covers |
|---|---|
| [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) | Plain-English explanation of the pipeline — for non-engineers |
| [`HANDOFF.md`](HANDOFF.md) | Session-handoff context — 14-step build status, locked decisions, channel inventory |
| [`DATA_SELECTION_MODULE_CONTEXT.md`](DATA_SELECTION_MODULE_CONTEXT.md) | Original planning doc — host repo background, upstream contract from Hypothesis Engine |
| [`INTEGRATION.md`](INTEGRATION.md) | Migration plan for folding into `IntentTerminal_V2.1` |
| [`link_extraction/README.md`](link_extraction/README.md) | Engineering-level overview, design decisions, channel weights |
| [`api/README.md`](api/README.md) | API quickstart, endpoint reference, CSV column spec, curl examples |

---

## Status

✅ **All 14 build steps complete.** Module is feature-complete; integration into the host repo is the next move (see [`INTEGRATION.md`](INTEGRATION.md)).

Run `python -m link_extraction._smoke_test` to verify all layers end-to-end against live APIs.
