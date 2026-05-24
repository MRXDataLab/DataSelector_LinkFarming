# Outtlyr Data Selection — Day-10 Demo

Standalone FastAPI app exposing the [`link_extraction`](../link_extraction/) pipeline (Steps 1–8) plus a single-page web UI for live demonstration.

---

## Run it

```bash
cd /Users/vdogg/Documents/mrxdatalabs_Station/MRX_DataSelector
"../MRX_Module_1 Claude/backend/venv/bin/uvicorn" api.app:app --port 8080
```

Then open **<http://localhost:8080>** — pick a sample hypothesis from the dropdown (h_006 cornflakes/India or h_test_genz wellness), pick a window, click **Run pipeline**. Live SSE event stream renders on the right; results grid appears at the bottom grouped into Supports / Refutes / Tangential.

---

## API surface

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/data-selection/start` | Kick off a pipeline job → returns `{job_id, status, ...}`. Body: `{hypothesis: {...}, window_label: "1y", use_llm: true, max_triage: 10}`. |
| `GET`  | `/api/data-selection/jobs` | List all in-memory jobs (summaries). |
| `GET`  | `/api/data-selection/jobs/{job_id}` | Job summary (status, verdict counts when done, elapsed). |
| `GET`  | `/api/data-selection/jobs/{job_id}/events` | SSE stream — replays history then streams live `PipelineEvent`s until `pipeline_complete` / `pipeline_error` / `stream_end`. |
| `GET`  | `/api/data-selection/jobs/{job_id}/results?wait=true` | Grouped triaged links (supports/refutes/tangential) with full `ShortVideoLink` payload. `wait=true` blocks until done. Add `&download=1` to get `Content-Disposition: attachment` for browser save. |
| `GET`  | `/api/data-selection/jobs/{job_id}/results.csv?wait=true` | Flat 28-column CSV (UTF-8 with BOM, Excel-friendly), one row per triaged link. Pipe-delimited inside list cells (hashtags, top_comments, signal_tags). `transcript_preview` capped at 500 chars. |
| `GET`  | `/api/data-selection/registry` | Which channel discoverers are currently registered & available. |
| `POST` | `/api/data-selection/batch/preview` | Multipart CSV upload → parsed preview `{hypothesis_count, core_problems[], detected_pairs[], errors[]}`. Does NOT start jobs. |
| `POST` | `/api/data-selection/batch/start` | JSON body `{csv_text, window_label, use_llm, max_triage, concurrency}` → `{batch_id, member_count, ...}` and kicks off bounded-parallel pipeline runs. |
| `GET`  | `/api/data-selection/batches` | List all in-memory batches (summary view). |
| `GET`  | `/api/data-selection/batch/{batch_id}` | Batch summary + member list with per-hypothesis status, verdict counts, elapsed. |
| `GET`  | `/api/data-selection/batch/{batch_id}/events` | SSE — merged stream of batch-level events + every member's pipeline events (tagged with `hypothesis_id`). |
| `GET`  | `/api/data-selection/batch/{batch_id}/results.csv?wait=true` | Aggregated 30-column CSV across every hypothesis. Prepends 2 batch columns (`core_problem_id`, `core_problem_statement`) to the standard 28 link columns. |
| `GET`  | `/api/data-selection/batch/{batch_id}/results.json?wait=true&download=1` | Full nested JSON: `core_problem` → `hypothesis` → `grouped` (supports/refutes/tangential link arrays). |

## SSE event kinds

Every event is named-event SSE (so the frontend can `addEventListener(kind, fn)`):

- `pipeline_start` — first event; carries `window_label`, `use_llm`, `available_channels`
- `stage_start` / `stage_done` — emitted once per stage (`L0_decompose`, `L1_source_select`, `L2_query_synth`, `L3_discover`, `L6_triage`)
- `channel_discovered` — per channel; carries `n_links`, `n_queries_run`, `skipped`, `reason`
- `pipeline_complete` — terminal success; carries `elapsed_sec`, `verdict_counts`
- `pipeline_error` — terminal failure; carries `error`
- `stream_end` — server bookend telling the client to close the EventSource (browsers otherwise auto-reconnect on completed jobs)

## Curl examples

```bash
# Start a job
curl -s http://localhost:8080/api/data-selection/start \
  -H 'Content-Type: application/json' \
  -d '{"hypothesis":{"hypothesis_id":"h_006","statement":"Corn Flakes is bland and misaligned with Indian taste preferences","expected_signals":["taste_complaints"],"expected_counter_signals":["taste_satisfaction"]},"window_label":"1y","max_triage":5}'

# Tail events (will stream until pipeline_complete then stream_end)
curl -N http://localhost:8080/api/data-selection/jobs/$JOB_ID/events

# Get results once done
curl -s http://localhost:8080/api/data-selection/jobs/$JOB_ID/results?wait=true | jq .verdict_counts

# Download CSV (28 columns, UTF-8 with BOM, opens cleanly in Excel/Sheets)
curl -O -J "http://localhost:8080/api/data-selection/jobs/$JOB_ID/results.csv?wait=true"

# Download JSON (full payload with decomposition + channel_fits + queries)
curl -O -J "http://localhost:8080/api/data-selection/jobs/$JOB_ID/results?wait=true&download=1"
```

## CSV columns (29)

```
hypothesis_id, verdict, confidence, channel, is_short_video,
url, canonical_url, title, snippet,
creator, duration_sec, view_count, like_count, comment_count, share_count,
engagement_score, hashtags, top_comments, transcript_preview,
signal_tags, also_found_on,
query_text, archetype, archetype_name, falsifier, target_signal,
backend_used, discovered_at, observed_at
```

`also_found_on` (Step 13) lists the other channels that L5 dedup found
discovering the same canonical URL / content hash. Pipe-delimited inside
the cell (e.g. `tiktok | reddit`). Empty when the link was discovered on
exactly one channel — the common case for narrow queries.

The batch CSV (`/batch/{id}/results.csv`) prepends two more columns —
`core_problem_id` and `core_problem_statement` — for a total of 31.

## Batch hypothesis upload — CSV input format

| Column | Required? | Notes |
|---|---|---|
| `statement` | **required** | the hypothesis sentence |
| `hypothesis_id` | optional | auto-generated as `h_auto_001`, `h_auto_002`, … if missing |
| `core_problem_id` | optional | groups rows; `_uncategorized` if absent |
| `core_problem_statement` | optional | first non-empty value per CP wins |
| `dimension` | optional | one of the 10 hypothesis dimensions |
| `force_assignment` | optional | one of 5 force names |
| `investigation_priority` | optional | `high` \| `medium` \| `low` |
| `expected_signals` | optional | **pipe-delimited inside the cell** (e.g. `taste_complaints\|preference_for_alternatives`) |
| `expected_counter_signals` | optional | same pipe format |
| `contrarian_pair_id` | optional | references another `hypothesis_id` — when both sides of a pair appear in the same CSV, the batch reports them in `detected_pairs` |
| `rationale` | optional | |
| `window_label` | optional | per-row override of the batch default (`7d`/`30d`/`90d`/`1y`/`5y`) |
| `max_triage` | optional | per-row override of the batch default |

Unknown columns are silently ignored. Empty rows are skipped. Errors are
reported per-row in `/batch/preview`'s `errors` field; the analyst can
fix the CSV and re-upload before committing via `/batch/start`.

A working example: [`static/sample_batch.csv`](static/sample_batch.csv) — 5 hypotheses
across 2 core problems with one MECE pair (`h_201` ↔ `h_202`) and one row
exercising the auto-generated `h_auto_001`.

### Batch behaviour decisions (locked)

- **Concurrency** defaults to **3** parallel pipelines per batch — capped via `asyncio.Semaphore`. YT API quota math: 3 × ~106 units in flight is comfortably under the 10K daily limit.
- **Errored hypotheses** are skipped-and-continued; a batch with any errored member finishes as `partial` (not `error`) so successful members are still exported.
- **MECE pair detection** is reported in `/batch/preview` and `/batch/{id}` (`detected_pairs` field). v1 still runs each row as a separate pipeline; v1.1 will swap in `synthesize_pair_pooled()` so both members of a pair share a discovery pool with verdicts attributed individually.
- **Per-hypothesis cost cap** is deferred to Step 14 (cost meter) — for now the only ceiling is the batch concurrency × max_triage product.

### Curl examples — batch path

```bash
# 1. Preview (does NOT start jobs)
curl -s -X POST http://localhost:8080/api/data-selection/batch/preview \
  -F "file=@api/static/sample_batch.csv" | jq

# 2. Start
CSV=$(python3 -c "import json; print(json.dumps(open('api/static/sample_batch.csv').read()))")
BATCH=$(curl -s -X POST http://localhost:8080/api/data-selection/batch/start \
  -H 'Content-Type: application/json' \
  -d "{\"csv_text\":$CSV,\"window_label\":\"1y\",\"max_triage\":5,\"concurrency\":3}" \
  | jq -r .batch_id)
echo "batch_id=$BATCH"

# 3. Watch SSE stream until batch_complete
curl -N http://localhost:8080/api/data-selection/batch/$BATCH/events

# 4. Download aggregated CSV
curl -O -J "http://localhost:8080/api/data-selection/batch/$BATCH/results.csv?wait=true"
```

---

## Currently-registered discoverers (default_registry)

Only **YouTube Shorts** ships in the Day-10 demo (priority #1 channel). All other channels (Reddit, Quora, Google PAA, News, TikTok, etc.) are produced by Step 5's query synthesizer but skipped during discovery with `channel_discovered { skipped: true, reason: "no_discoverer" }`. Steps 10–12 will fill in the remaining discoverers.

---

## Demo flow (h_006, 1y window, max_triage=10, use_llm=true)

Expected timing: ~60–90 seconds end-to-end.

1. `pipeline_start` — `available_channels=["youtube_shorts"]`
2. `L0_decompose` start/done — `primary_entity="Corn Flakes"`, pains=[bland, misaligned, decline], geo_hints=[india, indian]
3. `L1_source_select` start/done — top 5 channels with fit_scores (marketplace 82, reddit 81, youtube 76, google_paa 70, youtube_shorts 69)
4. `L2_query_synth` start/done — ~40 queries across 5 channels (8 per channel)
5. `L3_discover` start, then 5x `channel_discovered`: only `youtube_shorts` runs (~40s for 9 queries × YT API), the rest get `skipped: true`. Then `L3_discover` done.
6. `L6_triage` start/done — top 10 of ~49 discovered Shorts go to Gemini batch verdict
7. `pipeline_complete` — `verdict_counts={supports: 2, refutes: 0, tangential: 8}`, `elapsed_sec≈75`

Headline supports for h_006:
- **Bagrry Cornflakes** competitor Short @ confidence 0.90 (preference_for_alternatives)
- Hindi **"कॉर्न फ्लैक्स चिवड़ा"** — Indian consumers turning cornflakes into traditional savory poha (savory_preference, traditional_food_integration)

---

## Env vars

The app loads `.env` from `MRX_DataSelector/.env` first, then `MRX_Module_1 Claude/backend/.env`. Required:

- `YOUTUBE_API_KEY` (Step 6 — YT Data API v3)
- `BRAVE_API_KEY` (Step 1 — fallback chain primary)
- `VERTEX_AI_API_KEY` or `GEMINI_API_KEY` (Step 5 + Step 7 — Gemini for slot-fill & triage)
- `HYPOTHESIS_PROVIDER=gcp_gemini` (optional — controls Vertex-vs-Studio preference)

Optional:
- `USE_DUCKDUCKGO=1` (Step 1 — enables DDG fallback once `duckduckgo-search` is installed)
- `HEADLESS_CONCURRENCY=3` (Step 1 — Playwright Chromium cap)
- `YOUTUBE_API_KEY` quota is ~10K units/day; each `discover()` costs ~106 units. Demo run = ~1K units.
