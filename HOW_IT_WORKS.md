# How Outtlyr Data Selection Works

A plain-English walkthrough of what this app does, why each step exists, and what you get out the other end.

---

## The problem it solves

You have a hypothesis about a brand, product, or market. For example:

> *"Kellogg's Corn Flakes is losing ground in India because Indian consumers perceive it as bland and increasingly prefer traditional, savory breakfast options."*

You need to know: **is that actually true?**

Verifying it manually means:

- Hunting YouTube, TikTok, Reddit, Quora, news sites, and review sites yourself
- Writing dozens of search-query variations to avoid only finding what you already believed
- Reading 50+ web pages and 30+ short videos
- Deciding for each one whether it supports, contradicts, or is unrelated to the claim
- Doing it all again tomorrow for the next hypothesis

That's hours per hypothesis. Multiply by 30 in a typical project — weeks of analyst time.

**This tool does that hunting and reading for you in ~1-2 minutes per hypothesis.**

---

## The pipeline at a glance

```
HYPOTHESIS + TIME WINDOW
       │
       ▼
L0  Decompose         ── pull brand / geo / pains / aspirations from the sentence
       │
       ▼
L1  Source Selector   ── pick the top-5 most-relevant platforms (out of 13)
       │
       ▼
L2  Query Synthesizer ── generate 5-8 queries per channel across 9 search "angles"
       │
       ▼
L3  Discovery         ── search every chosen platform in parallel
       │
       ▼
L4  Temporal filter   ── enforce the user's time window
       │
       ▼
L5  Dedup             ── (Step 13 — coming soon) collapse cross-platform duplicates
       │
       ▼
L6  Triage            ── AI reads each link and verdicts: supports / refutes / tangential
       │
       ▼
L7  Emit              ── group by verdict, stream live progress to the UI
       │
       ▼
RANKED EVIDENCE GRID
```

---

## Stage-by-stage walkthrough

### Stage 1 — Read the hypothesis (L0 Decompose)

The tool parses your hypothesis sentence and extracts the key concepts:

| Slot | Example |
|---|---|
| **Brand / product** | `Corn Flakes`, `Kellogg's` |
| **Geography** | `India`, `Indian` |
| **Pain words** | `bland`, `misaligned`, `decline` |
| **Aspiration words** | `savory`, `traditional` |
| **Audience identity** | `Gen-Z`, `moms`, etc. |

No AI here — pure-Python with spaCy NER plus curated word lists. Instant and free.

### Stage 2 — Pick the right platforms (L1 Source Selector)

Different kinds of hypotheses live on different platforms. A hand-curated 10 × 13 grid scores every combination of (hypothesis dimension × strategic force) for each of 13 channels:

| If your hypothesis is about… | Top channels picked |
|---|---|
| **Gen-Z identity / cultural expression** | TikTok, YouTube Shorts, Instagram Reels, Reddit |
| **Product complaints / taste / quality** | Reddit, Trustpilot, Amazon reviews, Quora |
| **Crisis / brand reputation events** | News, Twitter, Brave News |
| **"How do I…" questions** | Quora, Google PAA, Reddit |

The selector picks the **top 5 channels** above a fit-score threshold of 50, and drops the rest entirely. Deterministic, no AI. This step saves ~60% of total LLM cost compared to "ask AI which platforms to use."

### Stage 3 — Write smart queries (L2 Query Synthesizer)

For each chosen channel, the tool generates **5-8 queries across 9 different "angles"**:

| # | Angle | Long-form template | Short-video template |
|---|---|---|---|
| 1 | Entity-pain | `{brand} {pain}` | `{brand} cringe` |
| 2 | Switching | `why I switched from {brand} to {alt}` | `pov: i stopped buying {brand}` |
| 3 | Comparison | `{brand} vs {alt} {dimension}` | `{brand} vs {alt}` |
| 4 | Aspiration | `best {category} for {aspiration}` | `{aspiration} {category} routine` |
| 5 | Question | `why is {brand} {trend}` | rarely used |
| 6 | **Counter (MANDATORY)** | `why I love {brand}` | `defending {brand}` |
| 7 | Expert | `{category} {expert_role}` | `nutritionist reacts {brand}` |
| 8 | Crisis | `{brand} {year}` | `{brand} scandal` |
| 9 | Hashtag | n/a | `#{brand}` |

**Why mandatory counter queries?** Without them you only find evidence confirming what you already suspected. The counter angle forces the pipeline to actively hunt for refuting evidence — turning a confirmation-bias machine into a falsification machine.

**Where the slots come from:**
- `{brand}`, `{pain}`, `{aspiration}` → directly from L0 decomposition
- `{alt}`, `{category}`, `{dimension}`, `{expert_role}` → one small Gemini call provides real-world knowledge ("muesli", "oats", "granola" as cornflakes alternatives)
- `{trend}` → **Google Trends** rising-queries — the *actual phrasings* consumers are typing about this brand right now in this country and time window. This injects live consumer language into the queries instead of marketer language.

### Stage 4 — Search every platform in parallel (L3 Discovery)

Nine discoverers fire simultaneously, each using its platform's native search:

| Channel | Discoverer | Special metadata it captures |
|---|---|---|
| **YouTube Shorts** ⭐ | Official YT Data API v3 | duration, view/like/comment counts, top 5 comments, full transcript, engagement score |
| **TikTok** | Brave `site:tiktok.com` | URL kind (video / discover / tag), hashtags |
| **YouTube long-form** | Official YT Data API v3 | same as Shorts, but ≥182s videos |
| **Reddit** | PRAW (if credentials) or Brave fallback | post body, subreddit, score, top 3 comments |
| **Quora** | Brave `site:quora.com` | question title, answer excerpt |
| **Google PAA** | Headless Chromium (only path) | People-Also-Ask question text |
| **News** | Brave News API | article date, source |
| **Substack** | Brave `site:substack.com` filtered to essays | newsletter author |
| **Marketplace** | Brave across Trustpilot / Amazon / Flipkart / etc. | review host as signal tag |

The **time window** (e.g., "past 1 year") is applied at search time wherever the platform's API supports it, so stale results get filtered before they reach the pipeline.

### Stage 5 — Pre-rank and shortlist (Triage budget)

Across all channels, discovery may return 50-200 candidate links. The tool can't afford to read all of them with AI, so it pre-ranks:

- **Short videos** rank by **engagement score** = (likes + comments × 3) ÷ views
- **Text content** preserves the search engine's relevance order

Then it cuts to the **top N** (the "triage budget" — default 30 for production, 10 for the demo). Below that line, links are discarded entirely and never appear in results.

**Tradeoffs of the triage budget:**

| Budget | Latency | $$ | Coverage |
|---|---|---|---|
| 5 | ~10s | $ | hero-rank only — misses borderlines |
| 10 (demo) | ~20s | $ | top engagement only — long-tail channels drop out |
| 30 (production) | ~60s | $$ | multi-channel coverage, supports + refutes both surface |
| 100 | ~3 min | $$$$ | full saturation — burns YT API quota |

### Stage 6 — Read each one and judge it (L6 Triage)

For each surviving link, the tool extracts the evidence text:

- **Short videos** → caption + top 5 comments + transcript (if captions available)
- **Long-form text pages** → HTTP fetch the URL, strip the HTML, take first 1,500 characters
- **Reddit threads** → post body + top 3 comments

A signal regex (auto-built from your `expected_signals` keywords) does a cheap first pass to count keyword hits.

Then the real judgment: batches of 6 links go to **Gemini** with this task:

> *"Here's the hypothesis. Here's each evidence snippet, numbered. For each one, classify it as **supports**, **refutes**, or **tangential**, with a confidence from 0.0 to 1.0, and tell me which signals you saw."*

Gemini's response shape:

```json
{
  "verdicts": [
    {
      "id": 1,
      "verdict": "supports",
      "confidence": 0.90,
      "signal_tags": ["preference_for_alternatives", "traditional_food_preference"]
    },
    ...
  ]
}
```

Verdicts attach to each link.

### Stage 7 — Group, rank, and stream (L7 Emit)

The classified links are grouped into 3 buckets:

- ✅ **Supports** — sorted by confidence, highest first
- ❌ **Refutes** — same
- ◯ **Tangential** — on-topic but doesn't actually answer the claim

**Every stage** (L0, L1, L2, L3, L6) emits a structured `PipelineEvent`. The web UI subscribes to a Server-Sent Events stream and renders these live, so instead of a frozen "loading…" spinner for 90 seconds you see:

```
pipeline_start                available_channels=[youtube_shorts,tiktok,reddit,...]
stage_start [L0_decompose]
stage_done  [L0_decompose]   entity=Corn Flakes pains=2 aspirations=2
stage_start [L1_source_select]
stage_done  [L1_source_select]   marketplace:82 reddit:81 youtube:76 google_paa:70 youtube_shorts:69
stage_start [L2_query_synth]
stage_done  [L2_query_synth]   34 queries across 4 channels
stage_start [L3_discover]
channel_discovered  marketplace      SKIPPED (no_discoverer)
channel_discovered  reddit           5 links from 9 queries
channel_discovered  youtube          1 link from 8 queries
channel_discovered  youtube_shorts   50 links from 9 queries
stage_done  [L3_discover]
stage_start [L6_triage]      n_candidates=56 max_triage=10
stage_done  [L6_triage]      n_triaged=10 verdicts={supports:2, refutes:0, tangential:8}
pipeline_complete            elapsed=78.5s
```

---

## What you see in the UI

The single-page demo (`http://localhost:8080`) renders:

- **3-column grid** of evidence cards — Supports (teal) / Refutes (red) / Tangential (grey)
- **Each card** carries: thumbnail (clickable to open the source), title, creator, view/like/comment counts, engagement bar, verdict badge with confidence, signal-tag chips, top comment, falsifier marker if applicable
- **Channel chips** showing every channel's fit_score from L1, with greyed-out chips for channels that got skipped
- **Live event log** scrolling on the right
- **Stats panel** — supports / refutes / tangential counts + elapsed seconds

---

## Real example: h_006 (Kellogg's Corn Flakes, India)

**Input:**

```json
{
  "hypothesis_id": "h_006",
  "statement": "The core Corn Flakes product is perceived as bland and misaligned with Indian taste preferences, which are shifting back towards traditional, savory breakfast options.",
  "dimension": "product",
  "force_assignment": "Demand Gravity",
  "expected_signals": ["taste_complaints", "product_feature_gaps", "preference_for_alternatives"],
  "expected_counter_signals": ["positive_product_feedback", "taste_satisfaction"]
}
```

**What happens (window = 1 year):**

1. **L0** extracts `primary_entity="Corn Flakes"`, `pains=[bland, misaligned, decline]`, `geo=[india, indian]`
2. **L1** picks top 5: marketplace(82), reddit(81), youtube(76), google_paa(70), youtube_shorts(69)
3. **L2** generates ~34 queries across those channels — including counter queries like "why I love Corn Flakes" so we don't only find supporting evidence
4. **L3** runs ~75s in parallel — returns ~50 YT Shorts + 5 Reddit threads + 5 Trustpilot reviews + 5 Substack essays + 1 long-form YT review (Quora and PAA come back empty for this narrow query)
5. **Top 30 by engagement** go to Gemini triage
6. **Headline supports:**
   - 🇮🇳 Hindi YT Short: **"कॉर्न फ्लैक्स चिवड़ा"** — Indian consumers turning cornflakes into traditional savory poha because the original is too bland (**0.85 confidence**) — direct lived evidence
   - **Bagrry Cornflakes** comparison Short — Indian competitor brand winning out (**0.90**)
   - **Long-form YT: "KELLOGG'S CORN FLAKES LAB TESTED"** — expert critique (**0.85**)
   - Reddit r/AskIndia: *"Need some breakfast recommendations"* — substitution-seeking
   - Reddit r/CasualIreland: *"Change of flavour in Bran Flakes"* — taste complaint
7. **Total elapsed: ~80 seconds.** Output streamed live as 16 SSE events.

---

## Design choices that matter

| Choice | Why |
|---|---|
| **Counter queries are mandatory, not optional** | Without them the system is a confirmation-bias machine. The most important search is the one looking for evidence AGAINST your hypothesis. |
| **Channel choice is deterministic, not LLM-driven** | Saves ~60% LLM cost. Makes outputs reproducible — same hypothesis always picks the same channels. |
| **Queries are composed from 9 fixed templates, not freewritten by an LLM** | Consistent query quality regardless of LLM mood. Easy to audit "why was this query run?" |
| **Top-30 triage budget** | Tradeoff: 30 = decisive verdicts + multi-channel coverage. 100 = full saturation but burns API quota. 10 = demo-fast but loses long-tail channels. |
| **Live SSE event stream** | Turns a 90-second black box into a real-time progress feed. Changes how users trust the output — they SEE the pipeline working rather than waiting on a frozen spinner. |
| **Every link carries hypothesis_id** | A link is never "topically about Kellogg's" — it's "evidence for hypothesis h_006." Multi-hypothesis runs never bleed evidence across claims. |
| **Sources serve hypotheses, not topics** | The pipeline is hypothesis-scoped throughout. Same link discovered for h_006 and h_007 is two separate triage decisions, not one shared judgment. |
| **YouTube Shorts as priority #1 channel** | Highest cultural-signal density for identity/aspiration hypotheses (Gen-Z palate shifts, wellness positioning). Captions + top comments + transcripts + engagement scores together give the LLM rich text for verdicts. |

---

## What you don't have to do anymore

Before this tool, a research analyst would:

1. Brainstorm which 5-10 platforms to even check
2. Write 30+ different query phrasings per hypothesis (and forget at least one angle every time)
3. Visit ~50 web pages and watch ~30 short videos
4. Read each carefully and judge "does this support, refute, or is it unrelated?"
5. Write up findings
6. **Start over tomorrow for hypothesis #2**

With this tool: paste the hypothesis JSON, click **Run pipeline**, get the answer in a minute and a half. Repeat for 30 hypotheses in a single afternoon.
