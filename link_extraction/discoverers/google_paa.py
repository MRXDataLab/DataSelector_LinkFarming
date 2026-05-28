"""Google PAA (People Also Ask) discoverer.

Wraps the existing headless Chromium backend at `vertical="paa"`. PAA is the
only Google SERP surface that exposes *questions consumers actually type
about a topic* — it's gold for hypothesis investigation because each PAA
node carries an accordion answer snippet plus a "Search for" link.

**v2 (this rewrite):** PAA questions are FIRST-CLASS evidence. The previous
implementation filtered out PAA results because the headless extractor
returns `RawResult(url="")` (PAA questions don't carry their own URL —
each is a search bait that expands to an accordion). We now mint a
synthetic `https://www.google.com/search?q=<question>` URL for every PAA
question so the question becomes a clickable, body-fetchable link.

Quirks:
- Headless-only; the registry routes `vertical="paa"` straight to Chromium.
- Slow (~3-6 sec per query depending on PAA depth + CAPTCHA risk).
- Backend's global concurrency semaphore (default 3) caps parallel pages.
- On CAPTCHA, the backend self-disables for 30s and returns []; we propagate
  that as an empty discovery rather than retrying — the cooldown is shared
  across all headless callers (TikTok / Quora to come) so any retry would
  just lengthen the disable window.

Returns `DiscoveredLink` whose URL is a Google-search URL keyed to the PAA
question. Each link is tagged `signal_tags=["paa_question"]` so downstream
triage / dedup / CSV-export can identify these as question-level signals
rather than article links.
"""
from __future__ import annotations

import logging
from typing import List, Optional
from urllib.parse import quote_plus

from ..backends import search_with_fallback, get_headless
from ..models import DiscoveredLink, TimeWindow, TypedQuery
from ._common import raw_result_to_link
from .base import Discoverer

log = logging.getLogger(__name__)


def _synthesize_paa_url(question: str) -> str:
    """Mint a clickable Google search URL for a PAA question."""
    return f"https://www.google.com/search?q={quote_plus(question)}"


class GooglePAADiscoverer(Discoverer):
    channel_id = "google_paa"

    def __init__(self) -> None:
        # Availability mirrors the headless backend (Chromium installed +
        # not in CAPTCHA cooldown). Re-checked on each discover() call.
        self.available = get_headless().available

    async def discover(
        self,
        query: TypedQuery,
        window: TimeWindow,
        count: int = 10,
    ) -> List[DiscoveredLink]:
        if not get_headless().available:
            return []
        raw = await search_with_fallback(
            query.text,
            vertical="paa",
            count=count,
            window=window,
            min_results=1,
        )

        out: List[DiscoveredLink] = []
        seen: set[str] = set()
        for r in raw:
            # PAA results carry the question text in `title`. The `url`
            # is empty because the headless extractor can't get a stable
            # URL from the accordion node — mint a search URL so the
            # question becomes a clickable, body-fetchable link.
            question = (r.title or "").strip()
            if not question:
                continue
            if len(question) < 8 or len(question) > 240:
                continue  # noise; PAA questions are usually 20-120 chars
            key = question.lower()
            if key in seen:
                continue
            seen.add(key)
            synthesized_url = r.url or _synthesize_paa_url(question)
            link = DiscoveredLink(
                url=synthesized_url,
                canonical_url=synthesized_url,
                title=question,
                snippet=r.snippet or question,
                channel=self.channel_id,
                hypothesis_id=query.hypothesis_id,
                query=query,
                backend_used=f"{r.backend}+paa",
                signal_tags=["paa_question"],
            )
            out.append(link)
        return out


_singleton: Optional[GooglePAADiscoverer] = None


def get_google_paa() -> GooglePAADiscoverer:
    global _singleton
    if _singleton is None:
        _singleton = GooglePAADiscoverer()
    return _singleton


def reset_google_paa() -> None:
    global _singleton
    _singleton = None
