"""Headless Chromium scrape of Google — FINAL FALLBACK plus the only path for
PAA and Related Searches (neither Brave nor DDG expose them).

Concurrency:
  - Global asyncio.Semaphore (env HEADLESS_CONCURRENCY, default 3) shared
    across all callers so PAA + TikTok + Quora don't oversaturate the same IP.

CAPTCHA handling:
  - On detection, backend enters cooldown for HEADLESS_CAPTCHA_COOLDOWN_SEC
    (default 30) and short-circuits to [] for the duration. Caller falls
    back to whatever's left.

Lifted from services/link_farming.py:241-335 (host's existing scraper) with:
  - Typed RawResult return instead of dict
  - Global semaphore for concurrency cap
  - Cooldown state machine
  - PAA + Related as first-class verticals
  - Time-window translation via temporal.to_google_tbs
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import List, Optional
from urllib.parse import quote_plus

from ..models import QueryVertical, RawResult, TimeWindow
from ..temporal import to_google_tbs
from .base import SearchBackend

log = logging.getLogger(__name__)

USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.5 Safari/605.1.15",
]

VERTICAL_TBM: dict[str, str] = {
    "videos": "&tbm=vid",
    "news": "&tbm=nws",
    "shopping": "&tbm=shop",
}


class HeadlessGoogleBackend(SearchBackend):
    """Singleton — one global semaphore and one cooldown clock per process."""

    id = "headless_google"
    _instance: Optional["HeadlessGoogleBackend"] = None
    _semaphore: Optional[asyncio.Semaphore] = None
    _captcha_until: float = 0.0

    def __new__(cls) -> "HeadlessGoogleBackend":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # __init__ runs on every call; guard re-init
        if getattr(self, "_initialised", False):
            return
        self._initialised = True

        self.concurrency = int(os.getenv("HEADLESS_CONCURRENCY", "3"))
        self.cooldown_sec = int(os.getenv("HEADLESS_CAPTCHA_COOLDOWN_SEC", "30"))
        if HeadlessGoogleBackend._semaphore is None:
            HeadlessGoogleBackend._semaphore = asyncio.Semaphore(self.concurrency)

        try:
            import playwright  # noqa: F401

            self.available = True
        except ImportError:
            log.info(
                "playwright not installed; HeadlessGoogleBackend disabled. "
                "Install: pip install playwright && playwright install chromium"
            )
            self.available = False

    # ── cooldown helpers ────────────────────────────────────────────────────

    def _in_cooldown(self) -> bool:
        return time.time() < HeadlessGoogleBackend._captcha_until

    def _enter_cooldown(self) -> None:
        HeadlessGoogleBackend._captcha_until = time.time() + self.cooldown_sec
        log.warning(
            "Headless Google entered CAPTCHA cooldown for %ds", self.cooldown_sec
        )

    # ── public ─────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        vertical: QueryVertical = "web",
        count: int = 10,
        window: Optional[TimeWindow] = None,
    ) -> List[RawResult]:
        if not self.available:
            return []
        if self._in_cooldown():
            log.debug("Headless in cooldown; skipping %r", query[:50])
            return []

        assert HeadlessGoogleBackend._semaphore is not None
        async with HeadlessGoogleBackend._semaphore:
            return await self._search_inner(query, vertical, count, window)

    # ── private ────────────────────────────────────────────────────────────

    async def _search_inner(
        self,
        query: str,
        vertical: QueryVertical,
        count: int,
        window: Optional[TimeWindow],
    ) -> List[RawResult]:
        from playwright.async_api import async_playwright

        tbs = to_google_tbs(window) if window else None
        params = f"?q={quote_plus(query)}&num={count}&hl=en"
        params += VERTICAL_TBM.get(vertical, "")
        if tbs:
            params += f"&tbs={quote_plus(tbs)}"
        url = "https://www.google.com/search" + params

        results: List[RawResult] = []
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                    ],
                )
                ctx = await browser.new_context(
                    user_agent=random.choice(USER_AGENTS),
                    viewport={"width": 1366, "height": 768},
                    locale="en-US",
                    timezone_id="America/New_York",
                )
                page = await ctx.new_page()
                await page.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => false});"
                )

                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                except Exception as e:
                    log.warning("Headless goto failed: %s", e)
                    await browser.close()
                    return []

                # anti-bot jitter — PAA gets a longer pause
                paa_like = vertical in ("paa", "related")
                await page.wait_for_timeout(
                    random.randint(3000, 5000) if paa_like else random.randint(2000, 4000)
                )

                content = await page.content()
                if (
                    "captcha" in content.lower()
                    or "unusual traffic" in content.lower()
                ):
                    self._enter_cooldown()
                    await browser.close()
                    return []

                if vertical == "paa":
                    results = await self._extract_paa(page, count)
                elif vertical == "related":
                    results = await self._extract_related(page, count)
                else:
                    results = await self._extract_organic(page, vertical, count)
                    if vertical == "web":
                        results += await self._extract_paa(page, 6)

                await browser.close()
        except Exception as e:
            log.warning("Headless Google failed for %r: %s", query[:60], e)

        return results

    async def _extract_organic(
        self, page, vertical: QueryVertical, count: int
    ) -> List[RawResult]:
        sel = (
            "div.g, div.tF2Cxc, div.MjjYud div.g"
            if vertical != "videos"
            else "div.g, div[data-vid]"
        )
        items = await page.query_selector_all(sel)
        out: List[RawResult] = []
        for item in items[:count]:
            try:
                link_el = await item.query_selector("a[href^='http']")
                title_el = await item.query_selector("h3")
                snippet_el = await item.query_selector(
                    "div.VwiC3b, span.aCOpRe, div.IsZvec"
                )
                link = await link_el.get_attribute("href") if link_el else ""
                title = (await title_el.inner_text()) if title_el else ""
                snippet = (await snippet_el.inner_text()) if snippet_el else ""
                if link:
                    out.append(
                        RawResult(
                            url=link,
                            title=title,
                            snippet=snippet,
                            backend=self.id,
                            vertical=vertical,
                        )
                    )
            except Exception:
                continue
        return out

    async def _extract_paa(self, page, count: int) -> List[RawResult]:
        out: List[RawResult] = []
        nodes = await page.query_selector_all(
            "div.related-question-pair, div[data-q]"
        )
        for node in nodes[:count]:
            try:
                txt = (await node.inner_text()).strip()
                if txt and len(txt) > 10:
                    out.append(
                        RawResult(
                            url="",
                            title=txt.split("\n")[0],
                            snippet="",
                            backend=self.id,
                            vertical="paa",
                        )
                    )
            except Exception:
                continue
        return out

    async def _extract_related(self, page, count: int) -> List[RawResult]:
        out: List[RawResult] = []
        # Related Searches sidebar — selectors vary across Google A/B variants
        nodes = await page.query_selector_all(
            "a.k8XOCe, p.nVcaUb, div.AJLUJb a"
        )
        for node in nodes[:count]:
            try:
                txt = (await node.inner_text()).strip()
                if txt and len(txt) > 3:
                    out.append(
                        RawResult(
                            url="",
                            title=txt,
                            snippet="",
                            backend=self.id,
                            vertical="related",
                        )
                    )
            except Exception:
                continue
        return out
