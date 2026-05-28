"""Headless Chromium scrape of Google — PRIORITY #1 by default (Change #2).

Tuned for low-CAPTCHA-rate operation. The "Google Free" backend label in
the UI is this module. The bundle of CAPTCHA-avoidance tactics:

  1. **playwright-stealth** — masks `navigator.webdriver`, languages,
     plugins, WebGL/Canvas fingerprints. Applied to every new page.
  2. **Persistent browser context** — one Chromium instance + one context
     alive across queries, so cookies/session look like a single human's
     browsing session, not fresh-incognito-per-query.
  3. **Per-host request pacing** — minimum 2s between Google hits + 1-3s
     random jitter on top. Prevents the burst pattern that flags bots.
  4. **Concurrency 1 by default** — only one Google scrape in flight at
     any moment. (`HEADLESS_CONCURRENCY` env var, was 3.)
  5. **5-minute CAPTCHA cooldown** — was 30s; that wasn't enough, Google
     remembers IPs for hours. 5 min is a softer signal that lets the
     server move past short-burst flags. (`HEADLESS_CAPTCHA_COOLDOWN_SEC`.)
  6. **UA + viewport rotation** — 4 user agents × 3 viewports (laptop /
     desktop / portrait tablet).
  7. **Realistic wait timings** — 4-7 sec after navigation to mimic human
     reading time before any scraping happens.

When CAPTCHA does fire, the global cooldown clock prevents further calls
for 5 min, and `search()` returns [] so the registry's fallback chain
escalates to Brave / DDG immediately.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, urlparse

from .. import backend_health
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

# 3 viewport sizes — laptop / desktop / tablet portrait. Real distribution
# is mostly desktop; we weight accordingly via the choice of indices.
VIEWPORTS = [
    {"width": 1366, "height": 768},   # laptop (most common)
    {"width": 1920, "height": 1080},  # desktop
    {"width": 1440, "height": 900},   # macbook-ish
]

VERTICAL_TBM: dict[str, str] = {
    "videos": "&tbm=vid",
    "news":   "&tbm=nws",
    # `scholar` and `local` route to different Google subdomains — see
    # `_url_for_vertical()`. The `&tbm=` value here is unused for those.
    "scholar": "",
    "local":   "",
    "shopping": "&tbm=shop",
}

# Per-host pacing: minimum gap (sec) between two requests to the same host.
_HOST_MIN_GAP_SEC = 2.0
# Extra random jitter on top of the minimum gap.
_HOST_GAP_JITTER_MIN = 0.5
_HOST_GAP_JITTER_MAX = 2.5
# Human-like dwell time after page load before scraping starts.
_DWELL_MIN_MS = 4000
_DWELL_MAX_MS = 7000

# Innocuous, high-volume queries used by the recovery probe. Picked at
# random per probe so we don't repeat the same string twice in a row —
# repeating "test" 50 times is itself a bot signature. These are queries
# Google sees billions of times a day from real users, so they blend in.
_PROBE_QUERIES: list[str] = [
    "weather today",
    "news today",
    "current time",
    "best restaurants nearby",
    "how to make coffee",
    "wikipedia",
    "youtube",
    "translate english to spanish",
    "what is the time in london",
    "stock market today",
]


class HeadlessGoogleBackend(SearchBackend):
    """Singleton — one browser, one cooldown clock, one per-host pacing table."""

    id = "headless_google"
    _instance: Optional["HeadlessGoogleBackend"] = None
    _semaphore: Optional[asyncio.Semaphore] = None
    _captcha_until: float = 0.0
    # Persistent browser state (created lazily on first use, never closed
    # explicitly — process exit handles cleanup)
    _pw_obj: Any = None
    _browser: Any = None
    _context: Any = None
    _ctx_lock: Optional[asyncio.Lock] = None
    # Per-host last-request-time tracker (monotonic seconds)
    _host_last_req: Dict[str, float] = {}
    _host_pace_lock: Optional[asyncio.Lock] = None

    def __new__(cls) -> "HeadlessGoogleBackend":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialised", False):
            return
        self._initialised = True

        # Defaults aligned with Change #2 anti-CAPTCHA stance.
        self.concurrency = int(os.getenv("HEADLESS_CONCURRENCY", "1"))
        self.cooldown_sec = int(os.getenv("HEADLESS_CAPTCHA_COOLDOWN_SEC", "300"))
        if HeadlessGoogleBackend._semaphore is None:
            HeadlessGoogleBackend._semaphore = asyncio.Semaphore(self.concurrency)
        if HeadlessGoogleBackend._ctx_lock is None:
            HeadlessGoogleBackend._ctx_lock = asyncio.Lock()
        if HeadlessGoogleBackend._host_pace_lock is None:
            HeadlessGoogleBackend._host_pace_lock = asyncio.Lock()

        try:
            import playwright  # noqa: F401
            self.available = True
            backend_health.report(
                "headless_google", "ok",
                message="playwright installed; ready",
            )
        except ImportError:
            log.info(
                "playwright not installed; HeadlessGoogleBackend disabled. "
                "Install: pip install playwright && playwright install chromium"
            )
            self.available = False
            backend_health.report(
                "headless_google", "not_installed",
                message="pip install playwright && playwright install chromium",
            )

    # ── cooldown helpers ────────────────────────────────────────────────────

    def _in_cooldown(self) -> bool:
        return time.time() < HeadlessGoogleBackend._captcha_until

    def _enter_cooldown(self) -> None:
        HeadlessGoogleBackend._captcha_until = time.time() + self.cooldown_sec
        backend_health.report(
            "headless_google", "captcha_cooldown",
            message=f"Google flagged CAPTCHA; backing off {self.cooldown_sec}s",
            cooldown_seconds=self.cooldown_sec,
        )
        log.warning(
            "Headless Google entered CAPTCHA cooldown for %ds", self.cooldown_sec
        )

    # ── Per-host pacing ─────────────────────────────────────────────────────

    async def _wait_for_host_slot(self, host: str) -> None:
        """Enforce minimum + jittered gap between requests to the same host."""
        assert HeadlessGoogleBackend._host_pace_lock is not None
        async with HeadlessGoogleBackend._host_pace_lock:
            now = time.monotonic()
            last = HeadlessGoogleBackend._host_last_req.get(host, 0.0)
            elapsed = now - last
            needed = _HOST_MIN_GAP_SEC + random.uniform(
                _HOST_GAP_JITTER_MIN, _HOST_GAP_JITTER_MAX,
            )
            sleep_for = needed - elapsed
            # Reserve the slot now so concurrent callers compute against an
            # updated timestamp (they'll sleep their own full gap).
            HeadlessGoogleBackend._host_last_req[host] = now + max(0.0, sleep_for)
        if sleep_for > 0:
            log.debug("host pacing: sleeping %.2fs before next %s request", sleep_for, host)
            await asyncio.sleep(sleep_for)

    # ── Persistent browser + context ────────────────────────────────────────

    async def _get_context(self) -> Any:
        """Lazy-init the persistent context. Recreates on failure."""
        if HeadlessGoogleBackend._context is not None:
            return HeadlessGoogleBackend._context

        assert HeadlessGoogleBackend._ctx_lock is not None
        async with HeadlessGoogleBackend._ctx_lock:
            if HeadlessGoogleBackend._context is not None:
                return HeadlessGoogleBackend._context
            await self._build_context()
            return HeadlessGoogleBackend._context

    async def _build_context(self) -> None:
        """Spin up playwright + Chromium + a fresh context with stealth."""
        from playwright.async_api import async_playwright

        cls = HeadlessGoogleBackend
        cls._pw_obj = await async_playwright().start()
        cls._browser = await cls._pw_obj.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                # Stealth flags — match real Chrome's default flags as
                # closely as headful Chrome would emit.
                "--disable-gpu",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        cls._context = await cls._browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport=random.choice(VIEWPORTS),
            locale="en-US",
            timezone_id="America/New_York",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
            },
            color_scheme="light",
        )
        # Stealth — patches `navigator.webdriver`, plugins, language list,
        # `chrome` object, WebGL vendor/renderer, etc. Applied to the
        # CONTEXT so every new page inherits the patches.
        try:
            from playwright_stealth import Stealth
            stealth = Stealth()
            await stealth.apply_stealth_async(cls._context)
            log.info("playwright-stealth applied to Chromium context")
        except ImportError:
            log.info("playwright-stealth not installed — manual init script only")
            # Manual minimal stealth fallback
            await cls._context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
                "window.chrome = {runtime: {}};"
                "Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});"
                "Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});"
            )
        except Exception as e:
            log.warning("stealth.apply_stealth_async failed: %s", e)

    async def _reset_context(self) -> None:
        """Drop persistent state — next call rebuilds. Use after errors."""
        cls = HeadlessGoogleBackend
        try:
            if cls._context is not None:
                await cls._context.close()
        except Exception:
            pass
        try:
            if cls._browser is not None:
                await cls._browser.close()
        except Exception:
            pass
        try:
            if cls._pw_obj is not None:
                await cls._pw_obj.stop()
        except Exception:
            pass
        cls._context = None
        cls._browser = None
        cls._pw_obj = None

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
            await self._wait_for_host_slot("google.com")
            try:
                return await self._search_inner(query, vertical, count, window)
            except Exception as e:
                log.warning("Headless _search_inner crashed, resetting ctx: %s", e)
                await self._reset_context()
                return []

    async def probe(self) -> bool:
        """Less-bot-like health probe for the recovery loop.

        Instead of going straight to ``/search?q=test`` (which is a
        textbook bot signature — no referrer, no cookies, no homepage
        visit, 4-char throwaway query), we simulate a human's actual
        navigation pattern:

          1. Land on ``google.com/`` and let the page settle.
          2. Dismiss the EU consent banner if it appears.
          3. Dwell + jitter the mouse — buys 2-4s of "reading" time.
          4. Focus the real search box, *type* a realistic query with
             per-keystroke delay, then press Enter — emulates the user
             flow rather than a synthetic GET.
          5. Wait for the SERP to render, then verify we have organic
             results.

        Any CAPTCHA detected along the way is treated identically to
        the search path: enter cooldown, reset the Chromium context.
        Returns True only on a clean SERP with ≥1 result.
        """
        if not self.available:
            return False
        if self._in_cooldown():
            log.debug("Headless probe — in cooldown, skipping")
            return False

        assert HeadlessGoogleBackend._semaphore is not None
        async with HeadlessGoogleBackend._semaphore:
            await self._wait_for_host_slot("google.com")
            try:
                return await self._probe_inner()
            except Exception as e:
                log.warning("Headless probe crashed, resetting ctx: %s", e)
                await self._reset_context()
                return False

    async def _probe_inner(self) -> bool:
        """Human-like probe — homepage → consent → dwell → typed search."""
        ctx = await self._get_context()
        page = await ctx.new_page()
        try:
            # ── Step 1 — homepage visit (the bit that makes the eventual
            # /search request look like a referred navigation, not a
            # cold deeplink).
            try:
                await page.goto(
                    "https://www.google.com/",
                    wait_until="domcontentloaded",
                    timeout=15000,
                )
            except Exception as e:
                log.warning("Headless probe: homepage goto failed: %s", e)
                return False

            # Early CAPTCHA check — Google sometimes serves /sorry/
            # straight off the homepage when an IP is heavily flagged.
            if await self._captcha_short_circuit(page, where="homepage"):
                return False

            # ── Step 2 — dismiss consent banner (EU/global rollout).
            # The dialog selectors vary by region; try a small set.
            await self._dismiss_consent(page)

            # ── Step 3 — dwell + mouse noise. Human reading time before
            # any action is taken.
            await page.wait_for_timeout(random.randint(1500, 3500))
            try:
                await page.mouse.move(
                    random.randint(200, 800), random.randint(100, 400),
                )
                await page.mouse.move(
                    random.randint(200, 800), random.randint(100, 400),
                    steps=random.randint(8, 20),
                )
            except Exception:
                pass

            # ── Step 4 — type a realistic query into the search box and
            # submit via Enter. Falls back to URL navigation if the box
            # selector isn't found (DOM markup occasionally shifts).
            realistic_q = random.choice(_PROBE_QUERIES)
            try:
                box = await page.query_selector(
                    'textarea[name="q"], input[name="q"]'
                )
                if box:
                    await box.click()
                    await page.wait_for_timeout(random.randint(300, 800))
                    await box.type(
                        realistic_q,
                        delay=random.randint(80, 180),  # per-keystroke ms
                    )
                    await page.wait_for_timeout(random.randint(400, 900))
                    await box.press("Enter")
                else:
                    log.debug(
                        "Headless probe: search box not found, "
                        "falling back to /search?q=%s", realistic_q,
                    )
                    await page.goto(
                        f"https://www.google.com/search?q={quote_plus(realistic_q)}",
                        wait_until="domcontentloaded",
                        timeout=15000,
                    )
            except Exception as e:
                log.warning("Headless probe: search submission failed: %s", e)
                return False

            # ── Step 5 — wait for SERP. The selectors below match both
            # the modern and the legacy Google result containers.
            try:
                await page.wait_for_selector(
                    "div.g, div.tF2Cxc, #search", timeout=10000,
                )
            except Exception:
                pass

            # Post-search CAPTCHA check.
            if await self._captcha_short_circuit(page, where="post-search"):
                return False

            # ── Step 6 — count organic results. A real SERP has ≥1 item.
            items = await page.query_selector_all("div.g, div.tF2Cxc")
            ok = len(items) > 0
            if ok:
                backend_health.report(
                    "headless_google", "ok",
                    message=(
                        f"probe ok: {len(items)} results for {realistic_q!r}"
                    ),
                )
                log.info(
                    "Headless Google probe succeeded — %d results for %r",
                    len(items), realistic_q,
                )
            return ok
        finally:
            try:
                await page.close()
            except Exception:
                pass

    async def _captcha_short_circuit(self, page: Any, *, where: str) -> bool:
        """If `page` is showing CAPTCHA or `/sorry/`, enter cooldown +
        reset the context, and return True. Otherwise return False.

        Used by `_probe_inner` at multiple checkpoints so the bot-wall
        gets caught early without forcing the caller to repeat boilerplate.
        """
        try:
            content = await page.content()
        except Exception:
            return False
        if (
            "captcha" in content.lower()
            or "unusual traffic" in content.lower()
            or "/sorry/" in (page.url or "")
        ):
            log.warning(
                "Headless probe: CAPTCHA detected at %s (url=%s)",
                where, (page.url or "")[:120],
            )
            self._enter_cooldown()
            try:
                await page.close()
            except Exception:
                pass
            await self._reset_context()
            log.info(
                "Headless Google context reset after CAPTCHA — next "
                "probe will use fresh fingerprint + clean cookies"
            )
            return True
        return False

    async def _dismiss_consent(self, page: Any) -> None:
        """Click through Google's cookie / consent dialog if present.

        The dialog appears for new-IP / no-cookie sessions in many
        regions. There are multiple selector variants — try a few in
        order. Silent best-effort; not finding the dialog is fine
        (most US users won't see one).
        """
        consent_selectors = [
            # Modern dialog — visible label is "Accept all" or local
            # equivalent; aria-label often spells the action explicitly.
            'button[aria-label*="Accept" i]',
            'button[aria-label*="agree" i]',
            # Text-based selectors as a fallback (Playwright extension).
            'button:has-text("Accept all")',
            'button:has-text("I agree")',
            'button:has-text("Accept")',
            # Generic dialog button — last-resort, matches anything that
            # looks like a primary action inside a modal.
            'div[role="dialog"] button',
        ]
        for sel in consent_selectors:
            try:
                btn = await page.query_selector(sel)
                if btn:
                    await btn.click()
                    await page.wait_for_timeout(random.randint(400, 900))
                    log.info(
                        "Headless probe: dismissed consent banner via %r",
                        sel,
                    )
                    return
            except Exception:
                continue

    # ── private ────────────────────────────────────────────────────────────

    async def _search_inner(
        self,
        query: str,
        vertical: QueryVertical,
        count: int,
        window: Optional[TimeWindow],
    ) -> List[RawResult]:
        tbs = to_google_tbs(window) if window else None
        # Phase 1.6 — Scholar + Maps use different subdomains. Route them
        # through dedicated URL builders so we don't accidentally send a
        # `&tbm=` parameter to `scholar.google.com` (which it ignores) or
        # try to scrape a Maps interactive page.
        if vertical == "scholar":
            url = (
                f"https://scholar.google.com/scholar"
                f"?q={quote_plus(query)}&num={count}&hl=en"
            )
            if tbs:
                url += f"&as_ylo={_year_lo_from_tbs(tbs)}"
        elif vertical == "local":
            # The local-pack appears in a plain web search for any
            # location-intent query — no special URL needed. We parse the
            # local card from the regular SERP in `_extract_local()`.
            params = f"?q={quote_plus(query)}&num={count}&hl=en"
            if tbs:
                params += f"&tbs={quote_plus(tbs)}"
            url = "https://www.google.com/search" + params
        else:
            params = f"?q={quote_plus(query)}&num={count}&hl=en"
            params += VERTICAL_TBM.get(vertical, "")
            if tbs:
                params += f"&tbs={quote_plus(tbs)}"
            url = "https://www.google.com/search" + params

        ctx = await self._get_context()
        page = await ctx.new_page()
        results: List[RawResult] = []
        try:
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            except Exception as e:
                log.warning("Headless goto failed: %s", e)
                return []

            # Human-like dwell time — longer for PAA which needs box expansion
            paa_like = vertical in ("paa", "related")
            dwell = (
                random.randint(_DWELL_MIN_MS + 1500, _DWELL_MAX_MS + 1500)
                if paa_like
                else random.randint(_DWELL_MIN_MS, _DWELL_MAX_MS)
            )
            await page.wait_for_timeout(dwell)

            # Mouse movement noise — makes the session look more human.
            try:
                await page.mouse.move(
                    random.randint(100, 800), random.randint(100, 600),
                )
                await page.mouse.move(
                    random.randint(100, 800), random.randint(100, 600),
                    steps=random.randint(5, 15),
                )
            except Exception:
                pass

            content = await page.content()
            if (
                "captcha" in content.lower()
                or "unusual traffic" in content.lower()
                or "/sorry/" in (page.url or "")
            ):
                self._enter_cooldown()
                # CRITICAL — burn the browser context too. Google has
                # marked this session's cookies + fingerprint as bot-
                # suspected; reusing it on the next retry guarantees
                # another CAPTCHA and traps us in a forever-cooldown
                # loop. Closing the page first (so `finally` doesn't
                # error on a dead ctx), then resetting the persistent
                # context forces the next call to rebuild Chromium with
                # a freshly-randomized UA + viewport + zero cookies.
                try:
                    await page.close()
                except Exception:
                    pass
                await self._reset_context()
                log.info(
                    "Headless Google context reset after CAPTCHA — next "
                    "probe will use fresh fingerprint + clean cookies"
                )
                return []

            if vertical == "paa":
                results = await self._extract_paa(page, count)
            elif vertical == "related":
                results = await self._extract_related(page, count)
            elif vertical == "scholar":
                results = await self._extract_scholar(page, count)
            elif vertical == "local":
                # Local-pack lives inline in a web SERP — also fall through
                # to organic if the pack is empty.
                results = await self._extract_local(page, count)
                if len(results) < count:
                    results += await self._extract_organic(page, "web", count - len(results))
            else:
                results = await self._extract_organic(page, vertical, count)
                if vertical == "web":
                    results += await self._extract_paa(page, 6)
        finally:
            try:
                await page.close()
            except Exception:
                pass

        # Reaching here means no CAPTCHA + no goto failure: mark healthy.
        backend_health.report(
            "headless_google", "ok",
            message=f"{vertical}: {len(results)} results",
        )
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

    # ── Phase 1.6 — Scholar + Local extractors ──────────────────────────────

    async def _extract_scholar(self, page, count: int) -> List[RawResult]:
        """Parse Google Scholar SERP — title + snippet + paper URL per result.

        Scholar's DOM differs from web SERP: each result lives in
        `div.gs_ri` with title link inside `h3.gs_rt > a` and the abstract
        snippet inside `div.gs_rs`.
        """
        out: List[RawResult] = []
        items = await page.query_selector_all("div.gs_ri")
        for item in items[:count]:
            try:
                title_link = await item.query_selector("h3.gs_rt a")
                href = await title_link.get_attribute("href") if title_link else ""
                title = (await title_link.inner_text()) if title_link else ""
                snippet_el = await item.query_selector("div.gs_rs")
                snippet = (await snippet_el.inner_text()) if snippet_el else ""
                # Citation metadata (year, authors) lives in div.gs_a
                meta_el = await item.query_selector("div.gs_a")
                meta = (await meta_el.inner_text()) if meta_el else ""
                if href:
                    out.append(
                        RawResult(
                            url=href, title=title.strip(), snippet=snippet.strip(),
                            backend=self.id, vertical="scholar",
                            raw_metadata={"citation_meta": meta.strip()[:200]},
                        )
                    )
            except Exception:
                continue
        return out

    async def _extract_local(self, page, count: int) -> List[RawResult]:
        """Parse the Google local-pack (3-pack of places) from a web SERP.

        Local results carry a place name + a snippet showing rating /
        review count / address. The link is the place card's "Website"
        anchor when present, otherwise the Maps `place_id` URL.
        """
        out: List[RawResult] = []
        # Local-pack containers — Google A/B tests these classes so we hedge.
        nodes = await page.query_selector_all(
            "div.VkpGBb, div[data-hveid] div.rllt__details, "
            "g-place-result, div.uMdZh"
        )
        for node in nodes[:count]:
            try:
                # Title — place name
                title_el = await node.query_selector(
                    "div.dbg0pd, span.OSrXXb, div.rllt__details > div"
                )
                title = (await title_el.inner_text()).strip().split("\n")[0] if title_el else ""
                # Snippet — rating + review count + address
                snippet_el = await node.query_selector(
                    "div.rllt__details, div.rllt__wrapped"
                )
                snippet = (await snippet_el.inner_text()).strip() if snippet_el else ""
                # Link — prefer the "Website" anchor, fall back to any link
                link_el = await node.query_selector(
                    "a[data-noner]:not([href^='/maps']), a[href^='http']"
                )
                href = (await link_el.get_attribute("href")) if link_el else ""
                if title and (href or snippet):
                    out.append(
                        RawResult(
                            url=href or "",
                            title=title,
                            snippet=snippet[:400],
                            backend=self.id,
                            vertical="local",
                        )
                    )
            except Exception:
                continue
        return out


def _year_lo_from_tbs(tbs: str) -> str:
    """Best-effort year-low for Scholar's `as_ylo` from a Google `tbs=qdr:y` value.

    Scholar uses a different time syntax than the main SERP: `as_ylo=2024`
    instead of `tbs=qdr:y`. We map only the year case (the only one Scholar
    cares about for our 7d / 30d / 90d / 1y / 5y windows — sub-year is
    irrelevant for academic papers).
    """
    import datetime
    if "qdr:" not in (tbs or ""):
        return ""
    # qdr:y1 ⇒ 1 year, qdr:y5 ⇒ 5 years, qdr:m ⇒ 1 month (ignored), etc.
    suffix = tbs.split("qdr:", 1)[1].rstrip(",").strip()
    years_back = 1
    if suffix.startswith("y"):
        n = suffix[1:]
        if n.isdigit():
            years_back = int(n)
    return str(datetime.datetime.utcnow().year - years_back)
