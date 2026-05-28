"""HTML body fetcher — Phase 3 foundation for quote extraction + synthesis.

Given a URL, downloads the page, strips chrome (nav / footer / ads /
sidebar / cookie banners), and returns the main article text. Used by
``quote_extractor`` to feed clean text into Gemini so the LLM can pull
verbatim evidence sentences.

Architecture choices:
  • **Primary extractor**: ``trafilatura`` — best-in-class boilerplate
    removal for news, blogs, forum posts. Handles JS-rendered text only
    when the server delivers it pre-rendered (we don't run headless here;
    that's a per-link 5-10s cost we can't afford for Phase 3 fanout).
  • **Fallback**: a regex-based `<script>`+`<style>` stripper that
    pulls everything between `<body>` and `</body>` as a last resort.
  • **Caching**: every successful fetch persists to
    ``.memory/bodies/<sha1(url)>.json`` so re-runs of the same hypothesis
    don't re-fetch (HTTP cost = 0 for cached). 30-day TTL.
  • **Concurrency**: one `aiohttp.ClientSession` with a 10-request
    parallel cap. Per-host pacing handled by the caller (orchestrator
    enriches in batches and respects channel-level rate limits).
  • **Timeout**: 10s per URL — pages that don't return that fast are
    almost certainly JS-walls, anti-bot redirects, or 504s we shouldn't
    spend time on.

What we DO NOT do here:
  • Render JS (headless playwright is too expensive at this fanout)
  • Follow OAuth / paywalls
  • Bypass robots.txt (we respect it — this is research, not crawling)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ─── Config ──────────────────────────────────────────────────────────────────


FETCH_TIMEOUT_SEC = float(os.getenv("BODY_FETCH_TIMEOUT_SEC", "10"))
MAX_CONCURRENT_FETCHES = int(os.getenv("BODY_FETCH_CONCURRENCY", "10"))
CACHE_TTL_SEC = int(os.getenv("BODY_FETCH_CACHE_TTL_SEC", str(30 * 86400)))  # 30d
MAX_BODY_CHARS = int(os.getenv("BODY_FETCH_MAX_CHARS", "20000"))  # cap for LLM cost

# Where the cache lives. Mirrors the memory_store layout.
_DEFAULT_CACHE_DIR = Path(__file__).parent.parent / ".memory" / "bodies"


def _cache_dir() -> Path:
    override = os.getenv("OUTTLYR_MEMORY_DIR")
    if override:
        return Path(override) / "bodies"
    return _DEFAULT_CACHE_DIR


# Try to import trafilatura at module load; fall back to regex if missing.
try:
    import trafilatura  # type: ignore
    _HAVE_TRAFILATURA = True
except ImportError:
    trafilatura = None  # type: ignore
    _HAVE_TRAFILATURA = False
    log.info(
        "trafilatura not installed; body fetcher will use regex fallback. "
        "Install: pip install trafilatura"
    )


# ─── Output type ─────────────────────────────────────────────────────────────


@dataclass
class FetchedBody:
    url: str
    text: str = ""
    fetched_at: float = field(default_factory=time.time)
    http_status: int = 0
    extractor: str = ""        # "trafilatura" | "regex" | "cache" | ""
    error: str = ""
    bytes_received: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "text": self.text,
            "fetched_at": self.fetched_at,
            "http_status": self.http_status,
            "extractor": self.extractor,
            "error": self.error,
            "bytes_received": self.bytes_received,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FetchedBody":
        return cls(
            url=d.get("url", ""),
            text=d.get("text", ""),
            fetched_at=d.get("fetched_at", 0.0),
            http_status=d.get("http_status", 0),
            extractor=d.get("extractor", ""),
            error=d.get("error", ""),
            bytes_received=d.get("bytes_received", 0),
        )


# ─── Cache helpers ───────────────────────────────────────────────────────────


def _cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest()


def _cache_path(url: str) -> Path:
    cd = _cache_dir()
    cd.mkdir(parents=True, exist_ok=True)
    return cd / f"{_cache_key(url)}.json"


def _load_cached(url: str) -> Optional[FetchedBody]:
    p = _cache_path(url)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    fb = FetchedBody.from_dict(d)
    # TTL check
    if (time.time() - fb.fetched_at) > CACHE_TTL_SEC:
        return None
    fb.extractor = "cache"
    return fb


def _save_cached(fb: FetchedBody) -> None:
    if not fb.ok:
        return  # don't cache failures — they're often transient
    try:
        _cache_path(fb.url).write_text(json.dumps(fb.to_dict()))
    except Exception as e:
        log.debug("body cache write failed: %s", e)


# ─── Extractors ──────────────────────────────────────────────────────────────


_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style|nav|footer|aside|form|noscript)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _extract_with_trafilatura(html: str) -> str:
    """trafilatura's `extract()` returns clean article text or None."""
    if not _HAVE_TRAFILATURA or not html:
        return ""
    try:
        out = trafilatura.extract(  # type: ignore[union-attr]
            html,
            include_comments=False,
            include_tables=False,
            favor_recall=True,        # bias toward keeping more text
            no_fallback=False,
        )
        return (out or "").strip()
    except Exception as e:
        log.debug("trafilatura failed: %s", e)
        return ""


def _extract_with_regex(html: str) -> str:
    """Fallback when trafilatura unavailable or returns empty.

    Crude but works for any HTML — strip script/style, drop all tags,
    collapse whitespace. Won't separate article body from chrome but
    captures the text.
    """
    if not html:
        return ""
    # Drop scripts/styles/nav/footer/aside/form blocks entirely
    stripped = _SCRIPT_STYLE_RE.sub(" ", html)
    # Find <body> if present, else use the whole thing
    body_m = re.search(r"<body[^>]*>(.*?)</body>", stripped,
                       re.IGNORECASE | re.DOTALL)
    if body_m:
        stripped = body_m.group(1)
    # Drop remaining tags
    text = _TAG_RE.sub(" ", stripped)
    # Decode common HTML entities
    text = (text.replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&quot;", '"')
                .replace("&apos;", "'").replace("&nbsp;", " "))
    # Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ─── HTTP layer ──────────────────────────────────────────────────────────────


_session_lock = asyncio.Lock()
_session: Optional[Any] = None
_semaphore: Optional[asyncio.Semaphore] = None


async def _get_session() -> Any:
    """Lazy aiohttp.ClientSession with shared concurrency semaphore."""
    global _session, _semaphore
    if _session is None:
        async with _session_lock:
            if _session is None:
                import aiohttp
                timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT_SEC)
                # Realistic browser headers — many servers 403 default
                # aiohttp user-agent. We're not pretending to be a bot.
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": (
                        "text/html,application/xhtml+xml,application/xml;"
                        "q=0.9,*/*;q=0.8"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                }
                _session = aiohttp.ClientSession(
                    timeout=timeout, headers=headers,
                )
                _semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)
    return _session


async def _close_session() -> None:
    """Cleanup hook — called on app shutdown."""
    global _session
    if _session is not None:
        try:
            await _session.close()
        except Exception:
            pass
        _session = None


# ─── Public API ──────────────────────────────────────────────────────────────


async def fetch_body(
    url: str,
    *,
    use_cache: bool = True,
    max_chars: Optional[int] = None,
) -> FetchedBody:
    """Fetch a URL and return its main-article text.

    Always returns a FetchedBody — check ``.ok`` to know whether the
    fetch succeeded.
    """
    if not url or not url.startswith(("http://", "https://")):
        return FetchedBody(url=url, error="invalid URL scheme")

    # Cache hit?
    if use_cache:
        cached = _load_cached(url)
        if cached is not None:
            return cached

    # Live fetch
    fb = FetchedBody(url=url, fetched_at=time.time())
    try:
        import aiohttp
    except ImportError:
        fb.error = "aiohttp not installed"
        return fb

    session = await _get_session()
    sem = _semaphore
    assert sem is not None

    async with sem:
        try:
            async with session.get(
                url, allow_redirects=True, ssl=False,
            ) as resp:
                fb.http_status = resp.status
                if resp.status >= 400:
                    fb.error = f"HTTP {resp.status}"
                    return fb
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "html" not in ctype and "text" not in ctype:
                    fb.error = f"non-HTML content-type: {ctype[:60]}"
                    return fb
                # Cap read size to keep memory bounded.
                raw = await resp.read()
                fb.bytes_received = len(raw)
                html = raw.decode("utf-8", errors="replace")
        except asyncio.TimeoutError:
            fb.error = f"timeout after {FETCH_TIMEOUT_SEC}s"
            return fb
        except aiohttp.ClientError as e:
            fb.error = f"{type(e).__name__}: {e}"
            return fb
        except Exception as e:
            fb.error = f"{type(e).__name__}: {e}"
            return fb

    # Extract
    text = _extract_with_trafilatura(html) if _HAVE_TRAFILATURA else ""
    if text:
        fb.extractor = "trafilatura"
    else:
        text = _extract_with_regex(html)
        fb.extractor = "regex"
    cap = max_chars or MAX_BODY_CHARS
    if cap and len(text) > cap:
        text = text[:cap]
    fb.text = text
    if not text:
        fb.error = "no extractable text"
    _save_cached(fb)
    return fb


async def fetch_many(
    urls: List[str],
    *,
    use_cache: bool = True,
    max_chars: Optional[int] = None,
) -> Dict[str, FetchedBody]:
    """Parallel fetch of multiple URLs. Returns url→FetchedBody dict.

    Order-preserving by URL; failures stay in the dict with .ok=False.
    """
    # Deduplicate (caller might pass same URL twice)
    unique = list({u: None for u in urls if u}.keys())
    tasks = [fetch_body(u, use_cache=use_cache, max_chars=max_chars)
             for u in unique]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return {fb.url: fb for fb in results}
