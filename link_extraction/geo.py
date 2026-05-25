"""Per-link geo scope tagger — India vs Rest-of-World vs unknown.

Every `DiscoveredLink` carries a `geo_scope` field set by `tag_geo_scope()`.
The classifier is host-pattern based (no LLM, no network):

  • **india** — ccTLDs (.in, .co.in, .org.in, .gov.in, .ac.in) OR a curated
    list of India-specific consumer/news/ecom hosts OR a `reddit.com/r/...`
    URL whose subreddit name is India-flavoured (r/india, r/Mumbai, etc.)
  • **row** — anything else with a recognisable non-India host
  • **unknown** — URL absent / malformed / TikTok discover-page (no host
    that maps to a country)

The classifier is deliberately CONSERVATIVE about claiming "india" — it
only fires when there's a strong host-level signal. False negatives are
acceptable; false positives confuse analysts.

The orchestrator calls `tag_links(links, decomp)` once after L3 discovery,
before L5 dedup. The hypothesis's `geo_hints` are a hint but NOT the
authority — a Kellogg's-India hypothesis can still discover ROW evidence
(e.g. competitor Substack from the US) that's relevant to triage.
"""
from __future__ import annotations

import contextvars
import logging
import re
from typing import Iterable, List, Optional, Sequence
from urllib.parse import urlparse

from .models import DiscoveredLink

log = logging.getLogger(__name__)


# ─── Geo-hint ContextVar ─────────────────────────────────────────────────────
# The orchestrator sets this from the hypothesis's decomposed geo_hints before
# the L3 discovery stage runs. Search backends (currently Brave) can read it
# to bias their country parameter and surface more region-appropriate results.

_geo_hints_cv: contextvars.ContextVar[Optional[List[str]]] = \
    contextvars.ContextVar("outtlyr_geo_hints", default=None)


def set_current_geo_hints(hints: Optional[List[str]]) -> contextvars.Token:
    """Bind the hypothesis's geo_hints to the current async context."""
    return _geo_hints_cv.set(list(hints) if hints else None)


def reset_current_geo_hints(token: contextvars.Token) -> None:
    _geo_hints_cv.reset(token)


def current_geo_hints() -> List[str]:
    return _geo_hints_cv.get() or []


# India ccTLDs — any URL with one of these ends is unambiguously India.
_INDIA_TLDS: tuple[str, ...] = (
    ".in", ".co.in", ".org.in", ".gov.in", ".ac.in", ".net.in",
    ".firm.in", ".gen.in", ".ind.in",
)

# India-specific consumer/news/ecom/quick-commerce hosts (without leading .).
# Conservative list — only sites whose primary audience is Indian.
_INDIA_HOSTS: frozenset[str] = frozenset({
    # News
    "ndtv.com", "indianexpress.com", "hindustantimes.com",
    "timesofindia.indiatimes.com", "economictimes.indiatimes.com",
    "thehindu.com", "thehindubusinessline.com", "livemint.com",
    "moneycontrol.com", "business-standard.com", "businesstoday.in",
    "news18.com", "firstpost.com", "indiatvnews.com", "republicworld.com",
    "thewire.in", "scroll.in", "thequint.com", "theprint.in",
    "outlookindia.com", "rediff.com", "mid-day.com",
    # Ecommerce
    "flipkart.com", "myntra.com", "nykaa.com", "ajio.com",
    "tatacliq.com", "jiomart.com", "croma.com", "snapdeal.com",
    "paytmmall.com", "meesho.com", "firstcry.com", "lenskart.com",
    "shopclues.com", "limeroad.com", "purplle.com", "pepperfry.com",
    "urbanic.com", "boat-lifestyle.com",
    # Quick commerce / hyperlocal
    "zepto.in", "blinkit.com", "swiggy.com", "instamart.com",
    "dunzo.com", "bigbasket.com", "zomato.com",
    # Streaming / entertainment
    "hotstar.com", "jiocinema.com", "sonyliv.com", "voot.com",
    "altbalaji.com", "zee5.com", "mxplayer.in",
    # Other consumer-tier sites that index well for India queries
    "magicbricks.com", "99acres.com", "naukri.com", "bookmyshow.com",
    "makemytrip.com", "yatra.com", "ixigo.com", "irctc.co.in",
    "policybazaar.com", "paisabazaar.com", "bankbazaar.com",
    "1mg.com", "pharmeasy.in", "netmeds.com", "apollopharmacy.in",
    "byjus.com", "unacademy.com", "vedantu.com",
})

# Subreddit names that strongly imply India audience (lowercased).
_INDIA_SUBREDDITS: frozenset[str] = frozenset({
    "india", "indiaspeaks", "askindia", "indianbusiness", "indianeconomy",
    "indianfood", "indiansocial", "indianpets", "indianmarketing",
    "bangalore", "mumbai", "delhi", "chennai", "hyderabad", "pune",
    "kolkata", "ahmedabad", "kerala", "tamilnadu", "karnataka",
    "maharashtra", "punjab", "westbengal", "rajasthan", "telangana",
    "indiandev", "indianfinance", "indianstreetbets", "indiandiscussion",
})

# Quora questions about India often have these tokens in the slug.
_INDIA_QUORA_TOKENS: tuple[str, ...] = (
    "india", "indian", "bharat", "hindi",
)

_REDDIT_SUBREDDIT_RE = re.compile(
    r"/r/([A-Za-z0-9_]+)", re.IGNORECASE,
)


def _host_of(url: str) -> Optional[str]:
    if not url:
        return None
    try:
        netloc = urlparse(url).hostname
    except Exception:
        return None
    if not netloc:
        return None
    h = netloc.lower()
    return h[4:] if h.startswith("www.") else h


def _path_of(url: str) -> str:
    if not url:
        return ""
    try:
        return (urlparse(url).path or "").lower()
    except Exception:
        return ""


def _has_india_tld(host: str) -> bool:
    return any(host.endswith(tld) for tld in _INDIA_TLDS)


def _is_india_reddit(host: str, path: str) -> bool:
    """Reddit URL pointing at an India-flavoured subreddit."""
    if "reddit.com" not in host:
        return False
    m = _REDDIT_SUBREDDIT_RE.search(path)
    if not m:
        return False
    return m.group(1).lower() in _INDIA_SUBREDDITS


def _is_india_quora(host: str, path: str) -> bool:
    if "quora.com" not in host:
        return False
    p = path.lower()
    return any(tok in p for tok in _INDIA_QUORA_TOKENS)


def classify_geo(url: str) -> str:
    """Return one of 'india', 'row', 'unknown' for a URL.

    Pure function — call it directly from tests.
    """
    host = _host_of(url)
    if not host:
        return "unknown"
    if _has_india_tld(host):
        return "india"
    if host in _INDIA_HOSTS:
        return "india"
    # Allow subdomain matches: any host ending in `.<india_host>` (e.g.
    # `m.flipkart.com`) — but only when the suffix is exactly the India host.
    for ih in _INDIA_HOSTS:
        if host.endswith("." + ih):
            return "india"
    path = _path_of(url)
    if _is_india_reddit(host, path):
        return "india"
    if _is_india_quora(host, path):
        return "india"
    return "row"


def tag_link(link: DiscoveredLink) -> str:
    """Set `link.geo_scope` based on its URL. Returns the verdict assigned."""
    scope = classify_geo(link.canonical_url or link.url)
    link.geo_scope = scope
    return scope


def tag_links(links: Iterable[DiscoveredLink]) -> dict[str, int]:
    """Tag every link in place. Returns counts per scope for logging."""
    counts = {"india": 0, "row": 0, "unknown": 0}
    for lk in links:
        scope = tag_link(lk)
        counts[scope] = counts.get(scope, 0) + 1
    return counts
