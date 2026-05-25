"""Per-job backend selection — which search engines to query, in what order.

The user-facing knob is three on/off toggles in the demo UI for the three
search backends:

  • google_free  — headless Chromium scrape (free, slow, CAPTCHA-prone)
  • brave        — Brave Search API (fast, requires API key, free 2K/mo)
  • duckduckgo   — duckduckgo-search package (free, fast, sometimes spotty)

Channel discoverers that route through `search_with_fallback()` honour the
ambient `BackendPreferences` ContextVar set by the orchestrator. Channels
that hit official APIs directly (`youtube_shorts`, `youtube`, `trends`)
ignore this — their authentication path doesn't go through the search
backends at all.

Default: all three enabled, **Google first** (per Change #2 — analyst
wants free-tier-first ordering). Brave second (cheap + reliable), DDG third
(works as belt-and-suspenders).
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import List, Optional


# Canonical backend ids. `headless_google` is the implementation; we expose
# it to the UI as "google_free" since that's the user-facing label.
ALL_BACKENDS: tuple[str, ...] = ("google_free", "brave", "duckduckgo")

# Mapping UI label → backend.id (as set in the SearchBackend subclass).
_UI_TO_BACKEND_ID = {
    "google_free": "headless_google",
    "brave": "brave",
    "duckduckgo": "duckduckgo",
}


@dataclass
class BackendPreferences:
    """Per-job backend selection. Order = priority (first = primary)."""

    enabled: List[str] = field(default_factory=lambda: list(ALL_BACKENDS))

    @property
    def priority(self) -> List[str]:
        """The active subset of ALL_BACKENDS, in user-specified order."""
        # Preserve user's order; drop anything not in ALL_BACKENDS.
        seen: set[str] = set()
        out: List[str] = []
        for name in self.enabled:
            if name in ALL_BACKENDS and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    @property
    def backend_ids_in_order(self) -> List[str]:
        """The internal `backend.id` strings in priority order."""
        return [_UI_TO_BACKEND_ID[n] for n in self.priority]

    def is_enabled(self, backend_id: str) -> bool:
        """Check by INTERNAL backend.id (e.g. 'headless_google')."""
        return backend_id in self.backend_ids_in_order

    def to_dict(self) -> dict:
        return {
            "enabled": list(self.enabled),
            "priority": self.priority,
            "backend_ids": self.backend_ids_in_order,
        }


# Default preference: all enabled, google-first (Change #2 default).
DEFAULT_PREFERENCES = BackendPreferences(enabled=list(ALL_BACKENDS))


# Ambient ContextVar — orchestrator sets this on pipeline start; the registry
# reads it on every search_with_fallback() call to determine routing.
_current_prefs_cv: contextvars.ContextVar[Optional[BackendPreferences]] = \
    contextvars.ContextVar("outtlyr_backend_prefs", default=None)


def set_current_preferences(prefs: Optional[BackendPreferences]) -> contextvars.Token:
    """Bind backend prefs to the current async context. Returns the Token
    that should be passed to `reset_current_preferences()` later."""
    return _current_prefs_cv.set(prefs)


def reset_current_preferences(token: contextvars.Token) -> None:
    _current_prefs_cv.reset(token)


def current_preferences() -> BackendPreferences:
    """Get the active BackendPreferences, falling back to default if unset."""
    p = _current_prefs_cv.get()
    return p if p is not None else DEFAULT_PREFERENCES
