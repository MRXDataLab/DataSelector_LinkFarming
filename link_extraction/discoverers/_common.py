"""Shared helpers used by multiple Discoverer implementations."""
from __future__ import annotations

from typing import Optional

from ..models import (
    ChannelId,
    DiscoveredLink,
    RawResult,
    TypedQuery,
)


def raw_result_to_link(
    raw: RawResult,
    query: TypedQuery,
    channel: ChannelId,
    *,
    backend_used: Optional[str] = None,
) -> DiscoveredLink:
    """Convert a backend-emitted `RawResult` into a `DiscoveredLink`.

    Most non-YouTube channels (Reddit fallback, Google PAA, News, Quora,
    Substack, Marketplace …) just wrap whatever Brave/DDG/headless returns.
    This kills boilerplate by centralising the field mapping.
    """
    return DiscoveredLink(
        url=raw.url,
        canonical_url=raw.url,
        title=raw.title,
        snippet=raw.snippet,
        channel=channel,
        hypothesis_id=query.hypothesis_id,
        query=query,
        observed_at=raw.observed_at,
        backend_used=backend_used or raw.backend,
    )
