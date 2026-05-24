"""Discoverer abstract base classes.

`Discoverer` is the single contract every channel implementation honours:
take one `TypedQuery` + `TimeWindow`, return a list of `DiscoveredLink`
(or a subclass like `ShortVideoLink`). Never raise; return `[]` on error
and log — the orchestrator (Step 8) treats `[]` as "channel was tried and
yielded nothing," and moves on.

`ShortVideoDiscoverer` is a thin marker subclass that promises
`List[ShortVideoLink]` instead of plain `DiscoveredLink`. YouTube Shorts,
TikTok, and (later) Instagram Reels all inherit from it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from ..models import (
    ChannelId,
    DiscoveredLink,
    ShortVideoLink,
    TimeWindow,
    TypedQuery,
)


class Discoverer(ABC):
    """Pluggable per-channel discoverer.

    Subclasses must set `channel_id` and report `available` based on
    key/dependency presence. `discover()` must never raise — return [] on
    any failure so the orchestrator can advance.
    """

    channel_id: ChannelId
    available: bool = False

    @abstractmethod
    async def discover(
        self,
        query: TypedQuery,
        window: TimeWindow,
        count: int = 10,
    ) -> List[DiscoveredLink]:
        raise NotImplementedError

    async def expand(
        self,
        seed: DiscoveredLink,
        depth: int = 1,
    ) -> List[DiscoveredLink]:
        """Walk the channel's native related-graph from a seed link.

        Default implementation returns []. Channels that expose a related
        graph (PAA, relatedToVideoId, crossposts, sidebar) override this.
        """
        return []

    async def batch_discover(
        self,
        queries: Sequence[TypedQuery],
        window: TimeWindow,
        count_per_query: int = 10,
        max_total: int = 30,
    ) -> List[DiscoveredLink]:
        """Run `discover()` for every query and dedup by URL.

        Concrete subclasses can override for native batch endpoints, but the
        default behaviour suits any channel without special multi-query
        support. Caps at `max_total` after dedup.
        """
        seen: set[str] = set()
        out: List[DiscoveredLink] = []
        for q in queries:
            if len(out) >= max_total:
                break
            results = await self.discover(q, window, count=count_per_query)
            for r in results:
                key = (r.canonical_url or r.url).strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(r)
                if len(out) >= max_total:
                    break
        return out


class ShortVideoDiscoverer(Discoverer):
    """Marker class for channels that return `ShortVideoLink`."""

    @abstractmethod
    async def discover(
        self,
        query: TypedQuery,
        window: TimeWindow,
        count: int = 10,
    ) -> List[ShortVideoLink]:  # type: ignore[override]
        raise NotImplementedError
