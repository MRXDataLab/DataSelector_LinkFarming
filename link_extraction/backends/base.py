"""SearchBackend abstract base class.

All backends return List[RawResult]. The registry composes them into the
Brave → DDG → Headless fallback chain.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models import QueryVertical, RawResult, TimeWindow


class SearchBackend(ABC):
    """Pluggable search backend.

    Subclasses must set `id` and report `available` based on key/dependency
    presence. `search()` must never raise — return [] and log on failure so
    the fallback chain can advance.
    """

    id: str = "base"
    available: bool = False

    @abstractmethod
    async def search(
        self,
        query: str,
        vertical: QueryVertical = "web",
        count: int = 10,
        window: Optional[TimeWindow] = None,
    ) -> List[RawResult]:
        raise NotImplementedError
