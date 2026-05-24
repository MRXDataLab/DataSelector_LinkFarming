"""SerpAPI backend — DISABLED PLACEHOLDER.

Returns [] until ENABLE_SERPAPI=1 AND a real implementation is wired here.
The original sync function lives in legacy `services/link_farming.py` (function
`serpapi_search`) and can be lifted when the placeholder is activated.

This file exists so the rest of the pipeline can reference SerpAPI as a
first-class backend without conditional imports — flip the env flag, replace
this stub with a real impl, no other files need to change.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from ..models import QueryVertical, RawResult, TimeWindow
from .base import SearchBackend

log = logging.getLogger(__name__)

_WARNED = False


class SerpApiStubBackend(SearchBackend):
    id = "serpapi"

    def __init__(self) -> None:
        self.enabled = os.getenv("ENABLE_SERPAPI", "0") in ("1", "true", "True")
        self.key = os.getenv("SERPAPI_KEY", "")
        # Stub always reports unavailable even when env is set — protects
        # callers from accidentally relying on it before implementation.
        self.available = False

    async def search(
        self,
        query: str,
        vertical: QueryVertical = "web",
        count: int = 10,
        window: Optional[TimeWindow] = None,
    ) -> List[RawResult]:
        global _WARNED
        if self.enabled and not _WARNED:
            log.info(
                "SerpAPI is enabled in env but the backend is a placeholder. "
                "Replace serpapi_stub.SerpApiStubBackend with a real impl."
            )
            _WARNED = True
        return []
