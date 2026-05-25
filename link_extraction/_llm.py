"""Minimal Gemini chat-completion shim — Vertex AI only (GCP connector).

Single backend by design: every LLM call in this module goes through the
**Gemini API on Vertex AI** (`aiplatform.googleapis.com`). Auth via
`VERTEX_AI_API_KEY` (an API key for the global publisher endpoint).

No fallback. No AI Studio path. No OpenRouter. If Vertex is unreachable,
`call_llm` returns `None` and the caller falls back to its deterministic
non-LLM path (e.g. `query_synthesizer`'s rule-based slot filler, or
`triage`'s regex-only verdict).

Same public signature as before:
    call_llm(system_prompt, user_prompt, expect_json=True) -> dict | str | None

When integrated into the host repo, swap this for `services/llm_client.py`
if and only if that one ALSO speaks Vertex natively — never route through
OpenRouter (see memory note `feedback_llm_provider.md`).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

# 2.5-flash is fast, cheap, JSON-reliable; override per call via `model` kwarg.
DEFAULT_MODEL = "gemini-2.5-flash"

_VERTEX_URL = "https://aiplatform.googleapis.com/v1beta1/publishers/google/models/{model}:generateContent"

_DEFAULT_TIMEOUT = 30  # seconds


# ─── Availability ────────────────────────────────────────────────────────────


def _vertex_key() -> str:
    return os.getenv("VERTEX_AI_API_KEY", "").strip()


def is_available() -> bool:
    """True when Vertex AI credentials are configured."""
    return bool(_vertex_key())


# ─── Request construction ───────────────────────────────────────────────────


def _build_payload(
    system_prompt: str,
    user_prompt: str,
    *,
    expect_json: bool,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Build the Gemini generateContent request body for Vertex."""
    body: Dict[str, Any] = {
        "contents": [
            {"role": "user", "parts": [{"text": user_prompt}]}
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    if expect_json:
        body["generationConfig"]["responseMimeType"] = "application/json"
    return body


def _extract_text(payload: Any) -> Optional[str]:
    """Pull the assistant text out of a Gemini response. None on any shape mismatch."""
    try:
        candidates = payload.get("candidates") or []
        if not candidates:
            return None
        parts = (candidates[0].get("content") or {}).get("parts") or []
        if not parts:
            return None
        # Concatenate text parts (usually exactly one)
        text_chunks: List[str] = []
        for p in parts:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                text_chunks.append(p["text"])
        if not text_chunks:
            return None
        return "".join(text_chunks).strip()
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def _parse_json_loose(text: str) -> Optional[Any]:
    """Best-effort JSON parse: strip ```json fences if present, then json.loads."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        log.warning("Gemini returned non-JSON despite responseMimeType=json; first 200 chars: %r",
                    text[:200])
        return None


# ─── Vertex caller ──────────────────────────────────────────────────────────


def _call_vertex(
    model: str, body: Dict[str, Any], timeout: int
) -> Optional[Dict[str, Any]]:
    key = _vertex_key()
    if not key:
        return None
    url = _VERTEX_URL.format(model=model)
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": key,
    }
    try:
        r = requests.post(url, json=body, headers=headers, timeout=timeout)
        if r.status_code >= 400:
            log.info("Vertex Gemini %d: %s", r.status_code, r.text[:300])
            return None
        return r.json()
    except Exception as e:
        log.info("Vertex Gemini call failed: %s", e)
        return None


def _usage_from_response(resp: Any) -> Tuple[int, int]:
    """Pull (input_tokens, output_tokens) from a Gemini response, defensively.

    Gemini's `usageMetadata` field carries:
      - `promptTokenCount`         → input
      - `candidatesTokenCount`     → output
      - `totalTokenCount`          → sum
    Returns (0, 0) when absent — caller charges nothing in that case.
    """
    try:
        um = resp.get("usageMetadata") or {}
        return int(um.get("promptTokenCount") or 0), int(um.get("candidatesTokenCount") or 0)
    except (AttributeError, TypeError):
        return 0, 0


# ─── Public API ──────────────────────────────────────────────────────────────


def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    expect_json: bool = True,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1500,
    timeout: int = _DEFAULT_TIMEOUT,
    job_id: Optional[str] = None,
) -> Optional[Any]:
    """One-shot Vertex AI Gemini call.

    - `expect_json=True`: requests `responseMimeType=application/json` and
      returns the parsed object (None on parse failure).
    - `expect_json=False`: returns the raw assistant text (None on failure).

    `job_id` (optional): when supplied OR when the ambient cost-meter
    ContextVar is set, every call's input/output tokens are charged against
    the per-job ledger so the cost meter can enforce caps.

    Never raises. Callers must treat `None` as "LLM unavailable; use the
    deterministic fallback."
    """
    if not is_available():
        log.info("VERTEX_AI_API_KEY not set; LLM call skipped (deterministic fallback path)")
        return None

    body = _build_payload(
        system_prompt, user_prompt,
        expect_json=expect_json,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    resp = _call_vertex(model, body, timeout)
    if resp is None:
        return None

    # Charge the cost meter regardless of whether parsing succeeds —
    # tokens cost $ even if we discard the response. The job_id can be
    # passed explicitly or read from the ambient ContextVar that the
    # orchestrator set before running the pipeline.
    try:
        from .cost_meter import get_meter, current_job_id
        effective_job_id = job_id or current_job_id()
        if effective_job_id:
            in_tok, out_tok = _usage_from_response(resp)
            get_meter().charge_llm(effective_job_id, model, in_tok, out_tok)
    except Exception as e:
        log.debug("cost_meter.charge_llm skipped: %s", e)

    text = _extract_text(resp)
    if text is None:
        log.info("Vertex Gemini returned empty/unexpected payload")
        return None
    log.debug("Vertex Gemini call OK model=%s", model)

    if not expect_json:
        return text
    parsed = _parse_json_loose(text)
    if parsed is None and expect_json:
        log.info("Vertex Gemini returned text but it didn't parse as JSON")
    return parsed


# ─── Diagnostics (used by smoke test) ────────────────────────────────────────


def diagnostics() -> Dict[str, Any]:
    """Snapshot of LLM client config — Vertex AI is the only backend."""
    return {
        "backend": "vertex_ai",
        "vertex_key_set": bool(_vertex_key()),
        "default_model": DEFAULT_MODEL,
        "endpoint": _VERTEX_URL.format(model=DEFAULT_MODEL),
    }
