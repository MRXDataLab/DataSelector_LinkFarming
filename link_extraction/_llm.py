"""Minimal Gemini chat-completion shim for in-module LLM calls.

Calls Google's Gemini family directly via REST — no SDK, no OpenRouter.
Two backends, tried in order:

  1. **Gemini API on Vertex AI** (`VERTEX_AI_API_KEY`)
     Endpoint: aiplatform.googleapis.com (global publisher endpoint)
     Auth: `x-goog-api-key` header. Enterprise-grade, GCP-billed.

  2. **Gemini API in Google AI Studio** (`GEMINI_API_KEY`)
     Endpoint: generativelanguage.googleapis.com/v1beta
     Auth: `?key=` query param. Developer-grade, AI-Studio-billed.

Order is driven by `HYPOTHESIS_PROVIDER` env var:
  - `gcp_gemini` (host default)  → Vertex first, Studio fallback
  - anything else / unset        → Studio first, Vertex fallback

Same public signature as before:
    call_llm(system_prompt, user_prompt, expect_json=True) -> dict | str | None

When integrated into the host repo, swap this for `services/llm_client.py`
and register `query_synthesis` in `services/stage_models.PRESETS`.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

# Both endpoints expose the same Gemini families. 2.5-flash is fast, cheap,
# JSON-reliable; override per call via the `model` kwarg.
DEFAULT_MODEL = "gemini-2.5-flash"

_STUDIO_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_VERTEX_URL = "https://aiplatform.googleapis.com/v1beta1/publishers/google/models/{model}:generateContent"

_DEFAULT_TIMEOUT = 30  # seconds


# ─── Availability ────────────────────────────────────────────────────────────


def _vertex_key() -> str:
    return os.getenv("VERTEX_AI_API_KEY", "").strip()


def _studio_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip()


def is_available() -> bool:
    return bool(_vertex_key() or _studio_key())


def _backend_order() -> List[str]:
    """Return ordered backend list based on HYPOTHESIS_PROVIDER hint."""
    prov = os.getenv("HYPOTHESIS_PROVIDER", "").strip().lower()
    prefer_vertex = prov == "gcp_gemini"
    if prefer_vertex:
        return ["vertex", "studio"]
    return ["studio", "vertex"]


# ─── Request construction ───────────────────────────────────────────────────


def _build_payload(
    system_prompt: str,
    user_prompt: str,
    *,
    expect_json: bool,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Build the shared Gemini generateContent request body.

    Same shape works on both Studio and Vertex endpoints.
    """
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


# ─── Backend callers ─────────────────────────────────────────────────────────


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


def _call_studio(
    model: str, body: Dict[str, Any], timeout: int
) -> Optional[Dict[str, Any]]:
    key = _studio_key()
    if not key:
        return None
    url = _STUDIO_URL.format(model=model) + f"?key={key}"
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(url, json=body, headers=headers, timeout=timeout)
        if r.status_code >= 400:
            log.info("Gemini Studio %d: %s", r.status_code, r.text[:300])
            return None
        return r.json()
    except Exception as e:
        log.info("Gemini Studio call failed: %s", e)
        return None


_CALLERS = {"vertex": _call_vertex, "studio": _call_studio}


# ─── Public API ──────────────────────────────────────────────────────────────


def _usage_from_response(resp: Any) -> tuple[int, int]:
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
    """One-shot Gemini call. Tries Vertex then Studio (or vice versa).

    - `expect_json=True`: requests `responseMimeType=application/json` and
      returns the parsed object (None on parse failure).
    - `expect_json=False`: returns the raw assistant text (None on failure).

    `job_id` (optional, Step 14): when supplied, every call is charged
    against the cost_meter under that job. Caller is responsible for
    checking `cost_meter.would_exceed()` BEFORE invoking — this function
    only records what it spent, it doesn't pre-empt.

    Never raises. Callers must treat `None` as "LLM unavailable; use the
    deterministic fallback."
    """
    if not is_available():
        log.info("No Gemini API key set (GEMINI_API_KEY / VERTEX_AI_API_KEY); LLM call skipped")
        return None

    body = _build_payload(
        system_prompt, user_prompt,
        expect_json=expect_json,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    last_text: Optional[str] = None
    for backend in _backend_order():
        caller = _CALLERS[backend]
        resp = caller(model, body, timeout)
        if resp is None:
            continue
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
            log.info("Gemini %s returned empty/unexpected payload", backend)
            continue
        last_text = text
        log.debug("Gemini call served by backend=%s model=%s", backend, model)
        if not expect_json:
            return text
        parsed = _parse_json_loose(text)
        if parsed is not None:
            return parsed
        # JSON parse failed — try next backend in case the model on the
        # other endpoint behaves better. Don't fail silently across both.

    if last_text and expect_json:
        log.info("All Gemini backends returned text but none parsed as JSON")
    return None


# ─── Diagnostics (used by smoke test) ────────────────────────────────────────


def diagnostics() -> Dict[str, Any]:
    """Snapshot of which keys are present and which backend would be tried first."""
    return {
        "vertex_key_set": bool(_vertex_key()),
        "studio_key_set": bool(_studio_key()),
        "hypothesis_provider": os.getenv("HYPOTHESIS_PROVIDER", "").strip() or None,
        "backend_order": _backend_order(),
        "default_model": DEFAULT_MODEL,
    }
