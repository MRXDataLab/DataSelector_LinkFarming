"""L1 — Channel Scorer (deterministic).

Loads `channel_weights.yaml` and produces a ranked `List[ChannelFit]` for one
hypothesis. **No LLM call in this layer** — rationale enrichment is a separate
optional pass that is wired in a follow-up step (it needs the still-pending
decision on `stage_models.PRESETS` registration for the `source_selection`
stage).

Composite formula per channel
─────────────────────────────
    score = w_dim   * dimension_affinity[hyp.dimension][channel]
          + w_force * force_affinity[hyp.force_assignment].weights[channel]
          + w_sig   * signal_detectability_base[channel] * dim_affinity / 100
          + w_aud   * audience_match_default
          + w_conf  * confirmation_balance[channel] * 100
          + w_acc   * data_accessibility[channel]

`w_*` come from `priority_weights[hyp.investigation_priority]`.
Channels listed in `force_affinity[force].anti_platforms` are hard-excluded.
Channels scoring below `fit_threshold` (default 50) are dropped.
Returns `top_n` (default 5), sorted descending.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .decomposer import Decomposition
from .models import ChannelFit

log = logging.getLogger(__name__)

_WEIGHTS_PATH = Path(__file__).parent / "channel_weights.yaml"


@lru_cache(maxsize=1)
def _load_weights() -> Dict[str, Any]:
    with open(_WEIGHTS_PATH) as f:
        return yaml.safe_load(f)


def score_channels(
    hypothesis: Dict[str, Any],
    decomposition: Optional[Decomposition] = None,
    *,
    top_n: int = 5,
    fit_threshold: int = 50,
) -> List[ChannelFit]:
    """Score every channel against one hypothesis using the YAML weights.

    Args:
        hypothesis: dict shaped like host's `hypothesis_engine` output. Reads
            `dimension`, `force_assignment`, `investigation_priority`.
        decomposition: optional — reserved for audience-match refinement in a
            later step (currently unused in deterministic scoring).
        top_n: max channels to return after thresholding.
        fit_threshold: minimum composite score (0–100) for inclusion.

    Returns:
        List[ChannelFit] sorted descending by `fit_score`, capped at `top_n`.
    """
    weights = _load_weights()

    dim = hypothesis.get("dimension", "")
    force = hypothesis.get("force_assignment", "")
    priority = (hypothesis.get("investigation_priority") or "medium").lower()

    priority_weights = (
        weights["priority_weights"].get(priority)
        or weights["priority_weights"]["medium"]
    )

    # Force block — weights table + anti_platforms exclusion list
    force_block = weights["force_affinity"].get(
        force, {"weights": {}, "anti_platforms": []}
    )
    anti_platforms: set[str] = set(force_block.get("anti_platforms") or [])
    force_weights: Dict[str, float] = force_block.get("weights") or {}

    dim_affinity: Dict[str, float] = weights["dimension_affinity"].get(dim, {})

    sig_detect_base: Dict[str, float] = weights["signal_detectability_base"]
    conf_balance: Dict[str, float] = weights["confirmation_balance"]
    data_access: Dict[str, float] = weights["data_accessibility"]
    audience_default: float = float(weights.get("audience_match_default", 70))

    if not dim:
        log.warning("hypothesis missing 'dimension' — dim_affinity will be 0 across the board")
    if not force:
        log.warning("hypothesis missing 'force_assignment' — force_alignment will be 0")

    all_channels = list(data_access.keys())

    fits: List[ChannelFit] = []
    for channel in all_channels:
        if channel in anti_platforms:
            log.debug("excluding %s — anti_platform for force %r", channel, force)
            continue

        dim_score = float(dim_affinity.get(channel, 0))
        force_score = float(force_weights.get(channel, 0))
        sig_base = float(sig_detect_base.get(channel, 0))
        # Signal detectability is gated by dimension fit
        # (a high-signal channel on an off-topic dimension yields nothing).
        sig_score = sig_base * (dim_score / 100.0)
        aud_score = audience_default
        conf_score = float(conf_balance.get(channel, 0)) * 100.0
        acc_score = float(data_access.get(channel, 0))

        composite = (
            priority_weights["dimension_affinity"] * dim_score
            + priority_weights["force_alignment"] * force_score
            + priority_weights["signal_detectability"] * sig_score
            + priority_weights["audience_match"] * aud_score
            + priority_weights["confirmation_balance"] * conf_score
            + priority_weights["data_accessibility"] * acc_score
        )

        fit = ChannelFit(
            channel=channel,  # type: ignore[arg-type]
            fit_score=int(round(composite)),
            rationale="",  # filled by enrich_with_rationales (deferred)
            expected_signal="",
            sub_scores={
                "dimension_affinity": dim_score,
                "force_alignment": force_score,
                "signal_detectability": round(sig_score, 1),
                "audience_match": aud_score,
                "confirmation_balance": conf_score,
                "data_accessibility": acc_score,
            },
        )

        if fit.fit_score >= fit_threshold:
            fits.append(fit)

    fits.sort(key=lambda f: f.fit_score, reverse=True)
    return fits[:top_n]


def get_all_channel_scores(
    hypothesis: Dict[str, Any],
    decomposition: Optional[Decomposition] = None,
) -> List[ChannelFit]:
    """Return ALL channel scores (no threshold, no cap, anti_platforms still
    excluded). Useful for debugging and for surfacing "below threshold"
    channels to a user-override UI."""
    return score_channels(
        hypothesis, decomposition, top_n=999, fit_threshold=0
    )
