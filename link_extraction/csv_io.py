"""Hypothesis CSV parsing — batch upload input + per-link export helpers.

Two responsibilities:

  1. **Parse incoming CSVs** of hypotheses for batch runs.
     `parse_hypothesis_csv(text) → ParsedBatch` returns:
       - `hypotheses`: list of dicts shaped like the host's hypothesis_engine
         output (the same shape `decompose()` accepts)
       - `errors`: list of row-level issues (missing required fields,
         malformed list cells)
       - `core_problems`: dict grouping rows by `core_problem_id`

  2. **Export helpers** that the API layer reuses for single-job and
     batch result CSVs. Pure functions, no FastAPI dependency.

CSV input format (column names, all case-insensitive):

    statement                  REQUIRED — the hypothesis sentence
    hypothesis_id              OPTIONAL — auto-generated h_auto_001+ if absent
    core_problem_id            OPTIONAL — groups rows; "_uncategorized" if absent
    core_problem_statement     OPTIONAL
    dimension                  OPTIONAL — one of 10 enum values
    force_assignment           OPTIONAL — one of 5 force names
    investigation_priority     OPTIONAL — high|medium|low
    expected_signals           OPTIONAL — pipe-delimited inside cell ("a|b|c")
    expected_counter_signals   OPTIONAL — pipe-delimited inside cell
    contrarian_pair_id         OPTIONAL — references another hypothesis_id
    rationale                  OPTIONAL
    window_label               OPTIONAL — per-row time-window override
    max_triage                 OPTIONAL — per-row triage budget override

Unknown columns are silently ignored — keeps the parser tolerant of analyst
CSVs that carry extra notes/labels columns.
"""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)


# Recognised columns (case-folded). Unknown columns are ignored, not errors.
_TEXT_COLS = {
    "statement", "hypothesis_id", "core_problem_id", "core_problem_statement",
    "dimension", "force_assignment", "investigation_priority",
    "contrarian_pair_id", "rationale",
}
_LIST_COLS = {"expected_signals", "expected_counter_signals"}
_OVERRIDE_COLS = {"window_label", "max_triage"}

_REQUIRED = ("statement",)

# Delimiter inside list cells (Pipe is robust against commas in CSV cells).
_INNER_DELIM = "|"

# Auto-generated hypothesis_id pattern when the CSV doesn't supply one.
def _auto_id(idx: int) -> str:
    return f"h_auto_{idx:03d}"


# ─── Output types ────────────────────────────────────────────────────────────


@dataclass
class ParsedHypothesis:
    """One row from the CSV, normalised into the hypothesis dict shape."""

    row_index: int                      # 1-based (excludes header)
    hypothesis: Dict[str, Any]          # the dict passed to decompose()
    core_problem_id: str                # "_uncategorized" when absent
    window_label_override: Optional[str] = None
    max_triage_override: Optional[int] = None


@dataclass
class ParseError:
    row_index: int                      # 1-based, 0 means "header-level"
    message: str
    raw_row: Optional[Dict[str, str]] = None


@dataclass
class ParsedBatch:
    hypotheses: List[ParsedHypothesis] = field(default_factory=list)
    errors: List[ParseError] = field(default_factory=list)
    # core_problem_id → list of row_index values; preserves CSV ordering
    core_problems: Dict[str, List[int]] = field(default_factory=dict)
    # core_problem_id → core_problem_statement (first non-empty wins)
    core_problem_statements: Dict[str, str] = field(default_factory=dict)
    # Detected contrarian pairs (both sides present in this CSV) — for the
    # batch runner to pool queries per locked recommendation #3.
    detected_pairs: List[tuple[str, str]] = field(default_factory=list)

    @property
    def hypothesis_count(self) -> int:
        return len(self.hypotheses)

    @property
    def core_problem_count(self) -> int:
        return len(self.core_problems)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _norm_header(name: str) -> str:
    """Case-fold + strip a CSV column name."""
    return (name or "").strip().lower().replace(" ", "_")


def _parse_list_cell(raw: str) -> List[str]:
    """Split a pipe-delimited list cell. Tolerates `;` and `,` as fallbacks."""
    if not raw or not raw.strip():
        return []
    s = raw.strip()
    # Prefer pipe; fall back to other delimiters if pipe is absent
    if _INNER_DELIM in s:
        parts = s.split(_INNER_DELIM)
    elif ";" in s:
        parts = s.split(";")
    elif "," in s and not (s.startswith("[") and s.endswith("]")):
        parts = s.split(",")
    else:
        # Strip Python-list-style brackets/quotes if present
        s2 = s.strip().lstrip("[").rstrip("]")
        parts = [s2]
    return [p.strip().strip("'\"") for p in parts if p.strip()]


def _coerce_int(raw: str) -> Optional[int]:
    if not raw or not str(raw).strip():
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


# ─── Public API ──────────────────────────────────────────────────────────────


def parse_hypothesis_csv(text: str) -> ParsedBatch:
    """Parse CSV text into a ParsedBatch.

    Never raises — every problem is reported on `ParsedBatch.errors` with
    the row index. Empty/blank rows are silently skipped.
    """
    out = ParsedBatch()

    # csv.DictReader handles BOM as part of the first key; explicitly strip it
    text = text.lstrip("﻿").lstrip("ï»¿")

    try:
        reader = csv.DictReader(io.StringIO(text))
        fieldnames = reader.fieldnames or []
    except Exception as e:
        out.errors.append(ParseError(0, f"CSV parse failed: {type(e).__name__}: {e}"))
        return out

    if not fieldnames:
        out.errors.append(ParseError(0, "CSV has no header row"))
        return out

    # Header normalisation map: original-name → normalized-name
    header_map = {orig: _norm_header(orig) for orig in fieldnames}
    normalized_set = set(header_map.values())

    missing = [c for c in _REQUIRED if c not in normalized_set]
    if missing:
        out.errors.append(ParseError(
            0, f"missing required column(s): {missing} (got {sorted(normalized_set)})",
        ))
        return out

    seen_ids: set[str] = set()
    auto_idx = 0
    for raw_idx, raw_row in enumerate(reader, start=1):
        # Normalize row keys to our internal names; preserve original values
        row = {header_map.get(k, _norm_header(k)): (v or "").strip()
               for k, v in raw_row.items()}

        # Skip wholly-empty rows
        if not any(row.values()):
            continue

        statement = row.get("statement", "").strip()
        if not statement:
            out.errors.append(ParseError(
                raw_idx, "missing required 'statement'",
                raw_row={k: v for k, v in row.items() if v},
            ))
            continue

        hyp_id = row.get("hypothesis_id", "").strip()
        if not hyp_id:
            auto_idx += 1
            hyp_id = _auto_id(auto_idx)
        if hyp_id in seen_ids:
            out.errors.append(ParseError(
                raw_idx, f"duplicate hypothesis_id={hyp_id!r}",
                raw_row={k: v for k, v in row.items() if v},
            ))
            continue
        seen_ids.add(hyp_id)

        cp_id = row.get("core_problem_id", "").strip() or "_uncategorized"
        cp_stmt = row.get("core_problem_statement", "").strip()

        hypothesis = {
            "hypothesis_id": hyp_id,
            "statement": statement,
        }
        for col in (
            "dimension", "force_assignment", "investigation_priority",
            "contrarian_pair_id", "rationale",
            "core_problem_id", "core_problem_statement",
        ):
            val = row.get(col, "").strip()
            if val:
                hypothesis[col] = val
        # Ensure cp_id present even if empty in row
        hypothesis.setdefault("core_problem_id", cp_id)

        for list_col in _LIST_COLS:
            parts = _parse_list_cell(row.get(list_col, ""))
            if parts:
                hypothesis[list_col] = parts

        window_override = row.get("window_label", "").strip() or None
        triage_override = _coerce_int(row.get("max_triage", ""))

        ph = ParsedHypothesis(
            row_index=raw_idx,
            hypothesis=hypothesis,
            core_problem_id=cp_id,
            window_label_override=window_override,
            max_triage_override=triage_override,
        )
        out.hypotheses.append(ph)
        out.core_problems.setdefault(cp_id, []).append(raw_idx)
        if cp_stmt and cp_id not in out.core_problem_statements:
            out.core_problem_statements[cp_id] = cp_stmt

    # Detect MECE pairs — both sides present in this CSV
    id_to_pair = {
        h.hypothesis["hypothesis_id"]: h.hypothesis.get("contrarian_pair_id")
        for h in out.hypotheses
        if h.hypothesis.get("contrarian_pair_id")
    }
    seen_pairs: set[tuple[str, str]] = set()
    for a, b in id_to_pair.items():
        if not b or b not in {h.hypothesis["hypothesis_id"] for h in out.hypotheses}:
            continue
        pair = tuple(sorted([a, b]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        out.detected_pairs.append(pair)

    return out


def preview_summary(batch: ParsedBatch) -> Dict[str, Any]:
    """JSON-safe preview dict for the `/batch/preview` endpoint."""
    cp_list = []
    for cp_id, row_indices in batch.core_problems.items():
        cp_list.append({
            "core_problem_id": cp_id,
            "statement": batch.core_problem_statements.get(cp_id, ""),
            "hypothesis_count": len(row_indices),
            "hypothesis_ids": [
                h.hypothesis["hypothesis_id"]
                for h in batch.hypotheses
                if h.core_problem_id == cp_id
            ],
        })
    return {
        "hypothesis_count": batch.hypothesis_count,
        "core_problem_count": batch.core_problem_count,
        "core_problems": cp_list,
        "detected_pairs": [list(p) for p in batch.detected_pairs],
        "errors": [
            {"row_index": e.row_index, "message": e.message}
            for e in batch.errors
        ],
    }
