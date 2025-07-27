"""
Utilities for the Time-Series -> LLM Lambda.

Responsibilities
----------------
1. Validate & coerce inbound timeseries payloads.
2. Compute lightweight descriptive stats that are cheap in tokens.
3. Generate natural-language context text (bounded length).
4. Rough token estimation & truncation helpers.

Design notes
------------
- Series represented as Python lists of numeric values (floats).
- We *do not* assume uniform sampling; stats are purely value-based.
- All functions are defensive: empty or 1-point series won't error.
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Optional
import math
import statistics as stats
# import numpy as np

# ------------------------------------------------------------------ #
# Validation / Coercion
# ------------------------------------------------------------------ #

def _coerce_val(x) -> Optional[float]:
    try:
        if x is None: 
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

def coerce_series(seq: Sequence) -> List[float]:
    """Return cleaned numeric list (drops bad entries)."""
    return [v for v in (_coerce_val(x) for x in seq) if v is not None]

def validate_payload(payload: Dict) -> Tuple[str, Dict[str, List[float]]]:
    """
    Validate top-level event body dict already JSON-decoded.

    Expected:
      {
        "prompt": "string",
        "timeseries": {
            "glucose": [...],
            "weight": [...],
            "bp_sys": [...],
            "bp_dia": [...]
        }
      }

    Returns: (prompt, cleaned_timeseries_dict)
    Raises: ValueError on required-field problems.
    """
    if "prompt" not in payload or not isinstance(payload["prompt"], str) or not payload["prompt"].strip():
        raise ValueError("Missing or empty 'prompt' field.")
    ts_in = payload.get("timeseries", {})
    if not isinstance(ts_in, dict):
        raise ValueError("'timeseries' must be an object.")
    cleaned = {k: coerce_series(v) for k, v in ts_in.items() if isinstance(v, (list, tuple))}
    # Optional: enforce max length to keep payloads small
    MAX_INPUT_POINTS = 10_000
    for k, v in cleaned.items():
        if len(v) > MAX_INPUT_POINTS:
            cleaned[k] = v[-MAX_INPUT_POINTS:]
    return payload["prompt"].strip(), cleaned

# ------------------------------------------------------------------ #
# Stats helpers (safe on short series)
# ------------------------------------------------------------------ #

def _safe_mean(x: Sequence[float]) -> Optional[float]:
    return stats.fmean(x) if len(x) else None

def _safe_std(x: Sequence[float]) -> Optional[float]:
    return stats.pstdev(x) if len(x) > 1 else None

def _delta(x: Sequence[float], n: int) -> Optional[float]:
    if len(x) < n:
        return None
    return x[-1] - x[-n]

def _pct_change(x: Sequence[float], n: int) -> Optional[float]:
    if len(x) < n or x[-n] == 0:
        return None
    return ((x[-1] - x[-n]) / abs(x[-n])) * 100.0

def _latest(x: Sequence[float]) -> Optional[float]:
    return x[-1] if len(x) else None

def describe_series(
    x: Sequence[float],
    label: str,
    short_window: int = 7,
    long_window: int = 30,
    units: str = "",
    fmt: str = ".1f",
) -> str:
    """
    Produce a concise textual bullet for one series.
    Windows are interpreted as "number of points" unless caller
    has pre-aggregated to daily values; keep caller-aware.
    """
    if not x:
        return f"No {label} data available."

    mean_short = _safe_mean(x[-short_window:]) if len(x) >= 1 else None
    mean_long  = _safe_mean(x[-long_window:])  if len(x) >= 1 else None
    latest     = _latest(x)
    delta_long = _delta(x, min(long_window, len(x)))
    pct_long   = _pct_change(x, min(long_window, len(x)))

    parts = []
    if latest is not None:
        parts.append(f"latest {label}: {latest:{fmt}}{units}")
    if mean_short is not None:
        parts.append(f"{short_window}pt avg: {mean_short:{fmt}}{units}")
    if mean_long is not None:
        parts.append(f"{long_window}pt avg: {mean_long:{fmt}}{units}")
    if delta_long is not None:
        parts.append(f"Δ{long_window}: {delta_long:+{fmt}}{units}")
    if pct_long is not None:
        parts.append(f"({pct_long:+.1f}%)")

    return " | ".join(parts)

# ------------------------------------------------------------------ #
# Domain-ish wrappers for expected vitals
# ------------------------------------------------------------------ #

def summarise_vitals(
    glucose: Optional[Sequence[float]] = None,
    weight: Optional[Sequence[float]] = None,
    bp_sys: Optional[Sequence[float]] = None,
    bp_dia: Optional[Sequence[float]] = None,
) -> str:
    """
    Build a human-readable multi-line context string for LLM.
    Only include sections that have data; order is clinically intuitive.
    """
    lines = []
    if glucose is not None:
        lines.append("Glucose: " + describe_series(glucose, "glucose", units=" mg/dL"))
    if weight is not None:
        lines.append("Weight: "  + describe_series(weight,  "weight",  units=" kg"))
    if bp_sys is not None and bp_dia is not None and len(bp_sys) and len(bp_dia):
        lines.append(_summ_bp(bp_sys, bp_dia))
    elif bp_sys is not None or bp_dia is not None:
        lines.append("Blood pressure: incomplete series supplied.")
    return "\n".join(lines) if lines else "No vitals data supplied."

def _summ_bp(sys: Sequence[float], dia: Sequence[float]) -> str:
    latest_sys, latest_dia = _latest(sys), _latest(dia)
    if latest_sys is None or latest_dia is None:
        return "Blood pressure: no data."
    # A few simple aggregates
    sys_m = _safe_mean(sys[-7:])
    dia_m = _safe_mean(dia[-7:])
    return (
        f"Blood pressure: latest {latest_sys:.0f}/{latest_dia:.0f} mmHg | "
        f"7pt avg: {sys_m:.0f}/{dia_m:.0f} mmHg" if sys_m and dia_m else
        f"Blood pressure: latest {latest_sys:.0f}/{latest_dia:.0f} mmHg"
    )

# ------------------------------------------------------------------ #
# Token estimation & truncation
# ------------------------------------------------------------------ #

AVG_CHARS_PER_TOKEN = 4.0  # crude heuristic; safe on Anthropic/GPT scale

def est_tokens(text: str) -> int:
    return int(len(text) / AVG_CHARS_PER_TOKEN) + 1

def trim_text_to_tokens(text: str, max_tokens: int) -> str:
    """
    Hard truncate text at a char boundary estimated to fit token budget.
    Prefer a soft cut at newline if possible.
    """
    max_chars = int(max_tokens * AVG_CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_nl = cut.rfind("\n")
    if last_nl > max_chars * 0.7:  # keep most of last line if near end
        cut = cut[:last_nl]
    return cut + "\n[...truncated for token budget...]"

# ------------------------------------------------------------------ #
# High-level helper used in handler
# ------------------------------------------------------------------ #

def build_context_from_payload(
    prompt: str,
    ts_dict: Dict[str, List[float]],
    max_context_tokens: int = 700,
) -> str:
    """
    Validate timeseries dict & produce bounded context string.

    Caller may *also* supply fetched backend series; merge by precedence:
    request (ts_dict) overrides backend.
    """
    g = ts_dict.get("glucose")
    w = ts_dict.get("weight")
    s = ts_dict.get("bp_sys")
    d = ts_dict.get("bp_dia")

    ctx = summarise_vitals(g, w, s, d)
    if est_tokens(ctx) > max_context_tokens:
        ctx = trim_text_to_tokens(ctx, max_context_tokens)
    return ctx
