from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Optional, TypedDict, Mapping
import math
import json
import statistics as stats  # if you use the rest of your helpers

class HistoryMsg(TypedDict):
    role: str
    content: str

# --- allowed series and aliases (CSV-style labels included) ---
_ALLOWED_SERIES = {"health_age","glucose", "bp_sys", "bp_dia", "bmi", "rhr"}

_ALIASES = {
    "Health age data": "health_age",
    "Health age predicted": "health_age",
    "FBG data": "glucose",
    "FBG Predict": "glucose",
    "SYS data": "bp_sys",
    "SYS-Predict": "bp_sys",
    "DIA-data": "bp_dia",
    "DIA-Predict": "bp_dia",
    "BMI Data": "bmi",
    "BMI-Predict": "bmi",
    "RHR data": "rhr",
    "RHR-Predict": "rhr",
    "fbg":  "glucose",
    "sbp":  "bp_sys",
    "dbp":  "bp_dia",
    "bmi":  "bmi",
    "rhr":  "rhr",
}

def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

def _point(actual: Optional[float], pred: Optional[float], week_index: int) -> Optional[dict]:
    a = _safe_float(actual) if actual is not None else None
    p = _safe_float(pred)    if pred    is not None else None
    if a is None and p is None:
        return None
    return {"week": int(week_index), "value_data": a, "value_predicted": p}

def _extract_from_weekly(payload: Dict) -> Dict[str, List[dict]]:
    """
    Build per-series lists of {week, value_data, value_predicted}
    from the weekly_timepoints + forecast structure.
    """
    out: Dict[str, List[dict]] = {k: [] for k in _ALLOWED_SERIES}

    weekly = payload.get("weekly_timepoints", [])
    if isinstance(weekly, list):
        for wk in weekly:
            if not isinstance(wk, Mapping): 
                continue
            w = wk.get("week_index")
            m = wk.get("metrics", {})
            if w is None or not isinstance(m, Mapping): 
                continue

            for raw_key, val in m.items():
                canon = _ALIASES.get(raw_key, raw_key)
                if canon not in _ALLOWED_SERIES or not isinstance(val, Mapping):
                    continue
                pt = _point(val.get("actual"), val.get("pred"), w)
                if pt:
                    out[canon].append(pt)

    # Append forecast weeks (pred only)
    fcast = payload.get("forecast", [])
    if isinstance(fcast, list):
        for wk in fcast:
            if not isinstance(wk, Mapping): 
                continue
            w = wk.get("week_index")
            m = wk.get("metrics", {})
            if w is None or not isinstance(m, Mapping): 
                continue
            for raw_key, val in m.items():
                canon = _ALIASES.get(raw_key, raw_key)
                if canon not in _ALLOWED_SERIES or not isinstance(val, Mapping):
                    continue
                pt = _point(None, val.get("pred"), w)
                if pt:
                    out[canon].append(pt)

    # drop empty series
    return {k: v for k, v in out.items() if v}

def _extract_legacy_timeseries(payload: Dict) -> Dict[str, List[dict]]:
    """
    Backwards-compat path if someone still posts {"timeseries": {...}}. 
    It tries to coerce lists of numbers or [{value:..}] to {week,..} shape.
    Weeks are assigned sequentially if none provided.
    """
    raw_ts = payload.get("timeseries", {})
    out: Dict[str, List[dict]] = {}
    if not isinstance(raw_ts, Mapping):
        return out

    for key, seq in raw_ts.items():
        if not isinstance(seq, (list, tuple)):
            continue
        canon = _ALIASES.get(key, key)
        if canon not in _ALLOWED_SERIES:
            continue

        pts: List[dict] = []
        for i, item in enumerate(seq):
            # allow numeric, {"value": x}, or {"week": i, "value_data": x, "value_predicted": y}
            if isinstance(item, Mapping):
                if "week" in item or "value_data" in item or "value_predicted" in item:
                    wk = int(item.get("week", i))
                    a  = item.get("value_data")
                    p  = item.get("value_predicted")
                    rec = _point(a, p, wk)
                else:
                    rec = _point(item.get("value"), None, i)
            else:
                rec = _point(item, None, i)
            if rec:
                pts.append(rec)
        if pts:
            out[canon] = pts
    return out

def validate_payload(payload: Dict) -> Tuple[str, Dict[str, List[dict]], List[HistoryMsg]]:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    prompt = payload.get("prompt") or payload.get("question") or payload.get("query")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Missing or empty 'prompt' field.")

    # Prefer the new weekly format; fall back to legacy
    ts = _extract_from_weekly(payload)
    if not ts:
        ts = _extract_legacy_timeseries(payload)

    # History
    history_raw = payload.get("history", [])
    history: List[HistoryMsg] = []
    if isinstance(history_raw, list):
        for item in history_raw:
            if (isinstance(item, Mapping)
                and isinstance(item.get("role"), str)
                and isinstance(item.get("content"), str)):
                role = item["role"].strip().lower()
                if role in ("user", "assistant"):
                    content = item["content"].strip()
                    if content:
                        history.append({"role": role, "content": content})

    return prompt.strip(), ts, history

# ----------------- context builder -----------------

AVG_CHARS_PER_TOKEN = 4.0
def est_tokens(text: str) -> int:
    return int(len(text) / AVG_CHARS_PER_TOKEN) + 1

def trim_text_to_tokens(text: str, max_tokens: int) -> str:
    max_chars = int(max_tokens * AVG_CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_nl = cut.rfind("\n")
    if last_nl > max_chars * 0.7:
        cut = cut[:last_nl]
    return cut + "\n[...truncated for token budget...]"

def build_context_from_payload(
    _prompt: str,
    ts_dict: Dict[str, List[dict]],
    max_context_tokens: int = 900,
) -> str:
    """
    Provide compact JSON of per-series points:
    { "<series>": [ {"week": int, "value_data": float|null, "value_predicted": float|null}, ... ] }
    """
    ctx = json.dumps(ts_dict, separators=(",", ":"))
    if est_tokens(ctx) > max_context_tokens:
        ctx = trim_text_to_tokens(ctx, max_context_tokens)
    return ctx

def prepare_history_for_llm(history: List[HistoryMsg], max_tokens: int = 1200) -> List[HistoryMsg]:
    out: List[HistoryMsg] = []
    running = 0
    for msg in reversed(history):
        running += est_tokens(msg["content"])
        if running > max_tokens:
            break
        out.append(msg)
    return list(reversed(out))