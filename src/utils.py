from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Optional, TypedDict
from typing import Mapping  # ← add this
import json
import math
import statistics as stats

class HistoryMsg(TypedDict):
    role: str
    content: str

# canonical series whitelist (clarity)
_ALLOWED_SERIES = {"glucose", "weight", "bp_sys", "bp_dia"}

_ALIASES = {"systolic": "bp_sys", "diastolic": "bp_dia"}

def _coerce_val(x) -> Optional[float]:
    if isinstance(x, dict):
        x = x.get("value")
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
    return [v for v in (_coerce_val(x) for x in seq) if v is not None]

def validate_payload(payload: Dict) -> Tuple[str, Dict[str, List[float]], List[HistoryMsg]]:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    prompt = payload.get("prompt") or payload.get("question") or payload.get("query")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Missing or empty 'prompt' field.")

    # 1) collect raw series (flat + nested "timeseries")
    raw_ts: Dict[str, Sequence] = {}
    raw_ts.update({k: v for k, v in payload.items()
                   if k not in ("prompt", "question", "query", "timeseries", "history")})
    nested = payload.get("timeseries", {})
    if isinstance(nested, dict):
        raw_ts.update(nested)

    # 2) normalize keys + coerce
    cleaned: Dict[str, List[float]] = {}
    for key, seq in raw_ts.items():
        if not isinstance(seq, (list, tuple)):
            continue
        canon = _ALIASES.get(key, key)
        if canon not in _ALLOWED_SERIES:
            continue
        cleaned[canon] = coerce_series(seq)

    # 3) cap length defensively
    MAX_INPUT_POINTS = 10_000
    for k, v in list(cleaned.items()):
        if len(v) > MAX_INPUT_POINTS:
            cleaned[k] = v[-MAX_INPUT_POINTS:]

    # 4) history (validate, minimal sanitation)
    history_raw = payload.get("history", [])
    history: List[HistoryMsg] = []
    if isinstance(history_raw, list):
        for item in history_raw:
            if (isinstance(item, Mapping) 
                & isinstance(item.get("role"), str) 
                & isinstance(item.get("content"), str)):
                role = item["role"].strip().lower()
                if role not in ("user", "assistant"):   # ignore system/tool
                    continue
                content = item["content"].strip()
                if content:
                    history.append({"role": role, "content": content})

    return prompt.strip(), cleaned, history

# --- everything below unchanged, but adding a helper to trim history tokens ---

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

def prepare_history_for_llm(history: List[HistoryMsg],
                            max_tokens: int = 1200) -> List[HistoryMsg]:
    """
    Cheap token-aware trimming of history (keep most recent turns).
    """
    out: List[HistoryMsg] = []
    running = 0
    # walk from newest to oldest, then reverse
    for msg in reversed(history):
        running += est_tokens(msg["content"])
        if running > max_tokens:
            break
        out.append(msg)
    return list(reversed(out))

def build_context_from_payload(
    _prompt: str,
    ts_dict: Dict[str, List[float]],
    max_context_tokens: int = 700,
) -> str:
    """
    Return **raw vitals JSON** (not pre-aggregated). The LLM is now
    responsible for any descriptive statistics.
    """
    ctx = json.dumps(ts_dict, separators=(",", ":"))  # compact JSON
    if est_tokens(ctx) > max_context_tokens:
        ctx = trim_text_to_tokens(ctx, max_context_tokens)
    return ctx