# tests/test_utils.py
import json
import string
import random
import utils

def test_validate_payload_nested_timeseries_and_aliases():
    payload = {
        "query": "How am I doing?",
        "timeseries": {
            "glucose": [100, "101", {"timestamp": "t", "value": 102}],
            "weight": [180.2, None, "181.4"],
            "systolic": [120, 118],    # alias → bp_sys
            "diastolic": [80, 77],     # alias → bp_dia
            "ignored_series": [1, 2, 3]
        },
        "history": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "nope"}  # should be ignored
        ]
    }
    prompt, ts, history = utils.validate_payload(payload)

    assert prompt == "How am I doing?"
    assert set(ts.keys()) == {"glucose", "weight", "bp_sys", "bp_dia"}
    assert ts["glucose"] == [100.0, 101.0, 102.0]
    assert ts["weight"] == [180.2, 181.4]
    assert ts["bp_sys"] == [120.0, 118.0]
    assert ts["bp_dia"] == [80.0, 77.0]

    # history keeps only user/assistant roles, strips system/tool
    assert history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_validate_payload_flat_structure_and_truncation():
    # make a long series to trigger truncation to last 10_000 points
    long_series = list(range(10_050))  # 0..10049
    payload = {
        "prompt": "Check please",
        "glucose": long_series,
    }
    prompt, ts, _ = utils.validate_payload(payload)
    assert prompt == "Check please"
    assert "glucose" in ts
    assert len(ts["glucose"]) == 10_000
    # last element preserved
    assert ts["glucose"][-1] == 10_049


def test_validate_payload_requires_prompt():
    for bad in [{}, {"timeseries": {}}, {"prompt": "   "}]:
        try:
            utils.validate_payload(bad)
            assert False, "Expected ValueError for missing/blank prompt"
        except ValueError as e:
            assert "Missing or empty 'prompt'" in str(e)


def test_prepare_history_for_llm_trims_from_oldest():
    # make messages with known content lengths
    # avg token heuristic ~4 chars → target trim threshold easily
    hist = [
        {"role": "user", "content": "a" * 1000},
        {"role": "assistant", "content": "b" * 1000},
        {"role": "user", "content": "c" * 1000},
    ]
    # max_tokens small so only the last two survive
    trimmed = utils.prepare_history_for_llm(hist, max_tokens=400)  # ~1600 chars
    assert trimmed == hist[1:]  # keeps most recent first two turns


def test_build_context_from_payload_compact_json():
    ts = {"glucose": [100.0, 101.0], "bp_sys": [120.0], "bp_dia": [80.0]}
    ctx = utils.build_context_from_payload("ignored", ts, max_context_tokens=9999)
    # should be compact JSON without spaces
    assert ctx == json.dumps(ts, separators=(",", ":"))
