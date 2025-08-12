# tests/test_handler.py
import os
import io
import json
import types

import handler

class DummyBedrockClient:
    def __init__(self):
        self.last_invocation = None

    def invoke_model(self, modelId, body):
        # capture inputs for assertions
        self.last_invocation = {"modelId": modelId, "body": body}
        # return a simple, Claude-style body with top-level "content"
        payload = {"content": "Hi there! (stubbed)"}
        return {"body": io.BytesIO(json.dumps(payload).encode("utf-8"))}

def test_handler_wires_history_and_vitals(monkeypatch):
    # set model id
    monkeypatch.setenv("MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
    handler.MODEL_ID = os.getenv("MODEL_ID")

    # stub bedrock client
    dummy = DummyBedrockClient()
    handler.bedrock = dummy

    # stub fetch_latest to avoid AWS calls when series missing
    monkeypatch.setattr(handler, "fetch_latest", lambda k: None)
    # stub RAG retrieval
    monkeypatch.setattr(handler, "retrieve_reference_material", lambda q, c: "")

    event = {
        "body": json.dumps({
            "query": "What do my last readings say?",
            "timeseries": {
                "glucose": [100, 105, 110],
                "systolic": [120, 122],
                "diastolic": [78, 80],
            },
            "history": [
                {"role": "user", "content": "hi how am i doing?"},
                {"role": "assistant", "content": "You are trending slightly up."}
            ]
        })
    }

    result = handler.handler(event, None)
    assert result["statusCode"] == 200
    body = json.loads(result["body"])
    assert body["answer"].startswith("Hi there!")

    # Validate the request sent to Bedrock
    sent = json.loads(dummy.last_invocation["body"].decode())
    assert sent["messages"][0]["role"] == "system"
    assert sent["messages"][1] == {
        "role": "assistant",
        "name": "vitals",
        "content": json.dumps(
            {"glucose": [100.0, 105.0, 110.0], "bp_sys": [120.0, 122.0], "bp_dia": [78.0, 80.0]},
            separators=(",", ":"),
        )
    }
    # history appears before latest user question
    assert sent["messages"][2:4] == [
        {"role": "user", "content": "hi how am i doing?"},
        {"role": "assistant", "content": "You are trending slightly up."},
    ]
    assert sent["messages"][-1]["role"] == "user"
    assert "What do my last readings say?" in sent["messages"][-1]["content"]

