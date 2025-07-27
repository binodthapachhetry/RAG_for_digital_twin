from src import handler
import json, importlib, pytest
import io

@pytest.fixture(autouse=True)
def reload():
    importlib.reload(handler)

def test_stub(monkeypatch):
    monkeypatch.setattr(handler, "fetch_latest",
                        lambda *_: [100.0]*10)
    monkeypatch.setattr(handler.bedrock, "invoke_model",
        lambda **_: {"body":io.BytesIO(
            json.dumps({"content":"OK"}).encode())})
    evt = {"body": json.dumps({"prompt": "Hello", "timeseries": {}})}
    out = handler.handler(evt, None)
    assert json.loads(out["body"])["answer"] == "OK"
