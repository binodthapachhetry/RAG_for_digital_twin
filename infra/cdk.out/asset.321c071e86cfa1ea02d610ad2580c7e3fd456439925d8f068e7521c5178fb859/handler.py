import json, os, boto3
import utils


# Bedrock is still required
bedrock = boto3.client("bedrock-runtime")

# ── TEMPORARILY DISABLE AWS TIMESTREAM ────────────────────────────────
# Replace the Timestream client with a stub that always returns an
# empty result set, preventing any network calls or data access.
def _noop(*_, **__):
    return {"Rows": []}

ts_q = type("NoTimestreamClient", (), {"query": staticmethod(_noop)})()

DB  = os.getenv("DB_NAME", "")
TBL = os.getenv("TABLE", "")
MODEL_ID = os.getenv("MODEL_ID")

def fetch_latest(series: str, hours: int = 168):  # pylint: disable=unused-argument
    """Timestream access disabled: return no data."""
    return []


def handler(event, _):
    try:
        payload = json.loads(event.get("body", "{}"))
        userQ, ts_in = utils.validate_payload(payload)
    except ValueError as err:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(err)})
        }

    ts_dict = {
        "glucose": ts_in.get("glucose") or fetch_latest("glucose"),
        "weight" : ts_in.get("weight")  or fetch_latest("weight"),
        "bp_sys" : ts_in.get("bp_sys")  or fetch_latest("bp_sys"),
        "bp_dia" : ts_in.get("bp_dia")  or fetch_latest("bp_dia"),
    }

    context = utils.build_context_from_payload(userQ, ts_dict)
    payload = {
        "messages": [
            {"role": "system",
             "content": "You are a helpful clinical assistant."},
            {"role": "assistant", "name": "vitals", "content": context},
            {"role": "user", "content": userQ},
        ],
        "max_tokens": 400,
    }
    if not MODEL_ID:                     # extra runtime safety
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "MODEL_ID not configured on Lambda"})
        }

    resp = bedrock.invoke_model(modelId=MODEL_ID,
                                body=json.dumps(payload).encode())
    answer = json.loads(resp['body'].read())['content']
    return {"statusCode":200,
            "headers":{"Content-Type":"application/json"},
            "body":json.dumps({"answer":answer})}
