import json, os, boto3
import utils


# Bedrock is still required
bedrock = boto3.client("bedrock-runtime")

# ------------------------------------------------------------------ #
#  SYSTEM PROMPT                                                     #
# ------------------------------------------------------------------ #
# All personalised statements **must** be grounded solely in the
# vitals data supplied in the assistant message named "vitals".
# Never invent statistics for periods (e.g. 7-day, 30-day averages)
# unless that span is represented in the provided data.
# If data is missing, explicitly say so.
# ------------------------------------------------------------------ #
SYSTEM_PROMPT = (
    "You are a helpful clinical assistant. "
    "When referring to the patient’s personal measurements "
    "(glucose, weight, blood pressure, etc.), rely ONLY on the data "
    "contained in the assistant message named 'vitals'. "
    "Do NOT fabricate or assume additional values or time-window "
    "statistics. If information is unavailable, state that it is "
    "missing. Use correct units and keep the answer concise and clear."
)

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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "name": "vitals", "content": context},
            {"role": "user", "content": userQ},
        ],
        "max_tokens": 500,
    }
    if not MODEL_ID:                     # extra runtime safety
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "MODEL_ID not configured on Lambda"})
        }

    # -----------------------------------------------------------------
    # Invoke Bedrock and robustly extract the assistant’s reply.
    # Different foundation models return slightly different response
    # shapes, so we fall-back through several possibilities instead of
    # assuming a top-level “content” key.
    # -----------------------------------------------------------------
    resp = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(payload).encode(),
    )

    raw_body = resp["body"].read()

    try:
        body_json = json.loads(raw_body)
    except json.JSONDecodeError:
        # Model returned plain text – use it directly.
        answer = raw_body.decode("utf-8", errors="ignore")
    else:
        # Common patterns across Bedrock providers
        answer = (
            body_json.get("content")                                  # Claude/Anthropic
            or (body_json.get("message") or {}).get("content")        # Meta/DeepSeek “message”
            or (
                body_json.get("choices", [{}])[0]                     # OpenAI-style
                .get("message", {})
                .get("content")
                if body_json.get("choices") else None
            )
            or body_json.get("results", [{}])[0].get("outputText")    # AI21-style
        )

        if answer is None:  # final fallback – return entire JSON
            answer = json.dumps(body_json)

    return {"statusCode":200,
            "headers":{"Content-Type":"application/json"},
            "body":json.dumps({"answer":answer})}
