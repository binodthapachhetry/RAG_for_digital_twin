import json, logging, uuid, time, os
from typing import Dict, List
import boto3
import utils                           # ← your helpers
from utils import validate_payload, build_context_from_payload, prepare_history_for_llm


# # --- Local testing bypass ---
# if os.getenv("LOCAL_TEST") == "1":
#     def handler(event, _):
#         return {
#             "statusCode": 200,
#             "headers": {"Content-Type": "application/json"},
#             "body": json.dumps({"answer": "[LOCAL MOCK] I parsed your timeseries and history just fine."})
#         }

 # ------------------------------------------------------------------ #                                                                                   
 #  RAG (Retrieval-Augmented Generation) Extension - Conceptual Stub  #                                                                                   
 # ------------------------------------------------------------------ #                                                                                   
 # This is a placeholder for future RAG integration. The actual                                                                                           
 # retriever module, vector DB config, and embedding logic are not                                                                                        
 # implemented here, but the handler is structured to allow easy                                                                                          
 # insertion of retrieved passages into the LLM context.                                                                                                  
                                                                                                                                                          
def retrieve_reference_material(userQ: str, vitals_context: str) -> str:
    """
    Retrieve top-N relevant knowledge base passages for the given query.

    Prerequisite setup steps (to be completed before implementing this function):
    1. Define knowledge domains and document sources (e.g., clinical guidelines, nutrition, user docs).
    2. Build a chunking & pre-processing pipeline to segment and store documents as plain text chunks with metadata.
    3. Select and benchmark an embedding model (e.g., Bedrock's amazon.titan-embed-text-v1).
    4. Provision a vector store (OpenSearch, Pinecone, etc.) with matching vector dimension and metadata fields.
    5. Implement an index build job to embed and upsert all chunks into the vector store.
    6. Configure IAM/network so Lambda can call embedding and vector store APIs.
    7. Add environment variables for endpoints, API keys, embedding model ID, and retrieval parameters.
    8. Add observability (CloudWatch metrics) and fallback logic (return "" on failure).
    9. Write validation assets: unit/integration/load tests.
    10. Ensure security/compliance: PHI scan, encryption, audit trail.
    11. Document the pipeline and provide a run-book.

    Once the above are in place, this function should:
    - Embed the concatenation of userQ (+ optionally vitals_context)
    - Query the vector store for top-k similar passages
    - Post-process: rank, merge, truncate to token budget, strip PII
    - Return the concatenated passages as a string

    For now, this is a stub.
    """
    # TODO: Implement retrieval logic after prerequisites are complete.
    return ""

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
    "You are a proactive life-coach / clinical assistant.\n\n"
    "You will see:\n"
    "• assistant/vitals – JSON of raw time-series vitals (glucose, weight, bp_sys, bp_dia), newest last.\n"
    "• optional assistant/reference_material – evidence snippets.\n"
    "• user/assistant message history – prior dialogue to maintain continuity.\n\n"
    "Ground your reply in the vitals and prior history; compute simple stats on the fly; "
    "state missing data explicitly; be concise, supportive, and numerically precise."
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

# ---------- structured logger -----------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def handler(event, _):
    req_id   = str(uuid.uuid4())
    t0       = time.time()

    try:
        incoming = json.loads(event.get("body", "{}"))
        userQ, ts_in, history_in = validate_payload(incoming)
    except ValueError as err:
        return _resp(400, {"error": str(err)})

    ts_dict = {
        "glucose": ts_in.get("glucose") or fetch_latest("glucose"),
        "weight" : ts_in.get("weight")  or fetch_latest("weight"),
        "bp_sys" : ts_in.get("bp_sys")  or fetch_latest("bp_sys"),
        "bp_dia" : ts_in.get("bp_dia")  or fetch_latest("bp_dia"),
    }

    context_json = build_context_from_payload(userQ, ts_dict)
    ref_mat = retrieve_reference_material(userQ, context_json)

    # --- assemble messages ---
    messages: List[Dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "name": "vitals", "content": context_json},
    ]
    if ref_mat:
        messages.append({"role": "assistant", "name": "reference_material", "content": ref_mat})

    # add trimmed/sanitized history BEFORE latest user question
    messages.extend(prepare_history_for_llm(history_in))

    messages.append({"role": "user", "content": userQ})

    if not MODEL_ID:
        return _resp(500, {"error": "MODEL_ID not configured on Lambda"})

    brq = {"messages": messages, "max_tokens": 2000}

    t1 = time.time()
    resp = bedrock.invoke_model(modelId=MODEL_ID, body=json.dumps(brq).encode())
    latency_ms = int((time.time() - t1) * 1000)

    # Prefer Bedrock's request id if present
    bedrock_req_id = (
        resp.get("ResponseMetadata", {}).get("RequestId")
        or resp.get("ResponseMetadata", {}).get("RequestID")
        or req_id
    )

    raw = resp["body"].read()

    answer, model_version, token_usage = _extract_answer_and_metadata(raw)

    logger.info(json.dumps({
        "req_id": bedrock_req_id,
        "model_id": MODEL_ID,
        "model_version": model_version,
        "token_usage": token_usage,
        "latency_ms": latency_ms,
        "has_glucose": bool(ts_dict.get("glucose")),
        "has_weight":  bool(ts_dict.get("weight")),
        "has_bp":      bool(ts_dict.get("bp_sys")) and bool(ts_dict.get("bp_dia")),
    }))

    return _resp(200, {
        "answer": answer,
        "model_id": MODEL_ID,
        "model_version": model_version,
        "token_usage": token_usage,
        "latency_ms": latency_ms,
        "request_id": bedrock_req_id,
    })


# ───────────────────────────────── helpers ───────────────────────────────────
def _extract_answer_and_metadata(body_bytes: bytes):
    """
    Best-effort extraction across Bedrock providers.
    Returns: (answer: str, model_version: Optional[str], token_usage: Optional[dict])
    """
    # defaults
    answer = None
    model_version = None
    token_usage = None

    # Try JSON first; fall back to plain text
    try:
        body = json.loads(body_bytes)
    except json.JSONDecodeError:
        return body_bytes.decode("utf-8", errors="ignore"), None, None

    # ----- answer (multiple common shapes) -----
    answer = (
        body.get("content")  # Anthropic-like
        or (body.get("message") or {}).get("content")  # Meta/DeepSeek-like
        or (
            body.get("choices", [{}])[0].get("message", {}).get("content")
            if body.get("choices") else None
        )  # OpenAI-like
        or body.get("results", [{}])[0].get("outputText")  # AI21-like
        or body.get("outputText")  # some Titan responses
    )

    # ----- model version (check several places) -----
    model_version = (
        body.get("modelVersion")
        or body.get("version")
        or (body.get("meta") or {}).get("model_version")
        or (body.get("message") or {}).get("model_version")
        or (body.get("model") if isinstance(body.get("model"), str) else None)
    )

    # ----- usage normalization -----
    # Common locations: top-level usage, meta.usage, message.usage, choices[0].usage
    raw_usage = (
        body.get("usage")
        or (body.get("meta") or {}).get("usage")
        or (body.get("message") or {}).get("usage")
        or (body.get("choices", [{}])[0].get("usage") if body.get("choices") else None)
    )
    if isinstance(raw_usage, dict):
        token_usage = {
            "input_tokens": (
                raw_usage.get("input_tokens")
                or raw_usage.get("prompt_tokens")
                or raw_usage.get("inputTokens")
                or raw_usage.get("promptTokens")
            ),
            "output_tokens": (
                raw_usage.get("output_tokens")
                or raw_usage.get("completion_tokens")
                or raw_usage.get("outputTokens")
                or raw_usage.get("completionTokens")
            ),
            "total_tokens": (
                raw_usage.get("total_tokens")
                or raw_usage.get("totalTokens")
            ),
        }
        # Compute total if provider didn’t supply it but parts exist
        if token_usage["total_tokens"] is None:
            it = token_usage["input_tokens"] or 0
            ot = token_usage["output_tokens"] or 0
            total = it + ot
            token_usage["total_tokens"] = total if (it or ot) else None

        # If all None, treat as absent
        if not any(v is not None for v in token_usage.values()):
            token_usage = None

    # Final fallback: return JSON if we couldn't find a content field
    if answer is None:
        answer = json.dumps(body)

    return answer, model_version, token_usage



def _resp(status: int, body: Dict):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json; charset=utf-8"},
        "body": json.dumps(body, ensure_ascii=False),
    }