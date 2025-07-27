import json, os, boto3
import utils

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
    "You are a helpful life-coach / clinical assistant.\n"
    "You will receive the patient’s raw vitals data in JSON format, "
    "contained in the assistant message named 'vitals'. Each key maps to "
    "an array of numeric values recorded chronologically.\n\n"
    "When answering the user, you may compute averages, trends, or other "
    "descriptive statistics **on-the-fly** as needed, but base every "
    "statement strictly on that supplied data set. If data for a metric "
    "or period is missing, state so explicitly. Keep responses concise, "
    "clear, and grounded in the provided numbers."
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
                                                                                                                                                        
    # --- RAG: Retrieve reference material (stubbed) ---                                                                                                 
    reference_material = retrieve_reference_material(userQ, context)                                                                                     
    messages = [                                                                                                                                         
        {"role": "system", "content": SYSTEM_PROMPT},                                                                                                    
        {"role": "assistant", "name": "vitals", "content": context},                                                                                     
    ]                                                                                                                                                    
    if reference_material:                                                                                                                               
        messages.append(                                                                                                                                 
            {"role": "assistant", "name": "reference_material", "content": reference_material}                                                           
        )                                                                                                                                                
    messages.append({"role": "user", "content": userQ})                                                                                                  
                                                                                                                                                        
    payload = {                                                                                                                                          
        "messages": messages,                                                                                                                            
        "max_tokens": 2000,                                                                                                                              
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
                                                                                                                                                        
    return {                                                                                                                                             
        "statusCode": 200,                                                                                                                               
        "headers": {"Content-Type": "application/json; charset=utf-8"},                                                                                  
        "body": json.dumps({"answer": answer}, ensure_ascii=False),                                                                                      
    }  
