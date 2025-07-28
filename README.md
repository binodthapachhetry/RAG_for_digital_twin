# Retrieval-Augmented Generation (RAG) — Implementation Prerequisite Checklist

1. **Scope & source definition**  
   - Identify the knowledge domains to search (clinical guidelines, nutrition tips, user-specific docs, etc.).
   - Decide on file formats (Markdown, PDF, HTML, JSON) and update cadence.

2. **Chunking & pre-processing pipeline**  
   - Write a batch script that:  
     1. Loads raw documents.  
     2. Converts each document to plain text.  
     3. Segments text into overlap-aware chunks (e.g., 500–1 000 chars).  
     4. Attaches metadata (title, URL, date).
   - Store results in an S3 staging bucket (or local folder for a PoC).

3. **Embedding model selection**  
   - Pick a Bedrock embedding model (e.g., `amazon.titan-embed-text-v1`) or open-source alternative.
   - Benchmark on a small corpus; confirm vector dimensionality and latency.

4. **Vector store provisioning**  
   - Choose service (OpenSearch, Pinecone, Milvus, pgvector, etc.).
   - Create index/collection with matching vector dimension + metadata fields.
   - Configure retention, replication, and encryption.

5. **Index build job**  
   - For each chunk: call embedding model, upsert `{id, vector, metadata}` into the store.
   - Automate with a one-off script and schedule (e.g., weekly AWS Batch or Step Functions).

6. **IAM & network wiring**  
   - Grant Lambda permission to call Bedrock embeddings and query the vector store.
   - OpenSearch: add VPC endpoint or public ACL + SigV4 auth.
   - Pinecone: store API key in AWS Secrets Manager.

7. **Runtime retrieval logic (inside Lambda)**  
   - Embed the concatenation of `userQ` + optionally `vitals_context`.
   - Perform top-k similarity search (`k≈3-5`) with score threshold.
   - Post-process: rank/merge, truncate to token budget, strip PII.

8. **Environmental configuration**  
   - Add env vars: `VECTOR_STORE_ENDPOINT`, `VECTOR_STORE_API_KEY`, `EMBED_MODEL_ID`, `RAG_TOP_K`, etc.
   - Update CDK stack to inject secrets.

9. **Observability & fallback**  
   - CloudWatch metrics: retrieval latency, hit/miss ratio, avg similarity score.
   - If retrieval fails, return empty string so LLM can still answer (current behaviour).

10. **Validation assets**  
    - Unit tests: mock vector store, assert deterministic top-k IDs.
    - Integration tests: end-to-end Lambda invocation with seeded corpus.
    - Load test: ensure <250 ms P95 retrieval to stay within Lambda budget.

11. **Security & compliance**  
    - Scan knowledge base for PHI; tag chunks with sensitivity level.
    - Enable encryption at rest + in transit.
    - Document data lineage for audit.

12. **Documentation & run-book**  
    - README for index build, env setup, failure recovery.
    - Architecture diagram showing flow:
      ```
      S3 → Batch → Embeddings → Vector DB → Lambda
      ```

Once the above pieces are in place, implementing `retrieve_reference_material` becomes a small wrapper: embed query → vector search → assemble passages → token-trim.
