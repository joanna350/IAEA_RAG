# IAEA Nuclear Document RAG Pipeline

RAG (Retrieval-Augmented Generation) pipeline built for IAEA nuclear safety and SMR licensing documents.

## Architecture

```
data/ (IAEA docs)
  └─ load → chunk → quality_check (+ near-dup) → embed → Qdrant collection
                                                       └─ BM25 index
query
  └─ hybrid_search (vector + BM25) → rerank (cross-encoder) → LLM → answer + metrics
                                                                        └─ JSONL log → offline RAGAS eval
```

## Features

- **Hybrid retrieval**: BM25 (sparse) + Qdrant vector search (dense, cosine similarity) with score fusion
- **Cross-encoder reranking**: `bge-reranker-base` re-scores the top fused candidates by reading query and chunk jointly, before the final top-k goes to the LLM
- **Data quality pipeline**: length filtering, boilerplate removal, exact-duplicate (MD5) and near-duplicate (embedding cosine similarity) detection, low-information-density filtering — runs on every `ingest()`/`load()`, not just the demo script
- **Monitoring**: per-query retrieval/LLM/total latency, token usage, cost estimation, retrieval scores/methods, and the full answer + retrieved contexts, all appended to a JSONL log
- **Offline RAGAS evaluation**: `scripts/evaluate_ragas.py` batch-scores logged queries for faithfulness and answer relevancy, outside the request path
- **FastAPI server**: REST endpoints for ingestion, querying, and streaming
- **Qdrant vector store**: a real service (not a local file), so the index survives container restarts and works from a stateful Kubernetes deployment — unlike FAISS's local-file index
- **Docker + docker-compose**: containerized deployment, API + Qdrant wired together
- **Kubernetes manifests**: Namespace, Qdrant (PVC + Deployment + Service), API (Deployment + Service), Secrets — verified on a local kind cluster
- **Admin API**: document upload/delete, reindex, and a stats dashboard, behind a shared-secret key

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start Qdrant (retrieval needs a running instance — see Docker section below)
docker compose up -d qdrant

# Run offline demo (no API key needed)
python scripts/demo.py

# Build index and start API server
python -c "from src.pipeline import IAEARagPipeline; p=IAEARagPipeline(); p.ingest()"
uvicorn src.api:app --reload

# Query via API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the passive safety requirements for SMR?"}'

# Same query, streamed as SSE
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the passive safety requirements for SMR?"}'

# Batch-score logged queries for faithfulness / answer relevancy
python scripts/evaluate_ragas.py --output logs/ragas_report.json
```

## Docker

The API needs Qdrant reachable at `QDRANT_URL` (default `http://localhost:6333`) — run both together with docker-compose rather than the API container alone:

```bash
# API + Qdrant, wired together (recommended)
docker compose up -d

# Just the API image, against an already-running Qdrant elsewhere
docker build -t iaea-rag .
docker run -e GROQ_API_KEY=sk-... -e QDRANT_URL=http://host.docker.internal:6333 -p 8000:8000 iaea-rag
```

## Kubernetes

Manifests in `k8s/`: `Namespace`, Qdrant (`PersistentVolumeClaim` + `Deployment` + `Service`), API (`Secret` template + `Deployment` + `Service`). No Ingress yet — access via `kubectl port-forward` below, or add one if the cluster has an ingress controller.

```bash
# Build the image the Deployment expects (must be named iaea-rag-api:latest)
docker compose build api

# Using kind for local testing — load the image in directly, no registry needed
kind create cluster --name iaea-rag
kind load docker-image iaea-rag-api:latest --name iaea-rag

kubectl apply -f k8s/00-namespace.yaml -f k8s/01-qdrant-pvc.yaml -f k8s/02-qdrant-deployment.yaml -f k8s/03-qdrant-service.yaml

# Secrets are created imperatively, not from the checked-in templates (see k8s/04-api-secret.example.yaml)
kubectl create secret generic groq-api-key -n iaea-rag --from-literal=GROQ_API_KEY=sk-...
kubectl create secret generic admin-api-key -n iaea-rag --from-literal=ADMIN_API_KEY=...

kubectl apply -f k8s/05-api-deployment.yaml -f k8s/06-api-service.yaml

kubectl get pods -n iaea-rag
kubectl port-forward -n iaea-rag svc/api 8080:8000
curl -X POST http://localhost:8080/query -H "Content-Type: application/json" -d '{"question": "..."}'
```

Verified: pods reach Ready, `/health`/`/query`/`/query/stream` work through port-forward, and — the actual point of moving off FAISS — deleting the Qdrant pod (`kubectl delete pod -n iaea-rag -l app=qdrant`) and letting it reschedule does **not** lose the indexed data, since it's backed by the PVC rather than the pod's own filesystem. Also verified with the API scaled to 2 replicas (`kubectl scale deployment/api -n iaea-rag --replicas=2`): both pods return identical `retrieval_scores` for the same query, including a pod that never ran `/ingest` itself — confirming they share Qdrant rather than each holding their own copy, which is the actual reason FAISS's local-file index doesn't work here.

Caveats:
- The API image doesn't bake in the embedding/reranker models, so every pod start re-downloads them from Hugging Face Hub (why the readiness probe's `initialDelaySeconds` is generous). Worth fixing by baking the model cache into the image or a shared volume before this goes anywhere near a real deployment.
- `logs/` has no PVC in these manifests (unlike docker-compose, which bind-mounts it), so `/admin/stats` resets on every pod restart in Kubernetes. Would need the same PVC treatment as Qdrant's storage to persist across restarts.
- Only Qdrant should ever run with `replicas: 1` here — its PVC is `ReadWriteOnce`, so scaling it would try to attach the same volume to two pods at once.

## Admin API

All `/admin/*` routes require a shared secret, either header (`X-Admin-Key: ...`) or query param (`?key=...`), checked against the `ADMIN_API_KEY` env var. If that env var isn't set at all, every admin route returns `503` rather than being open by default.

```bash
export ADMIN_API_KEY=...   # must match what the server was started with

curl http://localhost:8000/admin/documents -H "X-Admin-Key: $ADMIN_API_KEY"
curl -X POST http://localhost:8000/admin/documents -H "X-Admin-Key: $ADMIN_API_KEY" -F "file=@new_doc.txt"
curl -X DELETE http://localhost:8000/admin/documents/new_doc.txt -H "X-Admin-Key: $ADMIN_API_KEY"
curl -X POST http://localhost:8000/admin/reindex -H "X-Admin-Key: $ADMIN_API_KEY"
curl http://localhost:8000/admin/stats -H "X-Admin-Key: $ADMIN_API_KEY"

# Dashboard is a plain browser GET, so it takes the key as a query param instead of a header
open "http://localhost:8000/admin/dashboard?key=$ADMIN_API_KEY"
```

Uploading or deleting a document doesn't reindex automatically — call `/admin/reindex` afterward. This is not production-grade auth (no rate limiting, and a key in `?key=` can end up in access logs) — it's a minimal gate appropriate for a trusted network, not a public endpoint.

## Project Structure

```
iaea-rag/
├── data/                        # IAEA source documents (.txt / .pdf)
│   ├── iaea_safety_fundamentals.txt
│   ├── iaea_smr_design_safety.txt
│   └── iaea_nuclear_licensing_process.txt
├── src/
│   ├── pipeline.py              # Core RAG pipeline (load→chunk→embed→retrieve→generate)
│   ├── api.py                   # FastAPI server
│   ├── admin.py                 # Document management + log aggregation for /admin/*
│   ├── data_quality.py          # Chunk validation & quality checks
│   └── monitoring.py            # Latency, cost, retrieval metrics
├── scripts/
│   ├── demo.py                  # Offline demo (BM25 only, no API key required)
│   └── evaluate_ragas.py        # Batch RAGAS eval (faithfulness, answer relevancy)
├── logs/                        # Auto-created, gitignored — query_log.jsonl
├── k8s/                          # Namespace, Qdrant + API (Deployment/Service/PVC/Secret template)
├── Dockerfile
├── docker-compose.yml            # API + Qdrant, wired together
├── requirements.txt
└── .env
```

Qdrant's data lives in a Docker volume (`qdrant_data`), not a repo directory — nothing to gitignore for it, unlike the old `faiss_index/`.

## Chunking Strategy

Uses `RecursiveCharacterTextSplitter` with paragraph → sentence → word boundary hierarchy.
- chunk_size: 512 tokens
- chunk_overlap: 64 tokens

This preserves section context while keeping chunks small enough for precise retrieval.

## Data Quality

`validate_chunks()` runs five checks, in order, before a chunk is indexed — this happens at ingestion time, before anything reaches the vector store or BM25 index:

1. **too_short** — under 80 characters
2. **boilerplate** — page numbers, digit-only lines, separator lines
3. **duplicate** — exact match via MD5 hash
4. **near_duplicate** — embedding cosine similarity ≥ 0.95 (opt-in; only runs when an embeddings model is passed in). O(n²) pairwise comparison, fine at the current corpus scale — would need an approximate index (e.g. Qdrant's own ANN index, or LSH) in the tens-of-thousands-of-chunks range.
5. **low_info_density** — under 50% alphanumeric/whitespace characters

A per-reason rejection count prints on every `ingest()`/`load()`.

## Hybrid Search

Retrieval, at query time, happens in two sequential stages — not two alternative mechanisms. Hybrid search casts a wide, cheap net; reranking then narrows it with a slower, more precise model.

**Stage 1 — score fusion:**
```
final_score = 0.7 * vector_score + 0.3 * bm25_score
```

Vector search (Qdrant, cosine similarity) handles semantic similarity; BM25 handles exact keyword matching (e.g. regulation codes like "GSR Part 1", "LOCA", "FSAR"). The top `top_k_candidates` (default 10) fused results move on to reranking — the fusion score itself is discarded after that, it only decided who advances.

## Reranking

**Stage 2 — cross-encoder rescoring:** `bge-reranker-base` re-scores each of those 10 `(query, chunk)` pairs jointly, then the top `top_k_final` (default 4) go into the LLM prompt. Cross-encoders catch relevance signals that fused BM25/vector scores alone miss, since they read the query and the chunk together instead of comparing separately-computed embeddings/keyword scores. This rerank score — not the fusion score — is what ends up in the API response's `retrieval_scores`; `retrieval_methods` (e.g. `"hybrid+rerank"`) records which stage-1 method(s) surfaced each chunk plus the fact that it was reranked.

## Offline Evaluation

`scripts/evaluate_ragas.py` reads `logs/query_log.jsonl` and scores each query for **faithfulness** (is the answer grounded in the retrieved context?) and **answer relevancy** (does the answer address the question?), using RAGAS with the same Groq LLM as judge. This runs offline/batch rather than inline per-request, since each metric costs 1–2 extra LLM calls.

```bash
python scripts/evaluate_ragas.py --limit 20 --output logs/ragas_report.json
```

Note: `AnswerRelevancy` is pinned to `strictness=1` because Groq's API rejects `n>1` completions per call, which RAGAS's default self-consistency (n=3) relies on.
