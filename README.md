# IAEA Nuclear Document RAG Pipeline

RAG (Retrieval-Augmented Generation) pipeline built for IAEA nuclear safety and SMR licensing documents.

## Architecture

```
data/ (IAEA docs)
  └─ load → chunk → quality_check (+ near-dup) → embed → FAISS index
                                                       └─ BM25 index
query
  └─ hybrid_search (vector + BM25) → rerank (cross-encoder) → LLM → answer + metrics
                                                                        └─ JSONL log → offline RAGAS eval
```

## Features

- **Hybrid retrieval**: BM25 (sparse) + FAISS vector search (dense) with score fusion
- **Cross-encoder reranking**: `bge-reranker-base` re-scores the top fused candidates by reading query and chunk jointly, before the final top-k goes to the LLM
- **Data quality pipeline**: length filtering, boilerplate removal, exact-duplicate (MD5) and near-duplicate (embedding cosine similarity) detection, low-information-density filtering — runs on every `ingest()`/`load()`, not just the demo script
- **Monitoring**: per-query retrieval/LLM/total latency, token usage, cost estimation, retrieval scores/methods, and the full answer + retrieved contexts, all appended to a JSONL log
- **Offline RAGAS evaluation**: `scripts/evaluate_ragas.py` batch-scores logged queries for faithfulness and answer relevancy, outside the request path
- **FastAPI server**: REST endpoints for ingestion and querying
- **Docker support**: containerized deployment

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt


# Run offline demo (no API key needed)
python scripts/demo.py

# Build index and start API server
python -c "from src.pipeline import IAEARagPipeline; p=IAEARagPipeline(); p.ingest()"
uvicorn src.api:app --reload

# Query via API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the passive safety requirements for SMR?"}'

# Batch-score logged queries for faithfulness / answer relevancy
python scripts/evaluate_ragas.py --output logs/ragas_report.json
```

## Docker

```bash
docker build -t iaea-rag .
docker run -e GROQ_API_KEY=sk-... -p 8000:8000 iaea-rag
```

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
│   ├── data_quality.py          # Chunk validation & quality checks
│   └── monitoring.py            # Latency, cost, retrieval metrics
├── scripts/
│   ├── demo.py                  # Offline demo (BM25 only, no API key required)
│   └── evaluate_ragas.py        # Batch RAGAS eval (faithfulness, answer relevancy)
├── logs/                        # Auto-created, gitignored — query_log.jsonl
├── faiss_index/                 # Auto-created on ingest()
├── Dockerfile
├── requirements.txt
└── .env
```

## Chunking Strategy

Uses `RecursiveCharacterTextSplitter` with paragraph → sentence → word boundary hierarchy.
- chunk_size: 512 tokens
- chunk_overlap: 64 tokens

This preserves section context while keeping chunks small enough for precise retrieval.

## Hybrid Search

Score fusion formula:
```
final_score = 0.7 * vector_score + 0.3 * bm25_score
```

Vector search handles semantic similarity; BM25 handles exact keyword matching (e.g. regulation codes like "GSR Part 1", "LOCA", "FSAR").

The top `top_k_candidates` (default 10) fused results are passed to reranking, not straight to the LLM.

## Reranking

`bge-reranker-base` (cross-encoder) re-scores each `(query, chunk)` pair jointly, then the top `top_k_final` (default 4) go into the LLM prompt. Cross-encoders catch relevance signals that fused BM25/vector scores alone miss, since they read the query and the chunk together instead of comparing separately-computed embeddings/keyword scores.

## Data Quality

`validate_chunks()` runs five checks, in order, before a chunk is indexed:

1. **too_short** — under 80 characters
2. **boilerplate** — page numbers, digit-only lines, separator lines
3. **duplicate** — exact match via MD5 hash
4. **near_duplicate** — embedding cosine similarity ≥ 0.95 (opt-in; only runs when an embeddings model is passed in). O(n²) pairwise comparison, fine at the current corpus scale — would need an approximate index (FAISS/LSH) in the tens-of-thousands-of-chunks range.
5. **low_info_density** — under 50% alphanumeric/whitespace characters

A per-reason rejection count prints on every `ingest()`/`load()`.

## Offline Evaluation

`scripts/evaluate_ragas.py` reads `logs/query_log.jsonl` and scores each query for **faithfulness** (is the answer grounded in the retrieved context?) and **answer relevancy** (does the answer address the question?), using RAGAS with the same Groq LLM as judge. This runs offline/batch rather than inline per-request, since each metric costs 1–2 extra LLM calls.

```bash
python scripts/evaluate_ragas.py --limit 20 --output logs/ragas_report.json
```

Note: `AnswerRelevancy` is pinned to `strictness=1` because Groq's API rejects `n>1` completions per call, which RAGAS's default self-consistency (n=3) relies on.
