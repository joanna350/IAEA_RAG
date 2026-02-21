# IAEA Nuclear Document RAG Pipeline

RAG (Retrieval-Augmented Generation) pipeline built for IAEA nuclear safety and SMR licensing documents.

## Architecture

```
data/ (IAEA docs)
  └─ load → chunk → quality_check → embed → FAISS index
                                         └─ BM25 index
query
  └─ hybrid_search (vector + BM25) → rerank → LLM → answer + metrics
```

## Features

- **Hybrid retrieval**: BM25 (sparse) + FAISS vector search (dense) with score fusion
- **Data quality pipeline**: deduplication, length filtering, boilerplate removal
- **Monitoring**: per-query latency, token usage, cost estimation, JSONL log
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
```

## Docker

```bash
docker build -t iaea-rag .
docker run -GROQ_API_KEY=sk-... -p 8000:8000 iaea-rag
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
│   └── demo.py                  # Offline demo (BM25 only, no API key required)
├── logs/                        # Auto-created, stores query_log.jsonl
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
