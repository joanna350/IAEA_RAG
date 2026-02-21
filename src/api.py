"""
FastAPI server for IAEA RAG pipeline.

Endpoints:
    POST /ingest        - (re)build the index
    POST /query         - query the pipeline
    GET  /health        - liveness check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from src.pipeline import IAEARagPipeline, PipelineConfig

log = logging.getLogger(__name__)
app = FastAPI(title="IAEA Nuclear Document RAG API", version="1.0.0")

pipeline = IAEARagPipeline()
try:
    pipeline.load()
except:
    pass

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieval_scores: list[float]
    retrieval_methods: list[str]
    latency_sec: float
    total_latency_sec: float
    input_tokens: int
    output_tokens: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "index_ready": pipeline.vector_store is not None}


@app.post("/ingest")
def ingest():
    try:
        pipeline.ingest()
        return {"status": "ingestion complete", "chunks": len(pipeline.chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if pipeline.vector_store is None:
        raise HTTPException(status_code=400, detail="Index not built. Call /ingest first.")
    try:
        pipeline.cfg.top_k_final = req.top_k
        result = pipeline.query(req.question)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
