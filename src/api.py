"""
FastAPI server for IAEA RAG pipeline.

Endpoints:
    POST   /ingest              - (re)build the index
    POST   /query               - query the pipeline
    POST   /query/stream        - query the pipeline, streaming the answer via SSE
    GET    /health               - liveness check

    Admin (require X-Admin-Key header or ?key= query param == ADMIN_API_KEY):
    GET    /admin/documents      - list source documents
    POST   /admin/documents      - upload a document (.txt/.pdf)
    DELETE /admin/documents/{filename} - remove a document
    POST   /admin/reindex        - rebuild the index (same as /ingest, admin-gated)
    GET    /admin/stats          - aggregated latency/cost/token stats (JSON)
    GET    /admin/dashboard      - human-readable HTML dashboard
"""

import os

from fastapi import Depends, FastAPI, Header, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import logging

from src.pipeline import IAEARagPipeline
from src import admin
from src.monitoring import LOG_PATH

log = logging.getLogger(__name__)
app = FastAPI(title="IAEA Nuclear Document RAG API", version="1.0.0")

pipeline = IAEARagPipeline()
try:
    pipeline.load()
except:
    pass

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")


def require_admin(x_admin_key: str = Header(None), key: str = Query(None)):
    """
    Minimal shared-secret check — no rate limiting, no HTTPS enforcement,
    and a key passed as ?key= can end up in access logs. Good enough to keep
    /admin/* off-limits to casual requests; not a substitute for a real auth
    layer if this were ever exposed beyond a trusted network.
    """
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Admin API disabled: ADMIN_API_KEY is not set.")
    if (x_admin_key or key) != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid admin key.")

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
    retrieval_latency_sec: float
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


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    """
    Same retrieval/generation as /query, but streams the answer as it's
    generated instead of waiting for the full response. Server-Sent Events:
    one 'meta' event (sources/scores) up front, one 'token' event per answer
    fragment, one 'done' event with final latency/token counts.
    """
    if pipeline.vector_store is None:
        raise HTTPException(status_code=400, detail="Index not built. Call /ingest first.")
    pipeline.cfg.top_k_final = req.top_k
    return StreamingResponse(
        pipeline.query_stream(req.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------------------

@app.get("/admin/documents", dependencies=[Depends(require_admin)])
def admin_list_documents():
    return {"documents": admin.list_documents(pipeline.cfg.data_dir)}


@app.post("/admin/documents", dependencies=[Depends(require_admin)])
async def admin_upload_document(file: UploadFile):
    try:
        content = await file.read()
        saved_name = admin.save_document(pipeline.cfg.data_dir, file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "status": "uploaded",
        "filename": saved_name,
        "size_bytes": len(content),
        "note": "Call POST /admin/reindex to include this document in the index.",
    }


@app.delete("/admin/documents/{filename}", dependencies=[Depends(require_admin)])
def admin_delete_document(filename: str):
    try:
        admin.delete_document(pipeline.cfg.data_dir, filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return {
        "status": "deleted",
        "filename": filename,
        "note": "Call POST /admin/reindex to remove it from the index too.",
    }


@app.post("/admin/reindex", dependencies=[Depends(require_admin)])
def admin_reindex():
    try:
        pipeline.ingest()
        return {"status": "reindex complete", "chunks": len(pipeline.chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/stats", dependencies=[Depends(require_admin)])
def admin_stats():
    return {
        "logs": admin.summarize_logs(str(LOG_PATH)),
        "recent_queries": admin.recent_queries(str(LOG_PATH)),
        "documents": admin.list_documents(pipeline.cfg.data_dir),
    }


@app.get("/admin/dashboard", response_class=HTMLResponse, dependencies=[Depends(require_admin)])
def admin_dashboard():
    html = admin.render_dashboard_html(
        stats=admin.summarize_logs(str(LOG_PATH)),
        recent=admin.recent_queries(str(LOG_PATH)),
        documents=admin.list_documents(pipeline.cfg.data_dir),
    )
    return HTMLResponse(html)
