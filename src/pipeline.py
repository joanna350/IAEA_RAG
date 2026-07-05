"""
IAEA Document RAG Pipeline
--------------------------
Ingestion  : load text/PDF docs → chunk → embed → store in Qdrant
Retrieval  : hybrid search (BM25 + vector) → rerank → LLM answer
Monitoring : logs query latency, retrieval scores, token usage
"""

import os
import time
import json
import logging
from src.monitoring import log_query, print_metrics
from src.data_quality import validate_chunks, print_quality_report
from typing import Optional, Iterator
from dataclasses import dataclass

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    data_dir: str = "data"
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = "iaea_chunks"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_vector: int = 5
    top_k_bm25: int = 5
    top_k_candidates: int = 10  # candidates passed to reranker before final cut
    top_k_final: int = 4
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "llama-3.1-8b-instant"  # must match a key in monitoring.PRICING
    bm25_weight: float = 0.3   # hybrid score = bm25*w + vector*(1-w)
    vector_weight: float = 0.7
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_enabled: bool = True


# ---------------------------------------------------------------------------
# 1. Document Loading
# ---------------------------------------------------------------------------

def load_documents(data_dir: str) -> list[Document]:
    """Load .txt and .pdf files from data_dir."""
    docs: list[Document] = []
    data_path = Path(data_dir)

    for file in sorted(data_path.iterdir()):
        if file.suffix == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
        elif file.suffix == ".pdf":
            loader = PyMuPDFLoader(str(file))
        else:
            continue

        loaded = loader.load()
        # Attach source metadata
        for doc in loaded:
            doc.metadata["source"] = file.name
        docs.extend(loaded)
        log.info(f"Loaded {len(loaded)} page(s) from {file.name}")

    log.info(f"Total documents loaded: {len(docs)}")
    return docs


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------

def chunk_documents(docs: list[Document], cfg: PipelineConfig) -> list[Document]:
    """
    RecursiveCharacterTextSplitter — splits on paragraph → sentence → word boundaries.
    Preserves source metadata per chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Add chunk index for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    log.info(f"Total chunks after splitting: {len(chunks)}")
    return chunks


def prepare_chunks(docs: list[Document], cfg: PipelineConfig, embeddings=None) -> list[Document]:
    """
    Chunk documents and drop low-quality chunks (too short, boilerplate,
    duplicate, near-duplicate, low information density). chunk_id is
    reassigned densely over the *filtered* list, since hybrid_search
    indexes chunks by position — reusing pre-filter ids would misalign
    retrieval lookups.
    """
    raw_chunks = chunk_documents(docs, cfg)
    clean_chunks, report = validate_chunks(raw_chunks, embeddings=embeddings)
    print_quality_report(report, total=len(raw_chunks))
    for i, chunk in enumerate(clean_chunks):
        chunk.metadata["chunk_id"] = i
    return clean_chunks


# ---------------------------------------------------------------------------
# 3. Embedding & Vector Store
# ---------------------------------------------------------------------------

def build_vector_store(
    chunks: list[Document],
    cfg: PipelineConfig,
    embeddings: OpenAIEmbeddings,
) -> QdrantVectorStore:
    """Embed chunks and store in Qdrant. Recreates the collection from scratch,
    so re-running ingest() replaces stale data instead of appending to it."""
    log.info(f"Building Qdrant collection '{cfg.qdrant_collection}' at {cfg.qdrant_url} ...")
    t0 = time.time()
    store = QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=cfg.qdrant_url,
        collection_name=cfg.qdrant_collection,
        force_recreate=True,
    )
    log.info(f"Vector store built in {time.time() - t0:.1f}s → collection '{cfg.qdrant_collection}'")
    return store


def load_vector_store(cfg: PipelineConfig, embeddings: OpenAIEmbeddings) -> QdrantVectorStore:
    """Connect to an existing Qdrant collection (built by a prior ingest())."""
    log.info(f"Connecting to Qdrant collection '{cfg.qdrant_collection}' at {cfg.qdrant_url} ...")
    client = QdrantClient(url=cfg.qdrant_url)
    return QdrantVectorStore(client=client, collection_name=cfg.qdrant_collection, embedding=embeddings)


# ---------------------------------------------------------------------------
# 4. BM25 Index
# ---------------------------------------------------------------------------

def build_bm25_index(chunks: list[Document]) -> BM25Okapi:
    """Build BM25 sparse index over chunk texts."""
    tokenized = [doc.page_content.lower().split() for doc in chunks]
    return BM25Okapi(tokenized)


# ---------------------------------------------------------------------------
# 5. Hybrid Retrieval
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    doc: Document
    score: float
    retrieval_method: str


def hybrid_search(
    query: str,
    chunks: list[Document],
    vector_store: QdrantVectorStore,
    bm25: BM25Okapi,
    cfg: PipelineConfig,
) -> list[RetrievedChunk]:
    """
    Hybrid retrieval:
    1. Vector similarity search (dense)
    2. BM25 keyword search (sparse)
    3. Score fusion with configured weights
    4. Deduplicate and return top-k
    """
    # --- Vector search ---
    vec_results = vector_store.similarity_search_with_score(query, k=cfg.top_k_vector)
    # Qdrant (cosine distance) returns similarity directly, higher = better —
    # unlike FAISS's raw L2 distance, no inversion needed. Just normalize by
    # the top score so it's on the same 0-1ish scale as the BM25 side below.
    max_score = max((score for _, score in vec_results), default=0.0) or 1.0
    vec_scores: dict[int, float] = {}
    for doc, score in vec_results:
        chunk_id = doc.metadata.get("chunk_id", -1)
        vec_scores[chunk_id] = score / max_score

    # --- BM25 search ---
    tokenized_query = query.lower().split()
    bm25_raw = bm25.get_scores(tokenized_query)
    max_bm25 = bm25_raw.max() or 1.0
    top_bm25_ids = bm25_raw.argsort()[-cfg.top_k_bm25:][::-1]
    bm25_scores: dict[int, float] = {
        int(idx): float(bm25_raw[idx]) / max_bm25 for idx in top_bm25_ids
    }

    # --- Score fusion ---
    all_ids = set(vec_scores) | set(bm25_scores)
    fused: list[RetrievedChunk] = []
    for cid in all_ids:
        v_score = vec_scores.get(cid, 0.0)
        b_score = bm25_scores.get(cid, 0.0)
        combined = cfg.vector_weight * v_score + cfg.bm25_weight * b_score
        method = "hybrid" if (cid in vec_scores and cid in bm25_scores) else (
            "vector" if cid in vec_scores else "bm25"
        )
        fused.append(RetrievedChunk(doc=chunks[cid], score=float(combined), retrieval_method=method))

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[: cfg.top_k_candidates]


# ---------------------------------------------------------------------------
# 6. Reranking
# ---------------------------------------------------------------------------

def rerank_chunks(
    query: str,
    candidates: list[RetrievedChunk],
    reranker: CrossEncoder,
    cfg: PipelineConfig,
) -> list[RetrievedChunk]:
    """
    Re-score fused candidates with a cross-encoder for query-document relevance,
    then return the top_k_final. Cross-encoders read query+doc jointly, so they
    catch relevance signals that separate dense/sparse scoring miss.
    """
    if not candidates:
        return candidates

    pairs = [[query, c.doc.page_content] for c in candidates]
    scores = reranker.predict(pairs)

    reranked = [
        RetrievedChunk(doc=c.doc, score=float(s), retrieval_method=f"{c.retrieval_method}+rerank")
        for c, s in zip(candidates, scores)
    ]
    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked[: cfg.top_k_final]


# ---------------------------------------------------------------------------
# 7. LLM Answer Generation
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
You are an expert assistant specializing in IAEA nuclear safety standards and SMR licensing.
Answer the question using ONLY the provided context. If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Answer (cite document sections where possible):
""")


def build_prompt(query: str, retrieved: list[RetrievedChunk]):
    """Shared by generate_answer() and IAEARagPipeline.query_stream()."""
    context_parts = [
        f"[{i}] ({r.doc.metadata.get('source', 'unknown')})\n{r.doc.page_content}"
        for i, r in enumerate(retrieved, 1)
    ]
    context = "\n\n---\n\n".join(context_parts)
    return PROMPT_TEMPLATE.format_messages(context=context, question=query)


def generate_answer(
    query: str,
    retrieved: list[RetrievedChunk],
    llm: ChatOpenAI,
) -> dict:
    """Format context and generate LLM answer."""
    prompt = build_prompt(query, retrieved)

    t0 = time.time()
    response = llm.invoke(prompt)
    latency = time.time() - t0

    return {
        "answer": response.content,
        "latency_sec": round(latency, 2),
        "input_tokens": response.usage_metadata.get("input_tokens", 0),
        "output_tokens": response.usage_metadata.get("output_tokens", 0),
        "sources": [r.doc.metadata.get("source") for r in retrieved],
        "retrieval_scores": [round(r.score, 4) for r in retrieved],
        "retrieval_methods": [r.retrieval_method for r in retrieved],
        "contexts": [r.doc.page_content for r in retrieved],
    }


# ---------------------------------------------------------------------------
# 8. Pipeline Orchestrator
# ---------------------------------------------------------------------------

class IAEARagPipeline:
    """
    End-to-end RAG pipeline for IAEA nuclear licensing documents.

    Usage:
        pipeline = IAEARagPipeline()
        pipeline.ingest()          # build index from data/
        result = pipeline.query("What are SMR passive safety requirements?")
    """

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") #OpenAIEmbeddings(model=self.cfg.embedding_model)
        self.llm = ChatGroq(model=self.cfg.llm_model) #ChatOpenAI(model=self.cfg.llm_model, temperature=0)
        self.reranker = CrossEncoder(self.cfg.reranker_model) if self.cfg.rerank_enabled else None
        self.chunks: list[Document] = []
        self.vector_store: Optional[QdrantVectorStore] = None
        self.bm25: Optional[BM25Okapi] = None

    def ingest(self):
        """Load → chunk → filter → embed → index. Run once (or when docs change)."""
        docs = load_documents(self.cfg.data_dir)
        self.chunks = prepare_chunks(docs, self.cfg, self.embeddings)
        self.vector_store = build_vector_store(self.chunks, self.cfg, self.embeddings)
        self.bm25 = build_bm25_index(self.chunks)
        log.info("Ingestion complete.")

    def load(self):
        """Load pre-built index from disk (skip re-embedding)."""
        self.vector_store = load_vector_store(self.cfg, self.embeddings)
        # Reload + re-filter chunks for BM25 (lightweight, must match ingest()
        # exactly so chunk_id stays aligned with what's stored in the Qdrant collection)
        docs = load_documents(self.cfg.data_dir)
        self.chunks = prepare_chunks(docs, self.cfg, self.embeddings)
        self.bm25 = build_bm25_index(self.chunks)
        log.info("Pipeline loaded from disk.")

    def _retrieve(self, question: str) -> tuple[list[RetrievedChunk], float]:
        """Hybrid search + rerank. Shared by query() and query_stream()."""
        t_retrieval = time.time()
        candidates = hybrid_search(question, self.chunks, self.vector_store, self.bm25, self.cfg)
        if self.reranker is not None:
            retrieved = rerank_chunks(question, candidates, self.reranker, self.cfg)
        else:
            retrieved = candidates[: self.cfg.top_k_final]
        return retrieved, time.time() - t_retrieval

    def query(self, question: str) -> dict:
        """Run hybrid retrieval + LLM generation. Returns result dict with metrics."""
        if not self.vector_store or not self.bm25:
            raise RuntimeError("Pipeline not initialized. Call ingest() or load() first.")

        log.info(f"Query: {question}")
        t0 = time.time()

        retrieved, retrieval_latency = self._retrieve(question)

        result = generate_answer(question, retrieved, self.llm)
        result["retrieval_latency_sec"] = round(retrieval_latency, 2)
        result["total_latency_sec"] = round(time.time() - t0, 2)

        # Log quality metrics
        log.info(
            f"Retrieved {len(retrieved)} chunks | "
            f"scores={result['retrieval_scores']} | "
            f"latency={result['total_latency_sec']}s | "
            f"tokens={result['input_tokens']}+{result['output_tokens']}"
        )
        metrics = log_query(result, question, model=self.cfg.llm_model)
        print_metrics(metrics)
        return result

    def query_stream(self, question: str) -> Iterator[str]:
        """
        Same retrieval as query(), but streams the LLM answer token-by-token
        as SSE events instead of returning one dict. Retrieval/rerank are not
        streamed — they're a single fast call — only generation is
        incremental. Metrics are logged once the stream completes, exactly
        like the non-streaming path.

        Emits three event types:
          meta  - sources/retrieval_scores/retrieval_methods, sent once up front
          token - one answer text fragment per event
          done  - final latency/token counts, sent once at the end
        """
        if not self.vector_store or not self.bm25:
            raise RuntimeError("Pipeline not initialized. Call ingest() or load() first.")

        log.info(f"Query (stream): {question}")
        t0 = time.time()

        retrieved, retrieval_latency = self._retrieve(question)

        meta = {
            "sources": [r.doc.metadata.get("source") for r in retrieved],
            "retrieval_scores": [round(r.score, 4) for r in retrieved],
            "retrieval_methods": [r.retrieval_method for r in retrieved],
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        prompt = build_prompt(question, retrieved)

        t_llm = time.time()
        answer_parts: list[str] = []
        input_tokens = output_tokens = 0
        for chunk in self.llm.stream(prompt):
            if chunk.content:
                answer_parts.append(chunk.content)
                yield f"event: token\ndata: {json.dumps({'text': chunk.content})}\n\n"
            if chunk.usage_metadata:
                input_tokens = chunk.usage_metadata.get("input_tokens", input_tokens)
                output_tokens = chunk.usage_metadata.get("output_tokens", output_tokens)
        llm_latency = time.time() - t_llm

        result = {
            "answer": "".join(answer_parts),
            "latency_sec": round(llm_latency, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "contexts": [r.doc.page_content for r in retrieved],
            **meta,
        }
        result["retrieval_latency_sec"] = round(retrieval_latency, 2)
        result["total_latency_sec"] = round(time.time() - t0, 2)

        metrics = log_query(result, question, model=self.cfg.llm_model)
        print_metrics(metrics)

        yield f"event: done\ndata: {json.dumps({'total_latency_sec': result['total_latency_sec'], 'input_tokens': input_tokens, 'output_tokens': output_tokens})}\n\n"
