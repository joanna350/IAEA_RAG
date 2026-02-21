"""
IAEA Document RAG Pipeline
--------------------------
Ingestion  : load text/PDF docs → chunk → embed → store in FAISS
Retrieval  : hybrid search (BM25 + vector) → rerank → LLM answer
Monitoring : logs query latency, retrieval scores, token usage
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
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
    index_dir: str = "faiss_index"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_vector: int = 5
    top_k_bm25: int = 5
    top_k_final: int = 4
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    bm25_weight: float = 0.3   # hybrid score = bm25*w + vector*(1-w)
    vector_weight: float = 0.7


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


# ---------------------------------------------------------------------------
# 3. Embedding & Vector Store
# ---------------------------------------------------------------------------

def build_vector_store(
    chunks: list[Document],
    cfg: PipelineConfig,
    embeddings: OpenAIEmbeddings,
) -> FAISS:
    """Embed chunks and store in FAISS. Saves index to disk."""
    log.info("Building FAISS vector store ...")
    t0 = time.time()
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(cfg.index_dir)
    log.info(f"Vector store built in {time.time() - t0:.1f}s → saved to '{cfg.index_dir}'")
    return store


def load_vector_store(cfg: PipelineConfig, embeddings: OpenAIEmbeddings) -> FAISS:
    """Load existing FAISS index from disk."""
    log.info(f"Loading FAISS index from '{cfg.index_dir}' ...")
    return FAISS.load_local(cfg.index_dir, embeddings, allow_dangerous_deserialization=True)


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
    vector_store: FAISS,
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
    # FAISS returns L2 distance; convert to similarity
    max_dist = max(score for _, score in vec_results) or 1.0
    vec_scores: dict[int, float] = {}
    for doc, dist in vec_results:
        chunk_id = doc.metadata.get("chunk_id", -1)
        vec_scores[chunk_id] = 1.0 - (dist / max_dist)

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
        fused.append(RetrievedChunk(doc=chunks[cid], score=combined, retrieval_method=method))

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[: cfg.top_k_final]


# ---------------------------------------------------------------------------
# 6. LLM Answer Generation
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
You are an expert assistant specializing in IAEA nuclear safety standards and SMR licensing.
Answer the question using ONLY the provided context. If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Answer (cite document sections where possible):
""")


def generate_answer(
    query: str,
    retrieved: list[RetrievedChunk],
    llm: ChatOpenAI,
) -> dict:
    """Format context and generate LLM answer."""
    context_parts = []
    for i, r in enumerate(retrieved, 1):
        source = r.doc.metadata.get("source", "unknown")
        context_parts.append(f"[{i}] ({source})\n{r.doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = PROMPT_TEMPLATE.format_messages(context=context, question=query)

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
    }


# ---------------------------------------------------------------------------
# 7. Pipeline Orchestrator
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
        self.llm = ChatGroq(model="llama-3.1-8b-instant") #ChatOpenAI(model=self.cfg.llm_model, temperature=0)
        self.chunks: list[Document] = []
        self.vector_store: Optional[FAISS] = None
        self.bm25: Optional[BM25Okapi] = None

    def ingest(self):
        """Load → chunk → embed → index. Run once (or when docs change)."""
        docs = load_documents(self.cfg.data_dir)
        self.chunks = chunk_documents(docs, self.cfg)
        self.vector_store = build_vector_store(self.chunks, self.cfg, self.embeddings)
        self.bm25 = build_bm25_index(self.chunks)
        log.info("Ingestion complete.")

    def load(self):
        """Load pre-built index from disk (skip re-embedding)."""
        self.vector_store = load_vector_store(self.cfg, self.embeddings)
        # Reload chunks for BM25 (lightweight)
        docs = load_documents(self.cfg.data_dir)
        self.chunks = chunk_documents(docs, self.cfg)
        self.bm25 = build_bm25_index(self.chunks)
        log.info("Pipeline loaded from disk.")

    def query(self, question: str) -> dict:
        """Run hybrid retrieval + LLM generation. Returns result dict with metrics."""
        if not self.vector_store or not self.bm25:
            raise RuntimeError("Pipeline not initialized. Call ingest() or load() first.")

        log.info(f"Query: {question}")
        t0 = time.time()

        retrieved = hybrid_search(question, self.chunks, self.vector_store, self.bm25, self.cfg)
        result = generate_answer(question, retrieved, self.llm)
        result["total_latency_sec"] = round(time.time() - t0, 2)

        # Log quality metrics
        log.info(
            f"Retrieved {len(retrieved)} chunks | "
            f"scores={result['retrieval_scores']} | "
            f"latency={result['total_latency_sec']}s | "
            f"tokens={result['input_tokens']}+{result['output_tokens']}"
        )
        return result
