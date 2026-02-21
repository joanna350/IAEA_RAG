"""
CLI demo — runs the full pipeline without an OpenAI key by mocking embeddings/LLM.
Run: python scripts/demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import PipelineConfig
from data_quality import validate_chunks, print_quality_report
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


from pipeline import PipelineConfig, load_documents, chunk_documents, build_bm25_index
from data_quality import validate_chunks, print_quality_report


def run_demo():
    print("=" * 60)
    print("  IAEA RAG Pipeline — Offline Demo (no API key required)")
    print("=" * 60)

    cfg = PipelineConfig(data_dir="data")

    # 1. Load documents
    print("\n[1] Loading documents...")
    docs = load_documents(cfg.data_dir)

    # 2. Chunk
    print(f"\n[2] Chunking (size={cfg.chunk_size}, overlap={cfg.chunk_overlap})...")
    chunks = chunk_documents(docs, cfg)
    print(f"    Total chunks: {len(chunks)}")

    # 3. Data quality check
    print("\n[3] Running data quality checks...")
    clean_chunks, report = validate_chunks(chunks)
    print_quality_report(report, total=len(chunks))

    # 4. Show sample chunks
    print("[4] Sample chunks after quality filtering:\n")
    for i, chunk in enumerate(clean_chunks[:3], 1):
        print(f"  Chunk {i} | source={chunk.metadata['source']} | "
              f"chars={len(chunk.page_content)}")
        print(f"  Preview: {chunk.page_content[:120].strip()}...")
        print()

    # 5. BM25 keyword search
    print("[5] BM25 keyword search demo (no embedding API needed)...")
    bm25 = build_bm25_index(clean_chunks)

    test_queries = [
        "What are the passive safety requirements for SMR?",
        "What documents are required for nuclear licensing?",
        "What is the ALARA principle?",
    ]

    for query in test_queries:
        scores = bm25.get_scores(query.lower().split())
        top_idx = scores.argsort()[-3:][::-1]
        print(f"\n  Query: '{query}'")
        for rank, idx in enumerate(top_idx, 1):
            src = clean_chunks[idx].metadata.get("source", "?")
            preview = clean_chunks[idx].page_content[:80].replace("\n", " ").strip()
            print(f"    [{rank}] score={scores[idx]:.3f} | {src} | {preview}...")


if __name__ == "__main__":
    run_demo()
