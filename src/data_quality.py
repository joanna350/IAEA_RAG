"""
Data Quality Checker
--------------------
Validates chunks before indexing:
  - Minimum length filter
  - Exact-duplicate detection (MD5 hash of normalized text)
  - Near-duplicate detection (embedding cosine similarity, optional)
  - Empty / boilerplate detection
"""

import hashlib
import re
from collections import Counter

import numpy as np
from langchain_core.documents import Document


MIN_CHUNK_CHARS = 80
BOILERPLATE_PATTERNS = [
    r"^\s*page\s+\d+\s*$", # 페이지 번호
    r"^\s*\d+\s*$", # 숫자만 있는 줄
    r"^[\.\-\*\s]+$", # 점, 대시, 별표만 있는 구분선
]

# Not yet validated against real near-duplicate examples in this corpus —
# revisit once the dataset is large enough to have any.
DEFAULT_NEAR_DUP_THRESHOLD = 0.95


def _hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def find_near_duplicates(chunks: list[Document], embeddings, threshold: float) -> set[int]:
    """
    Pairwise cosine similarity over chunk embeddings, O(n^2) similarity matrix.
    Fine at the current corpus scale (tens of chunks); an approximate index
    (FAISS/LSH) would be needed if this ever runs over tens of thousands.

    Returns the set of indices (into `chunks`) to drop — for each similar
    pair, the later chunk is dropped and the earlier one is kept.
    """
    if len(chunks) < 2:
        return set()

    vectors = np.array(embeddings.embed_documents([c.page_content for c in chunks]))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / norms
    similarity = normalized @ normalized.T

    to_drop: set[int] = set()
    n = len(chunks)
    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j not in to_drop and similarity[i, j] >= threshold:
                to_drop.add(j)
    return to_drop


def validate_chunks(
    chunks: list[Document],
    embeddings=None,
    near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
) -> tuple[list[Document], dict]:
    """
    Returns (clean_chunks, report_dict).
    report_dict contains counts of each rejection reason.

    Near-duplicate detection only runs if `embeddings` is passed in — it
    requires an embed_documents() call per surviving chunk, so it's opt-in
    rather than always-on.
    """
    seen_hashes: set[str] = set()
    clean: list[Document] = []
    report: Counter = Counter()

    for chunk in chunks:
        text = chunk.page_content

        # 1. Too short
        if len(text.strip()) < MIN_CHUNK_CHARS:
            report["too_short"] += 1
            continue

        # 2. Boilerplate
        if any(re.match(p, text.strip(), re.IGNORECASE) for p in BOILERPLATE_PATTERNS):
            report["boilerplate"] += 1
            continue

        # 3. Duplicate
        h = _hash(text)
        if h in seen_hashes:
            report["duplicate"] += 1
            continue
        seen_hashes.add(h)

        # 4. Low information density (>80% non-alphanumeric)
        alpha_ratio = sum(c.isalnum() or c.isspace() for c in text) / len(text)
        if alpha_ratio < 0.5:
            report["low_info_density"] += 1
            continue

        report["passed"] += 1
        clean.append(chunk)

    # 5. Near-duplicate (optional, requires an embeddings model)
    if embeddings is not None:
        dup_indices = find_near_duplicates(clean, embeddings, near_dup_threshold)
        if dup_indices:
            clean = [c for i, c in enumerate(clean) if i not in dup_indices]
            report["near_duplicate"] = len(dup_indices)
            report["passed"] -= len(dup_indices)

    return clean, dict(report)


def print_quality_report(report: dict, total: int):
    print("\n=== Data Quality Report ===")
    print(f"Total chunks input : {total}")
    if total == 0:
        print("  (no chunks to report on)")
    else:
        for reason, count in report.items():
            pct = count / total * 100
            print(f"  {reason:<22}: {count:>4}  ({pct:.1f}%)")
    print("===========================\n")
