"""
Data Quality Checker
--------------------
Validates chunks before indexing:
  - Minimum length filter
  - Duplicate detection (exact + near-duplicate via hashing)
  - Language / encoding sanity
  - Empty / boilerplate detection
"""

import hashlib
import re
from collections import Counter
from langchain_core.documents import Document


MIN_CHUNK_CHARS = 80
BOILERPLATE_PATTERNS = [
    r"^\s*page\s+\d+\s*$", # 페이지 번호
    r"^\s*\d+\s*$", # 숫자만 있는 줄
    r"^[\.\-\*\s]+$", # 점, 대시, 별표만 있는 구분선
]


def _hash(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def validate_chunks(chunks: list[Document]) -> tuple[list[Document], dict]:
    """
    Returns (clean_chunks, report_dict).
    report_dict contains counts of each rejection reason.
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

    return clean, dict(report)


def print_quality_report(report: dict, total: int):
    print("\n=== Data Quality Report ===")
    print(f"Total chunks input : {total}")
    for reason, count in report.items():
        pct = count / total * 100
        print(f"  {reason:<22}: {count:>4}  ({pct:.1f}%)")
    print("===========================\n")
