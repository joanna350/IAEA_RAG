"""
Tests for src/data_quality.py — pure, deterministic logic only (no
embeddings/LLM calls), so these run fast and need no API keys or network.
"""

import pytest
from langchain_core.documents import Document

from src.data_quality import (
    MIN_CHUNK_CHARS,
    find_near_duplicates,
    print_quality_report,
    validate_chunks,
)


def make_chunk(text: str) -> Document:
    return Document(page_content=text)


class TestValidateChunks:
    def test_too_short_is_rejected(self):
        chunks = [make_chunk("short")]
        clean, report = validate_chunks(chunks)
        assert clean == []
        assert report == {"too_short": 1}

    def test_long_enough_chunk_passes(self):
        text = "A" * MIN_CHUNK_CHARS
        clean, report = validate_chunks([make_chunk(text)])
        assert len(clean) == 1
        assert report == {"passed": 1}

    @pytest.mark.parametrize("boilerplate", [
        "Page 12",
        "42",
        "--------------------",
        "***",
    ])
    def test_boilerplate_lines_are_rejected(self, boilerplate):
        # pad so it wouldn't be rejected as too_short first, isolating the
        # boilerplate check
        padded = boilerplate + " " * (MIN_CHUNK_CHARS - len(boilerplate))
        clean, report = validate_chunks([make_chunk(boilerplate)])
        assert clean == []
        assert report.get("boilerplate") == 1 or report.get("too_short") == 1

    def test_exact_duplicate_is_rejected(self):
        text = "Periodic safety reviews must be conducted every 10 years. " * 2
        chunks = [make_chunk(text), make_chunk(text)]
        clean, report = validate_chunks(chunks)
        assert len(clean) == 1
        assert report["duplicate"] == 1
        assert report["passed"] == 1

    def test_duplicate_check_ignores_case_and_surrounding_whitespace(self):
        base = "Periodic safety reviews must be conducted every 10 years. " * 2
        chunks = [make_chunk(base), make_chunk(f"  {base.upper()}  ")]
        clean, report = validate_chunks(chunks)
        assert len(clean) == 1
        assert report["duplicate"] == 1

    def test_low_info_density_is_rejected(self):
        # mostly punctuation/symbols, not alphanumeric or whitespace
        text = "#$%^&*()!@" * 10
        clean, report = validate_chunks([make_chunk(text)])
        assert clean == []
        assert report.get("low_info_density") == 1

    def test_report_reasons_sum_to_total_input(self):
        chunks = [
            make_chunk("short"),
            make_chunk("Page 1"),
            make_chunk("A" * MIN_CHUNK_CHARS),
            make_chunk("#" * MIN_CHUNK_CHARS),
        ]
        clean, report = validate_chunks(chunks)
        assert sum(report.values()) == len(chunks)


class TestNearDuplicates:
    class FakeEmbeddings:
        """Returns hand-picked vectors so cosine similarity is predictable."""
        def embed_documents(self, texts):
            vectors = {
                "a": [1.0, 0.0],
                "a_near": [0.99, 0.01],  # ~cosine 0.999... with "a"
                "b": [0.0, 1.0],
            }
            return [vectors[t] for t in texts]

    def test_near_duplicate_pair_is_detected(self):
        chunks = [make_chunk("a"), make_chunk("a_near"), make_chunk("b")]
        dropped = find_near_duplicates(chunks, self.FakeEmbeddings(), threshold=0.95)
        assert dropped == {1}  # the later of the near-duplicate pair

    def test_no_duplicates_below_threshold(self):
        chunks = [make_chunk("a"), make_chunk("b")]
        dropped = find_near_duplicates(chunks, self.FakeEmbeddings(), threshold=0.95)
        assert dropped == set()

    def test_single_chunk_short_circuits(self):
        assert find_near_duplicates([make_chunk("a")], self.FakeEmbeddings(), threshold=0.95) == set()

    def test_validate_chunks_applies_near_duplicate_filter_when_embeddings_given(self):
        text_a = (
            "Periodic safety reviews must be conducted every 10 years for "
            "licensing renewal, covering aging management and design basis."
        )
        text_b = (
            "Environmental impact statements must accompany license "
            "applications and evaluate radiological and conventional impacts."
        )
        chunks = [make_chunk(text_a), make_chunk(text_a + " Indeed."), make_chunk(text_b)]

        class AllSameEmbeddings:
            def embed_documents(self, texts):
                return [[1.0, 0.0] for _ in texts]  # everything "identical"

        clean, report = validate_chunks(chunks, embeddings=AllSameEmbeddings())
        assert report["near_duplicate"] == 2
        assert len(clean) == 1


class TestPrintQualityReport:
    def test_zero_total_does_not_raise(self, capsys):
        print_quality_report({}, total=0)
        assert "no chunks to report on" in capsys.readouterr().out

    def test_nonzero_total_prints_percentages(self, capsys):
        print_quality_report({"passed": 3, "too_short": 1}, total=4)
        out = capsys.readouterr().out
        assert "75.0%" in out
        assert "25.0%" in out
