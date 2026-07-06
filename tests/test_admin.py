"""Tests for src/admin.py — document management and log aggregation."""

import json

import pytest

from src import admin


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "a.txt").write_text("doc a content", encoding="utf-8")
    (d / "b.pdf").write_bytes(b"%PDF-fake")
    (d / "ignored.md").write_text("not a supported extension", encoding="utf-8")
    return d


class TestListDocuments:
    def test_lists_only_supported_extensions(self, data_dir):
        docs = admin.list_documents(str(data_dir))
        filenames = {d["filename"] for d in docs}
        assert filenames == {"a.txt", "b.pdf"}

    def test_each_entry_has_size_and_modified(self, data_dir):
        docs = admin.list_documents(str(data_dir))
        for d in docs:
            assert d["size_bytes"] > 0
            assert "modified" in d


class TestSaveDocument:
    def test_saves_file_with_stripped_name(self, data_dir):
        saved = admin.save_document(str(data_dir), "new_doc.txt", b"hello world")
        assert saved == "new_doc.txt"
        assert (data_dir / "new_doc.txt").read_bytes() == b"hello world"

    def test_strips_path_components_from_filename(self, data_dir):
        # even if a caller passes a path-y filename, only the basename is used
        saved = admin.save_document(str(data_dir), "../../etc/evil.txt", b"x")
        assert saved == "evil.txt"
        assert (data_dir / "evil.txt").exists()
        assert not (data_dir.parent.parent / "etc").exists()

    def test_rejects_unsupported_extension(self, data_dir):
        with pytest.raises(ValueError):
            admin.save_document(str(data_dir), "script.exe", b"x")


class TestDeleteDocument:
    def test_deletes_existing_file(self, data_dir):
        admin.delete_document(str(data_dir), "a.txt")
        assert not (data_dir / "a.txt").exists()

    def test_missing_file_raises_file_not_found(self, data_dir):
        with pytest.raises(FileNotFoundError):
            admin.delete_document(str(data_dir), "nope.txt")

    def test_path_traversal_attempt_is_rejected(self, data_dir, tmp_path):
        # a file that exists OUTSIDE data_dir must not be deletable via ../..
        outside = tmp_path / "outside.txt"
        outside.write_text("do not delete me", encoding="utf-8")
        with pytest.raises((FileNotFoundError, ValueError)):
            admin.delete_document(str(data_dir), "../outside.txt")
        assert outside.exists()


@pytest.fixture
def log_file(tmp_path):
    p = tmp_path / "query_log.jsonl"
    rows = [
        {"timestamp": "2026-01-01T00:00:00", "question": "q1", "total_latency_sec": 1.0,
         "retrieval_latency_sec": 0.4, "llm_latency_sec": 0.6, "estimated_cost_usd": 0.001,
         "input_tokens": 100, "output_tokens": 20, "sources": ["a.txt"]},
        {"timestamp": "2026-01-01T00:01:00", "question": "q2", "total_latency_sec": 2.0,
         "retrieval_latency_sec": 0.8, "llm_latency_sec": 1.2, "estimated_cost_usd": 0.002,
         "input_tokens": 200, "output_tokens": 40, "sources": ["b.txt"]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


class TestSummarizeLogs:
    def test_empty_log_returns_zeroed_stats(self, tmp_path):
        stats = admin.summarize_logs(str(tmp_path / "does_not_exist.jsonl"))
        assert stats["total_queries"] == 0
        assert stats["total_cost_usd"] == 0.0

    def test_aggregates_across_rows(self, log_file):
        stats = admin.summarize_logs(str(log_file))
        assert stats["total_queries"] == 2
        assert stats["avg_total_latency_sec"] == pytest.approx(1.5)
        assert stats["total_cost_usd"] == pytest.approx(0.003)
        assert stats["total_input_tokens"] == 300
        assert stats["total_output_tokens"] == 60


class TestRecentQueries:
    def test_most_recent_first(self, log_file):
        recent = admin.recent_queries(str(log_file), limit=20)
        assert [r["question"] for r in recent] == ["q2", "q1"]

    def test_respects_limit(self, log_file):
        recent = admin.recent_queries(str(log_file), limit=1)
        assert len(recent) == 1
        assert recent[0]["question"] == "q2"


class TestRenderDashboardHtml:
    def test_renders_without_error_and_includes_stats(self, log_file, data_dir):
        stats = admin.summarize_logs(str(log_file))
        recent = admin.recent_queries(str(log_file))
        documents = admin.list_documents(str(data_dir))

        html = admin.render_dashboard_html(stats, recent, documents)

        assert "<table>" in html
        assert "a.txt" in html
        assert "q1" in html and "q2" in html

    def test_escapes_html_in_question_text(self, data_dir):
        stats = admin.summarize_logs(str(data_dir / "missing.jsonl"))
        malicious = [{
            "timestamp": "2026-01-01T00:00:00",
            "question": "<script>alert(1)</script>",
            "total_latency_sec": 1.0,
            "estimated_cost_usd": 0.0,
            "sources": [],
        }]
        html = admin.render_dashboard_html(stats, malicious, [])
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_handles_empty_documents_and_queries(self):
        stats = admin.summarize_logs("/nonexistent/path.jsonl")
        html = admin.render_dashboard_html(stats, [], [])
        assert "No documents in data/" in html
        assert "No queries logged yet" in html
