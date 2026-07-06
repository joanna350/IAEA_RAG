"""Tests for src/monitoring.py — cost computation and JSONL logging."""

import json

import pytest

from src import monitoring
from src.monitoring import PRICING, compute_cost, log_query


class TestComputeCost:
    def test_known_model_computes_expected_cost(self):
        model = next(iter(PRICING))
        pricing = PRICING[model]
        cost = compute_cost(model, input_tokens=1000, output_tokens=1000)
        expected = pricing["input"] + pricing["output"]
        assert cost == pytest.approx(expected)

    def test_zero_tokens_is_zero_cost(self):
        model = next(iter(PRICING))
        assert compute_cost(model, 0, 0) == 0

    def test_unknown_model_returns_zero_and_warns(self, caplog):
        cost = compute_cost("some-model-not-in-pricing-table", 1000, 1000)
        assert cost == 0
        assert "No pricing entry" in caplog.text


class TestLogQuery:
    def test_appends_one_json_line_with_expected_fields(self, tmp_path, monkeypatch):
        log_path = tmp_path / "query_log.jsonl"
        monkeypatch.setattr(monitoring, "LOG_PATH", log_path)

        result = {
            "answer": "SMRs use passive safety systems.",
            "latency_sec": 0.5,
            "retrieval_latency_sec": 0.3,
            "total_latency_sec": 0.8,
            "input_tokens": 100,
            "output_tokens": 20,
            "retrieval_scores": [0.9, 0.5],
            "retrieval_methods": ["hybrid+rerank", "vector+rerank"],
            "sources": ["a.txt", "b.txt"],
            "contexts": ["chunk a text", "chunk b text"],
        }
        model = next(iter(PRICING))

        metrics = log_query(result, "What are SMR safety features?", model=model)

        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        row = json.loads(lines[0])
        assert row["question"] == "What are SMR safety features?"
        assert row["retrieval_latency_sec"] == 0.3
        assert row["llm_latency_sec"] == 0.5  # note: sourced from result["latency_sec"]
        assert row["answer"] == result["answer"]
        assert row["contexts"] == result["contexts"]
        assert row["answer_length"] == len(result["answer"])
        assert metrics.total_latency_sec == 0.8

    def test_two_calls_append_two_lines(self, tmp_path, monkeypatch):
        log_path = tmp_path / "query_log.jsonl"
        monkeypatch.setattr(monitoring, "LOG_PATH", log_path)

        minimal_result = {"answer": "x", "latency_sec": 0.1, "total_latency_sec": 0.1}
        log_query(minimal_result, "q1")
        log_query(minimal_result, "q2")

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_missing_fields_default_sensibly(self, tmp_path, monkeypatch):
        log_path = tmp_path / "query_log.jsonl"
        monkeypatch.setattr(monitoring, "LOG_PATH", log_path)

        metrics = log_query({}, "empty result")
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0
        assert metrics.sources == []
        assert metrics.answer == ""
