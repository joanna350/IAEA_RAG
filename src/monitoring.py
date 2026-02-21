"""
Pipeline Monitoring
-------------------
Tracks:
  - Query latency (retrieval + LLM)
  - Token usage & estimated cost
  - Retrieval score distribution
  - Simple JSONL query log for offline analysis
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

LOG_PATH = Path("logs/query_log.jsonl")
LOG_PATH.parent.mkdir(exist_ok=True)

# groq pricing (USD per 1K tokens)
PRICING = {
    "llama-3.1-8b-instant":     {"input": 0.00005, "output": 0.00008},
    "llama-3.3-70b-versatile":  {"input": 0.00059, "output": 0.00079},
}


@dataclass
class QueryMetrics:
    timestamp: str
    question: str
    retrieval_latency_sec: float
    llm_latency_sec: float
    total_latency_sec: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    retrieval_scores: list
    retrieval_methods: list
    sources: list
    answer_length: int


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, {"input": 0, "output": 0})
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1000


def log_query(result: dict, question: str, model: str = "gpt-4o-mini") -> QueryMetrics:
    cost = compute_cost(model, result.get("input_tokens", 0), result.get("output_tokens", 0))

    metrics = QueryMetrics(
        timestamp=datetime.utcnow().isoformat(),
        question=question,
        retrieval_latency_sec=result.get("latency_sec", 0),
        llm_latency_sec=result.get("latency_sec", 0),
        total_latency_sec=result.get("total_latency_sec", 0),
        input_tokens=result.get("input_tokens", 0),
        output_tokens=result.get("output_tokens", 0),
        estimated_cost_usd=round(cost, 6),
        retrieval_scores=result.get("retrieval_scores", []),
        retrieval_methods=result.get("retrieval_methods", []),
        sources=result.get("sources", []),
        answer_length=len(result.get("answer", "")),
    )

    # Append to JSONL log
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(asdict(metrics)) + "\n")

    return metrics


def print_metrics(m: QueryMetrics):
    print("\n--- Query Metrics ---")
    print(f"  Latency        : {m.total_latency_sec}s")
    print(f"  Tokens         : {m.input_tokens} in / {m.output_tokens} out")
    print(f"  Estimated cost : ${m.estimated_cost_usd}")
    print(f"  Sources        : {m.sources}")
    print(f"  Ret. scores    : {m.retrieval_scores}")
    print(f"  Methods        : {m.retrieval_methods}")
    print("---------------------\n")
