"""
RAGAS Offline Batch Evaluation
-------------------------------
Scores logged queries (logs/query_log.jsonl) for:
  - faithfulness      : is the answer grounded in the retrieved context?
  - answer_relevancy   : does the answer actually address the question?

Runs offline/batch instead of inline per-request. Each RAGAS metric is an
LLM-as-judge call — inlining this into /query would add seconds of extra
latency to every request for a quality signal that's only useful in
aggregate, not per-response.

Groq note: ResponseRelevancy defaults to strictness=3 (asks the judge LLM
for 3 completions in one call, n=3). Groq's API rejects n>1, so this script
pins strictness=1. That means each relevancy score reflects a single
generated question instead of an average over three, which is a weaker
(more single-shot / noisier) estimate — reasonable to revisit if this script
moves to an OpenAI-compatible judge model.

Usage:
    python scripts/evaluate_ragas.py
    python scripts/evaluate_ragas.py --limit 20
    python scripts/evaluate_ragas.py --output logs/ragas_report.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from src.pipeline import PipelineConfig


def load_samples(log_path: Path, limit: int | None):
    rows = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit:
        rows = rows[-limit:]

    # Rows logged before 'answer'/'contexts' were added can't be scored.
    usable = [r for r in rows if r.get("answer") and r.get("contexts")]
    return usable, len(rows) - len(usable)


def main():
    parser = argparse.ArgumentParser(description="RAGAS batch evaluation over logged queries.")
    parser.add_argument("--log-path", default="logs/query_log.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the N most recent queries.")
    parser.add_argument("--output", default=None, help="Optional path to write per-sample scores as JSON.")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"No log file found at {log_path}. Run some queries first.")
        return

    rows, skipped = load_samples(log_path, args.limit)
    if skipped:
        print(f"Skipping {skipped} row(s) missing 'answer'/'contexts' (logged before this script existed).")
    if not rows:
        print("No scoreable rows found.")
        return

    from ragas import evaluate, SingleTurnSample, EvaluationDataset
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings

    cfg = PipelineConfig()
    judge_llm = LangchainLLMWrapper(ChatGroq(model=cfg.llm_model))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=row["contexts"],
        )
        for row in rows
    ])

    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy(strictness=1)],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    scores = result.to_pandas()

    print("\n=== RAGAS Evaluation Report ===")
    print(f"Samples evaluated      : {len(rows)}")
    print(f"Mean faithfulness      : {scores['faithfulness'].mean():.3f}")
    print(f"Mean answer_relevancy  : {scores['answer_relevancy'].mean():.3f}")
    print("================================\n")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "timestamp": row.get("timestamp"),
                "question": row["question"],
                "faithfulness": float(score_row["faithfulness"]),
                "answer_relevancy": float(score_row["answer_relevancy"]),
            }
            for row, (_, score_row) in zip(rows, scores.iterrows())
        ]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"Per-sample scores written to {out_path}")


if __name__ == "__main__":
    main()
