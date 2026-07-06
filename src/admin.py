"""
Admin support functions: document management and query-log aggregation.
Kept separate from api.py so route handlers stay thin wrappers around these.
"""

import json
from datetime import datetime
from html import escape
from pathlib import Path

ALLOWED_EXTENSIONS = (".txt", ".pdf")


def list_documents(data_dir: str) -> list[dict]:
    """List source documents in data_dir with basic file metadata."""
    path = Path(data_dir)
    docs = []
    for f in sorted(path.iterdir()):
        if f.suffix.lower() in ALLOWED_EXTENSIONS:
            stat = f.stat()
            docs.append({
                "filename": f.name,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    return docs


def save_document(data_dir: str, filename: str, content: bytes) -> str:
    """
    Save an uploaded document into data_dir. Returns the saved filename.
    Strips any path components from the given filename so an upload can't
    escape data_dir (e.g. filename="../../etc/passwd").
    """
    safe_name = Path(filename).name
    if not safe_name or Path(safe_name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Only {ALLOWED_EXTENSIONS} files are supported.")
    dest = Path(data_dir) / safe_name
    dest.write_bytes(content)
    return safe_name


def delete_document(data_dir: str, filename: str) -> None:
    """Delete a document from data_dir. Raises FileNotFoundError if missing."""
    data_path = Path(data_dir).resolve()
    target = (data_path / Path(filename).name).resolve()
    if target.parent != data_path:
        raise ValueError("invalid filename")
    if not target.is_file():
        raise FileNotFoundError(filename)
    target.unlink()


def _read_log_rows(log_path: str) -> list[dict]:
    rows = []
    p = Path(log_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def summarize_logs(log_path: str) -> dict:
    """Aggregate cost/latency stats across every logged query."""
    rows = _read_log_rows(log_path)
    if not rows:
        return {
            "total_queries": 0,
            "avg_total_latency_sec": 0.0,
            "avg_retrieval_latency_sec": 0.0,
            "avg_llm_latency_sec": 0.0,
            "total_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

    n = len(rows)
    return {
        "total_queries": n,
        "avg_total_latency_sec": round(sum(r.get("total_latency_sec", 0) for r in rows) / n, 3),
        "avg_retrieval_latency_sec": round(sum(r.get("retrieval_latency_sec", 0) for r in rows) / n, 3),
        "avg_llm_latency_sec": round(sum(r.get("llm_latency_sec", 0) for r in rows) / n, 3),
        "total_cost_usd": round(sum(r.get("estimated_cost_usd", 0) for r in rows), 6),
        "total_input_tokens": sum(r.get("input_tokens", 0) for r in rows),
        "total_output_tokens": sum(r.get("output_tokens", 0) for r in rows),
    }


def recent_queries(log_path: str, limit: int = 20) -> list[dict]:
    """Most recent N logged queries, most-recent-first. Lightweight fields only."""
    rows = _read_log_rows(log_path)
    recent = rows[-limit:][::-1]
    return [
        {
            "timestamp": r.get("timestamp"),
            "question": r.get("question"),
            "total_latency_sec": r.get("total_latency_sec"),
            "estimated_cost_usd": r.get("estimated_cost_usd"),
            "sources": r.get("sources"),
        }
        for r in recent
    ]


def render_dashboard_html(stats: dict, recent: list[dict], documents: list[dict]) -> str:
    """
    Fully server-rendered dashboard — all data is embedded at request time,
    no client-side fetch/auth needed since the page itself is already behind
    the admin-key check.
    """
    doc_rows = "\n".join(
        f"<tr><td>{escape(d['filename'])}</td>"
        f"<td class='num'>{d['size_bytes']:,}</td>"
        f"<td>{escape(d['modified'][:19])}</td></tr>"
        for d in documents
    ) or "<tr><td colspan='3' class='empty'>No documents in data/</td></tr>"

    query_rows = "\n".join(
        f"<tr><td>{escape(r['timestamp'][:19] if r['timestamp'] else '')}</td>"
        f"<td>{escape(r['question'] or '')}</td>"
        f"<td class='num'>{r['total_latency_sec']}</td>"
        f"<td class='num'>${r['estimated_cost_usd']}</td></tr>"
        for r in recent
    ) or "<tr><td colspan='4' class='empty'>No queries logged yet</td></tr>"

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>IAEA RAG — Admin</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; background: #f4f6f5; color: #1a2422; margin: 0; padding: 32px; }}
  h1 {{ font-size: 20px; margin: 0 0 24px; }}
  h2 {{ font-size: 14px; text-transform: uppercase; letter-spacing: 0.04em; color: #5a6b66; margin: 32px 0 12px; }}
  .cards {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .card {{ background: #fff; border: 1px solid #dde4e2; border-radius: 8px; padding: 14px 18px; min-width: 140px; }}
  .card .label {{ font-size: 12px; color: #5a6b66; }}
  .card .value {{ font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; border: 1px solid #dde4e2; border-radius: 8px; overflow: hidden; }}
  th, td {{ text-align: left; padding: 8px 12px; font-size: 13px; border-bottom: 1px solid #eef1f0; }}
  th {{ background: #eef3f1; font-weight: 600; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.empty {{ text-align: center; color: #8a9793; padding: 20px; }}
</style>
</head>
<body>
  <h1>IAEA RAG — Admin Dashboard</h1>

  <div class="cards">
    <div class="card"><div class="label">Total queries</div><div class="value">{stats['total_queries']}</div></div>
    <div class="card"><div class="label">Avg total latency</div><div class="value">{stats['avg_total_latency_sec']}s</div></div>
    <div class="card"><div class="label">Avg retrieval latency</div><div class="value">{stats['avg_retrieval_latency_sec']}s</div></div>
    <div class="card"><div class="label">Avg LLM latency</div><div class="value">{stats['avg_llm_latency_sec']}s</div></div>
    <div class="card"><div class="label">Total cost</div><div class="value">${stats['total_cost_usd']}</div></div>
    <div class="card"><div class="label">Tokens (in/out)</div><div class="value">{stats['total_input_tokens']:,}/{stats['total_output_tokens']:,}</div></div>
  </div>

  <h2>Documents ({len(documents)})</h2>
  <table>
    <tr><th>Filename</th><th>Size (bytes)</th><th>Modified</th></tr>
    {doc_rows}
  </table>

  <h2>Recent Queries</h2>
  <table>
    <tr><th>Timestamp</th><th>Question</th><th>Latency</th><th>Cost</th></tr>
    {query_rows}
  </table>
</body>
</html>"""
