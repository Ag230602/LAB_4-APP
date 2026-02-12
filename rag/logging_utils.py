from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


COLUMNS = [
    "timestamp",
    "query_id",
    "retrieval_mode",
    "top_k",
    "latency_ms",
    "Precision@5",
    "Recall@10",
    "evidence_ids_returned",
    "faithfulness_pass",
    "missing_evidence_behavior",
]


def append_query_log(log_path: str, record: Dict):
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    row = {c: record.get(c, None) for c in COLUMNS}
    df = pd.DataFrame([row])

    if p.exists():
        df.to_csv(p, mode="a", header=False, index=False)
    else:
        df.to_csv(p, mode="w", header=True, index=False)
