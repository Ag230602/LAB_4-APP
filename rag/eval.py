from __future__ import annotations

from enum import Enum
from typing import List


class MissingEvidenceBehavior(str, Enum):
    OK = "ok"
    REFUSE_AND_ASK_CLARIFY = "refuse_and_ask_clarify"


def compute_pr_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int, metric: str) -> float:
    if k <= 0:
        return 0.0
    if not gold_ids:
        return 0.0
    topk = retrieved_ids[:k]
    hit = sum(1 for x in topk if x in set(gold_ids))
    if metric == "precision":
        return hit / float(k)
    if metric == "recall":
        return hit / float(len(set(gold_ids)))
    raise ValueError("metric must be 'precision' or 'recall'")


def faithfulness_heuristic(answer: str, evidence_ids: List[str]) -> bool:
    """
    Very simple heuristic:
    - Answer is "faithful" if it contains at least one evidence ID like [D1], [A2], etc.
    Replace with your rubric/judge if needed.
    """
    if not evidence_ids:
        return False
    return any(f"[{eid}]" in answer for eid in evidence_ids)
