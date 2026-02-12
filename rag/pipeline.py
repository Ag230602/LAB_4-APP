from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from .index import EvidenceStore, RetrievalMode
from .eval import compute_pr_at_k, faithfulness_heuristic, MissingEvidenceBehavior
from .logging_utils import append_query_log


@dataclass
class RunConfig:
    metadata_path: str = "data/metadata.json"
    log_path: str = "logs/query_metrics.csv"
    default_top_k: int = 10
    eval_p_at: int = 5
    eval_r_at: int = 10


def run_query(
    question: str,
    store: EvidenceStore,
    retrieval_mode: RetrievalMode,
    top_k: int,
    gold_evidence: Optional[List[str]],
    log_path: str,
    query_id: str,
) -> Dict:
    """
    Runs retrieval + simple grounded answer and appends a single row to logs/query_metrics.csv.
    """
    t0 = time.perf_counter()
    results = store.retrieve(question, mode=retrieval_mode, top_k=top_k)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    evidence_ids = [r["id"] for r in results]

    # "Missing evidence" behavior: if results look weak, refuse/ask for clarification.
    missing_behavior = MissingEvidenceBehavior.OK
    if len(results) == 0 or (results[0].get("score", 0.0) < store.missing_evidence_score_threshold):
        missing_behavior = MissingEvidenceBehavior.REFUSE_AND_ASK_CLARIFY

    answer = generate_grounded_answer(question, results, missing_behavior)

    # Metrics: only compute if gold_evidence is provided (Q4/Q5 often partial/empty)
    p_at = compute_pr_at_k(evidence_ids, gold_evidence or [], k=min(store.eval_p_at, 5), metric="precision")
    r_at = compute_pr_at_k(evidence_ids, gold_evidence or [], k=min(store.eval_r_at, 10), metric="recall")

    faithful = faithfulness_heuristic(answer=answer, evidence_ids=evidence_ids)

    append_query_log(
        log_path=log_path,
        record={
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "query_id": query_id,
            "retrieval_mode": retrieval_mode.value,
            "top_k": int(top_k),
            "latency_ms": round(latency_ms, 2),
            "Precision@5": None if gold_evidence is None else round(p_at, 4),
            "Recall@10": None if gold_evidence is None else round(r_at, 4),
            "evidence_ids_returned": " ".join(evidence_ids),
            "faithfulness_pass": bool(faithful),
            "missing_evidence_behavior": missing_behavior.value,
        },
    )

    return {
        "answer": answer,
        "results": results,
        "metrics": {
            "latency_ms": latency_ms,
            "Precision@5": p_at if gold_evidence is not None else None,
            "Recall@10": r_at if gold_evidence is not None else None,
            "faithfulness_pass": faithful,
            "missing_evidence_behavior": missing_behavior.value,
        },
    }


def generate_grounded_answer(question: str, retrieved: List[Dict], missing_behavior: str) -> str:
    """
    Works out of the box without any external LLM API.

    If you want an LLM, replace this with your Lab-3 generator OR add an API call
    (OpenAI, Gemini, etc.) that uses only retrieved evidence as context.
    """
    if missing_behavior == MissingEvidenceBehavior.REFUSE_AND_ASK_CLARIFY.value:
        return (
            "I don't have enough evidence in the retrieved context to answer confidently.\n\n"
            "Try one of these:\n"
            "- Rephrase the question with a specific entity/date\n"
            "- Increase top_k\n"
            "- Switch retrieval_mode (e.g., hybrid)\n"
            "- Add a relevant document/asset to the dataset\n"
        )

    # Simple evidence-grounded synthesis: cite evidence IDs inline.
    bullets = []
    for ev in retrieved[:5]:
        snippet = ev.get("snippet", "").strip().replace("\n", " ")
        if snippet:
            bullets.append(f"- {snippet} [{ev['id']}]")
    if not bullets:
        bullets = [f"- Retrieved evidence: {ev['title']} [{ev['id']}]" for ev in retrieved[:3]]

    return (
        f"**Answer (grounded):**\n\n"
        f"Question: {question}\n\n"
        f"Key evidence:\n" + "\n".join(bullets) + "\n\n"
        f"Note: Replace this generator with your Lab-3 RAG answerer if required."
    )
