import os
import sys
import json
import time
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Utilities / Metrics / Logging
# -----------------------------
LOG_COLUMNS = [
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

class MissingEvidenceBehavior(str, Enum):
    OK = "ok"
    REFUSE_AND_ASK_CLARIFY = "refuse_and_ask_clarify"

def append_query_log(log_path: str, record: Dict):
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    row = {c: record.get(c, None) for c in LOG_COLUMNS}
    df = pd.DataFrame([row])

    if p.exists():
        df.to_csv(p, mode="a", header=False, index=False)
    else:
        df.to_csv(p, mode="w", header=True, index=False)

def compute_pr_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int, metric: str) -> float:
    if k <= 0:
        return 0.0
    if not gold_ids:
        return 0.0
    topk = retrieved_ids[:k]
    gold = set(gold_ids)
    hit = sum(1 for x in topk if x in gold)
    if metric == "precision":
        return hit / float(k)
    if metric == "recall":
        return hit / float(len(gold))
    raise ValueError("metric must be 'precision' or 'recall'")

def faithfulness_heuristic(answer: str, evidence_ids: List[str]) -> bool:
    # Simple heuristic: must cite at least one evidence ID like [D1]
    return any(f"[{eid}]" in answer for eid in evidence_ids)

# -----------------------------
# Retrieval
# -----------------------------
class RetrievalMode(str, Enum):
    TFIDF = "tfidf"
    BM25 = "sparse"     # BM25
    DENSE = "dense"     # sentence-transformers
    HYBRID = "hybrid"   # fusion

@dataclass
class EvidenceItem:
    id: str
    type: str
    path: str
    title: str
    caption: str = ""

class EvidenceStore:
    """
    Loads evidence from data/metadata.json.
    Supports: TF-IDF, BM25, Dense, Hybrid.

    For images/figures: retrieval happens via title + caption (alt-text).
    For tables/docs: retrieval uses file text + title + caption.
    """
    def __init__(self, metadata_path: str, eval_p_at: int = 5, eval_r_at: int = 10):
        self.metadata_path = metadata_path
        self.eval_p_at = eval_p_at
        self.eval_r_at = eval_r_at

        self.items: List[EvidenceItem] = []
        self.texts: List[str] = []
        self.missing_evidence_score_threshold = 0.05

        self._tfidf = None
        self._tfidf_matrix = None

        self._bm25 = None
        self._bm25_tokens = None

        self._dense_model = None
        self._dense_matrix = None

        self._load()

    def _load(self):
        meta = json.loads(Path(self.metadata_path).read_text(encoding="utf-8"))
        for e in meta.get("evidence", []):
            self.items.append(
                EvidenceItem(
                    id=e["id"],
                    type=e.get("type", "document"),
                    path=e["path"],
                    title=e.get("title", e["id"]),
                    caption=e.get("caption", "") or e.get("text_hint", ""),
                )
            )
        self.texts = [self._as_text(it) for it in self.items]

    def _read_text_file(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8", errors="ignore")

    def _as_text(self, it: EvidenceItem) -> str:
        if it.type in ("document", "table", "text"):
            body = self._read_text_file(it.path)
        else:
            body = ""
        return f"TITLE: {it.title}\nCAPTION: {it.caption}\nBODY:\n{body}".strip()

    def build_indexes(self, enable_dense: bool = True):
        # TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(stop_words="english", max_features=50000)
        self._tfidf_matrix = self._tfidf.fit_transform(self.texts)

        # BM25
        from rank_bm25 import BM25Okapi
        self._bm25_tokens = [self._tokenize(t) for t in self.texts]
        self._bm25 = BM25Okapi(self._bm25_tokens)

        # Dense
        if enable_dense:
            from sentence_transformers import SentenceTransformer
            self._dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._dense_matrix = self._dense_model.encode(self.texts, normalize_embeddings=True)

    def retrieve(self, query: str, mode: RetrievalMode, top_k: int = 10) -> List[Dict]:
        if self._tfidf is None or self._bm25 is None:
            raise RuntimeError("Indexes not built. Call store.build_indexes() once at startup.")

        mode = RetrievalMode(mode)

        if mode == RetrievalMode.TFIDF:
            scores = self._score_tfidf(query)
        elif mode == RetrievalMode.BM25:
            scores = self._score_bm25(query)
        elif mode == RetrievalMode.DENSE:
            scores = self._score_dense(query)
        elif mode == RetrievalMode.HYBRID:
            s1 = self._minmax(self._score_tfidf(query))
            s2 = self._minmax(self._score_bm25(query))
            if self._dense_matrix is not None:
                s3 = self._minmax(self._score_dense(query))
                scores = 0.34 * s1 + 0.33 * s2 + 0.33 * s3
            else:
                scores = 0.5 * s1 + 0.5 * s2
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        top_idx = np.argsort(-scores)[:top_k]
        out = []
        for i in top_idx:
            it = self.items[int(i)]
            out.append({
                "id": it.id,
                "type": it.type,
                "title": it.title,
                "path": it.path,
                "score": float(scores[int(i)]),
                "snippet": self._make_snippet(self.texts[int(i)], max_chars=260),
                "caption": it.caption,
            })
        return out

    def _make_snippet(self, text: str, max_chars: int = 260) -> str:
        text = " ".join(text.split())
        return text[:max_chars] + ("…" if len(text) > max_chars else "")

    def _tokenize(self, s: str) -> List[str]:
        return [w.lower() for w in s.replace("\n", " ").split() if w.strip()]

    def _score_tfidf(self, query: str) -> np.ndarray:
        qv = self._tfidf.transform([query])
        scores = (self._tfidf_matrix @ qv.T).toarray().ravel()
        return scores

    def _score_bm25(self, query: str) -> np.ndarray:
        qtok = self._tokenize(query)
        return np.array(self._bm25.get_scores(qtok), dtype=float)

    def _score_dense(self, query: str) -> np.ndarray:
        if self._dense_model is None or self._dense_matrix is None:
            return np.zeros(len(self.items), dtype=float)
        q = self._dense_model.encode([query], normalize_embeddings=True)[0]
        return (self._dense_matrix @ q).astype(float)

    def _minmax(self, x: np.ndarray) -> np.ndarray:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-9:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

# -----------------------------
# Answer Generation (Grounded)
# -----------------------------
def generate_grounded_answer(question: str, retrieved: List[Dict], missing_behavior: str) -> str:
    if missing_behavior == MissingEvidenceBehavior.REFUSE_AND_ASK_CLARIFY.value:
        return (
            "I don't have enough evidence in the retrieved context to answer confidently.\n\n"
            "Try:\n"
            "- Rephrase with a specific entity/date\n"
            "- Increase top_k\n"
            "- Switch retrieval_mode (e.g., hybrid)\n"
            "- Add the missing document/asset to the dataset\n"
        )

    bullets = []
    for ev in retrieved[:5]:
        snippet = (ev.get("snippet") or "").strip().replace("\n", " ")
        if snippet:
            bullets.append(f"- {snippet} [{ev['id']}]")

    if not bullets:
        bullets = [f"- {ev['title']} [{ev['id']}]" for ev in retrieved[:3]]

    return (
        f"**Answer (grounded):**\n\n"
        f"**Question:** {question}\n\n"
        f"**Evidence used:**\n" + "\n".join(bullets)
    )

def run_query(
    question: str,
    store: EvidenceStore,
    retrieval_mode: RetrievalMode,
    top_k: int,
    gold_evidence: Optional[List[str]],
    log_path: str,
    query_id: str,
) -> Dict:
    t0 = time.perf_counter()
    results = store.retrieve(question, mode=retrieval_mode, top_k=top_k)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    evidence_ids = [r["id"] for r in results]

    missing_behavior = MissingEvidenceBehavior.OK
    if len(results) == 0 or (results[0].get("score", 0.0) < store.missing_evidence_score_threshold):
        missing_behavior = MissingEvidenceBehavior.REFUSE_AND_ASK_CLARIFY

    answer = generate_grounded_answer(question, results, missing_behavior.value)

    p_at = compute_pr_at_k(evidence_ids, gold_evidence or [], k=5, metric="precision") if gold_evidence is not None else None
    r_at = compute_pr_at_k(evidence_ids, gold_evidence or [], k=10, metric="recall") if gold_evidence is not None else None

    faithful = faithfulness_heuristic(answer=answer, evidence_ids=evidence_ids) if missing_behavior == MissingEvidenceBehavior.OK else True

    append_query_log(
        log_path=log_path,
        record={
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "query_id": query_id,
            "retrieval_mode": retrieval_mode.value,
            "top_k": int(top_k),
            "latency_ms": round(latency_ms, 2),
            "Precision@5": None if p_at is None else round(float(p_at), 4),
            "Recall@10": None if r_at is None else round(float(r_at), 4),
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
            "Precision@5": p_at,
            "Recall@10": r_at,
            "faithfulness_pass": faithful,
            "missing_evidence_behavior": missing_behavior.value,
        },
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Lab 4 — Ultimate Single File RAG", layout="wide")
st.title("CS 5542 Lab 4 — Ultimate Single-File RAG App")
st.caption("One file only: retrieval + UI + metrics + logging")

@st.cache_resource
def load_store(metadata_path: str):
    store = EvidenceStore(metadata_path=metadata_path, eval_p_at=5, eval_r_at=10)
    store.build_indexes(enable_dense=True)
    return store

st.sidebar.header("Dataset")
metadata_path = st.sidebar.text_input("metadata.json path", value="data/metadata.json")
if not Path(metadata_path).exists():
    st.error(f"metadata.json not found at: {metadata_path}")
    st.stop()

store = load_store(metadata_path)

st.sidebar.header("Retrieval Settings")
retrieval_mode = st.sidebar.selectbox("retrieval_mode", [m.value for m in RetrievalMode], index=3)
top_k = st.sidebar.slider("top_k", 1, 30, 10, 1)

st.sidebar.header("Logging")
log_path = st.sidebar.text_input("log file", value="logs/query_metrics.csv")

# ---- Update these to your Lab-3 query set + gold evidence IDs ----
GOLD = {
    "Q1": {"query": "What is the main objective of the project?", "gold": ["D1"]},
    "Q2": {"query": "Summarize the most important findings and constraints.", "gold": ["D1", "D2"]},
    "Q3": {"query": "Which components are used in the pipeline and why?", "gold": ["D2"]},
    "Q4": {"query": "Use the figure/table to support the answer and cite evidence.", "gold": ["A1", "T1"]},
    "Q5": {"query": "Ask something NOT in the dataset (should refuse/clarify).", "gold": None},
}

st.subheader("Query")
c1, c2 = st.columns([3, 1])
with c1:
    q = st.text_input("Enter your question", value=GOLD["Q1"]["query"])
with c2:
    query_id = st.selectbox("Query ID (for logging)", options=list(GOLD.keys()), index=0)

run_btn = st.button("Run")

if run_btn and q.strip():
    gold = GOLD[query_id]["gold"]
    out = run_query(
        question=q,
        store=store,
        retrieval_mode=RetrievalMode(retrieval_mode),
        top_k=top_k,
        gold_evidence=gold,
        log_path=log_path,
        query_id=query_id,
    )

    st.subheader("Generated Answer")
    st.markdown(out["answer"])

    st.subheader("Metrics")
    m = out["metrics"]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Latency (ms)", f"{m['latency_ms']:.1f}")
    mc2.metric("Precision@5", "N/A" if m["Precision@5"] is None else f"{m['Precision@5']:.3f}")
    mc3.metric("Recall@10", "N/A" if m["Recall@10"] is None else f"{m['Recall@10']:.3f}")
    mc4.metric("Faithfulness", "PASS" if m["faithfulness_pass"] else "FAIL")
    st.caption(f"Missing-evidence behavior: **{m['missing_evidence_behavior']}**")

    st.subheader("Retrieved Evidence")
    for r in out["results"]:
        with st.expander(f"{r['id']} — {r['title']} (score={r['score']:.4f})"):
            st.write(f"Type: `{r['type']}`  |  Path: `{r['path']}`")
            if r["type"] in ("image", "figure"):
                if Path(r["path"]).exists():
                    st.image(r["path"], caption=r.get("caption", ""))
                else:
                    st.warning("Image file not found. Check metadata.json paths.")
                if r.get("caption"):
                    st.write("Caption:", r["caption"])
            else:
                st.code(r.get("snippet", ""), language="text")
                if r.get("caption"):
                    st.write("Caption/Hint:", r["caption"])

    st.success(f"Logged 1 record to {log_path}")
else:
    st.info("Pick Q1–Q5, update metadata.json + GOLD evidence IDs, then click Run.")
