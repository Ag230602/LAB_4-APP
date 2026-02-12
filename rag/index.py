from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class RetrievalMode(str, Enum):
    TFIDF = "tfidf"
    BM25 = "sparse"
    DENSE = "dense"
    HYBRID = "hybrid"


@dataclass
class EvidenceItem:
    id: str
    type: str  # document | image | table | figure
    path: str
    title: str
    caption: str = ""


class EvidenceStore:
    """
    Evidence store that supports:
    - TF-IDF retrieval (fast, light)
    - BM25 retrieval (sparse)
    - Dense retrieval (sentence-transformers)
    - Hybrid score fusion

    Multimodal assets are included via captions/alt-text (simple but effective for Lab-4).
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
        evs = meta.get("evidence", [])
        for e in evs:
            item = EvidenceItem(
                id=e["id"],
                type=e.get("type", "document"),
                path=e["path"],
                title=e.get("title", e["id"]),
                caption=e.get("caption", "") or e.get("text_hint", ""),
            )
            self.items.append(item)

        # Build a text representation for each evidence item.
        self.texts = [self._as_text(it) for it in self.items]

    def _read_text_file(self, path: str) -> str:
        p = Path(path)
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8", errors="ignore")

    def _as_text(self, it: EvidenceItem) -> str:
        # For documents/tables: read file text. For images/figures: use caption.
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
            # Weighted fusion of TF-IDF + BM25 + Dense (if available)
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
                "snippet": self._make_snippet(self.texts[int(i)], max_chars=240),
                "caption": it.caption,
            })
        return out

    def _make_snippet(self, text: str, max_chars: int = 240) -> str:
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
        scores = np.array(self._bm25.get_scores(qtok), dtype=float)
        return scores

    def _score_dense(self, query: str) -> np.ndarray:
        if self._dense_model is None or self._dense_matrix is None:
            # Dense not enabled. Return zeros (lets hybrid still work).
            return np.zeros(len(self.items), dtype=float)
        q = self._dense_model.encode([query], normalize_embeddings=True)[0]
        return (self._dense_matrix @ q).astype(float)

    def _minmax(self, x: np.ndarray) -> np.ndarray:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi - lo < 1e-9:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)
