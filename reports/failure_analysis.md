# Failure Analysis (Lab 4)

## Failure 1 — Retrieval Failure
**Query ID:** (e.g., Q2)  
**Observed behavior:**  
- Top evidence did not include the correct document/asset.

**Root cause (likely):**
- Weak captions for multimodal assets OR sparse text in docs.
- Retrieval mode mismatch (e.g., TF-IDF only) for short queries.

**Proposed fix:**
- Improve captions/alt-text and add a short "asset summary" paragraph.
- Switch to HYBRID retrieval by default.
- Add query expansion (synonyms) for key entities.

## Failure 2 — Grounding / Missing-Evidence Failure
**Query ID:** (e.g., Q5)  
**Observed behavior:**  
- System answered despite missing evidence OR refused when evidence existed.

**Root cause (likely):**
- Missing-evidence threshold too low/high (`missing_evidence_score_threshold`).
- Generator not enforcing citations.

**Proposed fix:**
- Tune the score threshold using logged queries.
- Enforce “must cite at least 1 evidence ID” rule; otherwise refuse/ask clarify.
- If using an LLM, pass only retrieved context and require citations.
