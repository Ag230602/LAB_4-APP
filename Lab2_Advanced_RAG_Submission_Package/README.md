# Lab 2 — Advanced RAG Results

## Project dataset (full-credit)
- Folder: `project_data/`
- Documents: 8 plain-text `.txt` files (project-aligned to StormVision-3D emergency planning + recovery decision support)
- Total size: ~3–10 pages of content across the docs

## Queries
- **Q1 (normal):** 48 hours before landfall, what are the key criteria for choosing shelter locations under forecast uncertainty?
- **Q2 (normal):** After landfall, what factors make recovery slower, and how should limited repair crews be allocated to support recovery?
- **Q3 (ambiguous/edge):** Where should we deploy help in the next two days?

## Results table (Precision@5, Recall@10)

| Query | Method | Precision@5 | Recall@10 |
|---|---:|---:|---:|
| Q1 | keyword | 1.000 | 0.818 |
| Q1 | vector | 1.000 | 0.818 |
| Q1 | hybrid | 1.000 | 0.818 |
| Q2 | keyword | 1.000 | 0.909 |
| Q2 | vector | 1.000 | 0.909 |
| Q2 | hybrid | 1.000 | 0.909 |
| Q3 | keyword | 0.400 | 0.800 |
| Q3 | vector | 0.400 | 0.800 |
| Q3 | hybrid | 0.400 | 0.600 |


## Screenshots (include these in GitHub)
- Chunking comparison: `screenshots/chunking_comparison.png`
- Re-ranking before vs after: `screenshots/rerank_before_after_q1.png`
- Prompt-only vs RAG answer: `screenshots/prompt_only_vs_rag_q3.png`

## Short reflection
- **Failure case:** Q3 (“deploy help”) is ambiguous. The system can retrieve relevant background, but it may still return a single action if the generator is not forced to clarify.
- **Which layer failed:** **Generation / interaction policy** (it should ask a clarification question or present multiple interpretations by default for ambiguous queries).
- **System-level fix:** Add an “ambiguity detector” before retrieval (missing objective, missing geography, missing resource type). If triggered, the system should either (1) ask a clarification question first, or (2) return 2–3 evidence-grounded options labeled by interpretation (shelters vs medical vs repair crews), then re-run retrieval with the chosen interpretation.
