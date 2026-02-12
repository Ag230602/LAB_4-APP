<<<<<<< HEAD
# CS5542 Lab 4 — Single-Person RAG App (Streamlit + GitHub Deployment)

This repo is a **single-person** implementation of Lab 4 requirements:
- Streamlit UI: query input, retrieval controls, generated answer, evidence panel, metrics
- Dataset integration: 2–6 docs + 5–15 multimodal assets (images/tables/figures) via `data/metadata.json`
- Automatic logging: appends one row per query to `logs/query_metrics.csv`
- Failure analysis write-up in `/reports/`

## Quickstart (Local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
streamlit run app/main.py
```

## Dataset Setup

Edit `data/metadata.json`:

- Add **2–6** `document` items (markdown or extracted text)
- Add **5–15** assets (`image`, `figure`, `table`)
  - For images/figures, add a good `caption` (this enables retrieval without heavy vision models)
  - For tables, store as `data/assets/*.md` and include a caption

Each entry needs a unique `id` (these IDs are what you cite and what you put in the gold set).

## Evaluation Queries (Q1–Q5)

In `app/main.py`, update the `GOLD` dictionary:
- Q1–Q3: typical queries with complete gold evidence IDs
- Q4: **multimodal** evidence query (use image/table IDs)
- Q5: missing-evidence or ambiguous query (set `"gold": None`)

## Logging

A new row is appended to: `logs/query_metrics.csv` with columns:
- timestamp, query_id, retrieval_mode, top_k, latency_ms,
- Precision@5, Recall@10,
- evidence_ids_returned, faithfulness_pass, missing_evidence_behavior

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to Streamlit Cloud → **New app**.
3. Select your repo + branch.
4. Set the entry point to: `app/main.py`
5. Deploy.

Add your deployed link here in README once live.

## Failure Analysis

Write-ups:
- `reports/failure_analysis.md` (1 retrieval failure + 1 grounding/missing-evidence failure)
=======
# Big_data_2026_ag
>>>>>>> 89823e16ca7f60518803c0ce7ca42228368fc620
