# Public RAG

## Project layout

- `rag/` — reusable RAG code: loading, chunking, retrieval, prompting, generation, and evaluation.
- `ingest.py` — command-line entry point for indexing documents.
- `web_server.py` — local web application entry point.
- `evaluation.py` — compatibility entry point for the retrieval quality report.
- `web/` — browser interface.
- `data/raw/` — temporary upload queue.
- `data/processed/` — indexed source documents.
- `storage/` — local Chroma database and manifest; generated locally and ignored by Git.

## Run the project

From the project root:

```powershell
py ingest.py
py web_server.py
```

Then open `http://localhost:8000`.

Run the retrieval metrics report with:

```powershell
py evaluation.py
```

Keep the local `.env` file out of Git. It contains the API key and machine-specific configuration.
