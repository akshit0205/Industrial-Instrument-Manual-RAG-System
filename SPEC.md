# Project Specification: Industrial Instrument Manual RAG System

Build a local-first RAG system over technical PDF manuals. Do not invent APIs, manual content, or evaluation metrics. Ground every answer in retrieved chunks, using inline citations in the form `[source.pdf:page]`; where the retrieved context is insufficient, return exactly `Not found in the provided manuals.`

Use the pinned dependencies in `requirements.txt`. Keep the layout limited to `data`, `src`, `scripts`, `tests`, Dockerfile, requirements, and README. PDF extraction must use PyMuPDF page enumeration, normalize whitespace, skip pages shorter than 20 characters, and preserve source/page metadata in page-contained character chunks. Embeddings are normalized `all-MiniLM-L6-v2` vectors in `faiss.IndexFlatIP` and persist to `data/index`.

The service exposes `/health`, `/ingest`, and `/query`; generation uses an OpenAI-compatible HTTP endpoint and a deterministic `MOCK_LLM=true` mode. Evaluation sweeps 300/50, 600/100, and 1000/200 chunks and writes real Hit@k results only after an evaluation set based on actual manuals is supplied. Run pytest after every phase.
