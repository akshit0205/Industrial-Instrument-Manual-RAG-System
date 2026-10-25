# Industrial Instrument Manual RAG

A local-first retrieval-augmented generation service for technical PDF manuals. It uses PyMuPDF for page-aware extraction, `all-MiniLM-L6-v2` embeddings, FAISS, and a configurable OpenAI-compatible chat endpoint.

## Quick start

```powershell
python -m pip install -r requirements.txt
pytest
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Place real manuals in `data/manuals/`, then call `POST /ingest`. Query with `POST /query` using `{"question":"...","top_k":5}`. Set `MOCK_LLM=true` to run deterministic offline responses. The expected inline citation syntax is `[source.pdf:page]`.

## Evaluation

Copy `data/eval/eval_set_template.json` to `data/eval/eval_set.json`, replace every placeholder using real manuals, then run:

```powershell
python scripts/evaluate.py
```

| Configuration | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 |
|---|---:|---:|---:|
| A (300/50) | TBD | TBD | TBD |
| B (600/100) | TBD | TBD | TBD |
| C (1000/200) | TBD | TBD | TBD |

## Architectural tradeoffs

`all-MiniLM-L6-v2` is small enough for practical local CPU embedding while retaining useful semantic retrieval quality. `IndexFlatIP` performs exact inner-product search; with normalized embeddings this is cosine similarity and keeps the implementation straightforward, though it trades scale for accuracy. Smaller chunks can isolate specific instructions but lose surrounding context; larger chunks retain context but can dilute retrieval signals. The evaluation sweep measures that tradeoff on the actual manuals rather than assuming an optimal size.
