from fastapi import FastAPI, HTTPException

from .config import settings
from .ingest import ingest_manuals
from .rag import answer_question
from .schemas import QueryRequest, QueryResponse
from .vector_store import VectorStore

app = FastAPI(title="Industrial Manual RAG")
store = VectorStore.load(settings.index_dir)


@app.get("/health")
def health() -> dict[str, int | str]:
    return {"status": "ok", "indexed_chunks": store.size}


@app.post("/ingest")
def ingest() -> dict[str, int]:
    global store
    chunks = ingest_manuals(settings.manuals_dir, settings.chunk_size, settings.chunk_overlap)
    store = VectorStore()
    store.build(chunks)
    store.save(settings.index_dir)
    return {"indexed_chunks": store.size}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if store.size == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Ingest manuals first.")
    return answer_question(request.question, store, request.top_k)
