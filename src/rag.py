"""Grounded answer generation with citation enforcement."""
import re

import httpx

from .config import settings
from .schemas import CitedChunk, QueryResponse
from .vector_store import VectorStore

NOT_FOUND = "Not found in the provided manuals."
CITATION_PATTERN = re.compile(r"\[[^\[\]]+\.pdf:\d+\]")
SYSTEM_PROMPT = """You are a technical assistant for industrial instrumentation manuals.
Answer the user's question ONLY using the provided context chunks.
Rules:
1. Every factual claim MUST include an inline citation in the exact format [source.pdf:page].
2. If the context does not contain the answer, reply with the exact string: "Not found in the provided manuals." Do not guess or extrapolate."""


def format_context(results: list[tuple]) -> str:
    return "\n".join(f"\n{chunk.text}\n---" for chunk, _ in results)


def _mock_answer(results: list[tuple]) -> str:
    chunk, score = results[0]
    if score < 0.25:
        return NOT_FOUND
    return f"Retrieved manual context [${chunk.source}:{chunk.page}]".replace("[$", "[")


def answer_question(question: str, store: VectorStore, top_k: int) -> QueryResponse:
    results = store.retrieve(question, top_k)
    citations = [CitedChunk(
        source=chunk.source, page=chunk.page, score=score, text_snippet=chunk.text[:300]
    ) for chunk, score in results]
    if not results:
        return QueryResponse(answer=NOT_FOUND, citations=[])
    if settings.mock_llm:
        answer = _mock_answer(results)
    else:
        payload = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:{format_context(results)}\n\nQuestion: {question}"},
            ],
            "temperature": 0,
        }
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
        response = httpx.post(f"{settings.llm_base_url.rstrip('/')}/chat/completions", json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
    if answer != NOT_FOUND and not CITATION_PATTERN.search(answer):
        answer = NOT_FOUND
    return QueryResponse(answer=answer, citations=[] if answer == NOT_FOUND else citations)
