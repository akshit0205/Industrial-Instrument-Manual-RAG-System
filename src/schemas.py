from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    chunk_id: str
    source: str
    page: int
    text: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


class CitedChunk(BaseModel):
    source: str
    page: int
    score: float
    text_snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitedChunk]


class EvalQuestion(BaseModel):
    id: str
    question: str
    expected_source: str
    expected_page: int
    ground_truth_answer: str
