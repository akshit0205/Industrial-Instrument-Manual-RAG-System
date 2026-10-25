from fastapi.testclient import TestClient

from src import main
from src.schemas import ChunkMetadata


class FakeStore:
    size = 1

    def retrieve(self, query, top_k):
        return [(ChunkMetadata(chunk_id="x", source="manual.pdf", page=4, text="Pressure is 100 psi."), 0.9)]


def test_health_and_empty_index_refusal(monkeypatch):
    monkeypatch.setattr(main, "store", type("Empty", (), {"size": 0})())
    client = TestClient(main.app)
    assert client.get("/health").json() == {"status": "ok", "indexed_chunks": 0}
    assert client.post("/query", json={"question": "What is pressure?"}).status_code == 400


def test_mock_answer_has_citation(monkeypatch):
    monkeypatch.setattr(main, "store", FakeStore())
    monkeypatch.setattr("src.rag.settings.mock_llm", True)
    response = TestClient(main.app).post("/query", json={"question": "What is pressure?"})
    assert response.status_code == 200
    assert "[manual.pdf:4]" in response.json()["answer"]
