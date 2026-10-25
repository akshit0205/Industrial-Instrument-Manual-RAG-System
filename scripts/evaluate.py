"""Evaluate retrieval configurations against a real local evaluation set."""
import json
import sys
from pathlib import Path

from pydantic import TypeAdapter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest import ingest_manuals
from src.schemas import EvalQuestion
from src.vector_store import VectorStore

CONFIGS = {
    "A (300/50)": (300, 50),
    "B (600/100)": (600, 100),
    "C (1000/200)": (1000, 200),
}


def hit_at(results: list[tuple], question: EvalQuestion) -> bool:
    return any(
        chunk.source == question.expected_source and abs(chunk.page - question.expected_page) <= 1
        for chunk, _ in results
    )


def main() -> None:
    eval_path = ROOT / "data/eval/eval_set.json"
    if not eval_path.exists():
        raise FileNotFoundError("Create data/eval/eval_set.json from eval_set_template.json using real manuals.")
    questions = TypeAdapter(list[EvalQuestion]).validate_json(eval_path.read_text(encoding="utf-8"))
    if not questions:
        raise ValueError("eval_set.json must contain at least one question")
    lines = ["| Configuration | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 |", "|---|---:|---:|---:|"]
    for name, (size, overlap) in CONFIGS.items():
        store = VectorStore()
        store.build(ingest_manuals(ROOT / "data/manuals", size, overlap))
        rates = []
        for k in (1, 3, 5):
            hits = sum(hit_at(store.retrieve(question.question, k), question) for question in questions)
            rates.append(hits / len(questions))
        lines.append(f"| {name} | {rates[0]:.2%} | {rates[1]:.2%} | {rates[2]:.2%} |")
    table = "\n".join(lines) + "\n"
    print(table)
    (ROOT / "data/eval/results.md").write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
