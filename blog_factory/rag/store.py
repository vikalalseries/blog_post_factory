"""A tiny file-based vector store (JSON + cosine similarity in pure Python).

The source library is small, so a brute-force search is instant and avoids
heavy dependencies. The JSON file is read-only at runtime on serverless hosts;
build it locally with `python -m blog_factory.rag.ingest --reset` and deploy it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from blog_factory import config


@dataclass
class Record:
    id: str
    text: str
    source: str
    chunk: int
    embedding: list[float]


class VectorStore:
    def __init__(self, records: list[Record] | None = None) -> None:
        self.records: list[Record] = records or []

    def count(self) -> int:
        return len(self.records)

    def replace_source(self, source: str, new_records: list[Record]) -> None:
        """Drop stale chunks of `source`, then add the new ones."""
        self.records = [r for r in self.records if r.source != source] + new_records

    def query(self, embedding: list[float], k: int) -> list[tuple[Record, float]]:
        """Return the top-k records with their cosine similarity (vectors are unit length)."""
        scored = [(r, sum(a * b for a, b in zip(embedding, r.embedding))) for r in self.records]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def save(self) -> None:
        config.VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": config.EMBEDDING_MODEL,
            "dimensions": config.EMBEDDING_DIMENSIONS,
            "records": [r.__dict__ for r in self.records],
        }
        config.VECTOR_STORE_PATH.write_text(json.dumps(data), encoding="utf-8")


def load_store() -> VectorStore:
    """Load the store from disk (empty if it has not been built yet)."""
    path = config.VECTOR_STORE_PATH
    if not path.is_file():
        return VectorStore()
    data = json.loads(path.read_text(encoding="utf-8"))
    return VectorStore([Record(**r) for r in data.get("records", [])])
