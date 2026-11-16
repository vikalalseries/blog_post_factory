"""Retrieve stage: runs on every query."""

from __future__ import annotations

from dataclasses import dataclass

from blog_factory import config
from blog_factory.rag.embeddings import embed
from blog_factory.rag.store import load_store


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float  # cosine similarity, 1.0 = identical meaning


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Embed `query` and return the closest stored chunks."""
    store = load_store()
    if store.count() == 0:
        return []

    k = min(top_k or config.TOP_K, store.count())
    query_vector = embed([query], task_type="RETRIEVAL_QUERY")[0]
    return [
        RetrievedChunk(text=rec.text, source=rec.source, score=score)
        for rec, score in store.query(query_vector, k)
    ]
