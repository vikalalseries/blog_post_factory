"""Retrieve stage: runs on every query."""

from __future__ import annotations

from dataclasses import dataclass

from blog_factory import config
from blog_factory.rag.embeddings import embed
from blog_factory.rag.store import get_collection


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float  # cosine similarity, 1.0 = identical meaning


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Embed `query` and return the closest stored chunks."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    k = min(top_k or config.TOP_K, collection.count())
    result = collection.query(
        query_embeddings=embed([query]),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        RetrievedChunk(text=doc, source=meta.get("source", "unknown"), score=1.0 - dist)
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]
