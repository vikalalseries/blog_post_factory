"""Shared local embedding model.

The same model MUST be used for ingestion and for queries, otherwise the
vectors live in different "meaning spaces" and similarity search breaks.
Loading the model is slow, so it is created once and cached.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from blog_factory import config


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Turn a list of texts into a list of embedding vectors."""
    vectors = get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()
