"""Hosted Gemini embeddings (no local model, so the deployment stays tiny).

The same model MUST be used for ingestion and for queries, otherwise the
vectors live in different "meaning spaces" and similarity search breaks.
"""

from __future__ import annotations

import math

import httpx

from blog_factory import config

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_BATCH_LIMIT = 100  # API maximum requests per batchEmbedContents call


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def embed(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Turn texts into unit-length embedding vectors via the Gemini API.

    Use task_type="RETRIEVAL_QUERY" for search queries and
    "RETRIEVAL_DOCUMENT" for the chunks being stored.
    """
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set; it is required for embeddings.")

    model = config.EMBEDDING_MODEL
    vectors: list[list[float]] = []
    with httpx.Client(timeout=60) as client:
        for start in range(0, len(texts), _BATCH_LIMIT):
            batch = texts[start : start + _BATCH_LIMIT]
            payload = {
                "requests": [
                    {
                        "model": f"models/{model}",
                        "content": {"parts": [{"text": text}]},
                        "taskType": task_type,
                        "outputDimensionality": config.EMBEDDING_DIMENSIONS,
                    }
                    for text in batch
                ]
            }
            resp = client.post(
                f"{_BASE_URL}/models/{model}:batchEmbedContents",
                headers={"x-goog-api-key": config.GEMINI_API_KEY},
                json=payload,
            )
            resp.raise_for_status()
            vectors.extend(_normalize(e["values"]) for e in resp.json()["embeddings"])
    return vectors
