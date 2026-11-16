"""Access to the local Chroma vector database."""

from __future__ import annotations

import chromadb
from chromadb.api.models.Collection import Collection

from blog_factory import config


def get_collection() -> Collection:
    """Open (or create) the persistent collection that holds source chunks.

    Cosine distance is used because our embeddings are normalized.
    """
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection() -> Collection:
    """Delete and recreate the collection (used for a clean re-ingest)."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:  # collection did not exist yet
        pass
    return get_collection()
