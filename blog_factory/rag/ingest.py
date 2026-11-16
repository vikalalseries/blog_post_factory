"""One-time ingestion routine: gather -> chunk -> embed -> store.

Run it whenever you add or change files in the sources folder:

    python -m blog_factory.rag.ingest            # incremental upsert
    python -m blog_factory.rag.ingest --reset    # wipe and rebuild
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from blog_factory import config
from blog_factory.rag.embeddings import embed
from blog_factory.rag.store import get_collection, reset_collection

SUPPORTED_SUFFIXES = {".txt", ".md"}


# 1. Gather ---------------------------------------------------------------------
def gather_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)


# 2. Chunk ----------------------------------------------------------------------
def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into ~`size`-word chunks along paragraph boundaries.

    Paragraphs are packed together until a chunk reaches `size` words. A
    paragraph longer than `size` is split by words with `overlap` words of
    context carried between pieces.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current))
        current, current_len = [], 0

    for para in paragraphs:
        words = para.split()
        if len(words) > size:
            flush()
            step = max(1, size - overlap)
            for start in range(0, len(words), step):
                chunks.append(" ".join(words[start : start + size]))
                if start + size >= len(words):
                    break
            continue
        if current_len + len(words) > size:
            flush()
        current.append(para)
        current_len += len(words)
    flush()
    return chunks


def _chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{source}::{index}::{digest}"


# 3 + 4. Embed and store --------------------------------------------------------
def ingest(reset: bool = False) -> int:
    files = gather_files(config.SOURCES_DIR)
    if not files:
        print(f"No .txt/.md files found in {config.SOURCES_DIR}. Add some sources first.")
        return 0

    collection = reset_collection() if reset else get_collection()
    total = 0

    for path in files:
        source = path.relative_to(config.SOURCES_DIR).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
        if not chunks:
            continue

        # Remove stale chunks from a previous version of this file.
        collection.delete(where={"source": source})

        collection.add(
            ids=[_chunk_id(source, i, c) for i, c in enumerate(chunks)],
            documents=chunks,
            embeddings=embed(chunks),
            metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
        )
        print(f"  {source}: {len(chunks)} chunk(s)")
        total += len(chunks)

    print(f"Ingested {total} chunk(s) from {len(files)} file(s) into '{config.COLLECTION_NAME}'.")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the sources folder into Chroma.")
    parser.add_argument("--reset", action="store_true", help="wipe the collection before ingesting")
    args = parser.parse_args()
    ingest(reset=args.reset)


if __name__ == "__main__":
    main()
