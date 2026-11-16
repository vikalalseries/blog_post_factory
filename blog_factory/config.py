"""Central configuration for the Blog Post Factory.

All tunable values live here so the rest of the code never hard-codes paths,
model names, or secrets. Values can be overridden through environment
variables (or a local `.env` file, which is git-ignored).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = folder that contains this package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load secrets/overrides from <project>/.env if it exists.
load_dotenv(PROJECT_ROOT / ".env")

# --- Language model (Google Gemini free tier) ---------------------------------
# The Gemini connector reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment.
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# --- Web search (SerpAPI) -------------------------------------------------------
SERPAPI_API_KEY: str | None = os.getenv("SERPAPI_API_KEY") or os.getenv("serp_api_key")
# How many search results the web_search tool returns per query.
WEB_RESULTS: int = int(os.getenv("WEB_RESULTS", "5"))
# Hard cap on SerpAPI searches per blog post (free tier has a small monthly quota).
MAX_SEARCHES: int = int(os.getenv("MAX_SEARCHES", "3"))

# --- RAG ----------------------------------------------------------------------
SOURCES_DIR: Path = PROJECT_ROOT / os.getenv("SOURCES_DIR", "sources")
CHROMA_DIR: Path = PROJECT_ROOT / os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "blog_sources")

# Small, free, local embedding model (downloaded once on first use, ~90 MB).
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking: target words per chunk, and how many words consecutive chunks share.
CHUNK_SIZE_WORDS: int = int(os.getenv("CHUNK_SIZE_WORDS", "180"))
CHUNK_OVERLAP_WORDS: int = int(os.getenv("CHUNK_OVERLAP_WORDS", "30"))

# How many chunks the retrieval tool returns per query.
TOP_K: int = int(os.getenv("TOP_K", "5"))

# --- Output -------------------------------------------------------------------
OUTPUT_DIR: Path = PROJECT_ROOT / os.getenv("OUTPUT_DIR", "output")
