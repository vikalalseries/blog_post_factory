"""Where saved posts live: Vercel Blob when BLOB_READ_WRITE_TOKEN is set, else the local output/ folder."""

from __future__ import annotations

import os
import re
from pathlib import Path

from blog_factory import config

PREFIX = "posts/"
_access: str | None = None  # "private" or "public", detected from the store on first use


class PostNotFound(Exception):
    pass


def use_blob() -> bool:
    return bool(os.getenv("BLOB_READ_WRITE_TOKEN"))


def safe_name(name: str) -> str:
    name = Path(name).name
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.md", name):
        raise PostNotFound(name)
    return name


def _title(text: str, fallback: str) -> str:
    return next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), fallback)


# --- Vercel Blob -----------------------------------------------------------------

def _blob_put(name: str, content: str) -> str:
    global _access
    from vercel.blob.sync import put_blob

    data = content.encode("utf-8")
    kwargs = dict(content_type="text/markdown; charset=utf-8", allow_overwrite=True)
    for access in ([_access] if _access else ["private", "public"]):
        try:
            result = put_blob(PREFIX + name, data, access=access, **kwargs)
            _access = access
            return result.url
        except Exception:
            if access == "public" or _access:
                raise
    raise RuntimeError("Could not write to Vercel Blob")


def _blob_get(name: str) -> str:
    from vercel.blob.sync import get_blob

    last: Exception | None = None
    for access in ([_access] if _access else ["private", "public"]):
        try:
            return bytes(get_blob(PREFIX + name, access=access).body).decode("utf-8")
        except Exception as exc:
            last = exc
    raise PostNotFound(name) from last


# --- public API ------------------------------------------------------------------

def save(name: str, content: str) -> str:
    name = safe_name(name)
    if use_blob():
        _blob_put(name, content)
        return f"Vercel Blob: {PREFIX}{name}"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / name
    path.write_text(content + "\n", encoding="utf-8")
    return str(path)


def read(name: str) -> str:
    name = safe_name(name)
    if use_blob():
        return _blob_get(name)
    path = config.OUTPUT_DIR / name
    if not path.is_file():
        raise PostNotFound(name)
    return path.read_text(encoding="utf-8")


def list_posts() -> list[dict]:
    posts: list[dict] = []
    if use_blob():
        from vercel.blob.sync import query_blobs

        blobs = [b for b in query_blobs(prefix=PREFIX) if b.pathname.endswith(".md")]
        blobs.sort(key=lambda b: b.uploaded_at, reverse=True)
        for b in blobs:
            name = b.pathname[len(PREFIX):]
            # Titles come from the file name here; opening a post shows the real title.
            posts.append({"name": name, "title": name[:-3].replace("_", " ").title(),
                          "modified": b.uploaded_at.timestamp()})
        return posts
    if config.OUTPUT_DIR.is_dir():
        for path in sorted(config.OUTPUT_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
            text = path.read_text(encoding="utf-8", errors="ignore")
            posts.append({"name": path.name, "title": _title(text, path.stem), "modified": path.stat().st_mtime})
    return posts
