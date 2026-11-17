"""Where saved posts live.

Deployed (Vercel): Vercel Blob, selected when BLOB_READ_WRITE_TOKEN is set.
Local development: the output/ folder, when no blob token is set.

Blob settings (environment variables):
    BLOB_READ_WRITE_TOKEN     token for the store (enables Blob storage)
    BLOB_STORE_ID_STORE_ID    id of the store (used with the token, as in tests/test_storage.py)
    BLOB_ACCESS               "public" (default) or "private"
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from blog_factory import config

PREFIX = "posts/"


class PostNotFound(Exception):
    pass


def _token() -> str | None:
    # On Vercel a linked Blob store provides the store id and a per-deployment OIDC token
    # (VERCEL_OIDC_TOKEN); locally you use BLOB_READ_WRITE_TOKEN from .env.
    return os.getenv("BLOB_READ_WRITE_TOKEN") or os.getenv("VERCEL_OIDC_TOKEN")


def use_blob() -> bool:
    return bool(_token())


def safe_name(name: str) -> str:
    name = Path(name).name
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.md", name):
        raise PostNotFound(name)
    return name


def _title(text: str, fallback: str) -> str:
    return next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), fallback)


# --- Vercel Blob -----------------------------------------------------------------

def _access() -> str:
    access = os.getenv("BLOB_ACCESS", "public").strip().lower()
    return access if access in ("public", "private") else "public"


def _credentials():
    from vercel import blob

    return blob.BlobCredentials(
        token=_token() or "",
        store_id=os.environ["BLOB_STORE_ID_STORE_ID"],
        kind=blob.CredentialKind.OIDC,
    )


@contextmanager
def _blob_session() -> Iterator[None]:
    """Run blob calls with the explicit store credentials when a store id is configured."""
    if not os.getenv("BLOB_STORE_ID_STORE_ID"):
        yield  # let the SDK work out the store from the token alone
        return

    from vercel.api import session
    from vercel.blob.sync import SyncBlobServiceOptions

    options = SyncBlobServiceOptions(credentials_factory=_credentials)
    with session(service_options=[options]):
        yield


def _blob_put(name: str, content: str) -> str:
    from vercel.blob.sync import put_blob

    with _blob_session():
        result = put_blob(
            PREFIX + name,
            content.encode("utf-8"),
            access=_access(),
            content_type="text/markdown; charset=utf-8",
            allow_overwrite=True,
        )
    return result.url


def _blob_get(name: str) -> str:
    from vercel.blob import BlobNotFoundError
    from vercel.blob.sync import get_blob

    try:
        with _blob_session():
            return bytes(get_blob(PREFIX + name, access=_access()).body).decode("utf-8")
    except BlobNotFoundError as exc:
        raise PostNotFound(name) from exc


def _blob_list() -> list[dict]:
    from vercel.blob.sync import query_blobs

    with _blob_session():
        blobs = [b for b in query_blobs(prefix=PREFIX) if b.pathname.endswith(".md")]
    blobs.sort(key=lambda b: b.uploaded_at, reverse=True)
    # Titles come from the file name here; opening a post shows the real title.
    return [
        {
            "name": b.pathname[len(PREFIX):],
            "title": b.pathname[len(PREFIX):-3].replace("_", " ").title(),
            "modified": b.uploaded_at.timestamp(),
        }
        for b in blobs
    ]


# --- public API ------------------------------------------------------------------

def _require_writable_local() -> None:
    # Vercel's filesystem is read-only, so fail with a clear message instead of an OSError.
    if os.getenv("VERCEL"):
        seen = sorted(
            f"{k}{'' if v.strip() else ' (EMPTY)'}"
            for k, v in os.environ.items()
            if "BLOB" in k.upper() or "OIDC" in k.upper()
        )
        raise RuntimeError(
            "Storage is not configured: neither BLOB_READ_WRITE_TOKEN nor VERCEL_OIDC_TOKEN is "
            f"set in this deployment (env: {os.getenv('VERCEL_ENV')}). "
            f"Blob-related variables visible here: {seen or 'none'}."
        )


def save(name: str, content: str) -> str:
    name = safe_name(name)
    if use_blob():
        return _blob_put(name, content)
    _require_writable_local()
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
    if use_blob():
        return _blob_list()
    posts: list[dict] = []
    if config.OUTPUT_DIR.is_dir():
        for path in sorted(config.OUTPUT_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
            text = path.read_text(encoding="utf-8", errors="ignore")
            posts.append({"name": path.name, "title": _title(text, path.stem), "modified": path.stat().st_mtime})
    return posts
