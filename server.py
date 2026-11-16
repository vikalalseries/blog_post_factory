"""Web server that connects the UI (ui/index.html) to the Blog Post Factory.

Usage:
    uvicorn server:app --reload
    then open http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from blog_factory import config
from blog_factory.workflow import run_factory
from main import save_post

UI_FILE = Path(__file__).parent / "ui" / "index.html"
SUPPORTED_SUFFIXES = {".txt", ".md"}

app = FastAPI(title="Blog Post Factory")


class GenerateRequest(BaseModel):
    topic: str
    words_min: int = Field(400, ge=200, le=5000)
    words_max: int = Field(600, ge=200, le=5000)
    tone: Literal["normal", "exciting", "happy", "sad", "factual", "funny", "inspirational", "professional", "casual", "dramatic", "serious", "curious", "nostalgic", "persuasive"] = "normal"
    images: Literal["auto", "yes", "no"] = "auto"


class SaveRequest(BaseModel):
    post: str
    topic: str


class UploadFile(BaseModel):
    name: str
    content: str
    encoding: Literal["text", "base64"] = "text"


class LinkRequest(BaseModel):
    url: str


class UploadRequest(BaseModel):
    files: list[UploadFile]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_FILE)


def _list_sources() -> dict:
    from blog_factory.rag.ingest import chunk_text, gather_files

    files = []
    for path in gather_files(config.SOURCES_DIR):
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
        files.append({"name": path.relative_to(config.SOURCES_DIR).as_posix(), "chunks": len(chunks)})
    return {"folder": config.SOURCES_DIR.name, "files": files}


@app.get("/api/config")
def get_config() -> dict:
    return {"model": config.GEMINI_MODEL, **_list_sources()}


@app.get("/api/sources")
def get_sources() -> dict:
    return _list_sources()


@app.post("/api/sources")
def upload_sources(req: UploadRequest) -> dict:
    config.SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    from blog_factory import sources_import as si

    for f in req.files:
        name = Path(f.name).name
        suffix = Path(name).suffix.lower()
        if suffix == ".pdf":
            try:
                text = si.pdf_to_text(base64.b64decode(f.content))
                si.save_text(si.pdf_filename(name), text)
            except Exception as exc:
                raise HTTPException(400, f"{name}: could not read PDF ({exc})")
        elif suffix in SUPPORTED_SUFFIXES:
            (config.SOURCES_DIR / name).write_text(f.content, encoding="utf-8")
        else:
            raise HTTPException(400, f"{name}: only .txt, .md and .pdf files are supported")
    return _list_sources()


@app.post("/api/sources/link")
def add_link(req: LinkRequest) -> dict:
    from blog_factory import sources_import as si

    url = req.url.strip()
    try:
        si.save_text(si.link_filename(url), f"Source: {url}\n\n" + si.link_to_text(url))
    except Exception as exc:
        raise HTTPException(400, f"Could not read that link: {exc}")
    return _list_sources()


@app.post("/api/ingest")
def run_ingest() -> dict:
    from blog_factory.rag.ingest import ingest

    total = ingest(reset=True)
    return {"chunks": total, **_list_sources()}


@app.post("/api/generate")
async def generate(req: GenerateRequest) -> StreamingResponse:
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(400, "Topic is empty")
    if req.words_min > req.words_max:
        raise HTTPException(400, "Invalid word range")

    queue: asyncio.Queue = asyncio.Queue()

    def on_event(kind: str, stage: str, text: str) -> None:
        queue.put_nowait({"type": kind, "stage": stage, "text": text})

    async def worker() -> None:
        try:
            result = await run_factory(topic, verbose=False, on_event=on_event,
                                      words_min=req.words_min, words_max=req.words_max,
                                      tone=req.tone, images=req.images)
            final = result.final_post
            if not final or final.startswith("NO_FACTS_FOUND"):
                queue.put_nowait({"type": "empty"})
            else:
                queue.put_nowait({"type": "done", "post": final})
        except Exception as exc:  # surface any failure to the UI
            queue.put_nowait({"type": "error", "message": str(exc)})
        finally:
            queue.put_nowait(None)

    async def stream():
        task = asyncio.create_task(worker())
        try:
            while (item := await queue.get()) is not None:
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            task.cancel()

    return StreamingResponse(stream(), media_type="text/event-stream")


def _post_path(name: str) -> Path:
    path = config.OUTPUT_DIR / Path(name).name
    if path.suffix != ".md" or not path.is_file():
        raise HTTPException(404, "Post not found")
    return path


@app.get("/api/posts")
def list_posts() -> dict:
    posts = []
    if config.OUTPUT_DIR.is_dir():
        for path in sorted(config.OUTPUT_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
            text = path.read_text(encoding="utf-8", errors="ignore")
            title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), path.stem)
            posts.append({"name": path.name, "title": title, "modified": path.stat().st_mtime})
    return {"posts": posts}


@app.get("/api/posts/{name}")
def read_post(name: str) -> dict:
    return {"name": name, "content": _post_path(name).read_text(encoding="utf-8")}


@app.get("/api/posts/{name}/pdf")
def export_pdf(name: str) -> Response:
    from blog_factory.pdf_export import post_to_pdf

    path = _post_path(name)
    pdf = post_to_pdf(path.read_text(encoding="utf-8"))
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{path.stem}.pdf"'})


@app.post("/api/save")
def save(req: SaveRequest) -> dict:
    path = save_post(req.post, req.topic, None)
    return {"path": str(path)}
