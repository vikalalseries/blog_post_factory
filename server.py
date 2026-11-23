"""Web server that connects the UI (ui/index.html) to the Blog Post Factory.

Usage:
    uvicorn server:app --reload
    then open http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from blog_factory import config
from blog_factory.workflow import run_factory
from blog_factory import storage
from main import slugify

UI_FILE = Path(__file__).parent / "ui" / "index.html"
STATIC_DIR = Path(__file__).parent / "static"
SUPPORTED_SUFFIXES = {".txt", ".md"}

app = FastAPI(title="Blog Post Factory")
# Mounted before the other routes so /static/... is always served as files.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class GenerateRequest(BaseModel):
    topic: str
    words_min: int = Field(400, ge=200, le=5000)
    words_max: int = Field(600, ge=200, le=5000)
    tone: Literal["normal", "exciting", "happy", "sad", "factual", "funny", "inspirational", "professional", "casual", "dramatic", "serious", "curious", "nostalgic", "persuasive"] = "normal"
    images: Literal["auto", "yes", "no"] = "auto"
    # Research from an earlier run on the same topic; when sent, the researcher is skipped.
    facts: str | None = Field(None, max_length=30_000)


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


MAX_AUDIO_BYTES = 3_000_000   # one short recording (a few seconds of webm/opus is ~20-60 KB)
MIN_AUDIO_BYTES = 1_000       # smaller than this is silence / an empty recording


@app.post("/api/transcribe")
async def transcribe(request: Request) -> dict:
    """Voice typing, short-piece mode: raw audio body in, {"text"} out (Vercel-friendly, stateless)."""
    from blog_factory.speech import SpeechFailed, SpeechUnavailable, transcribe as dg_transcribe

    audio = await request.body()
    if len(audio) < MIN_AUDIO_BYTES:
        return {"text": ""}
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Recording is too large.")
    try:
        return {"text": await dg_transcribe(audio, request.headers.get("content-type") or "audio/webm")}
    except SpeechUnavailable as exc:
        raise HTTPException(503, str(exc))
    except SpeechFailed as exc:
        raise HTTPException(502, str(exc))


@app.post("/api/voice-token")
async def voice_token() -> dict:
    """Voice typing, live mode: a 60 s temporary token so the browser can stream to Deepgram
    directly. The real API key never leaves the server."""
    from blog_factory.speech import SpeechFailed, SpeechUnavailable, voice_token as dg_token

    try:
        return {"token": await dg_token(60)}
    except SpeechUnavailable as exc:
        raise HTTPException(503, str(exc))
    except SpeechFailed as exc:  # e.g. the key's role cannot create tokens -> browser falls back to /api/transcribe
        raise HTTPException(502, str(exc))


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


class ClassifyRequest(BaseModel):
    text: str


@app.post("/api/classify")
async def classify(req: ClassifyRequest) -> dict:
    """Intent detector: topic -> write a post, chat -> suggest topics, off_topic -> reject."""
    from blog_factory.intent import classify_intent

    return (await classify_intent(req.text)).model_dump()


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
                                      tone=req.tone, images=req.images, facts=req.facts)
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


def _read_post(name: str) -> str:
    try:
        return storage.read(name)
    except storage.PostNotFound:
        raise HTTPException(404, "Post not found")


@app.get("/api/img")
def proxy_image(url: str) -> Response:
    """Serve Wikimedia images through this server (with retries) so the browser never hotlinks them."""
    import time
    import urllib.error
    import urllib.parse
    import urllib.request

    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    if parsed.scheme not in ("http", "https") or not (host == "wikimedia.org" or host.endswith(".wikimedia.org")):
        raise HTTPException(400, "Only Wikimedia images can be loaded")
    req = urllib.request.Request(url, headers={"User-Agent": "BlogPostFactory/1.0 (image proxy)"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                ctype = resp.headers.get("Content-Type", "")
                data = resp.read(8_000_000)
            if not ctype.startswith("image/"):
                raise HTTPException(502, "Not an image")
            return Response(data, media_type=ctype, headers={"Cache-Control": "public, max-age=86400"})
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise HTTPException(502, f"Image host returned {exc.code}")
        except HTTPException:
            raise
        except Exception:
            if attempt == 2:
                raise HTTPException(502, "Could not fetch the image")
        time.sleep(1.5 * (attempt + 1))
    raise HTTPException(502, "Could not fetch the image")


@app.get("/api/posts")
def list_posts() -> dict:
    try:
        return {"posts": storage.list_posts()}
    except Exception as exc:
        raise HTTPException(500, f"Could not list saved posts: {exc}")


@app.get("/api/posts/{name}")
def read_post(name: str) -> dict:
    return {"name": name, "content": _read_post(name)}


@app.get("/api/posts/{name}/pdf")
def export_pdf(name: str) -> Response:
    from blog_factory.pdf_export import post_to_pdf

    pdf = post_to_pdf(_read_post(name))
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{Path(name).stem}.pdf"'})


@app.post("/api/save")
def save(req: SaveRequest) -> dict:
    try:
        # A timestamp in the name keeps every save as its own post instead of overwriting.
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        storage.save(f"{slugify(req.topic)}_{stamp}.md", req.post)
        return {"message": "Post saved successfully"}
    except Exception as exc:
        raise HTTPException(500, f"Save failed: {exc}")
