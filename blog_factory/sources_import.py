"""Turn uploaded PDFs and web links into plain-text files for the sources library."""

from __future__ import annotations

import io
import ipaddress
import re
import socket
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from blog_factory import config

MAX_BYTES = 10_000_000


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return _clean("\n\n".join(page.extract_text() or "" for page in reader.pages))


def _check_public(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Enter a full http(s) link")
    for info in socket.getaddrinfo(parsed.hostname, None):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise ValueError("Links to local or private addresses are not allowed")


def link_to_text(url: str) -> str:
    _check_public(url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 BlogPostFactory/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        ctype = resp.headers.get("Content-Type", "")
        data = resp.read(MAX_BYTES)
    if "pdf" in ctype or url.lower().split("?")[0].endswith(".pdf"):
        return pdf_to_text(data)
    charset = re.search(r"charset=([\w-]+)", ctype)
    html = data.decode(charset.group(1) if charset else "utf-8", errors="ignore")
    parser = _Text()
    parser.feed(html)
    return _clean("".join(parser.parts))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60] or "source"


def save_text(name: str, text: str) -> str:
    if len(text.split()) < 20:
        raise ValueError("No readable text found")
    config.SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    (config.SOURCES_DIR / name).write_text(text + "\n", encoding="utf-8")
    return name


def link_filename(url: str) -> str:
    p = urllib.parse.urlparse(url)
    return f"link_{slug(p.hostname + p.path)}.txt"


def pdf_filename(name: str) -> str:
    return f"{slug(Path(name).stem)}_pdf.txt"
