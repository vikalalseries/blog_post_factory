"""Render a saved Markdown post to PDF bytes (headings, paragraphs, lists, images, citations)."""

from __future__ import annotations

import io
import re
import urllib.request
from pathlib import Path

from fpdf import FPDF
from PIL import Image

FONT_SETS = [  # (regular, bold)
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/Library/Fonts/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"),
]
IMAGE = re.compile(r"^!\[([^\]]*)\]\((https?://[^)\s]+)\)$")
CITE = re.compile(r"\[(\d+)\]\((https?://[^)\s]+)\)")
BLUE = (67, 56, 202)
TEXT = (55, 65, 81)


def _fetch(url: str) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BlogPostFactory/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.read()
    except Exception:
        return None


def _plain(t: str) -> str:
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    return re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", t)


def post_to_pdf(markdown: str) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(True, margin=20)
    pdf.add_page()

    family, unicode_ok = "Helvetica", False
    for reg, bold in FONT_SETS:
        if Path(reg).is_file() and Path(bold).is_file():
            pdf.add_font("Body", "", reg)
            pdf.add_font("Body", "B", bold)
            family, unicode_ok = "Body", True
            break

    def fix(t: str) -> str:
        return t if unicode_ok else t.encode("latin-1", "replace").decode("latin-1")

    def write_para(text: str, size: float, bold: bool = False, color=TEXT, lh: float = 6.5) -> None:
        pdf.set_font(family, "B" if bold else "", size)
        pos = 0
        for m in CITE.finditer(text):
            pdf.set_text_color(*color)
            pdf.write(lh, fix(_plain(text[pos:m.start()]).rstrip(" ") + " "))
            pdf.set_text_color(*BLUE)
            pdf.write(lh, f"[{m.group(1)}]", link=m.group(2))
            pos = m.end()
        pdf.set_text_color(*color)
        pdf.write(lh, fix(_plain(text[pos:])))
        pdf.ln(lh)

    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            continue
        if (m := IMAGE.match(line)):
            data = _fetch(m.group(2))
            if not data:
                continue
            try:
                w, h = Image.open(io.BytesIO(data)).size
            except Exception:
                continue
            width = min(pdf.epw, 120.0)
            height = width * h / w
            if height > 85:
                width, height = width * 85 / height, 85
            if pdf.get_y() + height > pdf.page_break_trigger:
                pdf.add_page()
            pdf.image(io.BytesIO(data), x=(pdf.w - width) / 2, y=pdf.get_y(), w=width, h=height)
            pdf.set_y(pdf.get_y() + height + 5)
        elif line.startswith("# "):
            write_para(line[2:], 22, True, (17, 24, 39), 9)
            pdf.ln(4)
        elif line.startswith("#"):
            pdf.ln(3)
            write_para(line.lstrip("#").strip(), 15, True, (17, 24, 39), 8)
            pdf.ln(1)
        elif (m := re.match(r"^(?:([-*])|(\d+)\.)\s+(.*)", line)):
            label = "-" if m.group(1) else f"{m.group(2)}."
            pdf.set_font(family, "", 10)
            pdf.set_text_color(*TEXT)
            pdf.set_x(pdf.l_margin)
            pdf.write(6, fix(f"{label} "))
            write_para(m.group(3), 10, lh=6)
        else:
            write_para(line, 11)
            pdf.ln(2.5)
    return bytes(pdf.output())
