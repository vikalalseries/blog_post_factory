"""Post-processing that guarantees what the user asked for.

A single LLM pass from a handful of facts rarely reaches long word counts, and
the model may skip the image tool, so length and images are enforced here.
"""

from __future__ import annotations

import asyncio
import re
from typing import Callable

from agent_framework import Agent

from blog_factory.agents import TONE_GUIDE, make_client
from blog_factory.tools import find_image

IMAGE_LINE = re.compile(r"^!\[[^\]]*\]\(https?://[^)\s]+\)\s*$")
MAX_PASSES = 6


URL_TAG = re.compile(r"\[(https?://[^\]\s]+)\]")


def number_sources(facts: str) -> tuple[str, list[str]]:
    """Replace each fact's [url] tag with a [n] number; return the text and the url list."""
    urls: list[str] = []

    def repl(m: re.Match) -> str:
        url = m.group(1).rstrip(".,;")
        if url not in urls:
            urls.append(url)
        return f"[{urls.index(url) + 1}]"

    return URL_TAG.sub(repl, facts), urls


async def ensure_citations(post: str, numbered_facts: str, urls: list[str]) -> str:
    """If the writer/editor left no [n] markers, have the model add them (text otherwise unchanged)."""
    if not urls or re.search(r"\[\d+\]", post):
        return post
    agent = Agent(
        client=make_client(),
        name="citer",
        instructions="You add citation markers to blog posts. Return ONLY the full Markdown post.",
    )
    prompt = (
        "Add inline citation markers like [2] to the blog post below. After each sentence that "
        "states something taken from the numbered facts, append the number of the matching fact's "
        "source. Change nothing else: keep all wording, headings and image lines exactly.\n\n"
        f"FACTS:\n{numbered_facts}\n\nPOST:\n{post}"
    )
    try:
        text = _clean((await agent.run(prompt)).text)
    except Exception:
        return post
    return text if text and re.search(r"\[\d+\]", text) else post


def add_citations(post: str, urls: list[str]) -> str:
    """Turn [n] markers into [n](url) links and append a numbered Sources section."""
    post = re.split(r"^##\s+Sources\b", post, maxsplit=1, flags=re.M | re.I)[0].rstrip()

    # "[1, 2]" -> "[1][2]" so each number becomes its own link
    post = re.sub(
        r"\[(\d+(?:\s*[,;]\s*\d+)+)\]",
        lambda m: "".join(f"[{n}]" for n in re.findall(r"\d+", m.group(1))),
        post,
    )

    def link(m: re.Match) -> str:
        n = int(m.group(1))
        if not 1 <= n <= len(urls):
            return ""
        return f" [{n}]({urls[n - 1].replace('(', '%28').replace(')', '%29')})"

    post = re.sub(r"[ \t]*\[(\d+)\](?!\()", link, post)
    if urls:
        post += "\n\n## Sources\n\n" + "\n".join(f"{i}. {u}" for i, u in enumerate(urls, 1))
    return post


def _clean(text: str) -> str:
    """Strip code fences and any preamble before the '# Title' line; '' if none."""
    text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text.strip(), flags=re.I)
    m = re.search(r"^#\s+\S", text, re.M)
    return text[m.start():].strip() if m else ""


def count_words(post: str) -> int:
    """Words of prose: excludes image lines, headings and the Sources section."""
    body = re.split(r"^##\s+Sources\b", post, maxsplit=1, flags=re.M | re.I)[0]
    lines = [l for l in body.splitlines() if not IMAGE_LINE.match(l.strip()) and not l.lstrip().startswith("#")]
    return len(re.findall(r"\b[\w'’-]+\b", " ".join(lines)))


async def adjust_length(
    post: str, words_min: int, words_max: int, tone: str,
    note: Callable[[str], None] | None = None,
) -> str:
    target = (words_min + words_max) // 2
    agent = Agent(
        client=make_client(),
        name="expander",
        instructions="You rewrite blog posts to a requested length. Return ONLY the full Markdown post.",
    )
    bad = 0
    for _ in range(MAX_PASSES + 2):
        n = count_words(post)
        if words_min <= n <= words_max:
            break
        goal = min(target, n * 2) if n < words_min else target
        verb = "Expand" if n < words_min else "Condense"
        if note:
            note(f"{verb} to ~{goal} words (now {n})")
        prompt = (
            f"{verb} the blog post below from {n} words to about {goal} words "
            f"(final range {words_min}-{words_max}, excluding headings and Sources).\n"
            "- When expanding: add new '##' sections and deepen existing ones with explanation, "
            "context, examples and well-established background.\n"
            "- Do NOT invent specific dates, names, numbers, quotes or statistics that are not in the post.\n"
            f"- Keep the same title, a {tone} tone ({TONE_GUIDE[tone]}), every image line "
            "(![alt](url)) and every [n] citation marker exactly as they are.\n"
            "- Return ONLY the complete Markdown post, no preamble or code fences.\n\n"
            f"{post}"
        )
        try:
            text = _clean((await agent.run(prompt)).text)
        except Exception as exc:  # transient API error: try the next pass
            text = ""
            if note:
                note(f"Warning: expansion call failed ({type(exc).__name__}), retrying")
        if text:
            post = text
        elif (bad := bad + 1) >= 2:
            break
    n = count_words(post)
    if not words_min <= n <= words_max and note:
        note(f"Warning: finished at {n} words (target {words_min}-{words_max})")
    return post


def _lookup(query: str) -> list[tuple[str, str]]:
    out = find_image.func(query)
    results = []
    for line in out.splitlines():
        url, _, title = line.partition(" | ")
        if url.startswith("http"):
            results.append((url.strip(), title.strip()))
    return results


async def add_images(
    post: str, topic: str, mode: str, note: Callable[[str], None] | None = None
) -> str:
    """Insert Wikimedia Commons images between paragraphs if the writer added none."""
    if mode == "no" or any(IMAGE_LINE.match(l.strip()) for l in post.splitlines()):
        return post

    words = count_words(post)
    want = max(1, words // 250) if mode == "yes" else min(3, words // 500)
    if want == 0:
        return post

    lines = post.splitlines()
    # Anchor points: the first paragraph under the title and under each "##" heading.
    anchors: list[tuple[int, str]] = []  # (line index after which to insert, query)
    heading = topic
    seen_para = True
    for i, line in enumerate(lines):
        s = line.strip()
        if re.match(r"^##\s+Sources\b", s, re.I):
            break
        if s.startswith("#"):
            heading = f"{topic} {s.lstrip('#').strip()}" if s.startswith("##") else topic
            seen_para = False
        elif s and not seen_para:
            anchors.append((i, heading))
            seen_para = True
    if not anchors:
        return post

    # Spread the images evenly over the available anchors.
    step = max(1, len(anchors) // want)
    chosen = anchors[::step][:want]

    used: set[str] = set()
    inserts: list[tuple[int, str]] = []
    for idx, query in chosen:
        for q in (query, topic, topic.split()[-1]) * 2:
            hits = [h for h in await asyncio.to_thread(_lookup, q) if h[0] not in used]
            if hits:
                url, title = hits[0]
                used.add(url)
                alt = re.sub(r"[\[\]]", "", title) or topic
                inserts.append((idx, f"![{alt}]({url})"))
                break

    if not inserts and note:
        note("Warning: no matching images were found")
    for idx, md in sorted(inserts, reverse=True):
        lines[idx + 1:idx + 1] = ["", md]
    return "\n".join(lines)
