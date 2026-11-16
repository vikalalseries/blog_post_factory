"""Blog Post Factory entry point.

Usage:
    python main.py "the history of chess"
    python main.py                       # prompts for a topic
    python main.py "topic" -o my_post.md # custom output file
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

from blog_factory import config
from blog_factory.workflow import run_factory


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60] or "blog_post"


def save_post(post: str, topic: str, output: str | None) -> Path:
    path = Path(output) if output else config.OUTPUT_DIR / f"{slugify(topic)}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(post + "\n", encoding="utf-8")
    return path


async def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a grounded blog post with 3 agents.")
    parser.add_argument("topic", nargs="?", help="what the blog post should be about")
    parser.add_argument("-o", "--output", help="output Markdown file (default: output/<topic>.md)")
    parser.add_argument("-q", "--quiet", action="store_true", help="only show the final post")
    args = parser.parse_args()

    topic = args.topic or input("Blog topic: ").strip()
    if not topic:
        print("No topic given.")
        return 1

    print(f"Topic: {topic}")
    result = await run_factory(topic, verbose=not args.quiet)

    if not result.final_post or result.final_post.startswith("NO_FACTS_FOUND"):
        print("\nThe web search found nothing relevant for this topic.")
        return 1

    path = save_post(result.final_post, topic, args.output)
    print("\n================ FINAL POST ================\n")
    print(result.final_post)
    print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
