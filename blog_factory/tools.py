"""Tools the agents may call. Only the researcher gets these."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Annotated

from agent_framework import tool
from pydantic import Field
import serpapi

from blog_factory import config


@tool(approval_mode="never_require")
def retrieve_from_sources(
    topic: Annotated[str, Field(description="The topic or question to look up in the saved sources.")],
) -> str:
    """Search the local library of saved source material for passages relevant to a topic.

    Returns the most relevant passages, each labelled with the source file it came from.
    Call this several times with different phrasings to cover sub-topics.
    """
    from blog_factory.rag.retriever import retrieve  # lazy import

    chunks = retrieve(topic)
    if not chunks:
        return "NO_RESULTS: the sources library is empty or nothing matched. Run ingestion first."

    parts = [
        f"[{i}] (source: {c.source}, relevance: {c.score:.2f})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n---\n\n".join(parts)


_searches_used = 0


def _count_search() -> None:
    global _searches_used
    _searches_used += 1


def reset_search_count() -> None:
    """Start a new blog post with a fresh search budget."""
    global _searches_used
    _searches_used = 0


@tool(approval_mode="never_require")
def web_search(
    query: Annotated[str, Field(description="The topic or question to search the internet for.")],
) -> str:
    """Search the internet (Google via SerpAPI) and return the top results.

    Each result has a title, a URL, and a snippet. The number of searches per blog
    post is limited, so call this sparingly.
    """
    if not config.SERPAPI_API_KEY:
        return "NO_RESULTS: SERPAPI_API_KEY is not set in .env."

    client = serpapi.Client(api_key=config.SERPAPI_API_KEY)

    def search(q: str) -> list[dict]:
        # Hard cap on SerpAPI searches per blog post (free-tier credits are scarce).
        if _searches_used >= config.MAX_SEARCHES:
            return []
        _count_search()
        results = client.search(
            {"engine": "google", "q": q, "google_domain": "google.com", "hl": "en", "gl": "us"}
        )
        return results.get("organic_results", [])

    # Wikipedia first; only if that gives fewer than the limit, fill up with
    # the top organic results from the normal (unrestricted) search.
    limit = config.WEB_RESULTS
    # (`site:en.wikipedia.org` works; plain `site:wikipedia.org` is ignored by Google.)
    wiki = [r for r in search(f"site:en.wikipedia.org {query}") if "wikipedia.org/" in r.get("link", "")]
    organic = wiki[:limit]
    if len(organic) < limit:
        seen = {r.get("link") for r in organic}
        extra = [r for r in search(query) if r.get("link") not in seen]
        organic += extra[: limit - len(organic)]

    if not organic:
        if _searches_used >= config.MAX_SEARCHES:
            return (
                "SEARCH_LIMIT_REACHED: no more searches are allowed. Do not call web_search "
                "again; answer using only the results you already have."
            )
        return "NO_RESULTS: the web search returned nothing for this query."

    parts = [
        f"[{i}] {r.get('title', '')}\nurl: {r.get('link', '')}\n{r.get('snippet', '')}"
        for i, r in enumerate(organic, start=1)
    ]
    return "\n\n---\n\n".join(parts)


@tool(approval_mode="never_require")
def find_image(
    query: Annotated[str, Field(description="A short description of the image wanted, e.g. 'medieval chess pieces'.")],
) -> str:
    """Find a relevant, freely licensed image on Wikimedia Commons.

    Returns lines of 'url | title' (best match first), or NO_IMAGE if nothing fits.
    """
    params = urllib.parse.urlencode({
        "action": "query", "generator": "search", "gsrsearch": query, "gsrnamespace": 6,
        "gsrlimit": 5, "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": 800, "format": "json",
    })
    req = urllib.request.Request(
        f"https://commons.wikimedia.org/w/api.php?{params}",
        headers={"User-Agent": "BlogPostFactory/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            pages = json.load(resp).get("query", {}).get("pages", {})
    except Exception:
        return "NO_IMAGE: image lookup failed."

    found = []
    for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        info = (page.get("imageinfo") or [{}])[0]
        if info.get("mime") in ("image/jpeg", "image/png", "image/webp") and info.get("thumburl"):
            title = page.get("title", "").removeprefix("File:").rsplit(".", 1)[0]
            found.append(f"{info['thumburl']} | {title}")
    return "\n".join(found) or "NO_IMAGE: nothing found for this query."
