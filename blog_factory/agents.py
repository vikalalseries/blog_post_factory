"""The three agents. Each has one narrow job and only the tools it needs."""

from __future__ import annotations

from agent_framework import Agent
from agent_framework.gemini import GeminiChatClient

from blog_factory import config
from blog_factory.tools import find_image, web_search

RESEARCHER_INSTRUCTIONS = """You are the RESEARCHER in a blog-writing pipeline. You never write prose.

1. Call `web_search` with the topic. Searches are strictly limited (3 per post), so
   call it only once; call it again only if the first result is clearly insufficient.
   If a result says SEARCH_LIMIT_REACHED, stop searching and use what you have.
2. Reply with ONLY a bulleted list of 6-12 concise factual statements taken from the
   search results. End every bullet with its source URL in brackets, e.g.
   "- Chess originated in India around the 6th century CE. [https://example.com/chess]"
3. Report ONLY what the search results say. Do not add facts from your own knowledge.
4. If the tool returns NO_RESULTS or nothing relevant, reply exactly:
   "NO_FACTS_FOUND: the web search found nothing about this topic."
"""

TONE_GUIDE = {
    "normal": "neutral and clear",
    "exciting": "energetic and enthusiastic, with momentum",
    "happy": "warm, cheerful and upbeat",
    "sad": "melancholic and reflective",
    "factual": "objective, precise and information-focused, with minimal flourish",
    "funny": "light-hearted and witty, with gentle humour",
    "inspirational": "uplifting and motivating",
    "professional": "polished, authoritative and businesslike",
    "casual": "relaxed and conversational, like talking to a friend",
    "dramatic": "intense and cinematic, building tension",
    "serious": "grave, measured and thoughtful",
    "curious": "inquisitive and wonder-filled, inviting the reader to explore",
    "nostalgic": "wistful and sentimental about the past",
    "persuasive": "convincing and confident, guiding the reader to a view",
}

IMAGE_RULES = {
    "auto": """- Images: only where an image genuinely helps (many posts need none), call `find_image`
  with a short description, then place it on its own line between paragraphs as
  ![short alt text](url) using a URL the tool returned. Never invent image URLs.
  Use at most one image per few paragraphs, and none if the tool finds nothing.""",
    "yes": """- Images: the post MUST include relevant images. Call `find_image` with short
  descriptions and place each on its own line between paragraphs as
  ![short alt text](url) using a URL the tool returned (aim for one every 2-3 paragraphs).
  Never invent image URLs; if the tool finds nothing, skip that image.""",
    "no": "- Do NOT include any images.",
}


LONG_NOTE = (
    "- Because the post is long, you may add well-established general background and explanation, "
    "but never invent specific dates, names, numbers, quotes or statistics.\n"
)


def writer_instructions(
    words_min: int, words_max: int, tone: str = "normal", images: str = "auto"
) -> str:
    return f"""You are the WRITER in a blog-writing pipeline. You receive a bulleted list of facts.

Write a blog post in Markdown:
- A title on the first line as a level-1 heading (# Title).
- Length: the post body must be between {words_min} and {words_max} words. Organise longer
  posts with "##" section headings and develop each fact with explanation and context.
- Tone: write in a {tone} voice: {TONE_GUIDE[tone]}. Keep it consistent throughout.
- Use ONLY the facts you were given. Do not invent dates, names, numbers, or claims.
- Cite sources inline: every fact bullet ends with a source number like [2]. After each sentence
  that uses a fact, put that number in square brackets, e.g. "Chess began in India [1]."
  Use only numbers that appear in the facts, and never write URLs in the prose.
{LONG_NOTE if words_min > 600 else ""}{IMAGE_RULES[images]}
- Do NOT write a Sources section; it is added automatically.

If the input says NO_FACTS_FOUND, reply with exactly that line and nothing else.
"""


EDITOR_INSTRUCTIONS = """\
You are the EDITOR in a blog-writing pipeline. You receive a draft blog post.

- Tighten wordy sentences, fix grammar and spelling, and improve flow.
- Ensure there is exactly one clear, compelling level-1 title.
- Keep every fact as-is: do not add new information or remove any [n] citation markers.
- Keep every image line (![alt](url)) exactly as written, in place.
- Keep the post's length within its current range.
- Preserve the post's tone and voice; do not flatten it.
- Return ONLY the finished Markdown post: no preamble, no comments, no code fences.

If the input says NO_FACTS_FOUND, reply with exactly that line and nothing else.
"""


def make_client() -> GeminiChatClient:
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://aistudio.google.com"
        )
    # Retry temporary Gemini failures (overload 503, rate limit 429, timeouts) instead of
    # failing the whole post on the first hiccup.
    from google import genai
    from google.genai import types

    http = types.HttpOptions(
        timeout=120_000,
        retry_options=types.HttpRetryOptions(
            attempts=5,
            initial_delay=2.0,
            max_delay=30.0,
            exp_base=2.0,
            http_status_codes=[408, 429, 500, 502, 503, 504],
        ),
    )
    genai_client = genai.Client(api_key=config.GEMINI_API_KEY, http_options=http)
    return GeminiChatClient(client=genai_client, model=config.GEMINI_MODEL)


def build_agents(
    client: GeminiChatClient | None = None, words_min: int = 400,
    words_max: int = 600,
    tone: str = "normal",
    images: str = "auto",
) -> tuple[Agent, Agent, Agent]:
    client = client or make_client()

    researcher = Agent(
        client=client,
        name="researcher",
        description="Searches the internet and collects grounded facts.",
        instructions=RESEARCHER_INSTRUCTIONS,
        tools=[web_search],
    )
    writer = Agent(
        client=client,
        name="writer",
        description="Drafts a short blog post from the researcher's facts.",
        instructions=writer_instructions(words_min, words_max, tone, images),
        tools=[] if images == "no" else [find_image],
    )
    editor = Agent(
        client=client,
        name="editor",
        description="Polishes the draft into a finished Markdown post.",
        instructions=EDITOR_INSTRUCTIONS,
    )
    return researcher, writer, editor
