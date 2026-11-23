# Blog Post Factory — Student Notes (file by file, line by line)

These notes explain **what every file does, why it is written that way, and how it connects to the rest of the project**. Line numbers refer to the code as it is now; if you edit a file the numbers will drift, but the explanations still apply.

---

## 0. The big picture

**What the project does:** you type a topic ("the history of chess") into a chat box (or speak it with the mic).  Three AI agents work one after another:

```
Topic
  │
  ▼
RESEARCHER ──(searches Google via SerpAPI, writes bullet facts + source URLs)
  │
  ▼  NumberSources  (turns each [url] into [1], [2] …)
WRITER ──(writes a Markdown blog post from ONLY those facts)
  │
  ▼  ReplyToUserTurn  (hands only the draft text on)
EDITOR ──(polishes grammar, title, flow)
  │
  ▼  Post-processing (plain Python + extra LLM calls)
  │   adjust_length → add_images → ensure_citations → add_citations
  ▼
Final Markdown post  →  shown in the chat  →  Save  →  Vercel Blob (or output/ folder)
```

**Technology used**

| Piece | Tool | Why |
|---|---|---|
| LLM | Google Gemini (`google-genai`) | Free tier, good enough for writing |
| Agent orchestration | Microsoft **Agent Framework** (`agent_framework`) | Gives `Agent`, `tool`, and a `SequentialBuilder` workflow |
| Web search | **SerpAPI** (Google results) | Real, current facts with URLs |
| Images | Wikimedia Commons API | Free-licensed images, no key needed |
| Web server | **FastAPI** + **uvicorn** | Serves the UI, the JSON/streaming API, the voice-typing routes and `/static` |
| UI | One HTML file with plain JavaScript (chat-style layout) | No build step, easy to deploy |
| Voice typing | **Deepgram** over plain HTTP (`httpx`, model `nova-3`) | Works on Vercel; the key stays on the server (see §17) |
| Storage | **Vercel Blob** (deployed) or `output/` folder (local) | Vercel's disk is read-only, so posts need external storage |
| PDF export | `fpdf2` + `Pillow` | Turn saved Markdown into a PDF |
| RAG (extra) | Gemini embeddings + a JSON vector file | Search your own documents (see §6 — currently not wired to the agents) |

**Folder map**

```
blog_post_factory/
├── main.py                 CLI entry point (+ slugify helper used by the server)
├── server.py               FastAPI web server (API for the UI)
├── ui/index.html           The whole front end (chat-style UI)
├── static/deepgram_mic.js  Voice-typing script (served at /static/)
├── requirements.txt        Python packages
├── .env                    Secret keys (git-ignored, never deployed)
├── .env.example            Template: copy to .env and paste your own keys
├── sources/                Your own reference .txt/.md files (for RAG)
├── vector_store.json       Generated index of those files
├── output/                 Locally saved posts (git-ignored)
├── tests/test_storage.py   Your manual Vercel Blob upload test
└── blog_factory/           The real logic (a Python package)
    ├── config.py           All settings, read from .env
    ├── agents.py           The three agents + their instructions
    ├── tools.py            web_search, find_image, retrieve_from_sources
    ├── workflow.py         Connects the agents and runs them
    ├── postprocess.py      Length / images / citations enforcement
    ├── storage.py          Save/read/list posts (Blob or local)
    ├── pdf_export.py       Markdown → PDF
    ├── sources_import.py   PDFs/web links → text files for sources/
    ├── speech.py           Deepgram speech-to-text (httpx: transcribe + temporary token)
    └── rag/                embeddings.py, ingest.py, retriever.py, store.py
```

**Two ways to run it:** `python main.py "topic"` (command line) or `uvicorn server:app --reload` (web UI). Both call the same function, `run_factory()` in `workflow.py`.

---

## 1. `blog_factory/config.py` — all settings in one place

Every other file imports `config` instead of reading environment variables itself. **Why:** one place to change a model name or path; no secrets hard-coded in code.

| Lines | What / why |
|---|---|
| 1–6 | Docstring. Says: settings can be overridden by environment variables or `.env`. |
| 8 | `from __future__ import annotations` lets us write type hints like `str \| None` on any Python 3 version. |
| 10–13 | Imports: `os` (read env vars), `Path` (safe file paths), `load_dotenv` (read the `.env` file). |
| 16 | `PROJECT_ROOT` = the folder containing the package. `__file__` is this file; `.resolve()` makes it absolute; `.parent.parent` goes up from `blog_factory/` to the project folder. Used so paths work no matter where you run Python from. |
| 19 | `load_dotenv(...)` reads `.env` and puts its lines into `os.environ`. It does **not** overwrite variables that already exist. On Vercel there is no `.env`, so values come from the Vercel dashboard instead. |
| 23 | `GEMINI_API_KEY` — your Gemini key; falls back to `GOOGLE_API_KEY`. `str \| None` = may be missing. |
| 24 | `GEMINI_MODEL` — model name, with a default if not set. |
| 27 | `SERPAPI_API_KEY` — key for web search (accepts upper- or lower-case name; your `.env` uses `serp_api_key`). |
| 29 | `WEB_RESULTS` — how many search results to keep per search (default 5). |
| (new) | `DEEPGRAM_API_KEY` — key for voice typing. Reads `DEEPGRAM_API_KEY`, or the older name `text_to_speech` used in some `.env` files. Placed after the SerpAPI block. |
| 31 | `MAX_SEARCHES` — hard cap on SerpAPI calls per post (default 3) because free SerpAPI credits are limited. |
| 34 | `SOURCES_DIR` — folder of your own documents. |
| 36 | `VECTOR_STORE_PATH` — the JSON index file. Comment explains it is committed so a read-only server can still load it. |
| 40–41 | Embedding model and vector size (768). Smaller vectors = smaller index file. Changing them requires re-ingesting. |
| 44–45 | Chunking: ~180 words per chunk, 30 words overlap between chunks. |
| 48 | `TOP_K` — how many chunks the retriever returns (5). |
| 51 | `OUTPUT_DIR` — where posts are saved locally. |

`int(os.getenv("X", "5"))`: environment values are always text, so numbers are converted with `int()`.

---

## 2. `blog_factory/agents.py` — the three agents

An **agent** here = an LLM client + instructions (a "system prompt") + optional tools. Each agent has *one narrow job*; that is why the pipeline is reliable.

### 2.1 Imports (lines 3–9)
- `Agent` — the agent class from Agent Framework.
- `GeminiChatClient` — the connector that lets an `Agent` talk to Gemini.
- `find_image`, `web_search` — the tools (from `tools.py`).

### 2.2 `RESEARCHER_INSTRUCTIONS` (lines 11–22)
The prompt for the researcher. Key rules and **why**:
1. "Call `web_search` once" — saves SerpAPI credits.
2. "Reply with ONLY a bulleted list of 6–12 facts, each ending with `[url]`" — a strict format so that code later can parse it (the UI's Facts tab and `number_sources()` look for `- text [url]`).
3. "Report ONLY what the results say" — prevents the model inventing facts (hallucination).
4. "If nothing found, reply exactly `NO_FACTS_FOUND…`" — a magic word the rest of the code checks for.

### 2.3 `TONE_GUIDE` (lines 24–39)
Dictionary: tone name → description ("funny" → "light-hearted and witty…"). The writer and the length-adjuster insert this text into their prompts. The same 14 names appear in the server's allowed list (`server.py` line 36) and the UI dropdown.

### 2.4 `IMAGE_RULES` (lines 41–51)
Three prompts for the three image modes: `auto` (only if helpful), `yes` (must include), `no` (none). `writer_instructions` picks one.

### 2.5 `LONG_NOTE` (lines 54–57)
Extra rule added only for posts over 600 words: the writer may add general background but must never invent numbers, names, quotes.

### 2.6 `writer_instructions(...)` (lines 60–78)
A *function* (not a constant) because the prompt depends on the user's choices: min/max words, tone, images. It is an f-string; `{words_min}` etc. are filled in. Notable rules:
- Title as `# Title`; use `##` headings for long posts.
- "Use ONLY the facts you were given."
- Cite with `[n]` numbers (line 71–73) — these numbers match the `SOURCES:` table produced by `NumberSources` in `workflow.py`. Never write URLs in the prose.
- "Do NOT write a Sources section" — code adds it later (`add_citations`).
- `{LONG_NOTE if words_min > 600 else ""}` — conditional inclusion inside an f-string.
- `NO_FACTS_FOUND` passthrough: if the researcher found nothing, the writer repeats that line so the pipeline can stop cleanly.

### 2.7 `EDITOR_INSTRUCTIONS` (lines 81–93)
Editor rules: tighten, fix grammar, exactly one title, **keep every fact, `[n]` marker and image line**, keep length and tone, return only Markdown (no chatter or code fences).

### 2.8 `make_client()` (lines 96–118)
Builds the Gemini client.
- 97–101: if no API key, raise a friendly error.
- 104–105: import `genai` inside the function (lazy import) so importing `agents.py` stays cheap.
- 107–116: `HttpOptions` with a 120-second timeout and **retry options**: up to 5 attempts, waiting 2s, 4s, 8s … (max 30s) when Gemini answers with temporary errors — 408, 429 (rate limit), 500, 502, 503 (overloaded), 504. **Why:** the free tier is often busy; retrying avoids failing a whole post on one hiccup.
- 117–118: create the raw `genai.Client` and wrap it in `GeminiChatClient` with the model name.

### 2.9 `build_agents(...)` (lines 121–149)
Creates and returns `(researcher, writer, editor)`.
- `client = client or make_client()` — reuse one client for all three.
- **researcher**: tool `web_search`.
- **writer**: tool `find_image` (empty list if images == "no").
- **editor**: no tools — it only edits text.
- Each agent has a `name` (used as the id in events) and a `description`.

> Connection: `workflow.py` calls `build_agents()`; `postprocess.py` imports `TONE_GUIDE` and `make_client` from here.

---

## 3. `blog_factory/tools.py` — functions the LLM may call

A **tool** = a normal Python function the model can ask the framework to run. `@tool(approval_mode="never_require")` registers it and says "run automatically, no human approval". The function's docstring and `Annotated[..., Field(description=...)]` are what the model reads to decide how to call it, so they are written carefully.

### 3.1 `retrieve_from_sources` (lines 17–36)
Searches your own documents (RAG). Imports `retrieve` lazily (line 26), formats results as `[1] (source: file, relevance: 0.83)` + text, joined with `---`. Returns `NO_RESULTS…` if nothing. **Note:** this tool exists but `agents.py` does not currently give it to any agent (see §6).

### 3.2 Search budget (lines 39–50)
`_searches_used` is a module-level counter. `_count_search()` adds one. `reset_search_count()` sets it to 0 and is called at the start of each post (`run_factory`). `global` is needed to modify a module variable inside a function.

### 3.3 `web_search` (lines 53–100)
- 62–63: no key → return an error string (tools return strings so the model can read them).
- 65: create the SerpAPI client.
- 67–75: inner function `search(q)`: if the budget is used up return `[]`; else count it and call Google via SerpAPI; return `organic_results`.
- 79–82: **Wikipedia first**: search `site:en.wikipedia.org <query>` and keep only real Wikipedia links, up to `WEB_RESULTS`. **Why:** Wikipedia is a reliable factual source.
- 83–86: if fewer than the limit, run a normal search and add results not already seen (using a `set` of links to avoid duplicates).
- 88–94: if nothing: return `SEARCH_LIMIT_REACHED` (tells the model to stop searching) or `NO_RESULTS`.
- 96–100: format each result as `[i] title / url: … / snippet`.

So one `web_search` call can use **two** of the 3 searches (Wikipedia + general).

### 3.4 `find_image` (lines 103–131)
- 111–114: build a Wikimedia Commons API query (`urlencode` makes safe URL parameters): search in namespace 6 (files), 5 results, ask for image URL and mime type, thumbnail width 800.
- 115–118: `Request` with a `User-Agent` header (Wikimedia requires one).
- 119–123: open with a 10 s timeout; on any error return `NO_IMAGE`.
- 125–131: sort pages by search rank; keep only jpeg/png/webp that have a thumbnail URL; build lines `url | title`; return them or `NO_IMAGE`.

---

## 4. `blog_factory/workflow.py` — connecting the agents and running them

### 4.1 Imports (lines 5–22)
`dataclass` for a small result class; `Workflow`, `Message`, `Executor`, `handler`, `WorkflowContext`, `SequentialBuilder` from Agent Framework; our `build_agents`, post-processing functions and `reset_search_count`.

### 4.2 Constants and `_diagnose` (lines 24–49)
- `STAGE_LABELS` — human text printed to the terminal per agent.
- `EMPTY_HANDOFF` (line 32) — Gemini rejects empty messages, so if an agent returns nothing we pass this text on instead.
- `_diagnose(stage)` — fallback used only when a failure event has no details: it sends a tiny "ping" to Gemini to see whether the API itself is working, and builds an explanatory message.

### 4.3 `ReplyToUserTurn` (lines 52–62)
A custom **Executor** (a node in the workflow). `@handler` marks the method that receives the previous agent's response. It takes only `agent_response.text` and sends it onward as a **user** message. **Why:** each agent should see only the previous agent's final text — not tool calls or raw search data — and Gemini needs conversations that begin and end with a user turn.

### 4.4 `NumberSources` (lines 65–73)
Sits between researcher and writer. Calls `number_sources()` (from `postprocess.py`) to change `[https://…]` tags into `[1]`, `[2]`, then appends a `SOURCES:` table (`[1] https://…`). The writer therefore cites with small numbers instead of long URLs.

### 4.5 `build_workflow(...)` (lines 76–96)
`SequentialBuilder` runs participants in order:
`researcher → NumberSources → writer → ReplyToUserTurn → editor`.
`intermediate_output_from=[researcher, writer]` means "also stream the researcher's and writer's text", so the UI can show Facts and Draft live (the editor's text is the normal output).

### 4.6 `FactoryResult` (lines 99–102)
Dataclass: `final_post` (the finished Markdown) and `stages` (dict of each agent's text).

### 4.7 `run_factory(...)` (lines 105–165) — the heart of the project
Parameters: topic, `verbose` (print to terminal), `on_event` (callback the server uses to stream to the browser), word range, tone, images.

- 114: reset the search counter.
- 115: build the workflow.
- 119: `workflow.run(topic, stream=True)` — an **async generator** yielding events as work happens.
- 120–125: `executor_invoked` event = a stage began → remember it, notify the UI (`on_event("stage", …)`).
- 127–136: `intermediate`/`output` events carry small text *chunks*. They are appended per agent into `stages`, printed, and forwarded (`on_event("chunk", …)`) — this is the live typing effect.
- 138–145: failure events → raise `RuntimeError` with the real error type/message (and print the traceback to the server log). If no details exist, fall back to `_diagnose`.
- 148: the editor's text is the draft of the final post.
- 151–164: **Post-processing** (only if not `NO_FACTS_FOUND`):
  1. `adjust_length` — force the word range,
  2. `add_images` — insert images if allowed,
  3. `ensure_citations` — make sure `[n]` markers exist,
  4. `add_citations` — turn `[n]` into links and add a Sources list.
- 165: return the `FactoryResult`.

`note(msg)` (lines 152–156) is a helper that reports progress messages ("Expand to ~500 words") to the terminal and UI.

---

## 5. `blog_factory/postprocess.py` — guaranteeing what the user asked for

**Why this file exists (lines 1–5):** an LLM rarely hits exact word counts and may skip images or citations, so plain Python and extra LLM calls enforce them.

| Lines | Item | Explanation |
|---|---|---|
| 18 | `IMAGE_LINE` | Regex matching a Markdown image line `![alt](https://…)`. |
| 19 | `MAX_PASSES = 6` | Max rewrite attempts for length. |
| 22 | `URL_TAG` | Regex matching `[https://…]` tags. |
| 25–35 | `number_sources` | Finds every `[url]`, assigns numbers in order of first appearance (no duplicates), replaces with `[n]`, returns new text + url list. `rstrip(".,;")` removes punctuation stuck to a URL. |
| 38–57 | `ensure_citations` | If the post already has `[n]` markers, or there are no URLs, return unchanged. Otherwise ask a small "citer" agent to add markers without changing wording. Falls back to the original text on any error. |
| 60–80 | `add_citations` | (1) cut off any existing `## Sources` section; (2) turn `[1, 2]` into `[1][2]`; (3) replace each `[n]` with `[n](url)` (a clickable link, and delete invalid numbers); (4) append a numbered `## Sources` list. Parentheses in URLs are escaped as `%28`/`%29` so Markdown links don't break. |
| 83–87 | `_clean` | Removes ```` ``` ```` fences and any preamble before the first `# Title`. Returns `""` if there is no title (meaning "the model answer was bad"). |
| 90–94 | `count_words` | Counts only prose: ignores image lines, headings, and the Sources section. |
| 97–140 | `adjust_length` | Loop (up to 8 times): count words; if in range stop; else compute a goal (when too short, at most double the current length per pass), ask an "expander" agent to expand/condense, clean the result. Two consecutive empty results stop the loop (`bad` counter). At the end warns if still out of range. |
| 143–150 | `_lookup` | Calls the `find_image` tool function directly (`find_image.func`) and parses `url \| title` lines. |
| 153–203 | `add_images` | Skips if mode is "no" or the post already has images. Decides how many (`yes`: 1 per 250 words; `auto`: up to 3, 1 per 500 words). Finds "anchor" paragraphs (first paragraph under the title and each `##`), spreads images evenly across them, searches Commons using heading + topic, avoids duplicates, then inserts `![alt](url)` lines — in **reverse order** (line 201) so earlier inserts don't shift later line numbers. `asyncio.to_thread` runs the blocking HTTP call without freezing the server. |

---

## 6. `blog_factory/rag/` — "talk to your own documents" (RAG)

**RAG** = Retrieval-Augmented Generation: find relevant text first, then give it to the LLM.
Pipeline: `gather files → chunk → embed → store` (ingest, run once) and `embed query → compare → top matches` (retrieve, every question).

> **Important honest note:** the RAG code, the *Sources library* page, and the `retrieve_from_sources` tool all exist and work, but in `agents.py` only `web_search` and `find_image` are given to agents. So at the moment blog posts are built from **web search**, not from your `sources/` files. To use RAG you would add `retrieve_from_sources` to the researcher's `tools=[...]` and update its instructions.

### 6.1 `embeddings.py`
An **embedding** is a list of numbers representing the *meaning* of a text; similar meaning → similar numbers.
- 15–16: Gemini REST base URL; batch limit of 100 texts per request.
- 19–21 `_normalize`: scales a vector to length 1 (divide by its norm; `or 1.0` avoids dividing by zero). With unit vectors, the dot product equals cosine similarity.
- 24–56 `embed(texts, task_type)`: checks the API key; splits texts into batches of 100; POSTs to `batchEmbedContents` with `taskType` (`RETRIEVAL_DOCUMENT` for stored chunks, `RETRIEVAL_QUERY` for questions) and `outputDimensionality` 768; `raise_for_status()` turns HTTP errors into exceptions; returns normalized vectors. The **same model must be used for ingest and query** (docstring lines 3–4).

### 6.2 `store.py`
A tiny vector database stored as JSON — no heavy dependency.
- `Record` (16–22): one chunk: id, text, source file, chunk number, embedding.
- `VectorStore.replace_source` (32–34): remove old chunks of a file then add new ones (so re-ingesting doesn't duplicate).
- `query` (36–40): score every record with a dot product (`sum(a*b for …)`), sort high→low, return top *k*. Brute force is fine for a small library.
- `save` (42–49): write model, dimensions and records to `vector_store.json`.
- `load_store` (52–58): read the file, or return an empty store if missing.

### 6.3 `ingest.py`
- `gather_files` (24–27): find all `.txt`/`.md` files recursively, sorted.
- `chunk_text` (31–64): split text into paragraphs; pack paragraphs into chunks of about 180 words; a single huge paragraph is cut by words with 30 words overlap so no idea is cut off sharply. `nonlocal` lets the inner `flush()` change outer variables.
- `_chunk_id` (67–69): `source::index::hash` — a stable unique id.
- `ingest` (73–99): for each file → chunk → `embed` → `replace_source` → save once. `reset=True` starts with an empty store.
- `main` (102–106): command line with `--reset`.

### 6.4 `retriever.py`
`retrieve(query)`: load store; if empty return `[]`; embed the query as `RETRIEVAL_QUERY`; return `RetrievedChunk(text, source, score)` for the top-k.

---

## 7. `blog_factory/storage.py` — where saved posts live

Public functions used by `server.py`: `save`, `read`, `list_posts`, plus `PostNotFound`.

| Lines | Explanation |
|---|---|
| Docstring | Explains the two modes and the environment variables: `BLOB_READ_WRITE_TOKEN`, `BLOB_STORE_ID_STORE_ID`, `BLOB_ACCESS`. |
| `PREFIX = "posts/"` | All blobs are stored under a `posts/` "folder". |
| `PostNotFound` | Custom exception → the server turns it into HTTP 404. |
| `_token()` / `use_blob()` | Blob mode is on when `BLOB_READ_WRITE_TOKEN` or Vercel's `VERCEL_OIDC_TOKEN` exists. Otherwise local-folder mode. |
| `safe_name` | **Security.** `Path(name).name` strips directories (blocks `../../evil.md`); regex allows only letters, digits, `_ . -` ending in `.md`. Anything else → `PostNotFound`. |
| `_title` | First `# Heading` line of a post, or a fallback. |
| `_access()` | `public` (default) or `private`. |
| `_credentials()` | Builds `BlobCredentials(token, store_id, kind=OIDC)` — the same as your working `tests/test_storage.py`. |
| `_blob_session()` | A context manager. If a store id is set, runs Blob calls inside a `session(...)` that supplies those credentials; otherwise lets the SDK decide. |
| `_blob_put` | Uploads text as UTF-8 bytes, content type `text/markdown`, `allow_overwrite=True`; returns the URL. |
| `_blob_get` | Downloads and decodes; a Blob "not found" error becomes `PostNotFound`. |
| `_blob_list` | Lists blobs under `posts/`, only `.md`, newest first. Title is made from the file name; a trailing `_YYYYMMDD-HHMMSS` timestamp is removed with a regex. |
| `_require_writable_local` | If running on Vercel (`VERCEL` env var) but no token exists, raise a clear error instead of an obscure "read-only file system". |
| `save` | Validate name → Blob upload, or write to `output/` (creating the folder). |
| `read` | Validate name → Blob download, or read the local file. |
| `list_posts` | Blob list, or scan `output/*.md` sorted by modified time. |

**Why Blob?** Vercel serverless functions cannot keep files, so posts must go to external storage.

---

## 8. `blog_factory/pdf_export.py` — Markdown → PDF

Used by `GET /api/posts/{name}/pdf`.
- 13–17 `FONT_SETS`: font file paths for Windows, Linux, macOS. A Unicode TrueType font allows non-English characters; if none found, the built-in Helvetica is used and characters are converted to Latin-1 (`fix`, line 52).
- 18–19 regexes for image lines and `[n](url)` citations; 20–21 colours.
- `_fetch` (24–30): download an image (returns `None` on failure).
- `_plain` (33–35): remove `**bold**` and `*italic*` markers.
- `post_to_pdf` (38–105): create an A4 page with margins; choose font; `write_para` writes text and draws `[n]` citations in blue as real clickable links; then loop over each Markdown line:
  - **image** → download, read size with Pillow, scale (max 120 mm wide, 85 mm high), start a new page if it doesn't fit, centre it;
  - `# ` → big title; `##…` → sub-heading;
  - list item (`-`, `*`, `1.`) → bullet text;
  - otherwise a normal paragraph.
  Returns the PDF as bytes.

---

## 9. `blog_factory/sources_import.py` — turning PDFs and links into text

For the Sources library page.
- `_Text(HTMLParser)` (19–39): a small HTML-to-text converter. It skips `script/style/nav/footer/header/svg`, inserts newlines for paragraph-like tags, and collects text.
- `_clean` (42–44): collapse spaces and blank lines.
- `pdf_to_text` (47–51): `pypdf` reads each page's text.
- `_check_public` (54–61): **security (SSRF protection).** Only http/https; resolves the hostname and rejects private, loopback or link-local IP addresses, so users cannot make your server fetch internal addresses.
- `link_to_text` (64–76): download up to 10 MB; if PDF → `pdf_to_text`; else decode HTML and extract text.
- `slug`, `link_filename`, `pdf_filename` (79–97): make safe file names (`link_example_com_page.txt`, `name_pdf.txt`).
- `save_text` (83–88): rejects text under 20 words ("No readable text found"), writes into `sources/`.

---

## 10. `main.py` — command-line entry point (and a helper)

| Lines | Explanation |
|---|---|
| 9–18 | Imports. `config` and `run_factory` come from the package. |
| 21–22 `slugify` | lower-case, replace every run of non `a-z0-9` with `_`, trim `_`, cut to 60 characters, default `blog_post`. Example: "The History of Chess!" → `the_history_of_chess`. `server.py` imports this function. |
| 25–29 `save_post` | Path = `-o` argument or `output/<slug>.md`; create parent folders; write text. (CLI saves locally; it does not use `storage.py`.) |
| 32–55 `main` | `argparse` defines `topic`, `-o`, `-q`. If no topic, ask with `input()`. Run `run_factory`. If result is empty or `NO_FACTS_FOUND` → message and exit code 1. Otherwise save and print. |
| 58–59 | `asyncio.run(main())` starts the async function; its return value becomes the process exit code. |

`async/await`: `run_factory` waits on network calls; async lets the program do other work (like streaming to the browser) while waiting.

---

## 11. `server.py` — the web API

FastAPI turns Python functions into HTTP endpoints. **Pydantic models** (`BaseModel`) validate incoming JSON automatically.

### 11.1 Setup (lines 1–29)
Imports; `UI_FILE` = path of `ui/index.html`; `app = FastAPI(...)`.

### 11.2 Request models (lines 32–56)
- `GenerateRequest`: `topic`, `words_min`/`words_max` (200–5000, `Field(ge, le)`), `tone` restricted by `Literal[...]` to the 14 allowed names, `images` restricted to `auto|yes|no`. Invalid values → automatic HTTP 422.
- `SaveRequest`: post and topic. `UploadFile`/`UploadRequest`: uploaded source files (text or base64). `LinkRequest`: a URL.

### 11.3 Endpoints

| Endpoint | Lines | What it does |
|---|---|---|
| `GET /` | 59–61 | Returns the UI HTML file. |
| `_list_sources` | 64–72 | Lists files in `sources/` with their chunk counts. |
| `GET /api/config` | 75–77 | Model name + sources (the UI now only uses the sources list; the model name is no longer shown). |
| `POST /api/classify` | new | **Intent detector** (§18): `{text}` → `{intent, reply, suggestions}`. |
| `POST /api/voice-token` | new | 60 s temporary Deepgram token for live voice typing (§17). |
| `POST /api/transcribe` | new | Voice typing, short-piece mode: raw audio body → `{text}` via Deepgram (§17). |
| `GET /api/sources` | 80–82 | Source list. |
| `POST /api/sources` | 85–103 | Upload files. PDF → base64-decode → text → save; `.txt/.md` → write; other types → HTTP 400. `Path(f.name).name` blocks path tricks. |
| `POST /api/sources/link` | 106–115 | Fetch a web page/PDF link into a source file. |
| `POST /api/ingest` | 118–123 | Rebuild the vector index. |
| `POST /api/generate` | 126–162 | **Streaming generation**, explained below. Optional `facts` field: research from an earlier run, which makes the server skip the researcher (§19). |
| `GET /api/img` | 172–202 | Image proxy (see below). |
| `GET /api/posts` | 205–210 | List saved posts. |
| `GET /api/posts/{name}` | 213–215 | Read one post. |
| `GET /api/posts/{name}/pdf` | 218–224 | PDF download (`Content-Disposition: attachment`). |
| `POST /api/save` | 227–235 | Save a post (details below). |

**`/api/generate` in detail**
- Validate: empty topic or min>max → 400.
- `queue = asyncio.Queue()` — a mailbox between two coroutines.
- `on_event` puts small dicts into the queue (`stage`, `chunk`, `note`).
- `worker()` runs `run_factory`; then puts `done` (with the post), `empty`, or `error`; `finally` puts `None` as the "no more messages" signal.
- `stream()` starts the worker as a background task, then loops: take an item, yield `data: {json}\n\n`. This is the **Server-Sent Events** format; the browser reads it live. If the browser disconnects, `finally` cancels the worker.
- Returned as `StreamingResponse(..., media_type="text/event-stream")`.

**`/api/img` (image proxy):** the browser asks our server for the image instead of Wikimedia directly. It only allows Wikimedia hosts (security), retries 3 times on 429/5xx with growing waits, checks the content type is `image/*`, limits to 8 MB, and caches for a day. **Why:** Wikimedia often blocks hot-linking from other websites.

**`/api/save`:** builds a file name as `<topic-slug>_<YYYYMMDD-HHMMSS>.md` so every save is a **new** post (older code used just the topic, which overwrote the earlier save). Calls `storage.save` and returns `{"message": "Post saved successfully"}`. Errors become HTTP 500 with `Save failed: …`. (The timestamp uses the server clock — UTC on Vercel.)

---

## 12. `ui/index.html` — the whole front end

One file with three parts: `<style>` (design, using CSS variables in `:root`), HTML structure, and `<script>` (behaviour). The UI follows the chat-style wireframes: a sidebar of chats, a Research / Writer / Editor switcher and a chat box at the bottom. **The backend API is unchanged** — only the front end was redesigned.

### 12.1 Layout
- **Sidebar** (becomes a slide-out drawer under 900 px wide): *+ New chat*, a *Search chats* box, the chat history grouped **Today / Earlier** (each row has a delete ✕), then *Saved posts* and *Sources library*. (The old Settings page and model-name box were removed.)
- **Top bar**: chat title, the **Research / Writer / Editor** switcher (each button has a status dot: grey waiting, amber running, green done, red failed) and *Save to file*. On phones the switcher moves to its own row and *☰* / *+* buttons appear.
- **Chat view**: your topic as a bubble, then the assistant reply: three step chips (Researcher → Writer → Editor), and one **result card** for the agent you selected:
  - *Research* → fact cards with a source link each ("4 facts from 3 sources").
  - *Writer* → the draft with a word count, *Regenerate* and *Send to editor*.
  - *Editor* → the final post with *Copy Markdown* and *Save to file*.
  An empty new chat shows a greeting and three example topics (they disappear after the first message).
- **Saved posts view**: list on the left, selected post on the right with *Export PDF* / *Export TXT*.
- **Sources library view**: file list, *Add files*, *Re-run ingestion*, add-link field.

### 12.2 The chat box (composer)
- A white pill with a **green focus ring**. While the text fits on one line everything is on one row; once it wraps, the textarea moves to a full-width top row and the buttons go underneath (left: `+` and paperclip, right: mic and send). It switches back when the text fits again or after sending.
- How it decides: `updateComposer()` copies the text into a hidden "mirror" textarea that has the width of the **one-row** textarea and checks whether it wraps. Measuring the one-row width avoids flicker (switching rows would otherwise make the text fit again and bounce back).
- 14 px text (16 px on phones to stop iOS zooming), 40 px round buttons, hidden scrollbar, grows to 200 px (140 px on phones) then scrolls.
- **Send** is disabled while empty or while a post is generating.
- **Enter** sends on desktop (Shift+Enter = new line). On touch devices (`pointer: coarse`) Enter adds a new line and only the send button sends; the textarea is blurred after sending so the phone keyboard closes.
- **`+` options menu**: *Tone*, *Length*, *Images*. Tapping a row shows its choices; picking one saves it (`localStorage` key `pf.opts2`) and returns to the list. Default length is the lowest, 200 – 400 words. The `+` button has no hover or open-state styling on purpose.
  - Bug fixed: the menu redraws itself on each click, so the "click outside closes the menu" handler thought the click was outside and closed it. The menu's click handler now calls `stopPropagation()`.
- **Paperclip**: adds files to the sources library (then run *Re-run ingestion*).
- **Mic**: see §17.

### 12.3 JavaScript, function by function
| Area | Function(s) | Purpose |
|---|---|---|
| helpers | `$`, `$$`, `esc`, `toast`, `words`, `host` | Short selectors; HTML-escape (**XSS protection**); pop-up message; word count; hostname for source badges. |
| options | `TONES`, `LENGTHS`, `IMAGES`, `opts` | The three post options and their saved values. |
| chats | `chats`, `cur`, `getChat`, `persist` | Chat history lives in the browser (`localStorage` key `pf.chats`, max 50). A chat is `{id, topic, title, opts, facts, draft, final, status, stages, ts}`. A chat that was still `running` when the page reloaded is marked failed ("Interrupted"). |
| sidebar | `renderChats`, `openChat`, `newChat`, `delChat` | Draw/search the history; switch, start or delete chats. A new chat is blocked while one is generating. |
| render | `agentCard`, `stepChips`, `renderThread`, `renderSeg`, `renderHeader`, `renderAll` | Draw the selected agent's card, the step chips, the status dots and the title. `renderThread` keeps the scroll position unless you were already at the bottom. |
| parsing | `parseFacts`, `inline`, `md` | `parseFacts` extracts `{text, url}` per bullet; `md` is a **mini Markdown renderer** (bold, italic, `[n](url)` superscripts, links, images through `/api/img`, headings, lists). All text goes through `esc` first. |
| composer | `updateComposer`, `renderPop`, `setPop`, `submitTopic` | One-row/two-row switching, the `+` menu, and sending. |
| voice | `createMic(...)` (from `static/deepgram_mic.js`) | Voice typing; the page only creates it once and calls `mic.active()` / `mic.cancel()` on send (§17). |
| generate | `generate(topic, opts, existing)` | Creates the chat (or reuses it for *Regenerate*), **POSTs to `/api/generate` and reads the stream**: `res.body.getReader()` reads bytes, text is split on blank lines, each `data:` line is parsed as JSON and handled by `type`: `stage` (start a step and switch the card to that agent), `chunk` (append text), `done` (store the final post, use its `# Title` as the chat name), `empty`, `note` (warnings), `error`. If the stream ends without `done` → "connection closed" error. |
| actions | `copyMd`, `savePost` | Clipboard copy; POST to `/api/save`. |
| saved posts | `loadSaved`, `openSaved`, `toPlain`, `exportTxt`, `exportPdf` | List/open saved posts; export `.txt` (Blob + temporary link) or `.pdf` (from the server). |
| sources | `renderSources`, `loadConfig`, `addFiles`, `addLink`, `ingest` | Sources library actions. PDFs are base64-encoded in 32 KB pieces (so `String.fromCharCode(...)` doesn't overflow the stack). |
| navigation | `show` | Switch between Chat / Saved posts / Sources. |

**Each message starts a new chat**, because the backend runs one topic at a time (there are no follow-up turns in the same chat).

### 12.4 Responsive rules (phones)
Under 900 px the sidebar is a drawer with a dark scrim, the page is locked to the screen width and height (`100dvh`, no horizontal scrolling), scrollbars are hidden (you can still swipe), the footer disclaimer is hidden, and the options menu is limited to the screen width.

---

## 13. `tests/test_storage.py` — your manual Blob test

A **script**, not an automated unit test. It loads `.env`, builds `BlobCredentials` (token + store id, kind OIDC), opens a Blob `session`, and uploads one file with `put_blob(..., access="public", allow_overwrite=True)`, then prints the result. It runs `asyncio.run(main())` at import, so a test runner like `pytest` would execute it (and really upload). It proved that your credentials work locally.

---

## 14. Request lifecycle: "Generate" → "Save" (end to end)

0. *(new)* Every message is first checked by the **intent detector** (§18): only a real topic continues; chat and off-topic messages get a reply instead. A topic that was researched before also skips the check.
1. User types (or speaks) a topic and presses **Send** → `generate()` in the browser POSTs JSON to `/api/generate`. If the same topic was researched before, the old facts are sent along (§19).
2. `server.py` validates it and starts `run_factory` in a background task, streaming events through the queue.
3. `run_factory` builds the workflow: **Researcher** calls `web_search` (SerpAPI) → writes `- fact [url]` bullets. *(When old facts were sent, this step is skipped.)*
4. `NumberSources` converts `[url]` to `[1]…` and appends the `SOURCES:` table.
5. **Writer** drafts Markdown with `[n]` citations (may call `find_image`).
6. `ReplyToUserTurn` passes only the draft to the **Editor**, who polishes it.
7. Post-processing: length → images → citation check → links + Sources list.
8. Server sends `done` with the final Markdown; the browser renders it with `md()`.
9. User clicks **Save to file** → `/api/save` → `storage.save("<slug>_<timestamp>.md")` → Vercel Blob (or `output/`).
10. **Saved posts** page → `/api/posts` lists them; opening one → `/api/posts/{name}`; **Export PDF** → `pdf_export.py`.

---

## 15. Deployment notes (Vercel) and common problems

- **`.env` is never deployed** (it is in `.gitignore`). Every secret must be added in *Vercel → Project → Settings → Environment Variables* (`GEMINI_API_KEY`, `serp_api_key`, `BLOB_READ_WRITE_TOKEN`, `BLOB_STORE_ID_STORE_ID`, …) and the project must be **redeployed** afterwards.
- Vercel's disk is **read-only**; that is why posts go to Blob and why `vector_store.json` is committed.
- "Storage is not configured…" = the deployed function sees no Blob token.
- Gemini 503/429 = temporary overload or rate limit; the retry options in `make_client()` handle most of it.
- Writer-step failures: read the real error text now printed by `run_factory` (lines 138–145).

---

## 16. Concepts to remember

- **Agent** = LLM + instructions + (optional) tools. **Tool** = Python function the LLM may call.
- **Prompt engineering** = strict output formats and "ONLY"/"never" rules, so code can parse results.
- **Guard code** = never trust the LLM to obey; Python enforces length, images and citations.
- **Streaming (SSE)** = send partial results as they are produced, for a live UI.
- **Async** = wait for slow network calls without blocking everything else.
- **Security habits in this project:** `safe_name` (no path tricks), `esc` (no HTML injection), private-IP check for links, image proxy host allow-list, secrets only in environment variables.
- **RAG** = chunk → embed → store → retrieve nearest by cosine similarity.

---

## 17. Voice typing with Deepgram (`blog_factory/speech.py`, `static/deepgram_mic.js`, the mic button)

**Goal:** words appear in the chat box from the microphone, **on Vercel** and **without ever exposing the Deepgram key**. Vercel's serverless Python functions cannot hold a WebSocket open, so the earlier design (browser → our WebSocket → Deepgram) was replaced by two stateless HTTP routes. The SDK (`deepgram-sdk`) is no longer used: only `httpx`, which was already a dependency.

**Setup:** put the key in `.env` as `DEEPGRAM_API_KEY=...` (copy `.env.example`); on Vercel add the same name under *Settings → Environment Variables*. Without a key everything works except the mic (the routes answer 503 and the page shows the message once). The old name `text_to_speech` is still read as a fallback.

### 17.1 Server (`speech.py`, routes in `server.py`)
| Route | What it does |
|---|---|
| `POST /api/transcribe` | **Short-piece mode.** Body = one short recording as the raw request body. `< 1000` bytes → `{"text": ""}` (silence); `> 3,000,000` → 413. Forwards it to `POST https://api.deepgram.com/v1/listen` (`model=nova-3`, `language=multi` so English/Hindi/etc. work, `smart_format`, `punctuate`), with the same `Content-Type` (default `audio/webm`). Returns `{"text": transcript}`. Missing key → 503, Deepgram failure → 502. |
| `POST /api/voice-token` | **Live mode.** Asks `POST https://api.deepgram.com/v1/auth/grant` for a **60 second** temporary token and returns `{"token"}`. The real key stays on the server. If Deepgram answers 403 (the key's role is below *Member*) → 502, and the browser silently falls back to `/api/transcribe`. |
| `/static/*` | `app.mount("/static", StaticFiles(...))`, registered **before** the other routes so the script is served as JavaScript. |

### 17.2 Browser (`static/deepgram_mic.js`, one plain file)
`createMic({input, button, onError, onChange})` returns `{stop, cancel, active}`. The button is hidden if `getUserMedia` / `MediaRecorder` are missing. Tap to start, tap again to stop (text kept), **Esc** cancels and restores the text that was in the box before. While recording the button has class `rec` and `aria-pressed="true"`.

It picks a mode automatically after asking `/api/voice-token`:

```
start ──► POST /api/voice-token ──► token?  yes ─► MODE 1  live streaming
                                              no ─► MODE 2  short pieces (remembered until page reload)
```
- **Mode 1 – live:** `new WebSocket("wss://api.deepgram.com/v1/listen?...", ["bearer", token])`, `MediaRecorder.start(250)` sends every chunk. `Results` messages: `is_final` → appended to the finished text, otherwise shown as interim. Box = `[textBefore, finished, interim]`. On stop it sends `Finalize` then `CloseStream` and closes after 4 s if Deepgram has not. If the socket cannot even connect it falls back to Mode 2.
- **Mode 2 – short pieces (works with any key):** a new `MediaRecorder` per piece on the same stream; each blob is POSTed to `/api/transcribe` and appended **in order** (each piece has an index). Pieces are cut at natural pauses using an `AudioContext` + `AnalyserNode`: "quiet" = `max|sample-128| < 4`; cut when the piece is ≥ 2500 ms and quiet for ≥ 350 ms, never longer than 6500 ms. The next recorder starts immediately, so no speech is lost. Pieces smaller than 1000 bytes are not uploaded.
- **Cleanup (both modes):** timers cleared, recorder/socket/`AudioContext` closed, **all stream tracks stopped** (this turns the browser's recording light off). One error message per session. Results that arrive after the session ended or was cancelled are ignored (a bug found while testing: late results used to overwrite the box of the next recording).

### 17.3 Wiring in `ui/index.html`
`createMic({input: #topic, button: #micBtn, onError: toast, onChange: updateComposer})`. On send, `if (mic.active()) mic.cancel()` runs **after** the text was read (cancel restores the pre-recording text). The mic button is disabled while a post is generating. `.rec` = red pulsing background.

### 17.4 Things to remember
- The microphone only works on `localhost` or HTTPS (Vercel provides HTTPS).
- With a key below the *Member* role you get Mode 2 only: text appears after each pause (about 3–8 s of speech per piece) instead of word by word. Give the key the Member role (or higher) in the Deepgram console to get live streaming.
- `static/` must be deployed with the app so `/static/deepgram_mic.js` exists on Vercel.

### 17.5 What was tested (local)
- `py_compile` and `node --check` on all files.
- `/api/transcribe` with a spoken WAV (Windows System.Speech): returned *"Write a blog post about the history of chess in India."*; tiny body → `{"text": ""}`; 3.1 MB body → 413; `/static/deepgram_mic.js` served as JavaScript.
- `/api/voice-token` with the project's key → 502 (its role cannot create tokens), exactly the case the fallback is for.
- Headless Edge with a fake microphone playing the WAV (`--use-fake-device-for-media-stream --use-file-for-fake-audio-capture`): the mic button gets `rec`/`aria-pressed`, pieces were uploaded and the text appeared in the box; stop kept the text and cleared the state; Esc restored the old text and later results were ignored; **all media tracks ended** after stop and after Esc.
- **Not tested:** Mode 1 (live streaming) against real Deepgram, because this key cannot make tokens; real microphones; the deployed Vercel site.

---

## 18. Intent detector (`blog_factory/intent.py`, `POST /api/classify`)

**Why:** Blog Post Factory only writes blog posts. Without a gate, "hi" or "asdf" would start a full research → write → edit run and waste your free Gemini / SerpAPI quota.

### 18.1 Working process (step by step)

```
user presses Send
      │
      ▼
submitTopic()  (browser)
  1. read the box, clear it, create a chat entry with status "checking"
     → the chat shows "Checking your message…"
  2. was this exact topic researched before (findFacts)?  ── yes ─► skip the check (it is obviously a topic)
      │ no
      ▼
  3. POST /api/classify {text}
      │
      ▼
classify_intent()  (server, intent.py)
  • empty text                    → off_topic (no model call)
  • no GEMINI_API_KEY             → topic     (let the pipeline report the missing key)
  • otherwise ONE small Gemini call: prompt + the text as a JSON string,
    response_mime_type=application/json, response_schema=Intent, temperature 0.2
  • any error / timeout           → topic     (fail open: never block a real request)
      │
      ▼  {intent, reply, suggestions}
  4. browser decides:
        topic     → generate()  → the normal pipeline (§14)
        chat      → status "reply": friendly one-liner + up to 3 clickable topic ideas
                     └ click one → generate() for that topic (no second check needed)
        off_topic → status "reply": "this is not a topic…" + an example
```

| Intent | Examples | What the UI does |
|---|---|---|
| `topic` | "the history of chess", "chess", "who invented the telephone?", "800 words on volcanoes, funny tone" | Runs the three-agent pipeline. |
| `chat` | "hi how are you", "thanks!", "what can you do?", "I'm bored, chess is fun" | Replies in one line and shows up to 3 clickable ideas ("Did you mean…?"). |
| `off_topic` | "asdfgh jkl", "write python code…", "ignore previous instructions…" | Says plainly that this is not a topic and gives an example. |

### 18.2 Details worth knowing
- The user's text is passed to the model as a JSON string and the prompt says to treat it only as data, so "ignore previous instructions" can't hijack the detector.
- Only the first 500 characters are sent. Suggestions are trimmed to 3 and removed unless the intent is `chat`; if the model returns no reply text a built-in default is used.
- Chat entries have status `checking` then `reply`. They appear in the sidebar like any chat but have no Research/Writer/Editor switcher. A chat still `checking` when the page reloads is marked interrupted.
- The check happens in the UI, so someone calling `/api/generate` directly could skip it. If you ever expose the API publicly, call `classify_intent` inside `/api/generate` too.
- Tested against the real model on 11 sample messages (topics, greetings, gibberish, a code request, a prompt-injection attempt): all classified as expected.

---

## 19. Reusing earlier research on the same topic (fewer API calls)

**Why:** a full run costs SerpAPI searches (free quota is small) plus a research LLM call. If you want a **rewrite on the same topic** (Regenerate, a different tone or length, or typing the same topic again) the facts are still valid, so they are reused and only the writer and editor run.

### 19.1 Working process (step by step)

```
generate(topic, opts, chat, fresh=false)   (browser)
  1. reuse = this chat's own facts  — or — facts of any earlier chat whose topic
     matches after normalising (lower-case, punctuation removed, spaces collapsed:
     "The History of Chess!" == "the history of chess")
     (skipped when fresh=true, i.e. the "Search the web again" button)
  2. POST /api/generate { topic, words, tone, images, facts: reuse | null }
      │
      ▼
server.py  → run_factory(..., facts=...)  (workflow.py)
  3. no facts  → normal chain:   researcher → NumberSources → writer → ReplyToUserTurn → editor
     facts     → shorter chain:  CachedFacts ───────────────► writer → ReplyToUserTurn → editor
                 • CachedFacts is an Executor that numbers the sources ([1], [2]…) and hands
                   them to the writer exactly like NumberSources does (shared helper _numbered_text)
                 • no web_search, no research LLM call → 0 SerpAPI searches, 1 Gemini call fewer
                 • the server still streams events so the UI looks the same: a "researcher"
                   stage, the old facts as one chunk, then a note "Reusing earlier research…"
      │
      ▼
  4. post-processing is unchanged (length → images → citation check → links),
     using the reused facts for the Sources list
  5. browser: the Research chip shows "Research · reused" and a toast says no new web search was made
```

### 19.2 What counts as "the same topic"
- Regenerate / Try again on a chat that already has facts.
- Sending the same topic text again (case, spaces and punctuation ignored), even in a new chat.
- Changing tone, length or images in the `+` menu and regenerating — those don't change the facts.
- A different topic always researches fresh. A failed run that never produced facts has nothing to reuse.

### 19.3 Getting fresh research
Open the **Research** view of a finished chat and press **Search the web again** (`fresh=true`): the facts are not sent, so the researcher runs normally.

### 19.4 Where the facts live
The browser keeps them in the chat history (`localStorage`, key `pf.chats`) and sends them with the request, so the server stays stateless — it works the same on Vercel, where memory is not kept between requests. The server accepts up to 30,000 characters in `facts`, and ignores empty text or a `NO_FACTS_FOUND` marker.

### 19.5 Savings per rewrite
| | New topic | Rewrite on the same topic |
|---|---|---|
| Intent-detector call | 1 small Gemini call | 0 (known topic) |
| Research LLM call | 1 (with tool calls) | 0 |
| SerpAPI searches | up to `MAX_SEARCHES` (3) | 0 |
| Writer + editor + post-processing | yes | yes |

### 19.6 Testing notes
Tested with the real framework: with facts supplied the executor order was `input-conversation → cached_facts → writer → editor`, no researcher ran, and a 297-word cited post came back. Gemini sometimes answers `503 UNAVAILABLE (high demand)`; it showed up as "Failed during the writer step" once during testing and the retry succeeded. That is a temporary Google overload, not a bug in the reuse code.

---

## 20. Change log (newest first)

| Change | Where | Notes |
|---|---|---|
| Voice typing made Vercel-safe | `blog_factory/speech.py`, `server.py`, `static/deepgram_mic.js`, `ui/index.html` | WebSocket relay and `deepgram-sdk` removed. Two HTTP routes (`/api/transcribe`, `/api/voice-token`) + one browser script with live and short-piece modes (§17). Fixed: late results overwriting the next recording. |
| Reuse research on the same topic | `workflow.py` (`CachedFacts`), `server.py` (`facts` field), `ui/index.html` | Rewrites skip the researcher: 0 searches, 1 fewer LLM call (§19). |
| Intent detector | `blog_factory/intent.py`, `POST /api/classify`, `ui/index.html` | Topic → pipeline; chat → "did you mean…" ideas; off-topic → rejected (§18). |
| Dotted interactive background | `ui/index.html` | Canvas grid of dots that move away from the cursor (ported from `DottedBackground.jsx`). |
| Chat-style UI redesign | `ui/index.html` | Sidebar of chats, Research/Writer/Editor switcher, ChatGPT-style chat box, mobile layout (§12). |
| `.env.example` | `.env.example` | Template with placeholder keys and instructions. |
