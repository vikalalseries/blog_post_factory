"""Wire researcher -> writer -> editor into a sequential workflow and run it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from agent_framework import Message, Workflow
from agent_framework._workflows._agent_executor import AgentExecutorResponse
from agent_framework._workflows._executor import Executor, handler
from agent_framework._workflows._workflow_context import WorkflowContext
from agent_framework.orchestrations import SequentialBuilder

from blog_factory.agents import build_agents
from blog_factory.postprocess import (
    add_citations,
    add_images,
    adjust_length,
    ensure_citations,
    number_sources,
)
from blog_factory.tools import reset_search_count

STAGE_LABELS = {
    "researcher": "Researcher: searching the web",
    "writer": "Writer: drafting the post",
    "editor": "Editor: polishing the draft",
}


# Gemini rejects empty messages, so an agent that returns nothing is handed on as this.
EMPTY_HANDOFF = "NO_FACTS_FOUND: the previous step returned no content."


async def _diagnose(stage: str) -> str:
    """The framework hides Gemini's real error, so probe the API to explain a failure."""
    from agent_framework import Agent

    from blog_factory.agents import make_client

    try:
        probe = Agent(client=make_client(), name="probe", instructions="Reply with OK.")
        await probe.run("ping")
    except Exception as exc:
        return f"Gemini request failed during the {stage} step: {exc}"
    return (
        f"Gemini failed during the {stage} step, but the API is reachable now. "
        "It was most likely a temporary overload or a blocked/empty response. Please try again."
    )


class ReplyToUserTurn(Executor):
    """Pass only the previous agent's final text to the next agent, as a user message.

    This keeps each agent isolated (no tool calls or raw passages leak through) and
    gives Gemini a conversation that starts and ends on a user turn.
    """

    @handler
    async def forward(self, prior: AgentExecutorResponse, ctx: WorkflowContext[list[Message]]) -> None:
        text = prior.agent_response.text.strip() or EMPTY_HANDOFF
        await ctx.send_message([Message(role="user", contents=[text])])


class NumberSources(Executor):
    """Give the writer the researcher's facts with numbered sources ([1], [2], ...)."""

    @handler
    async def forward(self, prior: AgentExecutorResponse, ctx: WorkflowContext[list[Message]]) -> None:
        numbered, urls = number_sources(prior.agent_response.text.strip() or EMPTY_HANDOFF)
        table = "\n".join(f"[{i}] {u}" for i, u in enumerate(urls, 1))
        text = f"{numbered}\n\nSOURCES:\n{table}" if urls else numbered
        await ctx.send_message([Message(role="user", contents=[text])])


def build_workflow(
    words_min: int = 400, words_max: int = 600, tone: str = "normal", images: str = "auto"
) -> Workflow:
    researcher, writer, editor = build_agents(
        words_min=words_min, words_max=words_max, tone=tone, images=images
    )
    return SequentialBuilder(
        name="blog_post_factory",
        # Each agent sees only the previous agent's reply: the writer gets the
        # facts (not raw passages), and the editor gets only the draft.
        participants=[
            researcher,
            NumberSources(id="to_writer"),
            writer,
            ReplyToUserTurn(id="to_editor"),
            editor,
        ],
        # Surface every agent's reply (not just the editor's), so each stage
        # can be streamed to the terminal as it runs.
        intermediate_output_from=[researcher, writer],
    ).build()


@dataclass
class FactoryResult:
    final_post: str
    stages: dict[str, str] = field(default_factory=dict)


async def run_factory(
    topic: str,
    verbose: bool = True,
    on_event: Callable[[str, str, str], None] | None = None,
    words_min: int = 400,
    words_max: int = 600,
    tone: str = "normal",
    images: str = "auto",
) -> FactoryResult:
    reset_search_count()
    workflow = build_workflow(words_min, words_max, tone, images)
    stages: dict[str, str] = {}
    current_stage = ""

    async for event in workflow.run(topic, stream=True):
        if event.type == "executor_invoked" and event.executor_id in STAGE_LABELS:
            current_stage = event.executor_id
            if verbose:
                print(f"\n\n>>> {STAGE_LABELS[event.executor_id]} ...\n", flush=True)
            if on_event:
                on_event("stage", event.executor_id, "")

        elif event.type in ("intermediate", "output"):
            # While streaming, each event carries one chunk (AgentResponseUpdate)
            # of an agent's reply; stitch the chunks together per agent.
            who = event.executor_id or "unknown"
            chunk = getattr(event.data, "text", None) or ""
            stages[who] = stages.get(who, "") + chunk
            if on_event and chunk:
                on_event("chunk", who, chunk)
            if verbose and chunk:
                print(chunk, end="", flush=True)

        elif event.type in ("failed", "error", "executor_failed"):
            raise RuntimeError(await _diagnose(current_stage or "pipeline"))

    # The editor is the last agent in the chain; its reply is the finished post.
    final_post = stages.get("editor", "").strip()

    # Enforce the requested length and images (the model alone rarely does).
    if final_post and not final_post.startswith("NO_FACTS_FOUND"):
        def note(msg: str) -> None:
            if verbose:
                print(f"\n>>> {msg}", flush=True)
            if on_event:
                on_event("note", "editor", msg)

        final_post = await adjust_length(final_post, words_min, words_max, tone, note)
        if images != "no":
            note("Adding images")
            final_post = await add_images(final_post, topic, images, note)
        numbered, urls = number_sources(stages.get("researcher", ""))
        final_post = await ensure_citations(final_post, numbered, urls)
        final_post = add_citations(final_post, urls)
    return FactoryResult(final_post=final_post.strip(), stages=stages)
