"""A small Textual TUI that streams an agent run as collapsible items.

Each reasoning block, assistant message, tool call and tool result becomes a
`Collapsible` row. Everything starts collapsed; the row title is a one-line
summary, and expanding a row reveals the full detail (streamed text, tool
arguments, or tool output).

Wiring: `FastContextTUI` runs `Agent.run` in a Textual worker and passes an
`event_sink` that forwards each agent event onto the Textual message queue (see
`on_agent_event`). The LLM streams tokens, so reasoning/assistant text fills in
live even while the row is collapsed.
"""

from __future__ import annotations

import json

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message as TextualMessage
from textual.widgets import Collapsible, Footer, Header, Static

from fastcontext.agent.agent import Agent
from fastcontext.agent.events import (
    AgentErrored,
    AgentFinished,
    Event,
    StreamClose,
    StreamDelta,
    StreamOpen,
    ToolCallStarted,
    ToolResultReady,
    TurnStarted,
    UsageUpdated,
)

_KIND_META = {
    "reasoning": ("🧠", "reasoning"),
    "content": ("💬", "assistant"),
}


class AgentEvent(TextualMessage):
    """Carries one agent `Event` onto the Textual message queue."""

    def __init__(self, event: Event) -> None:
        self.event = event
        super().__init__()


def _format_args(arguments: str) -> tuple[str, str]:
    """Return a (one-line preview, pretty body) pair for raw JSON tool args."""
    try:
        data = json.loads(arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        return (arguments[:60], arguments)
    preview = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in data.items())
    if len(preview) > 60:
        preview = preview[:57] + "..."
    return (preview, json.dumps(data, indent=2, ensure_ascii=False))


class FastContextTUI(App):
    """Live, collapsible view of a single agent run."""

    CSS = """
    #log {
        padding: 1 2;
    }
    .query {
        color: $accent;
        margin-bottom: 1;
    }
    .divider {
        color: $text-muted;
        text-style: dim;
        margin-top: 1;
    }
    .final {
        color: $success;
    }
    .error {
        color: $error;
    }
    #tokens {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: $panel;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("e", "expand_all", "Expand all"),
        Binding("c", "collapse_all", "Collapse all"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, agent: Agent, prompt: str, max_turns: int, citation: bool) -> None:
        super().__init__()
        self.agent = agent
        self.prompt = prompt
        self.max_turns = max_turns
        self.citation = citation
        self.final_answer: str | None = None
        self.error: str | None = None
        self._stream_body: Static | None = None
        self._stream_text: str = ""
        # Running token totals across the whole run.
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._context_tokens: int = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="log")
        # Footer docks first so it sits at the very bottom; the token bar
        # docks next and stacks just above it.
        yield Footer()
        yield Static(self._token_bar_text(), id="tokens")

    @property
    def log_view(self) -> VerticalScroll:
        return self.query_one("#log", VerticalScroll)

    def on_mount(self) -> None:
        self.title = "FastContext"
        self.sub_title = "running…"
        self.log_view.mount(Static(f"[b]Query:[/b] {self.prompt}", classes="query"))
        self.run_worker(self._run_agent(), name="agent")

    async def _run_agent(self) -> None:
        def sink(event: Event) -> None:
            # Same event loop as the worker; post_message is ordered and safe.
            self.post_message(AgentEvent(event))

        try:
            answer = await self.agent.run(
                prompt=self.prompt,
                max_turns=self.max_turns,
                citation=self.citation,
                event_sink=sink,
            )
            self.final_answer = answer
        except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
            self.post_message(AgentEvent(AgentErrored(message=str(exc))))

    async def on_agent_event(self, message: AgentEvent) -> None:
        ev = message.event
        log = self.log_view
        # Capture this before new content grows the scroll height: only keep
        # following the tail if the user hasn't scrolled up to read something.
        follow = log.scroll_offset.y >= log.max_scroll_y

        if isinstance(ev, TurnStarted):
            await log.mount(Static(f"── turn {ev.n} ──", classes="divider"))

        elif isinstance(ev, StreamOpen):
            icon, label = _KIND_META[ev.kind]
            body = Static("", markup=False)
            self._stream_body = body
            self._stream_text = ""
            await log.mount(Collapsible(body, title=f"{icon} {label} · turn {ev.n}", collapsed=True))

        elif isinstance(ev, StreamDelta):
            if self._stream_body is not None:
                self._stream_text += ev.text
                self._stream_body.update(self._stream_text)

        elif isinstance(ev, StreamClose):
            self._stream_body = None
            self._stream_text = ""

        elif isinstance(ev, ToolCallStarted):
            preview, body_text = _format_args(ev.arguments)
            await log.mount(
                Collapsible(Static(body_text, markup=False), title=f"🔧 {ev.name}({preview})", collapsed=True)
            )

        elif isinstance(ev, ToolResultReady):
            icon = "⚠️" if ev.failed else "📄"
            n_lines = len(ev.output.splitlines())
            title = f"{icon} result · {ev.name} · {n_lines} line{'' if n_lines == 1 else 's'}"
            await log.mount(Collapsible(Static(ev.output or "(empty)", markup=False), title=title, collapsed=True))

        elif isinstance(ev, UsageUpdated):
            self._input_tokens += ev.prompt_tokens
            self._output_tokens += ev.completion_tokens
            self._context_tokens = ev.prompt_tokens + ev.completion_tokens
            self.query_one("#tokens", Static).update(self._token_bar_text())

        elif isinstance(ev, AgentFinished):
            self.final_answer = ev.answer
            await log.mount(
                Collapsible(
                    Static(ev.answer or "(empty)", markup=False, classes="final"),
                    title="✅ final answer",
                    collapsed=False,
                )
            )
            self.sub_title = "done — press q to quit"

        elif isinstance(ev, AgentErrored):
            self.error = ev.message
            await log.mount(Static(f"[b red]Error:[/b red] {ev.message}", classes="error"))
            self.sub_title = "error — press q to quit"

        if follow:
            log.scroll_end(animate=False)

    def _token_bar_text(self) -> str:
        return (
            f"📊 input {self._input_tokens:,} · "
            f"output {self._output_tokens:,} · "
            f"context {self._context_tokens:,}"
        )

    def action_expand_all(self) -> None:
        for collapsible in self.query(Collapsible):
            collapsible.collapsed = False

    def action_collapse_all(self) -> None:
        for collapsible in self.query(Collapsible):
            collapsible.collapsed = True
