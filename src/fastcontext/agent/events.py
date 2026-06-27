"""Lightweight events emitted by the agent loop for live (TUI) rendering.

The agent and LLM are normally headless: they build up a `Context` and return a
final string. When an `event_sink` callable is provided, they additionally emit
the small dataclasses below as things happen (token deltas, tool calls, tool
results). A consumer such as the Textual TUI turns these into UI updates.

The sink is a plain ``Callable[[Event], None]``. It must be cheap and
non-blocking; the TUI implementation simply forwards each event onto the Textual
message queue. When no sink is supplied, behaviour is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

StreamKind = Literal["reasoning", "content"]


@dataclass
class TurnStarted:
    """A new agent turn (one LLM call) has begun."""

    n: int


@dataclass
class StreamOpen:
    """A streamed text block of the given kind started for turn ``n``."""

    kind: StreamKind
    n: int


@dataclass
class StreamDelta:
    """A chunk of streamed text for the currently-open block."""

    kind: StreamKind
    text: str


@dataclass
class StreamClose:
    """The currently-open streamed block of ``kind`` finished."""

    kind: StreamKind


@dataclass
class ToolCallStarted:
    """The model asked to call a tool. ``arguments`` is the raw JSON string."""

    id: str
    name: str
    arguments: str


@dataclass
class ToolResultReady:
    """A tool finished running."""

    tool_call_id: str
    name: str
    output: str
    failed: bool


@dataclass
class AgentFinished:
    """The agent produced its final answer."""

    answer: str


@dataclass
class AgentErrored:
    """The agent stopped because of an error."""

    message: str


Event = (
    TurnStarted
    | StreamOpen
    | StreamDelta
    | StreamClose
    | ToolCallStarted
    | ToolResultReady
    | AgentFinished
    | AgentErrored
)

EventSink = Callable[[Event], None]
