import os
import sys
from typing import Any, Literal

import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel, model_serializer

from fastcontext.agent.events import EventSink, StreamClose, StreamDelta, StreamOpen

# How long ONE response may be. Not the model's window: a completion cap is a bound on what the
# model writes in a single turn, and FastContext's turns are tool calls and a final answer, both of
# which are short. Sizing it from the window is what made this dangerous -- see resolve_max_context.
DEFAULT_MAX_COMPLETION_TOKENS = 4096

# Field names different OpenAI-compatible servers use to advertise a model's context window, in
# priority order.
#
# SERVED beats TRAINED. A model trained at 131072 but served with --ctx-size 32768 can only hold
# 32768: believing the trained figure sets a budget four times the real window, so the budget never
# trips and the run dies with the overflow it exists to prevent. Under-reading is survivable (the run
# finalizes early but answers); over-reading is not. So n_ctx (served) is checked before n_ctx_train
# (a property of the weights), and max_position_embeddings -- also an architectural figure -- is last.
_CONTEXT_LENGTH_KEYS: tuple[str, ...] = (
    "max_model_len",  # vLLM: the served length
    "max_context_length",
    "context_length",
    "context_window",
    "n_ctx",  # llama.cpp: served context
    "max_total_tokens",  # TGI: served
    "n_ctx_train",  # llama.cpp: what the weights were trained for -- may exceed what is served
    "max_position_embeddings",  # architectural ceiling, not a promise about this server
)


class LLMAPIError(Exception):
    """Exception for LLM API call failures."""


def _coerce_positive_int(value: Any) -> int | None:
    """Return ``value`` as a positive int, or ``None`` if it isn't one."""
    if isinstance(value, bool):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _context_from_arg_tokens(tokens: Any) -> int | None:
    """Extract usable context from a llama.cpp-style launch-args token list.

    Some routers (e.g. llama.cpp's model swapper) don't expose a context field but
    list the server's launch flags. ``--ctx-size`` is the total context, shared
    across ``--parallel`` slots, so usable context per request is
    ``ctx-size // parallel``.
    """
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        return None
    ctx_size = parallel = None
    for i, tok in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if tok in ("--ctx-size", "-c"):
            ctx_size = _coerce_positive_int(nxt)
        elif tok in ("--parallel", "-np"):
            parallel = _coerce_positive_int(nxt)
    if not ctx_size:
        return None
    return ctx_size // parallel if parallel and parallel > 1 else ctx_size


def _scan_for_context_length(obj: Any) -> int | None:
    """Recursively search a decoded ``/models`` entry for a context-length field.

    At each dict level the recognised keys are checked in priority order before
    descending, so a top-level ``max_model_len`` wins over a nested value. Lists
    are also probed as possible llama.cpp launch-args.
    """
    if isinstance(obj, dict):
        for key in _CONTEXT_LENGTH_KEYS:
            if key in obj:
                found = _coerce_positive_int(obj[key])
                if found:
                    return found
        for value in obj.values():
            found = _scan_for_context_length(value)
            if found:
                return found
    elif isinstance(obj, list):
        from_args = _context_from_arg_tokens(obj)
        if from_args:
            return from_args
        for item in obj:
            found = _scan_for_context_length(item)
            if found:
                return found
    return None


def fetch_provider_context_length(
    base_url: str,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = 5.0,
) -> int | None:
    """Best-effort lookup of a model's context WINDOW from its provider.

    This is the size of the whole conversation the model can hold, not how long one response may
    be -- the two were conflated under the name ``max_tokens``, which fed a window into the
    completion cap and disabled exploration. It feeds ``--max-context``.

    Queries the OpenAI-compatible ``GET {base_url}/models`` endpoint and scans the
    response for a known context-length field. Returns the discovered value, or
    ``None`` if the endpoint is unreachable or advertises nothing we recognise.
    Never raises — auto-detection must not break a run.
    """
    if not base_url:
        return None
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None

    entries: list[Any]
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        entries = payload["data"]
    elif isinstance(payload, dict):
        entries = [payload]
    elif isinstance(payload, list):
        entries = payload
    else:
        return None

    # Prefer the entry whose id matches our model, then fall back to any entry.
    if model:
        preferred = [e for e in entries if isinstance(e, dict) and e.get("id") == model]
        for entry in preferred:
            found = _scan_for_context_length(entry)
            if found:
                return found
    for entry in entries:
        found = _scan_for_context_length(entry)
        if found:
            return found
    return None


def resolve_max_completion_tokens(explicit: Any = None, *, verbose: bool = False) -> int:
    """How long one response may be.

    Deliberately does NOT consult the provider. What a provider advertises is its context
    *window* -- the whole conversation -- and using that as a per-response cap is not merely
    imprecise, it disables the run: the context reserve is sized as 2 x this value, so a window-sized
    completion cap makes the reserve exceed the window and the agent finalizes before its first turn,
    answering with no exploration at all. The window belongs in resolve_max_context().
    """
    raw = None if explicit is None else str(explicit).strip()
    if raw and raw.lower() != "auto":
        forced = _coerce_positive_int(raw)
        if forced:
            return forced
        print(
            f"[fastcontext] ignoring invalid max_completion_tokens {raw!r}; "
            f"using {DEFAULT_MAX_COMPLETION_TOKENS}",
            file=sys.stderr,
        )
    elif raw and raw.lower() == "auto" and verbose:
        # "auto" used to mean "ask the provider". It now has nothing to ask for: a response cap is
        # not something a provider advertises.
        print(
            f"[fastcontext] max_completion_tokens='auto' is no longer detected from the provider "
            f"(that value is the context window, see --max-context); using "
            f"{DEFAULT_MAX_COMPLETION_TOKENS}",
            file=sys.stderr,
        )
    return DEFAULT_MAX_COMPLETION_TOKENS


def resolve_max_context(
    explicit: Any = None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    verbose: bool = False,
) -> int:
    """The usable context window in tokens, which is what the provider can actually tell us.

    An explicit integer wins (``0`` disables the budget outright). ``"auto"`` or a missing value asks
    the provider, and falls back to ``0`` when it advertises nothing -- the budget stays off rather
    than guessing, because a wrong window is worse than none: too high and the run dies mid-flight,
    too low and it finalizes early.
    """
    raw = None if explicit is None else str(explicit).strip()
    if raw and raw.lower() != "auto":
        forced = _coerce_positive_int(raw)
        if forced:
            return forced
        if raw == "0":
            return 0
        print(f"[fastcontext] ignoring invalid max_context {raw!r}; auto-detecting", file=sys.stderr)

    fetched = fetch_provider_context_length(base_url, api_key=api_key, model=model) if base_url else None
    if fetched:
        if verbose:
            print(f"[fastcontext] using provider-reported context window={fetched}", file=sys.stderr)
        return fetched

    if verbose:
        print("[fastcontext] provider context window unavailable; context budget stays off", file=sys.stderr)
    return 0


type Role = Literal[
    "system",
    "user",
    "assistant",
    "tool",
]


class FunctionCall(BaseModel):
    id: str
    name: str
    arguments: str

    @model_serializer(mode="wrap")
    def serialize_call(self, handler, info):
        return {
            "id": self.id,
            "type": "function",
            "function": {"arguments": self.arguments, "name": self.name},
        }


class Message(BaseModel):
    id: str | None = None
    role: Role
    content: str | None = None
    reasoning_content: str | None = None

    # [{"name": name, "arguments": arguments, "id": id} ... ]
    tool_calls: list[dict | FunctionCall] | None = None
    tool_call_id: str | None = None
    model: str | None = None
    usage: dict | None = None

    def to_dict(self, exclude_none: bool = True) -> dict:
        return self.model_dump(exclude_none=exclude_none)


class LLM:
    def __init__(self, model: str, api_key: str | None, base_url: str, **kwargs) -> None:
        self.model = model
        self.base_url = base_url
        self.client = AsyncOpenAI(api_key=api_key or "ollama", base_url=base_url)
        self.max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_COMPLETION_TOKENS)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.95)
        self.reasoning_effort = kwargs.get("reasoning_effort")

    async def acall(
        self,
        messages: list[dict | Message],
        tools: list[dict[str, Any]] | None,
        event_sink: EventSink | None = None,
        turn: int = 0,
        tool_choice: str | None = None,
    ) -> Message:

        if isinstance(messages[0], Message):
            messages = [message.to_dict(exclude_none=True) for message in messages]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        reasoning_effort = self.reasoning_effort or os.getenv("FC_REASONING_EFFORT")
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if "qwen" in self.model:
            payload["extra_body"] = {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }

        if tools:
            payload["tools"] = tools
            # Forbidding tool calls with tool_choice keeps the tool schemas in the prompt, so the
            # cached prefix stays valid. Dropping the tools instead would change the prompt prefix
            # and invalidate the provider's prompt cache for the whole conversation.
            if tool_choice:
                payload["tool_choice"] = tool_choice

        # Token-by-token streaming is only used when a live consumer (the TUI) asks for it.
        if event_sink is not None:
            try:
                return await self._acall_stream(payload, event_sink, turn)
            except Exception as e:
                raise LLMAPIError(f"LLM API call failed: {str(e)}") from e

        try:
            response = await self.client.chat.completions.create(**payload)
            usage = response.usage.to_dict()
            content = None
            reasoning_content = None
            tool_calls: list[ChatCompletionMessageToolCall] = []
            role = response.choices[0].message.role

            if len(response.choices) == 1:
                content = response.choices[0].message.content
                if hasattr(response.choices[0].message, "reasoning_content"):
                    reasoning_content = response.choices[0].message.reasoning_content
                elif hasattr(response.choices[0].message, "reasoning_text"):
                    reasoning_content = response.choices[0].message.reasoning_text
                elif hasattr(response.choices[0].message, "reasoning"):
                    reasoning_content = response.choices[0].message.reasoning
                tool_calls = response.choices[0].message.tool_calls
            elif len(response.choices) == 2:
                reasoning_content = response.choices[0].message.reasoning_text
                content = response.choices[0].message.content
                for choice in response.choices:
                    if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                        tool_calls.extend(choice.message.tool_calls)
            elif len(response.choices) > 2:
                raise ValueError(f"Unexpected number of choices returned: {len(response.choices)}")
            else:
                raise ValueError("No choices returned from LLM API call.")

            if tool_calls:
                function_calls = [
                    FunctionCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments) for tc in tool_calls
                ]
                return Message(
                    role=role,
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=function_calls,
                    tool_call_id=tool_calls[0].id,
                    model=self.model,
                    usage=usage,
                )
            return Message(
                role=role, content=content, reasoning_content=reasoning_content, model=self.model, usage=usage
            )
        except Exception as e:
            raise LLMAPIError(f"LLM API call failed: {str(e)}") from e

    @staticmethod
    def _delta_reasoning(delta: Any) -> str | None:
        """Reasoning tokens arrive under different field names across providers."""
        for attr in ("reasoning_content", "reasoning_text", "reasoning"):
            value = getattr(delta, attr, None)
            if value:
                return value
        return None

    async def _acall_stream(self, payload: dict, event_sink: EventSink, turn: int) -> Message:
        """Stream a completion token-by-token, emitting events as text arrives.

        Reasoning and content are forwarded to ``event_sink`` as they stream in;
        tool-call deltas are accumulated by index and assembled into the returned
        ``Message``, matching the shape produced by the non-streaming path.
        """
        payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        stream = await self.client.chat.completions.create(**payload)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, str]] = {}
        usage: dict | None = None
        role: str = "assistant"
        open_kind: str | None = None

        def _switch(kind: str) -> None:
            nonlocal open_kind
            if open_kind == kind:
                return
            if open_kind is not None:
                event_sink(StreamClose(kind=open_kind))
            event_sink(StreamOpen(kind=kind, n=turn))
            open_kind = kind

        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage.to_dict()
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "role", None):
                role = delta.role

            reasoning = self._delta_reasoning(delta)
            if reasoning:
                _switch("reasoning")
                reasoning_parts.append(reasoning)
                event_sink(StreamDelta(kind="reasoning", text=reasoning))

            if getattr(delta, "content", None):
                _switch("content")
                content_parts.append(delta.content)
                event_sink(StreamDelta(kind="content", text=delta.content))

            for tcd in getattr(delta, "tool_calls", None) or []:
                acc = tool_calls_acc.setdefault(tcd.index, {"id": "", "name": "", "arguments": ""})
                if tcd.id:
                    acc["id"] = tcd.id
                if tcd.function and tcd.function.name:
                    acc["name"] += tcd.function.name
                if tcd.function and tcd.function.arguments:
                    acc["arguments"] += tcd.function.arguments

        if open_kind is not None:
            event_sink(StreamClose(kind=open_kind))

        content = "".join(content_parts) or None
        reasoning_content = "".join(reasoning_parts) or None
        function_calls = [
            FunctionCall(id=acc["id"], name=acc["name"], arguments=acc["arguments"])
            for acc in tool_calls_acc.values()
            if acc["id"]
        ]

        if function_calls:
            return Message(
                role=role,
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=function_calls,
                tool_call_id=function_calls[0].id,
                model=self.model,
                usage=usage,
            )
        return Message(role=role, content=content, reasoning_content=reasoning_content, model=self.model, usage=usage)
