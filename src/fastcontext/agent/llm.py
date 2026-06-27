import os
from typing import Any, Literal

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel, model_serializer

from fastcontext.agent.events import EventSink, StreamClose, StreamDelta, StreamOpen


class RequestyAPIError(Exception):
    """Exception for Requesty LLM API errors."""


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
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.95)
        self.debug = kwargs.get("debug", False)

    async def acall(
        self,
        messages: list[dict | Message],
        tools: list[dict[str, Any]] | None,
        event_sink: EventSink | None = None,
        turn: int = 0,
    ) -> Message:

        if isinstance(messages[0], Message):
            messages = [message.to_dict(exclude_none=True) for message in messages]
        payload = {
            "model": self.model,
            "messages": messages,
            # "max_tokens": self.max_tokens,
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        reasoning_effort = os.getenv("FC_REASONING_EFFORT")
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if "qwen" in self.model:
            payload["extra_body"] = {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }

        if tools:
            payload["tools"] = tools

        if self.debug:
            print("LLM Payload:", payload)

        # Token-by-token streaming is only used when a live consumer (the TUI)
        # asks for it and the model goes through the OpenAI-compatible endpoint.
        # The claude path uses a separate, non-streaming client.
        if event_sink is not None and "claude" not in self.model:
            try:
                return await self._acall_stream(payload, event_sink, turn)
            except Exception as e:
                raise RequestyAPIError(f"LLM API call failed: {str(e)}") from e

        try:
            if "claude" in self.model:
                # Use the custom API call for claude models
                from fastcontext.agent.llm_api import call_completion

                response = call_completion(model=self.model, messages=messages, tools=tools)
            else:
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
            raise RequestyAPIError(f"LLM API call failed: {str(e)}") from e

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
