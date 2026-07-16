import asyncio
import json
from asyncio import Future
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from fastcontext.agent.budget import cap_tool_output, cap_turn_outputs
from fastcontext.agent.llm import Message

MAX_TOOLRUN_TIMEOUT = 10


class ToolResult(BaseModel):
    tool_call_id: str
    output: str
    failed: bool


ToolResultFuture = Future[ToolResult]

type ToolOutput = ToolResult | ToolResultFuture


class Tool:
    name: str
    description: str
    parameters: dict[str, Any]

    async def call(self, parameters: str, **kwargs) -> str:
        raise NotImplementedError("Tool.call must be implemented by subclasses.")

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @staticmethod
    def load_desc(path: str) -> str:
        desc = Path(path).read_text(encoding="utf-8")
        return desc


class ToolSet:
    _tool_dict: dict[str, Tool] = {}

    def __init__(
        self,
        tools: list[Tool],
        work_dir: str,
        max_tool_output_chars: int = 0,
        max_tool_result_chars: int = 0,
    ):
        self._tool_dict = {tool.name: tool for tool in tools}
        self.work_dir = work_dir
        # Two distinct bounds, both in characters; 0 disables either.
        #
        # `max_tool_output_chars` is the budget for the WHOLE turn. It is what the context reserve is
        # sized against, so it is the one that keeps the final-answer turn sendable. A single Read can
        # otherwise return 2000 lines x 500 chars (~250k tokens), enough to blow past any context
        # window in one call and leave no room to even ask for an answer.
        self.max_tool_output_chars = max_tool_output_chars
        # `max_tool_result_chars` bounds ONE result. The turn budget alone is spent greedily in call
        # order, so a single huge result can consume all of it and starve the later calls of the same
        # turn -- the model asked three questions and gets one answer. Capping each result first gives
        # every call in the turn a chance to survive.
        self.max_tool_result_chars = max_tool_result_chars

    def schema_list(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tool_dict.values()]

    async def _single_tool_call(self, tool_name: str, parameters: str, toll_call_id: str) -> ToolOutput:
        if tool_name not in self._tool_dict:
            return ToolResult(
                tool_call_id=toll_call_id,
                failed=True,
                output=f"Tool `{tool_name}` not found.",
            )

        tool = self._tool_dict[tool_name]
        try:
            json.loads(parameters or "{}")
        except json.JSONDecodeError:
            return ToolResult(
                tool_call_id=toll_call_id,
                failed=True,
                output=f"Tool `{tool_name}` arguments are invalid.",
            )

        async def _call():
            try:
                output = await tool.call(parameters, cwd=self.work_dir)
                return ToolResult(tool_call_id=toll_call_id, failed=False, output=output)
            except Exception as e:
                return ToolResult(tool_call_id=toll_call_id, failed=True, output=str(e))

        # return asyncio.create_task(_call())
        return await _call()

    async def call(self, msg: Message) -> list[Message]:
        if not msg.tool_calls:
            return []

        tool_results: list[ToolResult] = []
        for c in msg.tool_calls:
            try:
                result = await asyncio.wait_for(
                    self._single_tool_call(c.name, c.arguments, c.id), timeout=MAX_TOOLRUN_TIMEOUT
                )
            except TimeoutError:
                result = ToolResult(
                    tool_call_id=c.id, failed=True, output=f"Tool `{c.name}` timed out after {MAX_TOOLRUN_TIMEOUT}s."
                )
            tool_results.append(result)

        # Per result first, so one oversized result cannot eat the whole turn's allowance and starve
        # the calls after it; then the turn total, because N results each just under the per-result
        # cap still add N x cap to the prompt in one step.
        outputs = [cap_tool_output(tr.output, self.max_tool_result_chars) for tr in tool_results]
        outputs = cap_turn_outputs(outputs, self.max_tool_output_chars)
        tools_result_messages = []
        for tr, output in zip(tool_results, outputs):
            tools_result_messages.append(
                Message(
                    role="tool",
                    content=output,
                    tool_call_id=tr.tool_call_id,
                )
            )
        return tools_result_messages
