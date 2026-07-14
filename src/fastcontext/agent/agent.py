from uuid import uuid4

from fastcontext.agent.budget import FINALIZE_MESSAGE, ContextBudget
from fastcontext.agent.context import Context
from fastcontext.agent.events import (
    AgentFinished,
    EventSink,
    ToolCallStarted,
    ToolResultReady,
    TurnStarted,
    UsageUpdated,
)
from fastcontext.agent.llm import LLM, Message, RequestyAPIError
from fastcontext.agent.observed import (
    ObservedLines,
    correction_message,
    record_tool_results,
    unverified_citations,
)
from fastcontext.agent.tool import ToolSet
from fastcontext.agent.utils import get_final_answer, parse_citations

# How many times the agent is asked to fix a final answer that cites line ranges it never opened,
# before the remaining unverified citations are dropped from the answer.
MAX_CITATION_CORRECTIONS = 2


class Agent:
    """The loaded agent."""

    name: str
    system_prompt: str
    llm: LLM
    toolset: ToolSet
    context: Context

    work_dir: str

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: LLM,
        toolset: ToolSet,
        trajectory_file: str,
        work_dir: str,
        budget: ContextBudget | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.toolset = toolset
        self.context = Context(trajectory_file)
        self.work_dir = work_dir
        self.budget = budget or ContextBudget()
        self.run_id = str(uuid4())
        self.n_turn = 0

    async def _remember(self, message: Message | list[Message]) -> None:
        """Add to the conversation and charge it against the context budget."""
        await self.context.add(message)
        messages = [message] if isinstance(message, Message) else message
        self.budget.add_pending([m.to_dict(exclude_none=True) for m in messages])

    async def _agent_loop(
        self,
        prompt: str,
        max_turns: int,
        verbose: bool,
        citation: bool,
        event_sink: EventSink | None = None,
    ) -> str:
        # user promp -> tool calls -> tool results -> tool calls ... -> assistant final answer
        n_turn = 0
        # line numbers the model actually observed, used to drop hallucinated citations
        observed: ObservedLines = {}
        corrections = 0
        # Set once the context budget trips: the next turn is the last, and it runs without
        # tools so the model has no way to keep exploring and must answer.
        finalizing = False
        await self._remember(Message(role="system", content=self.system_prompt))
        await self._remember(Message(role="user", content=prompt))

        while True:
            n_turn += 1
            if n_turn > max_turns + 1:
                return f"No final answer after {max_turns} turns."
            if n_turn == max_turns + 1:
                await self._remember(
                    Message(
                        role="user",
                        content="Max number of turns reached. Please provide the final answer based on the information you have gathered.",
                    )
                )

            tools = self.toolset.schema_list()
            if not finalizing and self.budget.must_finalize(self.context.get_messages(), tools):
                # Stop while a request still fits. Withholding the tools is what guarantees the
                # run ends with an answer instead of exploring its way into an unsendable prompt.
                finalizing = True
                await self._remember(Message(role="user", content=FINALIZE_MESSAGE))
            if finalizing:
                tools = None

            if event_sink is not None:
                event_sink(TurnStarted(n=n_turn))

            # call LLM to get next action
            try:
                step_msg = await self.llm.acall(
                    messages=self.context.get_messages(),
                    tools=tools,
                    event_sink=event_sink,
                    turn=n_turn,
                )
            except RequestyAPIError as e:
                error_msg = f"LLM API call failed. So stopping the agent.\nError details:\n{str(e)}"
                await self.context.add(Message(role="assistant", content=error_msg))
                return error_msg
            self.n_turn = n_turn
            await self._remember(step_msg)
            # The provider's own prompt-token count replaces our estimate for everything sent
            # so far, so estimation error cannot accumulate across turns.
            self.budget.record_usage(step_msg.usage)
            if event_sink is not None and step_msg.usage:
                usage = step_msg.usage
                prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
                event_sink(UsageUpdated(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))
            if verbose:
                print(f"Turn {n_turn}: \n {step_msg.to_dict()} \n")
            if step_msg.tool_calls and not finalizing:
                if event_sink is not None:
                    for call in step_msg.tool_calls:
                        event_sink(ToolCallStarted(id=call.id, name=call.name, arguments=call.arguments))
                tools_result_msg = await self.toolset.call(step_msg)
                await self._remember(tools_result_msg)
                if citation:
                    record_tool_results(observed, step_msg.tool_calls, tools_result_msg, self.work_dir)
                if event_sink is not None:
                    name_by_id = {call.id: call.name for call in step_msg.tool_calls}
                    for result in tools_result_msg:
                        event_sink(
                            ToolResultReady(
                                tool_call_id=result.tool_call_id,
                                name=name_by_id.get(result.tool_call_id, "tool"),
                                output=result.content or "",
                                failed=False,
                            )
                        )
            else:
                if not citation:
                    if event_sink is not None:
                        event_sink(AgentFinished(answer=step_msg.content or ""))
                    return step_msg.content
                # Ask the model to fix citations referencing line ranges it never opened; once the
                # retries are exhausted, the still-unverified ones are dropped by get_final_answer.
                unverified = unverified_citations(observed, parse_citations(step_msg.content or ""), self.work_dir)
                if unverified and corrections < MAX_CITATION_CORRECTIONS and n_turn <= max_turns:
                    corrections += 1
                    await self._remember(Message(role="user", content=correction_message(unverified)))
                    continue
                answer = get_final_answer(step_msg.content or "", observed=observed, cwd=self.work_dir)
                if event_sink is not None:
                    event_sink(AgentFinished(answer=answer or ""))
                return answer

    async def run(
        self,
        prompt: str,
        max_turns: int = 4,
        verbose: bool = False,
        citation: bool = False,
        event_sink: EventSink | None = None,
    ) -> str:
        if verbose:
            print("=== Agent Runtime Info ===")
            print(f"Agent: {self.name}")
            print(f"LLM: {self.llm.model}")
            print(f"Working Directory: {self.work_dir}")
            print("Agent Tools: " + " / ".join(self.toolset._tool_dict.keys()))
            print(f"User prompt:\n{prompt}\n")
            print("=== Agent Trajectory ===")
        return await self._agent_loop(prompt, max_turns, verbose, citation, event_sink)
