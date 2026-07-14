"""Keep the conversation inside the model's context window.

Without this the agent simply grows its message list until the provider rejects the
request (``exceeds the available context size``) and the run dies mid-exploration,
producing no answer at all. Two mechanisms, both bounding growth rather than
rewriting history:

- ``cap_tool_output`` bounds a single tool result. A ``Read`` can otherwise return
  2000 lines x 500 chars -- ~250k tokens, several times a typical window -- so one
  call can make the conversation unsendable in a single step, leaving no room even
  to ask for a final answer.
- ``ContextBudget`` tracks how close the conversation is to the window and tells the
  agent when to stop exploring and produce its final answer while a request still
  fits.

Token accounting is exact for the prefix the provider has already seen (its response
carries ``usage.prompt_tokens``) and estimated only for the messages appended since.
The estimate deliberately over-counts: over-counting finalizes a little early, while
under-counting hits the very failure this module exists to prevent.
"""

import json

# Chars per token. Real tokenizers land around 3.5-4 chars/token on source code; 3.0
# over-counts on purpose so the estimate errs toward finalizing early.
_CHARS_PER_TOKEN = 3.0

# Per-message overhead (role, delimiters, tool-call scaffolding) the chat template adds.
_MESSAGE_OVERHEAD_TOKENS = 4

# Held back from the window for the completion itself plus the final-answer turn, so that when
# the budget trips there is still room for a request to succeed. See required_reserve().
DEFAULT_CONTEXT_RESERVE = 8192

# Headroom on top of the computed reserve, covering the finalize message, the chat template,
# estimation error, and the per-result truncation notices a many-call turn appends.
_RESERVE_SLACK_TOKENS = 2048

# ~10k tokens. Generous enough for a real file, small enough that no single result can exhaust a
# window on its own. Read's own 2000-line x 500-char ceiling allows ~250k tokens without this.
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 30_000

TRUNCATION_NOTICE = (
    "\n<system-reminder>Output truncated: it exceeded the per-result limit of {limit} characters. "
    "Read a specific line range with the Read tool's offset/limit parameters, or narrow the search, "
    "to see the rest.</system-reminder>"
)

FINALIZE_MESSAGE = (
    "You are approaching the context limit, so this is your last turn: no further tool calls will be "
    "available. Provide your final answer now, based only on what you have already read. Cite only "
    "line ranges you actually opened."
)


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def estimate_message_tokens(message: dict) -> int:
    """Estimate the tokens a single chat message contributes to the prompt."""
    total = _MESSAGE_OVERHEAD_TOKENS + estimate_tokens(message.get("content"))
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        total += estimate_tokens(function.get("name"))
        total += estimate_tokens(function.get("arguments"))
    return total


def estimate_messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def estimate_tools_tokens(tools: list[dict] | None) -> int:
    """Tool schemas are resent on every request and are not free -- FastContext's three
    tool descriptions run to several thousand tokens."""
    if not tools:
        return 0
    return estimate_tokens(json.dumps(tools))


def cap_tool_output(output: str, limit: int) -> str:
    """Bound a single tool result so that one call cannot exhaust the window.

    ``limit`` <= 0 disables the cap.
    """
    if limit <= 0 or len(output) <= limit:
        return output
    return output[:limit] + TRUNCATION_NOTICE.format(limit=limit)


def cap_turn_outputs(outputs: list[str], limit: int) -> list[str]:
    """Bound the *total* tool output a single turn can add.

    A per-result cap alone is not enough: the model can issue several tool calls in one
    turn, and N results each just under the cap still add N x cap to the prompt. That can
    overshoot the window in a single step, leaving the final-answer turn itself unsendable
    -- the exact failure the budget exists to prevent.

    The allowance is spent greedily in call order, so early small results survive intact and
    only the results that actually exhaust the turn's budget are truncated.
    """
    if limit <= 0:
        return list(outputs)
    capped: list[str] = []
    remaining = limit
    for output in outputs:
        if len(output) <= remaining:
            capped.append(output)
            remaining -= len(output)
            continue
        # Truncate to whatever the turn has left -- which may be nothing. Note this cannot
        # delegate to cap_tool_output(): there a limit of 0 means "no cap", so an exhausted
        # allowance would let the rest of the turn's output through untouched.
        capped.append(output[:remaining] + TRUNCATION_NOTICE.format(limit=limit))
        remaining = 0
    return capped


def required_reserve(max_tool_output_chars: int, max_completion_tokens: int) -> int:
    """The smallest reserve that keeps the final-answer turn sendable.

    The budget trips only *after* a turn's tool results have landed, so the reserve has to
    absorb a full turn's worth of tool output plus the completion the model still has to
    write. Anything less and the agent can cross the limit and discover that even asking for
    a final answer no longer fits.
    """
    turn_tokens = estimate_tokens("x" * max_tool_output_chars) if max_tool_output_chars > 0 else 0
    return max(DEFAULT_CONTEXT_RESERVE, turn_tokens + max_completion_tokens + _RESERVE_SLACK_TOKENS)


class ContextBudget:
    """Decide when the agent must stop exploring and answer.

    ``max_context`` is the model's usable context window in tokens; <= 0 disables the
    budget entirely (the previous, unbounded behavior). ``reserve`` is held back for the
    completion itself plus the final-answer turn, so that when the budget trips there is
    still room for a request to succeed.
    """

    def __init__(self, max_context: int = 0, reserve: int = DEFAULT_CONTEXT_RESERVE):
        self.max_context = max_context
        self.reserve = reserve
        # Exact prompt size of the last request, as reported by the provider.
        self._measured_prompt_tokens: int | None = None
        # Messages appended since that measurement, which the provider has not counted yet.
        self._pending_tokens = 0

    @property
    def enabled(self) -> bool:
        return self.max_context > 0

    @property
    def limit(self) -> int:
        """The projected-token ceiling at which the agent must finalize."""
        return self.max_context - self.reserve

    def record_usage(self, usage: dict | None) -> None:
        """Adopt the provider's exact prompt-token count for everything sent so far."""
        if not usage:
            return
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        if not prompt_tokens:
            return
        self._measured_prompt_tokens = int(prompt_tokens)
        self._pending_tokens = 0

    def add_pending(self, messages: list[dict] | dict) -> None:
        """Account for messages appended since the last provider measurement."""
        if isinstance(messages, dict):
            messages = [messages]
        self._pending_tokens += estimate_messages_tokens(messages)

    def projected_tokens(self, messages: list[dict], tools: list[dict] | None = None) -> int:
        """Estimated prompt size of the next request.

        Uses the provider's exact count for the measured prefix and adds an estimate for
        what has been appended since. Before the first response there is nothing measured,
        so the whole conversation is estimated.
        """
        if self._measured_prompt_tokens is None:
            return estimate_messages_tokens(messages) + estimate_tools_tokens(tools)
        # The measured count already includes the tool schemas from the previous request.
        return self._measured_prompt_tokens + self._pending_tokens

    def must_finalize(self, messages: list[dict], tools: list[dict] | None = None) -> bool:
        """True when the next request would leave too little room to answer safely."""
        if not self.enabled:
            return False
        return self.projected_tokens(messages, tools) >= self.limit
