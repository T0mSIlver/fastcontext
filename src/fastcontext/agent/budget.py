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

# Chars per token for ASCII text. Real tokenizers land around 3.5-4 chars/token on source code;
# 3.0 over-counts on purpose so the estimate errs toward finalizing early.
_ASCII_CHARS_PER_TOKEN = 3.0

# Non-ASCII text tokenizes far denser -- CJK is roughly one token per character, and can be worse.
# Charging 1 token per non-ASCII character keeps the estimate conservative on i18n resources, CJK
# source and base64 blobs, where an ASCII-calibrated ratio would UNDER-count by tens of thousands
# of tokens on a single turn and let the agent walk straight into the overflow it exists to avoid.
_NON_ASCII_TOKENS_PER_CHAR = 1.0

# Per-message overhead (role, delimiters, tool-call scaffolding) the chat template adds.
_MESSAGE_OVERHEAD_TOKENS = 4

# Held back from the window for the completion itself plus the final-answer turn, so that when
# the budget trips there is still room for a request to succeed. See required_reserve().
DEFAULT_CONTEXT_RESERVE = 8192

# Headroom on top of the computed reserve, covering the finalize message, the chat template,
# estimation error, and the per-result truncation notices a many-call turn appends.
_RESERVE_SLACK_TOKENS = 2048

# Total tool output ONE turn may add, in characters (not per result -- see cap_turn_outputs).
# ~4k tokens of ASCII source, and the reserve is sized against it at worst-case density, so the
# value is a direct trade: raising it buys the model more per turn and costs usable context.
# Read's own 2000-line x 500-char ceiling would otherwise allow ~250k tokens from a single call.
DEFAULT_MAX_TURN_OUTPUT_CHARS = 16_000

TRUNCATION_NOTICE = (
    "\n<system-reminder>Output truncated: it exceeded the {scope} limit of {limit} characters. "
    "Read a specific line range with the Read tool's offset/limit parameters, or narrow the search, "
    "to see the rest.</system-reminder>"
)

# The two caps truncate for different reasons, and the model can act on the difference: "this one
# result was too big" means narrow this call, while "this turn's calls together were too big" means
# ask for less at once. Labelling both "per-result" told it to fix the wrong thing.
_RESULT_SCOPE = "per-result"
_TURN_SCOPE = "per-turn total output"

FINALIZE_MESSAGE = (
    "You are approaching the context limit, so this is your last turn: no further tool calls will be "
    "available. Provide your final answer now, based only on what you have already read. Cite only "
    "line ranges you actually opened."
)


def estimate_tokens(text: str | None) -> int:
    """Conservatively estimate the tokens ``text`` costs.

    Deliberately over-counts. Over-counting finalizes the run slightly early; under-counting
    walks into the context overflow this module exists to prevent.
    """
    if not text:
        return 0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ascii_chars = len(text) - non_ascii
    return int(ascii_chars / _ASCII_CHARS_PER_TOKEN + non_ascii * _NON_ASCII_TOKENS_PER_CHAR) + 1


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
    return output[:limit] + TRUNCATION_NOTICE.format(limit=limit, scope=_RESULT_SCOPE)


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

    notice = TRUNCATION_NOTICE.format(limit=limit, scope=_TURN_SCOPE)
    # Every result may need a truncation notice, and the notices are not free. Set the payload
    # allowance aside for them up front so the turn's total stays under `limit` for ANY number of
    # tool calls -- otherwise a turn with many truncated calls appends one notice each and drifts
    # past the ceiling the reserve was sized against.
    payload_allowance = max(0, limit - len(notice) * len(outputs))

    capped: list[str] = []
    remaining = payload_allowance
    for output in outputs:
        if len(output) <= remaining:
            capped.append(output)
            remaining -= len(output)
            continue
        # Truncate to whatever the turn has left -- which may be nothing. Note this cannot
        # delegate to cap_tool_output(): there a limit of 0 means "no cap", so an exhausted
        # allowance would let the rest of the turn's output through untouched.
        capped.append(output[:remaining] + notice)
        remaining = 0
    return capped


def required_reserve(max_turn_output_chars: int, max_completion_tokens: int) -> int:
    """The smallest reserve that keeps the final-answer turn sendable.

    The budget can only trip *after* a turn's tool results have landed, so the reserve has to
    absorb everything one turn can add on top of a prompt that was still under the limit:

      - the turn's tool output, at worst-case token density (see below),
      - the assistant message that requested the tools, bounded by the completion limit,
      - the completion the model still has to write for the final answer.

    Tool output is charged at **one token per character**, not the ASCII ratio. A turn is capped
    in *characters*, and CJK / i18n resources tokenize at roughly one token per character, so an
    ASCII-calibrated reserve would be several times too small on exactly the content that already
    strains the window.
    """
    if max_turn_output_chars <= 0:
        # No cap on tool output means no bound on what one turn can add; the reserve cannot make
        # any promise, so fall back to the flat default.
        return DEFAULT_CONTEXT_RESERVE
    worst_case_turn_tokens = max_turn_output_chars
    needed = (
        worst_case_turn_tokens
        + 2 * max_completion_tokens
        # The budget is checked BEFORE the finalize message is appended, so its own cost has to be
        # reserved too.
        + estimate_tokens(FINALIZE_MESSAGE)
        + _RESERVE_SLACK_TOKENS
    )
    return max(DEFAULT_CONTEXT_RESERVE, needed)


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
        # How many messages that measurement covered. Everything after this index in the
        # conversation is estimated. Deriving the pending tail from the conversation itself --
        # rather than from a counter the caller must remember to update -- means a message can
        # never be appended without being charged.
        self._measured_message_count = 0

    @property
    def enabled(self) -> bool:
        return self.max_context > 0

    @property
    def limit(self) -> int:
        """The projected-token ceiling at which the agent must finalize."""
        return self.max_context - self.reserve

    def record_usage(self, usage: dict | None, message_count: int) -> None:
        """Adopt the provider's exact prompt-token count for the messages it was given.

        ``message_count`` is how many messages were in the request the usage came back from;
        everything appended after them is estimated until the next measurement.
        """
        if not usage:
            return
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        if not prompt_tokens:
            return
        self._measured_prompt_tokens = int(prompt_tokens)
        self._measured_message_count = message_count

    def projected_tokens(self, messages: list[dict], tools: list[dict] | None = None) -> int:
        """Estimated prompt size of the next request.

        Exact for the prefix the provider has already counted, estimated for the tail appended
        since. Before the first response nothing is measured, so the whole conversation and the
        tool schemas are estimated.
        """
        if self._measured_prompt_tokens is None:
            return estimate_messages_tokens(messages) + estimate_tools_tokens(tools)
        # The measured count already includes the tool schemas from that request.
        tail = messages[self._measured_message_count :]
        return self._measured_prompt_tokens + estimate_messages_tokens(tail)

    def must_finalize(self, messages: list[dict], tools: list[dict] | None = None) -> bool:
        """True when the next request would leave too little room to answer safely."""
        if not self.enabled:
            return False
        return self.projected_tokens(messages, tools) >= self.limit
