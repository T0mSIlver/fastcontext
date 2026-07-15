"""Live LLM tests. Skipped unless FC_MODEL and FC_BASE_URL are set (see conftest)."""

from fastcontext.agent.llm import LLM
from fastcontext.agent.tool.read import ReadTool


async def test_llm(llm_endpoint):
    llm = LLM(**llm_endpoint)
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    msg = await llm.acall(messages=messages, tools=None)
    assert msg.role == "assistant"
    assert msg.content is not None


async def test_llm_tools(llm_endpoint):
    llm = LLM(**llm_endpoint, temperature=0.0, max_tokens=1024)
    messages = [
        {"role": "system", "content": "You are a powerful AI agent."},
        {"role": "user", "content": "read file content from ./test_llm.py and ./README.md"},
    ]
    msg = await llm.acall(messages=messages, tools=[ReadTool().schema()])
    # The model should either answer or ask to call the Read tool.
    assert msg.content is not None or msg.tool_calls


async def test_llm_tools_result(llm_endpoint):
    llm = LLM(**llm_endpoint, temperature=0.0, max_tokens=1024)
    messages = [
        {"role": "system", "content": "You are a powerful AI agent."},
        {"role": "user", "content": "please show me the current time"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_0", "function": {"arguments": '{"command": "date"}', "name": "bash"}, "type": "function"},
            ],
        },
        {"role": "tool", "content": "Thu Aug 21 17:42:44 CST 2025", "tool_call_id": "call_0"},
    ]
    msg = await llm.acall(messages=messages, tools=None)
    assert msg.role == "assistant"
    assert msg.content is not None
