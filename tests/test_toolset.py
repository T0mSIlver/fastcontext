import json

from fastcontext.agent.llm import FunctionCall, Message
from fastcontext.agent.tool import ToolSet
from fastcontext.agent.tool.read import ReadTool


async def test_toolset_schema_list(tmp_path):
    toolset = ToolSet(tools=[ReadTool()], work_dir=str(tmp_path))
    schema_list = toolset.schema_list()
    assert len(schema_list) == 1
    assert schema_list[0]["function"]["name"] == "Read"


async def test_toolset_call_reads_file(tmp_path):
    (tmp_path / "README.md").write_text("hello world\nsecond line\n", encoding="utf-8")
    toolset = ToolSet(tools=[ReadTool()], work_dir=str(tmp_path))

    tool_call_msg = Message(
        role="assistant",
        content=None,
        tool_call_id="call_1",
        tool_calls=[
            FunctionCall(id="call_1_1", name="Read", arguments=json.dumps({"path": str(tmp_path / "README.md")})),
        ],
    )
    results = await toolset.call(tool_call_msg)

    assert len(results) == 1
    assert results[0].role == "tool"
    assert results[0].tool_call_id == "call_1_1"
    assert "hello world" in results[0].content


async def test_toolset_reports_unknown_tool(tmp_path):
    toolset = ToolSet(tools=[ReadTool()], work_dir=str(tmp_path))
    msg = Message(
        role="assistant",
        content=None,
        tool_call_id="call_x",
        tool_calls=[FunctionCall(id="call_x_1", name="Nope", arguments="{}")],
    )
    results = await toolset.call(msg)
    assert len(results) == 1
    assert "not found" in results[0].content.lower()
