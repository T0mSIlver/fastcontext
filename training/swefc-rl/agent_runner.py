import json

from api_sglang import SGLangCompletionClient, mask_assistant_content
from utils import parse_final_answer, read_citations_content

from swefc.agent.llm import FunctionCall, Message
from swefc.agent.tool.glob import GlobTool
from swefc.agent.tool.grep import GrepTool
from swefc.agent.tool.read import ReadTool
from swefc.agent.tool.tool import ToolSet

g_n_sample_debug = 0


class AgentRunner:
    def __init__(self, *, model: str, max_iterations: int = 6, api_client: SGLangCompletionClient) -> None:
        self.model = model
        self.max_iterations = max_iterations
        # self.url = f"http://{sglang_router_ip}:{sglang_router_port}/generate"
        # e.g. http://0.0.0.0:30000/generate
        self.api_client = api_client
        self.tools = None

    def save_messages_to_file(self, messages, max_samples=10):
        # create a temporary file to save the messages
        global g_n_sample_debug
        if g_n_sample_debug >= max_samples:
            return
        import os
        import tempfile

        os.makedirs("/tmp/swefc_rl", exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", prefix="messages_", dir="/tmp/swefc_rl")
        msgs = [message.to_dict(exclude_none=True) for message in messages]
        with open(tmp.name, "w") as f:
            json.dump(msgs, f, indent=4)
        g_n_sample_debug += 1

    async def run(self, query: str | list[dict[str, str]], work_dir):
        toolset = ToolSet([ReadTool(), GlobTool(), GrepTool()], work_dir=work_dir)
        self.tools = toolset.schema_list()

        if isinstance(query, str):
            messages: list[Message] = [
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content=query),
            ]
        else:
            messages = [Message(role=msg["role"], content=msg["content"]) for msg in query]

        prompt_ids = []
        input_ids = []
        output_token_ids = []

        final_response = ""
        finish_reason = None
        n_turn = 0
        n_max_parallel_tool_calls = 1
        for turn_id in range(self.max_iterations):
            # if is the last turn, we don't need to call the model again, just return the final response
            if turn_id == self.max_iterations - 1:
                messages.append(
                    Message(role="assistant", content="Stop exploring and return the final answer directly.")
                )

            input_msg_list = [message.to_dict(exclude_none=True) for message in messages]
            completion, input_ids, output_token_ids, output_token_logprobs = await self.api_client.complete(
                input_msg_list, self.tools
            )
            if turn_id == 0:
                prompt_ids = input_ids
            n_turn += 1
            finish_reason = completion.choices[0].finish_reason
            final_response = completion.choices[0].message.content
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }
            tool_calls = completion.choices[0].message.tool_calls
            if finish_reason == "tool_calls" and tool_calls:
                n_max_parallel_tool_calls = max(n_max_parallel_tool_calls, len(tool_calls))
                function_calls = [
                    FunctionCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments) for tc in tool_calls
                ]
                msg = Message(
                    role="assistant",
                    content=completion.choices[0].message.content,
                    tool_calls=function_calls,
                    tool_call_id=tool_calls[0].id,
                    model=self.model,
                    usage=usage,
                )
                tools_result_messages = await toolset.call(msg=msg)
                messages.append(msg)
                messages.extend(tools_result_messages)
            else:
                msg = Message(
                    role="assistant",
                    content=completion.choices[0].message.content,
                    model=self.model,
                    usage=usage,
                )
                messages.append(msg)
                break
        self.save_messages_to_file(messages, max_samples=10)
        tokens = input_ids + output_token_ids
        tokens_loss_mask = mask_assistant_content(tokens)
        loss_mask = tokens_loss_mask[len(prompt_ids) :]  # only keep the loss mask for the output tokens
        final_answer_result = parse_final_answer(final_response)
        citation_contents = read_citations_content(final_answer_result["citations"])
        runtime_info = {
            "n_ctx_tokens": len(tokens),
            "n_turn": n_turn,
            "n_max_parallel_tool_calls": n_max_parallel_tool_calls,
            "n_citations": final_answer_result.get("n_citations", 0),
            "n_citations_files": final_answer_result.get("n_citations_files", 0),
            "n_citations_lines": final_answer_result.get("n_citations_lines", 0),
            "citation_contents": citation_contents,
        }
        return finish_reason, tokens, prompt_ids, loss_mask, final_response, final_answer_result, runtime_info


if __name__ == "__main__":
    import asyncio

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/local/models/swefc_sft_qwen3.5_4b_hf_iter_0000137", trust_remote_code=True
    )

    async def main():
        url = "http://0.0.0.0:30000/generate"
        sampling_params = {"max_new_tokens": 2048, "temperature": 1.0}
        api_client = SGLangCompletionClient(
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            url=url,
            chat_template_kwargs={"enable_thinking": False},
            tool_call_parser="qwen3_coder",
        )
        agent_runner = AgentRunner(model="qwen3p5-4b", max_iterations=8, api_client=api_client)
        finish_reason, tokens, prompt_ids, loss_mask, final_response, final_answer_result, runtime_info = (
            await agent_runner.run("Searching system_prompt in `/root/swefc`", work_dir="/root/swefc")
        )
        print("runtime_info:", runtime_info)
        print("Finish reason:", finish_reason)
        print("Prompt IDs:", prompt_ids)
        print("Loss mask:", loss_mask)
        print("Final response:", final_response)
        print("Final answer result:", final_answer_result)

    asyncio.run(main())
