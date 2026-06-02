import datetime
import json
import uuid
import httpx

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage
from sglang.srt.entrypoints.openai.protocol import Function as SglFunction
from sglang.srt.entrypoints.openai.protocol import Tool as SglTool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from slime.utils.http_utils import post


def mask_assistant_content(input_ids: list[int], start_ids=[151644, 77091, 198], end_ids=[151645]) -> list[int]:
    # {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    # <|im_start|>assistant\n...content...<|im_end|>\n
    mask = [0] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i : i + len(start_ids)] == start_ids:
            j = i + len(start_ids)
            while j < len(input_ids) - len(end_ids) + 1:
                if input_ids[j : j + len(end_ids)] == end_ids:
                    start_index = i + len(start_ids)
                    end_index = j + len(end_ids)
                    for k in range(start_index, end_index):
                        mask[k] = 1
                    print(input_ids[start_index:end_index])
                    i = j + len(end_ids) - 1
                    break
                j += 1
        i += 1
    return mask


async def _post(url, payload, max_retries=3):
    # from slime.utils.http_utils import post
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                output = response.json()
                return output
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed. Retrying...")
            else:
                print(f"Attempt {attempt + 1} failed. No more retries left: {e}")
    return None


def parse_tool_calls(
    text: str,
    tools: list,
    tool_call_parser: str | None,
    finish_reason: str,
) -> tuple[
    str,
    str,
    list[ChatCompletionMessageFunctionToolCall] | None,
]:
    tools = [SglTool(type=tool["type"], function=SglFunction(**tool["function"])) for tool in tools]
    parser = FunctionCallParser(tools, tool_call_parser)

    tool_calls = None
    normal_text = text
    if parser.has_tool_call(text):
        if finish_reason == "stop":
            finish_reason = "tool_calls"
        try:
            normal_text, call_info_list = parser.parse_non_stream(text)
            tool_calls = [
                ChatCompletionMessageFunctionToolCall(
                    type="function",
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    function=Function(name=call_info.name, arguments=call_info.parameters),
                )
                for call_info in call_info_list
            ]
        except Exception:
            return normal_text, finish_reason, tool_calls

    return normal_text, finish_reason, tool_calls


class SGLangCompletionClient:
    def __init__(
        self,
        *,
        tokenizer,
        sampling_params: dict,
        url: str,
        chat_template_kwargs: dict | None = None,
        tool_call_parser: str | None = None,
        model: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.url = url
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.tool_call_parser = tool_call_parser
        self.model = model

    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> ChatCompletion:

        # convert tools arguments to dict
        for msg in messages:
            if msg.get("tool_calls", None):
                for tc in msg["tool_calls"]:
                    if isinstance(tc["function"]["arguments"], str):
                        tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
        result = self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            add_generation_prompt=True,
            tokenize=True,
            **self.chat_template_kwargs,
        )
        input_ids = result["input_ids"]

        payload = {
            "input_ids": input_ids,
            "sampling_params": self.sampling_params,
            "return_logprob": True,
        }
        output = await post(self.url, payload, max_retries=3)
        # output = await _post(self.url, payload, max_retries=3)
        output_text: str = output.get("text", "")
        meta_info = output["meta_info"]
        finish_reason: str = meta_info["finish_reason"]["type"]

        if "output_token_logprobs" in meta_info:
            raw_logprobs = meta_info["output_token_logprobs"]
            output_token_ids: list[int] = [x[1] for x in raw_logprobs]
            output_token_logprobs: list[float] = [x[0] for x in raw_logprobs]
        else:
            output_token_ids = []
            output_token_logprobs = []

        tool_calls = None
        if tools:
            content, finish_reason, tool_calls = parse_tool_calls(
                output_text,
                tools,
                self.tool_call_parser,
                finish_reason,
            )
        else:
            content = output_text

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        chat_completion = ChatCompletion(
            id=completion_id,
            choices=[
                Choice(
                    finish_reason=finish_reason,
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(
                        content=content,
                        role="assistant",
                        tool_calls=tool_calls,
                    ),
                )
            ],
            created=int(datetime.datetime.now().timestamp()),
            model=self.model or "unknown",
            object="chat.completion",
            service_tier=None,
            system_fingerprint=None,
            usage=CompletionUsage(
                prompt_tokens=len(input_ids),
                completion_tokens=len(output_token_ids),
                total_tokens=len(input_ids) + len(output_token_ids),
            ),
        )
        return chat_completion, input_ids, output_token_ids, output_token_logprobs


if __name__ == "__main__":
    """
python3 -m sglang.launch_server \
  --model-path /mnt/local/models/swefc_sft_qwen3.5_4b_hf_iter_0000137 \
  --served-model-name qwen3p5-4b \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --context-length 262144 \
  --trust-remote-code \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --mem-fraction-static 0.8
    """

    import asyncio

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/local/models/swefc_sft_qwen3_4b_hf_iter_0000137/", trust_remote_code=True
    )

    async def main():
        client = SGLangCompletionClient(
            tokenizer=tokenizer,
            sampling_params={"max_new_tokens": 1024, "temperature": 1.0},
            url="http://0.0.0.0:30000/generate",
            chat_template_kwargs={"enable_thinking": False},
            tool_call_parser="qwen3_coder",
            model="qwen3p5-4b",
        )
        start_ids = client.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        start_ids_non_think = client.tokenizer.encode(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False
        )
        end_ids = client.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        print(start_ids)
        print(start_ids_non_think)
        print(end_ids)
        messages = [
            {"role": "system", "content": "You are a powerful AI agent."},
            {"role": "user", "content": "What is in the current directory?"},
            {
                "role": "assistant",
                "content": "ok let me run ls",
                "tool_calls": [
                    {
                        "id": "call_12345678",
                        "function": {"arguments": {"command": "ls"}, "name": "bash"},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "content": "api_sglang.py", "tool_call_id": "call_12345678"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute a bash command in the terminal.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
                        "required": ["command"],
                    },
                },
            }
        ]

        completion, input_ids, output_token_ids, output_token_logprobs = await client.complete(messages, tools)
        print("Completion:", completion)
        print("Input IDs:", input_ids)
        print("Output Token IDs:", output_token_ids)
        print("Output Token Logprobs:", output_token_logprobs)
        print("Messages:", messages)

        tokens = input_ids + output_token_ids
        msgs_loss_mask = mask_assistant_content(tokens, start_ids=start_ids, end_ids=end_ids)
        print("Loss Mask:", msgs_loss_mask)
        print("len(input_ids):", len(input_ids))
        print("len(output_token_ids):", len(output_token_ids))
        print("len(msgs_loss_mask):", len(msgs_loss_mask))

    asyncio.run(main())
