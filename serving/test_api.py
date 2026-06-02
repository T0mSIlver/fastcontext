from openai import OpenAI

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


def completions(
    base_url: str,
    model: str,
    api_key: str,
    tools: list[dict],
    messages: list[dict],
):
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        max_completion_tokens=32_000,
        temperature=0.7,
        top_p=0.8,
        tools=tools,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return response


def example_qwen3p5(base_url: str, model: str, api_key: str = "anything", tools=None):
    messages = [
        {"role": "system", "content": "You are a powerful AI agent."},
        {"role": "user", "content": "please show me the current time by calling bash command"},
    ]
    return completions(base_url=base_url, model=model, api_key=api_key, tools=tools, messages=messages)


def example_qwen3p5_tool_calls(base_url: str, model: str, api_key: str = "anything", tools=None):
    messages = [
        {"role": "system", "content": "You are a powerful AI agent."},
        {"role": "user", "content": "please show me the current time"},
        {
            "role": "assistant",
            "content": "let me check the current time for you.",
            "reasoning_content": "The user wants to know the current time and explicitly asks me to call a bash command to show it. I should use the `date` command which is the standard way to display the current date and time in Unix-like systems.",
            "tool_calls": [
                {
                    "id": "call_de82428323644869a1ac3e47",
                    "function": {"arguments": '{"command": "date"}', "name": "bash"},
                    "type": "function",
                }
            ],
        },
        {"role": "tool", "content": "Thu Aug 21 17:42:44 CST 2025", "tool_call_id": "call_de82428323644869a1ac3e47"},
    ]
    return completions(base_url=base_url, model=model, api_key=api_key, tools=tools, messages=messages)


if __name__ == "__main__":
    import os

    base_url = os.environ.get("BASE_URL", "http://0.0.0.0:30000/v1")
    model = os.environ.get("MODEL", "qwen3p5-4b")
    api_key = os.environ.get("API_KEY", "anything")
    print(f"base_url: {base_url}, model: {model}")
    response = example_qwen3p5(base_url, model, api_key, tools=tools)
    print(response)
    response = example_qwen3p5_tool_calls(base_url, model, api_key, tools=tools)
    print(response)
