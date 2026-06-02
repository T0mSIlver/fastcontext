import json

from apis import call_llm_api

system_file = "./system.md"
tools_file = "./tools.json"
with open(tools_file, "r", encoding="utf-8") as f:
    tools = json.load(f)
with open(system_file, "r", encoding="utf-8") as f:
    system = f.read()


TASK_SYSTEM_PROMPT = """
You are a helpful data synthesis assistant that generates parallel tool calls to explore a codebase based on a user query.

# Task
Focusing on the below <query>, you MUST follow the rules to generate parallel tool calls in your **first turn** to explore the codebase.

## Rules (in priority order)

1. **Parallel execution**: All tool calls in the first turn MUST be issued simultaneously in a single response — do NOT wait for one to finish before calling the next.
2. **No redundancy**: Each call must target distinct information. Do not issue two calls that would return overlapping results (e.g., searching the same keyword in two different ways).
3. **Coverage-driven count**: Issue as many calls as needed to achieve broad, diverse coverage of the query — typically **4 tool calls**.
4. **Diversity of signal**: Calls should vary in strategy — e.g., combine file-pattern search, symbol search and directory listing rather than repeating the same approach.

## Decision logic for parallel call count
- Simple / narrow query → 1 - 2 tool calls
- Moderate scope → 3 - 4 tool calls
- Broad / multi-faceted query → 5 - 6 tool calls (justify each extra call)
""".strip()

work_env_template = """
## Working Environment

OS Version: Linux

Shell: bash

Workspace Path:${WORK_DIR}

The directory listing of the workspace is:
```
${WORK_DIR_LS}
```
"""


def prepare_samples_message(prompt_template: str, tools: list, samples: list, verbose: bool = False):
    global TASK_PROMPT
    for i, sample in enumerate(samples):
        query = sample["subagent"]["tool_calls"][0]["arguments"]["prompt"]
        input_query = f"<query>{query}</query>"
        workspace = sample.get("workspace", "Unknown")
        workspace_ls = sample.get("workspace_ls", "Unknown")
        system = (
            TASK_SYSTEM_PROMPT
            + "\n\n"
            + prompt_template.replace("${WORK_DIR}", workspace).replace("${WORK_DIR_LS}", workspace_ls)
        )

        task_prompt = (
            input_query
            + "\n"
            + "Now, let's generate parallel tool calls to explore the codebase based on the above query."
        )
        prompt_msgs = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": task_prompt,
            },
        ]

        if i == 0 and verbose:
            print(json.dumps(prompt_msgs, indent=2))
            print(json.dumps(tools, indent=2))
        sample["payload"] = {
            "messages": prompt_msgs,
            "tools": tools,
        }

    return samples


def gen_parallel_tool_calls(model: str, sample: dict) -> str | None:
    msgs = sample["payload"]["messages"]
    tools = sample["payload"]["tools"]
    response = call_llm_api(
        model=model,
        messages=msgs,
        tools=tools,
    )
    tool_msg = None
    if response is not None:
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        tool_calls = []
        for choice in response.choices:
            if choice.message.tool_calls:
                tool_calls.extend(choice.message.tool_calls)
        if finish_reason == "tool_calls" and tool_calls:
            tool_msg = {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        "id": tc.id,
                        "type": "function",
                    }
                    for tc in tool_calls
                ],
            }
    else:
        print(f"Instance {sample['instance_id']} failed after retries of API calls.")
    return tool_msg


def run():
    model = "claude-sonnet-4.6"
    input_file = "./query_1010samples.jsonl"
    output_file = "./query_1010samples_parallel_tool_calls.jsonl"
    with open(input_file, "r") as f:
        samples = [json.loads(line) for line in f]
    samples = prepare_samples_message(work_env_template, tools, samples, verbose=True)

    n = 0
    num_parallel_calls = []
    with open(output_file, "w") as f:
        for i, sample in enumerate(samples):
            tool_msg = gen_parallel_tool_calls(model, sample)
            if tool_msg is not None:
                sample["message_parallel_tool_calls"] = tool_msg
                n_tool_calls = len(tool_msg.get("tool_calls", []))
                num_parallel_calls.append(n_tool_calls)
                if i == 0:
                    print(json.dumps(tool_msg, indent=2))
                del sample["payload"]
                f.write(json.dumps(sample) + "\n")
                n += 1
            print(f"Processed sample {i+1}/{len(samples)}, parallel: {n_tool_calls}")

    avg_parallel_calls = sum(num_parallel_calls) / len(num_parallel_calls) if num_parallel_calls else 0
    print(f"Generated parallel tool calls for {n} out of {len(samples)} samples.")
    print(f"Average number of parallel tool calls per sample: {avg_parallel_calls}")
    print(f"Output saved to {output_file}.")


def construct_trajectory_with_tool_calls(system: str, tools: list[dict]) -> list[dict]:
    input_file = "./query_swerebenchv2_1010samples_parallel_tool_calls.jsonl"
    output_file = "./query_swerebenchv2_1010samples_parallel_tool_calls_traj.jsonl"
    with open(input_file, "r") as f:
        samples = [json.loads(line) for line in f]

    for sample in samples:
        query = sample["subagent"]["tool_calls"][0]["arguments"]["prompt"]
        input_query = f"<query>{query}</query>"
        workspace = sample.get("workspace", "Unknown")
        workspace_ls = sample.get("workspace_ls", "Unknown")
        system = system.replace("${OS_KIND}", "Linux").replace("${SHELL_NAME}", "bash")
        system = system.replace("${WORK_DIR}", workspace)
        system = system.replace("${WORK_DIR_LS}", workspace_ls)
        message_parallel_tool_calls = sample["message_parallel_tool_calls"]
        message_parallel_tool_calls["content"] = ""
        for tc in message_parallel_tool_calls["tool_calls"]:
            tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
        traj = [
            {"role": "system", "content": system},
            {"role": "user", "content": input_query},
            message_parallel_tool_calls,
        ]
        sample["trajectory"] = traj

    with open(output_file, "w") as f:
        for sample in samples:
            row = {
                "source": "parallel_toolcalls",
                "instance_id": sample["instance_id"],
                "messages": sample["trajectory"],
                "tools": tools,
            }
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(samples)} samples to {output_file}.")


if __name__ == "__main__":
    # run()
    construct_trajectory_with_tool_calls(system=system, tools=tools)
