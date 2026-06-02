import json
import re

system = """
Your main goal is to locate relevant file paths and line ranges that can help answer the <query>.

## Final Answer Format

Provide the final answer in the form of absolute file paths and line ranges, which are denoted by the <final_answer> tag:
<example>
<final_answer>
/absolute/path/to/file_1.py:10-15 (3 ~ 5 words brief explanation of relevance in parentheses, optional but helpful)
/absolute/path/to/file_2.js:102-123
</final_answer></example>

## Guidelines for Final Answer

- **Relevance**: Only include file paths and line ranges that are highly relevant to the query
- **Precision**: Include only the necessary lines that directly address the query
- **Conciseness**: Keep line ranges as narrow as possible while capturing all relevant information
- The provided file contents may contain irrelevant information - carefully filter based on the query
- Avoid large line ranges that include mostly irrelevant content
- Provide only the final answer in the specified format - no explanations needed
"""

task_prompt = """
## Task
You can assume that the <files> contents are the tool results from the previous steps, which are explored based on the <query>.
The <files> contents may contain a large amount of information, and only a small portion of it is relevant to the <query>.
Your task is to identify the relevant file paths and line ranges that can help answer the <query>, and provide the final answer in the specified format.

let's get started!
""".strip()


def extract_final_answer(text: str) -> dict[str, str] | None:
    match = re.search(r"<final_answer>(.*?)</final_answer>", text, re.DOTALL)
    if match is None:
        return None
    prefix = text[: match.start()]
    final_answer = match.group(1)
    tail = text[match.end() :]
    return {
        "prefix": prefix.strip(),
        "final_answer": final_answer.strip(),
        "tail": tail.strip(),
    }


def task_messages(system, files, task_prompt):
    file_content = ""
    for fpath, start, end, content in files:
        file_content += f"\n```{fpath}:{start}-{end}\n{content}\n```"
    file_content = f"<files>\n{file_content}\n</files>"
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": file_content + "\n\n" + task_prompt},
    ]
    return msgs


def add_line_numbers_to_file_content(file_content: str) -> str:
    lines = file_content.splitlines()
    MAX_LINE_LENGTH = 2000
    lines = [line if len(line) <= MAX_LINE_LENGTH else line[:MAX_LINE_LENGTH] + "...\n" for line in lines]

    if len(lines) > 2000:
        lines = lines[:2000]
        lines.append("... (truncated lines)\n")

    numbered_lines = [f"{i+1}|{line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)


input_file = "query-claude-sonnet-4.5-final-answer.jsonl"
output_file = "query_1k_linerange_traj.jsonl"
samples = []
with open(input_file, "r") as f:
    for line in f:
        samples.append(json.loads(line))
print(f"Loaded {len(samples)} samples from {input_file}")

sft_samples = []
for sample in samples:
    messages = sample["payload"]["messages"]
    if sample["message_final_answer"]:
        text = sample["message_final_answer"]["content"]
        final_answer = extract_final_answer(text)
        if final_answer is not None and final_answer["final_answer"]:
            messages.append(
                {
                    "role": "assistant",
                    "content": "<final_answer>\n" + final_answer["final_answer"] + "\n</final_answer>",
                }
            )
            sft_samples.append({"source": "linerange", "instance_id": sample["instance_id"], "messages": messages})

with open(output_file, "w") as f:
    for sample in sft_samples:
        f.write(json.dumps(sample) + "\n")
print(f"Saved {len(sft_samples)} samples to {output_file}")
