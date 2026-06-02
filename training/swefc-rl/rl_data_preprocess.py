import json
import os

from utils import parse_patch


def dir_is_empty(dir_path):
    """Check if a directory is empty."""
    return not any(os.scandir(dir_path))


system_prompt_file = "system.md"
with open(system_prompt_file, "r") as f:
    system_prompt = f.read()

input_file = "samples.jsonl"
output_file = "swefc_rl_training.jsonl"
with open(input_file, "r") as f:
    samples = [json.loads(line) for line in f]

rl_samples = []
for i, sample in enumerate(samples):
    instance_id = sample["instance_id"]
    query = sample["subagent"]["tool_calls"][0]["arguments"]["prompt"]
    input_query = f"<query>{query}</query>"
    workspace = "/testbed" + sample["workspace"]
    workspace_ls = sample["workspace_ls"]
    patch = sample["patch"]
    if not os.path.exists(workspace):
        print(f"Workspace does not exist: {workspace}")
        continue
    if dir_is_empty(workspace):
        print(f"Workspace is empty: {workspace}")
        continue

    edits = parse_patch(patch)
    label = "\n".join([f"{edit['path']}:{edit['start_line']}-{edit['end_line']}" for edit in edits])
    label = "<final_answer>\n" + label.strip() + "\n</final_answer>"

    system = system_prompt.replace("${WORK_DIR}", workspace).replace("${WORK_DIR_LS}", workspace_ls)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": input_query},
    ]
    rl_samples.append(
        {
            "messages": messages,
            "label": label,
            "metadata": {"workspace": workspace, "instance_id": instance_id},
        }
    )

with open(output_file, "w") as f:
    for s in rl_samples:
        f.write(json.dumps(s) + "\n")
