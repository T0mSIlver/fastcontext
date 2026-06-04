import json
import os

input_file = "./query_swerebench_1k.jsonl"
workdir = "/workspace/fastcontext/sft-claude-sonnet-4.6/"
save_file = "./query_swerebench_1k_multiturn_traj.jsonl"

with open(input_file, "r") as f:
    samples = [json.loads(line) for line in f]

n = 0
with open(save_file, "w") as fw:
    n_miss = 0
    n_not_assistant = 0
    for sample in samples:
        instance_id = sample["instance_id"]
        traj_file = os.path.join(workdir, instance_id, "traj.jsonl")
        if not os.path.exists(traj_file):
            n_miss += 1
            continue
        # get the last line of traj_file
        with open(traj_file, "r") as f:
            lines = f.readlines()
            msgs = [json.loads(line) for line in lines]
            if msgs[-1]["role"] != "assistant":
                n_not_assistant += 1
                continue
            if len(msgs) < 3:
                n_not_assistant += 1
                continue
        # only keep role, content, tool_calls
        traj = []
        for msg in msgs:
            line = {
                "role": msg["role"],
            }
            if "content" in msg and msg["content"]:
                line["content"] = msg["content"]
            if "tool_calls" in msg and msg["tool_calls"]:
                for tc in msg["tool_calls"]:
                    tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
                line["tool_calls"] = msg["tool_calls"]
            traj.append(line)
        row = {
            "source": "multiturn_traj",
            "instance_id": instance_id,
            "messages": traj,
        }
        fw.write(json.dumps(row) + "\n")
        n += 1

    print(f"Missing traj files: {n_miss}/{len(samples)}")
    print(f"Last line not from assistant: {n_not_assistant}/{len(samples)}")
    print(f"Saved {n} samples to {save_file}.")
