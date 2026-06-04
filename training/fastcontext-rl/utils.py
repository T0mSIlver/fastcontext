import json
import os
import re


def load_json_file(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        traj = json.load(f)
    # steps: "message", "reasoning_content", "tool_calls", "observation"
    return traj


def parse_final_answer(text: str, drop_prefix: str = None) -> list[dict[str, str]]:

    citations = []
    n_citations = 0
    n_citations_files = 0
    n_citations_lines = 0
    n_broken_lines = 0
    n_overlap_line_citation = 0

    result = {
        "valid": False,
        "prefix": "",
        "final_answer": "",
        "tail": "",
        "citations": citations,
        "n_citations": n_citations,
        "n_citations_files": n_citations_files,
        "n_citations_lines": n_citations_lines,
        "n_broken_lines": n_broken_lines,
        "n_overlap_line_citation": n_overlap_line_citation,
    }
    if text is None:
        return result
    fa_match = re.search(r"<final_answer>(.*?)</final_answer>", text, re.DOTALL)
    if fa_match is None:
        return result

    if drop_prefix is not None and not drop_prefix.endswith("/"):
        drop_prefix += "/"

    prefix = text[: fa_match.start()]
    final_answer = fa_match.group(1)
    tail = text[fa_match.end() :]
    entries = final_answer.strip().splitlines()

    entries = [e for e in entries if e.strip()]

    file_paths = set()
    for entry in entries:
        # /absolute/path/to/file_1.py:10-15 (explanation 1)
        # /absolute/path/to/file_1.py:10 (explanation 2)
        # /absolute/path/to/file_1.py:10-15
        match = re.match(r"(.+?):(\d+(?:-\d+)?)\s*(.*)", entry.strip())
        if match:
            file_path = match.group(1).strip()
            if drop_prefix and file_path.startswith(drop_prefix):
                file_path = file_path[len(drop_prefix) :]
            line_range = match.group(2).strip()
            explanation = match.group(3).strip() if match.group(3) else ""
            start_line, end_line = line_range.split("-") if "-" in line_range else (line_range, line_range)
            start_line = int(start_line.strip())
            end_line = int(end_line.strip())
            file_paths.add(file_path)
            n_citations_lines += end_line - start_line + 1
            n_citations_files = len(file_paths)
            citations.append(
                {
                    "path": file_path,
                    "line_range": line_range,
                    "start_line": start_line,
                    "end_line": end_line,
                    "explanation": explanation,
                }
            )
        else:
            n_broken_lines += 1
    n_overlap_line_citation = line_range_overlap(citations)

    result["valid"] = True
    result["prefix"] = prefix
    result["final_answer"] = final_answer
    result["tail"] = tail
    result["citations"] = citations
    result["n_citations"] = len(citations)
    result["n_citations_files"] = n_citations_files
    result["n_citations_lines"] = n_citations_lines
    result["n_broken_lines"] = n_broken_lines
    result["n_overlap_line_citation"] = n_overlap_line_citation

    return result


def read_citations_content(citations: list[dict[str, str]], workspace: str = None) -> str:
    full_content = ""
    max_lines = 500
    max_line_length = 500
    for citation in citations:
        file_path = citation["path"]
        if workspace is not None:
            file_path = os.path.join(workspace, file_path)
        if not os.path.exists(file_path):
            content = "Error: the file does not exist.\n"
        else:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    start_line = citation["start_line"]
                    end_line = citation["end_line"]
                    lines = lines[start_line - 1 : end_line]
                    lines = [line[:max_line_length] for line in lines[:max_lines]]
                    content = "".join(lines)
            except Exception:
                content = "Error: failed to read the file.\n"
        full_content += f"```{citation['path']}:{citation['line_range']}\n{content}```\n"
    return full_content


def calculate_explore_score(precision, recall, n_citation, n_label=3.0, beta=0.5, lmbda=0.1):
    if (precision + recall) == 0:
        f_beta = 0
    else:
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    penalty = lmbda * max(0, (n_citation - n_label) / n_label)
    final_score = f_beta - penalty
    return final_score


def calculate_score_file(edit_true: list[dict], citations_pred: list[dict]) -> dict[str, float]:
    true_files = set(e["path"] for e in edit_true)
    pred_files = set(c["path"] for c in citations_pred)

    overlap = true_files & pred_files

    precision = len(overlap) / len(pred_files) if len(pred_files) > 0 else 0.0
    recall = len(overlap) / len(true_files) if len(true_files) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    score = calculate_explore_score(precision, recall, len(pred_files), n_label=len(true_files))

    return {
        "score": score,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_citation_files": len(pred_files),
        "n_patch_files": len(true_files),
    }


def calculate_score_line(edit_true: list[dict], citations_pred: list[dict]) -> dict[str, float]:
    """
    edit_true, e.g. [{'path': 'docs/querying/post-aggregations.md', 'start_line': 36, 'end_line': 42}, ...]
    citations_pred, e.g. [{'path': 'docs/querying/post-aggregations.md', 'start_line': 30, 'end_line': 50}, ...]
    """
    # caculate line-level precision, recall, F1
    true_lines = set()
    for edit in edit_true:
        for line in range(edit["start_line"], edit["end_line"] + 1):
            true_lines.add((edit["path"], line))

    pred_lines = set()
    for edit in citations_pred:
        for line in range(edit["start_line"], edit["end_line"] + 1):
            pred_lines.add((edit["path"], line))

    overlap = true_lines & pred_lines

    precision = len(overlap) / len(pred_lines) if len(pred_lines) > 0 else 0.0
    recall = len(overlap) / len(true_lines) if len(true_lines) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    score = calculate_explore_score(precision, recall, len(pred_lines), n_label=len(true_lines))

    return {
        "score": score,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_citation_lines": len(pred_lines),
        "n_patch_lines": len(true_lines),
    }


def parse_patch(text: str, workspace: str = None, file_types: list[str] = None):
    file_pattern = re.compile(r"^diff --git a/(.+) b/(.+)$", re.MULTILINE)
    file_sections = re.split(r"(?=^diff --git a/)", text, flags=re.MULTILINE)
    file_sections = [s for s in file_sections if len(s)]  # skip the first empty element

    edit_files = []
    for file_patch in file_sections:
        files = file_pattern.findall(file_patch)
        if len(files) != 1:
            # skip files that are not in the format of a/b
            continue
        file = files[0][0]
        if file_types is not None and not any(file.endswith(ft) for ft in file_types):
            continue
        lines = file_patch.splitlines()
        _patch = "\n".join(lines[3:]).strip()
        hunks = re.split(r"(?=^@@ -\d+,\d+ \+\d+,\d+ @@)", _patch, flags=re.MULTILINE)

        for h in hunks:
            hunk_header_pattern = re.compile(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@")
            hunk_header_match = hunk_header_pattern.match(h)
            if not hunk_header_match:
                continue
            old_line_start = int(hunk_header_match.group(1))
            old_n_lines = int(hunk_header_match.group(2))
            # new_start_line = int(hunk_header_match.group(3))
            # new_n_lines = int(hunk_header_match.group(4))
            # skip new file, e.g. @@ -0,0 +1,145 @@
            if old_line_start == 0 and old_n_lines == 0:
                continue

            old_line_end = old_line_start + old_n_lines - 1
            if workspace is not None:
                file = os.path.join(workspace, file)
            edit_files.append(
                {
                    "path": file,
                    "start_line": old_line_start,
                    "end_line": old_line_end,
                }
            )
            if old_line_end < old_line_start:
                print(h)

        if len(hunks) == 0:
            continue
    return edit_files


def line_range_overlap(citations: dict):

    n_overlap_line_citation = 0
    files_lines = {}
    for citation in citations:
        if citation["path"] not in files_lines:
            files_lines[citation["path"]] = []
        files_lines[citation["path"]].append((citation["start_line"], citation["end_line"]))

    for _, line_ranges in files_lines.items():
        line_ranges.sort()
        for i in range(1, len(line_ranges)):
            _, prev_end = line_ranges[i - 1]
            curr_start, _ = line_ranges[i]
            if curr_start < prev_end:
                n_overlap_line_citation += 1
                print(f"Warning: overlapping line ranges: {line_ranges}")
    return n_overlap_line_citation


def caculte_score(pred: str, label: str, drop_prefix: str = None) -> dict:
    ground_truth = parse_final_answer(label, drop_prefix=drop_prefix)["citations"]
    pred_edits = parse_final_answer(pred, drop_prefix=drop_prefix)["citations"]
    score_file = calculate_score_file(ground_truth, pred_edits)
    score_line = calculate_score_line(ground_truth, pred_edits)

    return {
        "score_file": score_file,
        "score_line": score_line,
    }


def log_runtime_info(runtime_info: dict):
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(runtime_info)
    except ImportError:
        pass  # wandb not available


if __name__ == "__main__":
    c = read_citations_content(
        [
            {"path": "flexx/README.md", "line_range": "35-42", "start_line": 35, "end_line": 42},
            {"path": "flexx/src/llm/llm.py", "line_range": "10-15", "start_line": 10, "end_line": 15},
        ],
        workspace="/testbed/",
    )
    print(c)
