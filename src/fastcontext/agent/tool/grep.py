import json
from pathlib import Path

from .tool import Tool
from .utils import RG_PATH, resolve_path


class GrepTool(Tool):
    name = "Grep"
    description: str = Tool.load_desc(Path(__file__).parent / "grep.md")
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The regular expression pattern to search for in file contents",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search in (rg pattern -- PATH). Defaults to current working directory.",
            },
            "glob": {
                "type": "string",
                "description": 'Glob pattern to filter files (e.g. "*.js", "*.{ts,tsx}") - maps to rg --glob',
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
                "description": 'Output mode: "content" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), "files_with_matches" shows file paths (supports head_limit), "count" shows match counts (supports head_limit). Defaults to "content".',
            },
            "-B": {
                "type": "number",
                "description": 'Number of lines to show before each match (rg -B). Requires output_mode: "content", ignored otherwise.',
            },
            "-A": {
                "type": "number",
                "description": 'Number of lines to show after each match (rg -A). Requires output_mode: "content", ignored otherwise.',
            },
            "-C": {
                "type": "number",
                "description": 'Number of lines to show before and after each match (rg -C). Requires output_mode: "content", ignored otherwise.',
            },
            "-n": {
                "type": "boolean",
                "description": 'Show line numbers in output (rg -n). Requires output_mode: "content", ignored otherwise. Defaults to true.',
            },
            "-i": {
                "type": "boolean",
                "description": "Case insensitive search (rg -i)",
            },
            "type": {
                "type": "string",
                "description": "File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than include for standard file types.",
            },
            "head_limit": {
                "type": "number",
                "minimum": 0,
                "description": 'Limit output to first N lines/entries, equivalent to "| head -N". Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count (limits count entries). When unspecified, results are capped at the first 100 lines.',
            },
            "multiline": {
                "type": "boolean",
                "description": "Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false.",
            },
        },
        "required": ["pattern"],
    }

    async def call(self, parameters: str, **kwargs) -> str:
        params: dict = json.loads(parameters)
        cwd = kwargs.get("cwd", str(Path.cwd()))
        # ripgrep parameters
        pattern = params.get("pattern")
        path = params.get("path", cwd)
        glob = params.get("glob")
        output_mode = params.get("output_mode", "content")
        before_context = params.get("-B")
        after_context = params.get("-A")
        context = params.get("-C")
        line_number = params.get("-n", True)
        ignore_case = params.get("-i", False)
        type = params.get("type")
        head_limit = params.get("head_limit")
        multiline = params.get("multiline")

        path = resolve_path(path, cwd)
        if not Path(path).resolve().is_relative_to(Path(cwd).resolve()):
            return f"<system-reminder>Permission error: `{path}` is not within the working directory `{cwd}`</system-reminder>"

        output = run_rg(
            RG_PATH,
            pattern,
            path,
            cwd=cwd,
            glob=glob,
            output_mode=output_mode,
            before_context=before_context,
            after_context=after_context,
            context=context,
            line_number=line_number,
            ignore_case=ignore_case,
            type=type,
            multiline=multiline,
        )
        if not output:
            output = "No matches found"
        else:
            limit = 100
            if head_limit is not None and head_limit > 0:
                limit = head_limit

            lines = output.splitlines()
            if len(lines) > limit:
                output = "\n".join(lines[:limit])
                truncated_hit = f"Results truncated to first {limit} lines"
                output += f"\n{truncated_hit}"

        return output


def run_rg(rg_path: str, pattern: str, path: str, **kwargs) -> str:
    import subprocess

    command = [rg_path]
    command.append(pattern)
    if path:
        command.append(path)
    if kwargs.get("glob"):
        command.append("--glob")
        command.append(kwargs["glob"])
    if kwargs.get("ignore_case"):
        command.append("--ignore-case")
    if kwargs.get("type"):
        command.append("--type")
        command.append(kwargs["type"])
    if kwargs.get("multiline"):
        command.append("--multiline")
        command.append("--multiline-dotall")
    output_mode = kwargs.get("output_mode")
    if output_mode == "content":
        if kwargs.get("before_context") is not None:
            command.append("-B")
            command.append(str(kwargs["before_context"]))
        if kwargs.get("after_context") is not None:
            command.append("-A")
            command.append(str(kwargs["after_context"]))
        if kwargs.get("context") is not None:
            command.append("-C")
            command.append(str(kwargs["context"]))
        if kwargs.get("line_number"):
            command.append("-n")
    elif output_mode == "files_with_matches":
        command.append("--files-with-matches")
    elif output_mode == "count":
        command.append("--count-matches")

    # --heading and --color never
    command.append("--heading")
    command.append("--color")
    command.append("never")
    # rg omits the filename when the search target is a single explicit file, so a Read-then-Grep on
    # one file came back as bare `12:match` with the path nowhere in the output -- the model then has
    # to remember which file it asked about in order to cite it. Force the heading in every mode so
    # every result carries the path it belongs to.
    command.append("--with-filename")

    cwd = kwargs.get("cwd", str(Path.cwd()))

    timeout = 10  # seconds
    try:
        output = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return f"<system-reminder>Grep timed out after {timeout}s</system-reminder>"

    if output.returncode == 0:
        output_text = (
            output.stdout if isinstance(output.stdout, str) else output.stdout.decode("utf-8", errors="replace")
        )
    else:
        output_text = (
            output.stderr if isinstance(output.stderr, str) else output.stderr.decode("utf-8", errors="replace")
        )
    return output_text
