import os
import platform
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from jinja2 import Environment as JinjaEnvironment
from jinja2 import StrictUndefined

from fastcontext.agent.observed import citation_observed


@dataclass(frozen=True, slots=True, kw_only=True)
class SystemPromptArgs:
    OS_KIND: str
    SHELL_NAME: str
    WORK_DIR: str
    WORK_DIR_LS: str


def _load_system_prompt(path: Path, builtin_args: SystemPromptArgs) -> str:
    system_prompt = path.read_text(encoding="utf-8").strip()
    env = JinjaEnvironment(
        keep_trailing_newline=True,
        lstrip_blocks=True,
        trim_blocks=True,
        variable_start_string="${",
        variable_end_string="}",
        undefined=StrictUndefined,
    )
    try:
        template = env.from_string(system_prompt)
        return template.render(asdict(builtin_args))
    except Exception as e:
        raise RuntimeError(f"Failed to render system prompt template: {e}") from e


def load_system_prompt(work_dir: str) -> str:

    os_kind = platform.system()
    if os_kind == "Windows":
        shell_name = os.getenv("COMSPEC", "powershell.exe")
    else:
        shell_name = os.getenv("SHELL", "bash")
    # List entries with their absolute paths so the model is primed to pass
    # absolute paths to the tools (Read/Grep/Glob all expect absolute paths),
    # instead of inventing a short prefix from the workspace name.
    base = Path(work_dir)
    work_dir_ls = "\n".join(str(base / name) for name in sorted(os.listdir(work_dir)))

    return _load_system_prompt(
        path=Path(__file__).parent / "system.md",
        builtin_args=SystemPromptArgs(
            OS_KIND=os_kind,
            SHELL_NAME=shell_name,
            WORK_DIR=work_dir,
            WORK_DIR_LS=work_dir_ls,
        ),
    )


def parse_citations(text: str) -> list:
    final_answer = re.search(r"<final_answer>(.*?)</final_answer>", text, re.DOTALL)
    if final_answer is None:
        return []

    entries = final_answer.group(1).strip().splitlines()

    entries = [e for e in entries if e.strip()]

    citations = []
    for entry in entries:
        # /absolute/path/to/file_1.py:10-15
        # /absolute/path/to/file_1.py:10-15 (explanation 1)
        # /absolute/path/to/file_1.py:10 (explanation 2)
        match = re.match(r"(.+?):(\d+(?:-\d+)?)\s*(.*)", entry.strip())
        if match:
            file_path = match.group(1).strip()
            line_range = match.group(2).strip()
            explanation = match.group(3).strip() if match.group(3) else ""
            start_line, end_line = line_range.split("-") if "-" in line_range else (line_range, line_range)
            start_line = int(start_line.strip())
            end_line = int(end_line.strip())
            citations.append(
                {
                    "path": file_path,
                    "line_range": line_range,
                    "start_line": start_line,
                    "end_line": end_line,
                    "explanation": explanation,
                }
            )
    return citations


def format_citations(
    citations: list, validate: bool = True, observed: dict | None = None, cwd: str | None = None
) -> str:

    if validate:
        validated_citations = []
        for c in citations:
            # if not file or not existing, skip this citation. Resolve a relative path against the
            # working directory (as the observed-lines side does) rather than the process cwd, or a
            # relative citation would be checked against the wrong directory and silently dropped.
            path = c["path"]
            if cwd and not os.path.isabs(path):
                path = os.path.join(cwd, path)
            if not os.path.isfile(path):
                continue
            validated_citations.append(c)

        citations = validated_citations

    if observed is not None:
        # Drop citations whose lines were never opened during exploration: a hallucinated range is
        # worse than a missing one.
        citations = [c for c in citations if citation_observed(observed, c, cwd)]

    formatted = []
    for c in citations:
        if c["explanation"]:
            formatted.append(f"{c['path']}:{c['line_range']} {c['explanation']}")
        else:
            formatted.append(f"{c['path']}:{c['line_range']}")
    return "<final_answer>\n" + "\n".join(formatted) + "\n</final_answer>"


def get_final_answer(text: str, observed: dict | None = None, cwd: str | None = None) -> str:
    citations = parse_citations(text)
    final_answer = format_citations(citations, observed=observed, cwd=cwd)
    return final_answer


if __name__ == "__main__":
    print(load_system_prompt(os.getcwd()))
