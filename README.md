# FastContext

**A read-only repository-exploration agent that a coding agent delegates to over `bash`.**

Instead of letting a main coding agent burn its own context window on broad file reads and code
searches, it shells out a natural-language question to FastContext. FastContext explores the
repository with read-only tools, then returns a short, cited answer — file paths and line ranges the
main agent can act on directly. Exploration happens in a *separate* process, so the noise of grepping
and reading never lands in the main agent's context.

```text
main agent ──"where is request validation done?"──▶  fastcontext -q ... (read-only explore)
          ◀──────────  <final_answer> src/router.py:42-58 … </final_answer>  ──────────┘
```

This is a fork focused on making that harness robust and pleasant to operate: a live **TUI** to watch
a run, **context-budget** management so long explorations answer instead of dying, **provider
auto-detection** of token limits, **citation validation** that drops hallucinated line ranges, and a
small **eval harness**. See [Fork features](#fork-features) for the full list.

---

## Table of contents

- [What it does](#what-it-does)
- [Fork features](#fork-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick start — watch a run in the TUI](#quick-start--watch-a-run-in-the-tui)
- [How a coding agent uses it (over bash)](#how-a-coding-agent-uses-it-over-bash)
- [CLI reference](#cli-reference)
- [Sizing tokens and the context budget](#sizing-tokens-and-the-context-budget)
- [Programmatic use](#programmatic-use)
- [Evaluation harness](#evaluation-harness)
- [Background](#background)

---

## What it does

FastContext runs a small agent loop against an **OpenAI-compatible** chat-completions endpoint
(llama.cpp, Ollama, vLLM, or a hosted API). Each turn the model may call read-only tools; when it has
enough evidence it emits a `<final_answer>` block:

| Tool | Purpose |
| --- | --- |
| `Read` | Read a file (optionally a line range). |
| `Glob` | List files by glob pattern, newest first. |
| `Grep` | Search file contents with ripgrep (`rg`). |

The final answer is compact, machine-parseable evidence:

```text
<final_answer>
src/fastcontext/agent/tool/read.py:13-64
src/fastcontext/agent/tool/grep.py:9-120
</final_answer>
```

The contract is deliberately narrow: **FastContext finds the relevant code; your main agent decides
what to do with it.** It never edits files.

## Fork features

Everything below is in this fork on top of the original explorer:

- 🖥️ **Live TUI run inspector** (`--tui`) — stream a run as collapsible rows (reasoning, each tool
  call, each result, the final answer) with a docked token-usage bar. Far more legible than dumping
  raw JSON. See [Quick start](#quick-start--watch-a-run-in-the-tui).
- 🧮 **Context-budget management** (`--max-context`, `--max-tool-output-chars`) — tracks the growing
  conversation and, as it approaches the window, tells the agent to answer *now* instead of exploring
  its way into a prompt the server rejects. A per-tool-output cap stops one giant `Read` from blowing
  the window in a single turn.
- 🔎 **Provider token auto-detection** (`--max-tokens auto`) — queries the provider's `/models`
  endpoint to discover the model's real context length (including llama.cpp swapper launch args,
  where usable context is `ctx-size ÷ parallel`) instead of guessing a hardcoded default.
- 📌 **Citation validation** — line ranges the model never actually opened during exploration are
  dropped from the answer, so a hallucinated citation can't slip through.
- 🧭 **Robust path handling** — mangled or relative tool-call paths are resolved against the working
  directory; absolute workspace paths are primed into the prompt.
- 🪟 **Cross-platform ripgrep** — `rg` is located with `shutil.which`, output is decoded as UTF-8, and
  ripgrep invocations have a subprocess timeout.
- 🧾 **Trajectory recording** — every run is written to a JSONL trajectory (`--traj`) for replay and
  scoring.
- 📊 **Eval harness** (`eval/`) — run branches from isolated worktrees against (repo, query) tasks and
  compare turns, tool calls, failed/duplicate calls, citation quality, and tokens. See
  [Evaluation harness](#evaluation-harness).

## Installation

FastContext requires **Python 3.12+** and uses [`uv`](https://docs.astral.sh/uv/). It also needs
[ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) on `PATH` for the `Grep` tool.

```bash
# install the CLI from the repo root
uv tool install .

# …or set up a dev environment
uv sync --all-groups
```

## Configuration

FastContext is configured through a [config file](#configuration-file), environment variables, or CLI
flags. Only the model and endpoint are required.

| Variable | Required | Default | Meaning |
| --- | --- | --- | --- |
| `FC_MODEL` | ✅ | — | Model name/id served by the endpoint (legacy alias: `MODEL`). |
| `FC_BASE_URL` | ✅ | — | OpenAI-compatible base URL, e.g. `http://127.0.0.1:11434/v1` (alias: `BASE_URL`). |
| `FC_API_KEY` | — | — | Sent as a bearer token when the endpoint requires auth (alias: `API_KEY`). |
| `FC_TEMPERATURE` | — | `0.7` | Sampling temperature. |
| `FC_MAX_TOKENS` | — | `auto` → `4096` | Completion cap per response. An integer, or `auto` to detect the model's context length from the provider. |
| `FC_MAX_CONTEXT` | — | `0` (off) | Usable context window in tokens; the budget finalizes the run before it overruns. `0` disables the budget. |
| `FC_MAX_TOOL_OUTPUT_CHARS` | — | `16000` | Truncate a single tool result above this many characters (`0` disables). |
| `FC_CONTEXT_RESERVE` | — | auto | Tokens held back from the budget for one turn of tool output + the completion. |
| `FC_REASONING_EFFORT` | — | — | Passed through to servers that support it (`none`/`low`/`medium`/`high`/`max`). |

CLI flags override the matching environment variable for a single run.

### Configuration file

Re-exporting these on every call is wasteful — especially when a coding agent drives FastContext over
`bash`. Set them once in a TOML file instead; keys mirror the `FC_*` variables without the prefix.

Scaffold one with `fastcontext init` — it writes a starter file (owner-readable only) and bakes in any
`FC_*` variables you already have exported, so you can freeze a working shell into a config in one step:

```bash
fastcontext init            # -> $XDG_CONFIG_HOME/fastcontext/config.toml (--path / --force available)
```

```toml
# ~/.config/fastcontext/config.toml
base_url = "http://127.0.0.1:11434/v1"
model    = "fastcontext"
api_key  = "dummy"                 # omit if your endpoint needs no auth

# optional tuning
max_tokens = "auto"                # int, or "auto" to detect from the provider
max_context = 70000                # usable window; enables the context budget
max_tool_output_chars = 16000
reasoning_effort = "none"
temperature = 0.0
```

With that in place, `fastcontext -q "…"` just works — no env vars required. Settings resolve with this
precedence (highest first):

```text
CLI flag  >  FC_* env var  >  project config  >  user config  >  built-in default
```

| Source | Location |
| --- | --- |
| **User config** | `${XDG_CONFIG_HOME:-~/.config}/fastcontext/config.toml` — set the endpoint once per machine. |
| **Project config** | `./.fastcontext/config.toml` (searched upward from the working directory) — pin per-repo settings. |
| **Explicit** | `--config PATH` or `FC_CONFIG=PATH` — bypasses discovery and uses that file. |

Environment variables and CLI flags remain the highest-priority overrides, so existing setups keep
working — a config file only makes them optional.

## Quick start — watch a run in the TUI

The TUI is the easiest way to *see* what FastContext does. Point it at a local endpoint and pass
`--tui`:

```bash
export FC_BASE_URL="http://127.0.0.1:11434/v1"   # e.g. Ollama
export FC_MODEL="hf.co/mitkox/FastContext-1.0-4B-RL-Q4_K_M-GGUF:latest"

# run from inside the repository you want to explore
fastcontext -q "Where is authentication handled and where would I add a new provider?" --tui
```

You get a live, collapsible view of the run:

```text
 FastContext · running…
 Query: Where is authentication handled and where would I add a new provider?
 ── turn 1 ──
 ▸ 🧠 reasoning · turn 1
 ▸ 🔧 Grep(pattern="login", output_mode="content")
 ▸ 📄 result · Grep · 12 lines
 ── turn 2 ──
 ▸ 🔧 Read(path="auth/session.py")
 ▸ 📄 result · Read · 40 lines
 ▾ ✅ final answer
     auth/session.py:1-40
 ────────────────────────────────────────────────────────────
 📊 input 5,412 · output 318 · context 5,730
```

- Every reasoning block, tool call, and tool result is its own **collapsible row** (streamed live,
  even while collapsed). The final answer is expanded automatically.
- Keys: **`e`** expand all · **`c`** collapse all · **`q`** quit.
- The bottom bar tracks cumulative **input / output** tokens and the latest **context** size.

When the run finishes, the final answer is also printed to stdout, so `--tui` stays scriptable.

### Local endpoints

**Ollama** (easiest on macOS):

```bash
brew install ollama && brew services start ollama
ollama pull hf.co/mitkox/FastContext-1.0-4B-RL-Q4_K_M-GGUF

export FC_BASE_URL="http://127.0.0.1:11434/v1"
export FC_MODEL="hf.co/mitkox/FastContext-1.0-4B-RL-Q4_K_M-GGUF:latest"
export FC_REASONING_EFFORT="none"   # FastContext/Qwen models emit reasoning separately
```

**llama.cpp** works the same way — just point `FC_BASE_URL` at its `/v1`. Note that with
`--parallel N`, the *usable* window per request is the configured `--ctx-size` divided by `N`;
`--max-tokens auto` accounts for this automatically (see below).

## How a coding agent uses it (over bash)

FastContext is built to be driven by another agent. The main agent shells out one command per
question and reads the result from **stdout**:

```bash
fastcontext -q "Locate the request-validation logic and the tests that cover it" --citation
```

- **`--citation`** makes stdout *only* the `<final_answer>` block — clean to parse, nothing else.
- **stdout** carries the answer; **stderr** carries diagnostics (token auto-detection notes, budget
  warnings). A parsing agent should read stdout and ignore stderr.
- Exit code is `0` for a normal run. On an LLM failure the agent still exits `0` and writes
  `LLM API call failed…` — check stdout for that marker rather than relying on the exit code.
- Each run records a JSONL trajectory under `.fastcontext/` (override with `--traj`).

A typical delegation loop the main agent runs:

```bash
ANSWER=$(fastcontext -q "$QUESTION" --citation --max-context 60000)
# parse file:line ranges out of <final_answer>…</final_answer>, then read only those spans
```

Because exploration runs in this separate process, none of the intermediate greps and file dumps
enter the main agent's context — only the compact citation block does.

## CLI reference

```text
fastcontext -q "<query>" [options]
```

| Option | Description |
| --- | --- |
| `--query`, `-q` | Natural-language exploration request (required). |
| `--traj`, `-t` | JSONL trajectory output path (default: `.fastcontext/trajectory_<timestamp>.jsonl`). |
| `--max-turns` | Maximum exploration turns before the agent is forced to answer (default `4`). |
| `--citation` | Print only the `<final_answer>` block — the machine-readable path. |
| `--tui` | Watch the run in the collapsible Textual TUI. |
| `--verbose` | Print runtime info and each turn to the terminal. |
| `--max-tokens` | Completion cap per response: an integer, or `auto` to detect from the provider. Overrides `FC_MAX_TOKENS`. |
| `--max-context` | Usable context window in tokens; the budget finalizes before overrunning it. `0` disables. Overrides `FC_MAX_CONTEXT`. |
| `--max-tool-output-chars` | Truncate any single tool result above this size (`0` disables). Overrides `FC_MAX_TOOL_OUTPUT_CHARS`. |
| `--config` | Path to a TOML config file. Overrides `FC_CONFIG` and config-file discovery. |

## Sizing tokens and the context budget

Two independent knobs control token limits; both matter for reliable runs:

- **`--max-tokens` — the per-response completion cap** (sent to the API as `max_completion_tokens`).
  With `auto` (the default), FastContext queries the provider's `/models` endpoint and uses the
  model's advertised context length, recognising vLLM (`max_model_len`), llama.cpp
  (`n_ctx_train`/`n_ctx`), TGI (`max_total_tokens`), and llama.cpp-swapper launch args
  (`ctx-size ÷ parallel`). If nothing is detected it falls back to `4096`. An explicit integer always
  wins, and the diagnostic is printed to **stderr** so it never pollutes a parsed answer.

- **`--max-context` — the exploration budget.** As the conversation approaches this size, the agent
  is told to produce its final answer instead of continuing to explore — which is what prevents a long
  run from growing the prompt until the server rejects it (`exceeds the available context size`). It
  is **off by default** (`0`), because a safe value can't be guessed blindly: a server's usable window
  is often well below its configured one (llama.cpp `--parallel 2` halves it per slot). Set it to your
  model's real usable window; pair it with `--max-tool-output-chars` so one large `Read` can't
  overshoot the window in a single turn.

A good default for a local llama.cpp preset serving 160k context with `--parallel 2`:

```bash
export FC_MAX_CONTEXT=70000        # ~80k usable, minus headroom for the final turn
export FC_MAX_TOOL_OUTPUT_CHARS=16000
```

## Programmatic use

```python
import asyncio
from fastcontext.agent.agent_factory import make_fastcontext_agent

async def explore(question: str) -> str:
    agent = make_fastcontext_agent(
        trajectory_file=".fastcontext/trajectory.jsonl",
        work_dir=".",
        max_tokens="auto",        # or an int; None reads FC_MAX_TOKENS
        max_context=60000,        # 0 disables the budget
    )
    return await agent.run(prompt=question, max_turns=6, citation=True)

print(asyncio.run(explore("Where is the retry logic for tool calls?")))
```

To render a run live instead, hand the agent to the TUI:

```python
from fastcontext.tui import FastContextTUI

app = FastContextTUI(agent=agent, prompt=question, max_turns=6, citation=True)
app.run()
print(app.final_answer)
```

The agent also accepts an `event_sink` callable (`agent.run(..., event_sink=fn)`) that receives
streaming `Event` objects (turn started, token deltas, tool calls/results, usage, finished) — the same
mechanism the TUI uses, so you can build your own live view.

## Evaluation harness

`eval/` contains a small harness for comparing branches on the same tasks:

```bash
cd eval
uv run fc-eval -c config.yaml run        # run each branch from its own worktree
uv run fc-eval analyze                    # score trajectories
uv run fc-eval dashboard                  # build a self-contained HTML dashboard
uv run fc-eval all                        # run + analyze + dashboard
```

It reports, per branch, means for turns, tool calls, failed/duplicate calls, self-corrections,
citation counts and citation quality, and tokens — with failed runs retried and excluded from the
means so a broken run is never averaged in as a zero. See [`eval/README.md`](eval/README.md).

## Background

FastContext originates from research on training efficient repository-exploration models (4B–30B) with
supervised fine-tuning and task-grounded RL, delegating broad exploration out of the main coding
agent's trajectory to improve the score-per-token tradeoff. This fork keeps that exploration contract
and hardens the surrounding harness for day-to-day use.

```bibtex
@misc{zhang2026fastcontexttrainingefficientrepository,
      title={FastContext: Training Efficient Repository Explorer for Coding Agents},
      author={Shaoqiu Zhang and Maoquan Wang and Yuling Shi and Yuhang Wang and Xiaodong Gu and Yongqiang Yao and Tori Gong and Sheng Chen and Rao Fu and Anisha Agarwal and Spandan Garg and Gabriel Ryan and Colin Merkel and Yufan Huang and Shengyu Fu},
      year={2026},
      eprint={2606.14066},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2606.14066},
}
```
