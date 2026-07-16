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
- 🧮 **Context-budget management** (`--max-context`, `--max-turn-output-tokens`,
  `--max-result-output-tokens`) — tracks the growing conversation and, as it approaches the window,
  tells the agent to answer *now* instead of exploring its way into a prompt the server rejects. A
  per-turn output cap stops one giant `Read` from blowing the window in a single turn; a per-result
  cap stops it starving the other calls of the same turn.
- 🔎 **Provider window auto-detection** (`--max-context auto`) — queries the provider's `/models`
  endpoint to discover the model's real context window (including llama.cpp swapper launch args,
  where usable context is `ctx-size ÷ parallel`) and turns the context budget on with it, instead of
  leaving it off because the value could not be guessed.
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
| `FC_MAX_COMPLETION_TOKENS` | — | `4096` | How long **one response** may be. Not the model's window — see `FC_MAX_CONTEXT`. (Was `FC_MAX_TOKENS`, still honored with a warning.) |
| `FC_MAX_CONTEXT` | — | `auto` | Usable context window in tokens; the budget finalizes the run before it overruns. An integer, `0` to disable, or `auto` to ask the provider (off if it advertises nothing). |
| `FC_MAX_TURN_OUTPUT_TOKENS` | — | `12000` | Total **tokens** of tool output one **turn** may add, across all its tool calls (`0` disables). The reserve is sized against it. Supersedes the `*_CHARS` names, which are converted with a warning. |
| `FC_MAX_RESULT_OUTPUT_TOKENS` | — | `0` (off) | Truncate a **single** tool result above this many **tokens** (`0` disables). Stops one big result eating the whole turn budget. |
| `FC_MAX_CITATIONS` | — | `25` | Cap the number of citations in the final answer; a safety bound on a runaway list (`0` disables). |
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
max_completion_tokens = 4096        # how long ONE response may be (not the window)
max_context = "auto"                # window; int, 0 to disable, or "auto" to ask the provider
max_turn_output_tokens = 12000      # per-TURN total (tokens), across all of a turn's tool calls
max_result_output_tokens = 0        # per-RESULT cap (tokens); 0 = off
max_citations = 25                  # safety cap on the final answer's citation count (0 disables)
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
`--max-context auto` accounts for this automatically (see below).

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
| `--max-turns` | Maximum exploration turns before the agent is forced to answer (default `12`; see [Choosing `--max-turns`](#choosing---max-turns)). |
| `--citation` | Print only the `<final_answer>` block — the machine-readable path. |
| `--tui` | Watch the run in the collapsible Textual TUI. |
| `--verbose` | Print runtime info and each turn to the terminal. |
| `--max-completion-tokens` | How long **one response** may be, in tokens. Not the model's window — see `--max-context`. Overrides `FC_MAX_COMPLETION_TOKENS`. (Deprecated alias: `--max-tokens`.) |
| `--max-context` | Usable context window in tokens: an integer, `0` to disable, or `auto` (default) to ask the provider. Overrides `FC_MAX_CONTEXT`. |
| `--max-turn-output-tokens` | Total tool output, in **tokens**, one **turn** may add across all its calls (`0` disables). Overrides `FC_MAX_TURN_OUTPUT_TOKENS`. |
| `--max-result-output-tokens` | Truncate a **single** tool result above this many **tokens** (`0` disables, the default). Overrides `FC_MAX_RESULT_OUTPUT_TOKENS`. |
| `--max-citations` | Cap the number of citations in the final answer; a safety bound (`0` disables). Overrides `FC_MAX_CITATIONS` (default `25`). |
| `--config` | Path to a TOML config file. Overrides `FC_CONFIG` and config-file discovery. |

## Sizing tokens and the context budget

Two independent knobs control token limits; both matter for reliable runs:

- **`--max-completion-tokens` — how long one response may be** (sent to the API as
  `max_completion_tokens`). Default `4096`. FastContext's responses are tool calls and a final
  answer, both short, so this rarely needs changing.

  It is **never detected from the provider**, and that is the point. What a provider advertises is
  its context *window*; feeding that in here is not merely imprecise, it disables the run — the
  reserve is `2 x` this value, so a window-sized cap makes the reserve exceed the window and the
  agent finalizes before its first turn, answering with no exploration at all. The window belongs in
  `--max-context`, below. Was `--max-tokens` / `FC_MAX_TOKENS`, whose name named neither which tokens
  nor whose; the old name still works and warns.

- **`--max-context` — the exploration budget.** As the conversation approaches this size, the agent
  is told to produce its final answer instead of continuing to explore — which is what prevents a long
  run from growing the prompt until the server rejects it (`exceeds the available context size`).

  `auto` (the default) asks the provider's `/models` endpoint, recognising vLLM (`max_model_len`),
  llama.cpp (`n_ctx_train`/`n_ctx`), TGI (`max_total_tokens`), and llama.cpp-swapper launch args
  (`ctx-size ÷ parallel`, so a `--parallel 2` server reports its real per-slot window). If nothing is
  advertised the budget stays **off** rather than guessing: a wrong window is worse than none — too
  high and the run dies mid-flight, too low and it finalizes early. Pass an integer to set it
  yourself, or `0` to disable.

- **`--max-turn-output-tokens` — the per-turn tool-output budget.** The total a single turn may add
  across *all* of its tool calls, not per result. This is the one the reserve is sized against, so it
  is what keeps the final-answer turn sendable.

  **Raise it before you lower it.** Truncating a repo-exploration agent is self-defeating: the model
  pages the same file back over several turns, spending the context the cap was protecting, and any
  range it never pages back in is simply missing from the answer. The default is sized so that whole
  source files usually arrive in one piece.

  It is a **token** budget. It was a character budget under the `*_CHARS` names, which forced the
  reserve to assume one token per character (CJK really does tokenize that densely) and so held back
  ~4x what a turn of ASCII source actually costs. Those names are converted with a warning, at the
  ASCII ratio they were chosen against — so a config saying `16000` chars becomes ~5300 tokens.

- **`--max-result-output-tokens` — the per-result cap.** Bounds one tool result. **Off by default**,
  because the per-turn budget already protects the window; this only changes how that budget is
  *shared*. The turn budget is spent greedily in call order, so a model that issues a big `Read` and
  two `Grep`s in one turn can have the `Read` consume the entire allowance and get empty results for
  the greps — it asked three questions and got one answer. Setting this to roughly
  `max_turn_output_tokens ÷ expected calls per turn` gives every call in a turn room to survive.

A good default for a local llama.cpp preset serving 160k context with `--parallel 2`:

```bash
export FC_MAX_CONTEXT=70000              # or leave unset: auto detects ~80k usable from the server
export FC_MAX_TURN_OUTPUT_TOKENS=12000    # tokens per turn, across all its tool calls
# export FC_MAX_RESULT_OUTPUT_TOKENS=2500 # optional: keep one big Read from starving the same turn's greps
```

### Choosing `--max-turns`

The turn cap is a bound on the worst case, not a target: an exploration stops as soon as it has the
answer. What actually limits a long run is context, so the affordable number of turns follows from the
budget rather than from a fixed rule:

```
usable turns ≈ (max_context − reserve) ÷ tokens_per_turn
    reserve  =  required_reserve(max_turn_output_tokens, max_completion_tokens)
```

The reserve is substantial — it holds back a full turn of tool output plus the completion so the
final-answer request still fits. At the defaults (`max_turn_output_tokens=12000`,
`max_completion_tokens=4096`) it is **22,315** tokens, and it does not shrink as the window does: it
leaves **49,685** tokens to explore with at `max_context=72000`, but only ~47.7k at the `70000`
suggested above — and nothing at all below ~22.3k, where every run would finalize on turn one.

**Measured** — 12 runs over 6 repos (186 to 47k source files), easy lookups and hard multi-module
traces, each at `--max-turns 16` against `fastcontext-1.0-4b-rl-q8_0` on llama.cpp
(`--ctx-size 160000 --parallel 2`, ~80k usable per slot):

| Metric | Result |
| --- | --- |
| Tokens per turn | **~2.0k median**, 786 min, 4.1k max |
| Peak prompt size | 14.6k – 46.9k (median ~25k) |
| Turns when the agent answered on its own | 5, 6, 9, 10, 13 (median ~9) |
| Runs that used all 16 turns | 6 / 12 |
| Runs the budget stopped early | 1 / 12 (a `llama.cpp` KV-cache trace, finalized at turn 13) |
| Runs returning a usable answer | 12 / 12 |

At the median burn rate the 49,685-token allowance covers ~25 turns; at the heaviest sustained rate
observed (3.4k/turn) it covers ~14. So **`--max-turns 12`** stays inside the context budget in every case measured,
and `16` is a reasonable ceiling for hard traces. `4` and `8` cut exploration short on over half the
runs. Note the cap is not a promise of sufficiency: 6 runs would have kept exploring past 16 had they
been allowed to, so a higher cap costs latency, not correctness.

Two caveats. Repo size barely predicts cost — a 10.8k-file repo burned 804 tokens/turn while a
47k-file one converged in 6 turns; question shape dominates. And these numbers scale with *your*
budget: halving `max_context`, or raising `max_turn_output_tokens`, roughly halves the usable turns.

## Programmatic use

```python
import asyncio
from fastcontext.agent.agent_factory import make_fastcontext_agent

async def explore(question: str) -> str:
    agent = make_fastcontext_agent(
        trajectory_file=".fastcontext/trajectory.jsonl",
        work_dir=".",
        max_completion_tokens=4096,   # how long one response may be; None reads the config
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
