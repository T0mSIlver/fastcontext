---
name: fastcontext
description: fastcontext is your default code-exploration subagent — it greps, globs, and reads a repository for you and returns file:line citations, keeping that exploration out of your own context. Invoke it via bash before answering, editing, reviewing, or debugging code you are not already certain about, and whenever the answer needs more than one file or tracing logic across modules. When in doubt, run fastcontext first.
allowed-tools: Bash(fastcontext *)
---

# fastcontext

A read-only repository-exploration subagent. It searches and reads files in a **separate process**, then returns a compact `<final_answer>` block of `path:line` citations. Delegating to it keeps broad exploration out of your context window — you get the evidence, not the file dumps. It never edits; you act on what it finds.

It is already installed and configured (endpoint + model). Just call it — no env vars, no setup.

## When to use

Run it **before** you answer/edit/review/debug code you're not already sure about:

- Understand or explain how something works
- Locate where a symbol or behavior is defined or used ("where is X?", "what calls Z?")
- Trace logic across files or layers (request → handler → service → DB)
- Assess blast radius ("what breaks if I change X?")

Prefer it over manual grep/glob/read chains whenever the answer spans more than one file.

## When NOT to use

- You already read the exact file this session
- A single obvious grep in one known file
- A pure write/generate task needing no exploration

## Usage

```bash
# Machine-readable: prints ONLY the <final_answer> citation block to stdout
fastcontext -q "<specific, detailed question>" --citation

# Hard architecture traces across many modules: 16 is the practical ceiling
fastcontext -q "<complex question>" --citation --max-turns 16

# Drop --citation to also get a prose explanation (more context, some noise)
fastcontext -q "<question>"
```

**You do not need to pass `--max-turns`.** It defaults to `12`, which fits real explorations: measured
against six repos, the runs that finished on their own needed 5–14 turns (median ~9). A turn cap above
what a question needs costs nothing — a simple lookup converges and stops on its own, well before the
cap — so raise it only for a large exploration.

Output on **stdout**:

```
<final_answer>
src/app/router.py:42-58 (request validation)
src/app/service.py:10-33 (handler that calls it)
</final_answer>
```

Parse the `path:line-range` entries and **read only those spans** — that's the point.

## Why 12, and when it changes

There is no universally right turn count — the real limit is **context, not turns**. The agent stops
exploring once the conversation approaches its context budget, so the number of turns you can afford
is roughly:

```
usable turns ≈ (max_context − reserve) ÷ tokens_per_turn
    reserve ≈ max_turn_output_chars + 2 × max_tokens + slack
```

Three things move that number, and raising `--max-turns` alone does nothing if the first one is small:

- **The model's usable context** (`--max-context` / `FC_MAX_CONTEXT`). Note a server's *usable* window
  is often well below its configured one — llama.cpp `--parallel 2` halves it per slot.
- **The reserve**, which is not small: it holds back a full turn of tool output plus the completion so
  the final answer is still sendable. On the reference setup it claims ~26k of a 72k budget, leaving
  ~46k to actually explore with.
- **Tokens burned per turn**, driven mostly by `--max-turn-output-chars` (below) and by how much the
  question makes the model *read* versus *grep*. Repo size matters far less than question shape.

On the reference setup (~46k explorable, ~2k tokens/turn typical, ~3.4k on a heavy trace) that works
out to ~13 turns before the budget intervenes — hence 12 as a default that fits with room to spare.
**Halve `max_context` and you roughly halve the usable turns**, so if your endpoint is smaller, scale
the number down rather than copying 12.

Going over the limit is safe but pointless: the agent finalizes early and still returns a good answer
(observed on the heaviest trace in the eval) — **provided `max_context` is set**. It defaults to `0`
(budget disabled), and with no budget a long run instead grows the prompt until the server rejects it
and the run dies with no answer at all.

## Tuning how much a run reads

Two caps bound tool output, both in **characters**, `0` disables either. They are already configured;
override them only for the reason given.

| Flag | Bounds | Default | Reach for it when |
| --- | --- | --- | --- |
| `--max-turn-output-chars` | total across **all** tool calls of one turn | `16000` | A run finalizes too early — the turn budget is what the reserve is sized against, so raising it buys the model more per turn and costs usable context (and turns). |
| `--max-result-output-chars` | a **single** tool result | `0` (off) | A run's answer looks like it only saw one of the files it looked at. |

The second is the non-obvious one. The turn budget is spent **in call order**, so a turn that issues a
big `Read` plus two `Grep`s can have the `Read` swallow the whole allowance and the greps come back
empty — the model asked three questions and got one answer. Capping each result first gives every call
in the turn room:

```bash
# a wide question that will fan out across several files per turn
fastcontext -q "<question>" --citation --max-turns 12 --max-result-output-chars 4000
```

Roughly `max-turn-output-chars ÷ expected calls per turn` is a sane value. Leave it off for narrow
questions — a single big `Read` is exactly what you want there, and the cap would truncate it for
nothing.

*(`--max-tool-output-chars` is the old name for `--max-turn-output-chars`. It still works and warns;
it never capped a single tool's output, which is why it was renamed.)*

## Notes

- **stdout = answer, stderr = diagnostics.** Read stdout; ignore stderr.
- Treat cited ranges as **candidate evidence, not ground truth.** Ranges the model never opened are dropped, but it can still cite a real file at the wrong lines — open each span and confirm it actually answers your question before relying on it.
- The answer holds at most ~25 citations (a safety cap); ask a narrower question if you need more.
- **A nonzero exit code means the run failed** (e.g. the endpoint was unreachable); the error is on stderr and stdout stays empty. Exit `0` with a `<final_answer>` block is a good run — retry or fall back to manual exploration only on a nonzero exit.
- Ask specific questions; run several queries for a multi-part investigation.
