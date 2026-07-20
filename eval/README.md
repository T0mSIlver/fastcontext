# FastContext Eval

A small [uv](https://docs.astral.sh/uv/) project that launches FastContext runs,
saves their trajectories, scores them, and renders a simple HTML dashboard.

It is built to **compare branches of FastContext** (e.g. `main` vs a feature
branch) on the same set of repo + query tasks, so you can see how a change
moves the metrics.

## What it measures

For every run (one trajectory) it computes:

| Metric | Meaning |
| --- | --- |
| **Turns** | Number of LLM steps in the conversation. |
| **Tool calls** | Total `Read` / `Glob` / `Grep` calls, plus a per-tool breakdown. |
| **Failed calls** | Tool results that returned a `Permission error` / `Error` reminder (not "No files found", which is a valid empty result). |
| **Duplicate calls** | Tool calls with an identical `(name, arguments)` issued more than once. |
| **Self-corrections** | Times the citation self-correction loop fired (the agent retries up to 2× when it cites unopened lines). |
| **Citations** | Citations in the final answer pointing at files that exist. |
| **Unverified cites** | Cited line ranges the model never actually opened via `Read`/`Grep` (likely hallucinations). |
| **Missing-file cites** | Citations whose file does not exist. |
| **Tokens** | Prompt + completion tokens summed across all turns. |
| **Final?** | Whether the run produced a `<final_answer>` block. |

The scorer re-implements FastContext's observed-lines logic, so it works on any
trajectory JSONL regardless of which branch produced it.

## Configure

1. Point the FastContext endpoint at your model (see the repo README — remote
   OpenAI-compatible endpoint or local Ollama):

   ```bash
   export FC_BASE_URL=... FC_MODEL=... [FC_API_KEY=...]
   ```

2. Edit [`config.yaml`](config.yaml): set the `branches` to compare and the
   `tasks` (each is a repo path + a natural-language query).

## Run

```bash
cd eval
uv run fc-eval run        # launch branches × tasks, save trajectories under results/
uv run fc-eval analyze    # score trajectories -> results/summary.json
uv run fc-eval dashboard  # build results/dashboard.html  (--open / --serve)
uv run fc-eval all        # run + analyze + dashboard
```

`-c/--config` and `-r/--results` are accepted before or after the subcommand
(`fc-eval -r out run` and `fc-eval run -r out` both work); relative paths are
resolved against the current directory.

## Failed runs

A run whose LLM endpoint fails (the agent exits nonzero / prints
`LLM API call failed`), times out, or crashes is retried (`retries` / `retry_backoff` in `config.yaml`, default
2 attempts extra with 5s backoff; `--retries` overrides). A run that still fails
is marked **errored**: it is reported separately (`errors: N/M` per branch in the
summary and dashboard) and **excluded from the branch means**, so a broken run is
never averaged in as a legitimate zero. One failing run never aborts the sweep.

Each branch is checked out as a git worktree of `fastcontext_repo` and run from
inside the target repo via `uv run --project <worktree> fastcontext`, so the
query is scored against that branch's code with the repo as `work_dir`.

Results layout:

```
results/
  runs/<branch>/<task>/{trajectory.jsonl, final_answer.txt, meta.json, stderr.log}
  summary.json
  dashboard.html
```

## Try it without an endpoint

`config.yaml` shows how to list pre-recorded trajectories as `reference_runs`.
They are scored and shown in the dashboard without launching anything:

```bash
uv run fc-eval dashboard --open
```

## Tests

```bash
uv run pytest
```
