# FastContext

FastContext is a lightweight repository-exploration agent for software engineering tasks. It is designed to be called by a main coding agent when the main agent needs fast, read-only context about a codebase before editing files.

Given a natural-language query, FastContext explores the current repository with three tools:

- `Read`: read files with optional line offsets and limits.
- `Glob`: find files by glob patterns.
- `Grep`: search file contents with regular expressions.

FastContext then returns a concise answer with file paths and line ranges inside a `<final_answer>` block. The intended use case is repository triage: locating relevant files, functions, configuration, tests, or call paths before a larger coding agent starts making changes.

## Highlights

- Read-only by design: the agent searches and reads code but does not edit files.
- OpenAI-compatible model interface through the `openai` Python SDK.
- CLI-first workflow for easy integration into benchmark containers and external agents.
- Trajectory logging in JSONL for debugging, analysis, and evaluation.
- Built-in tool schema generation for function-calling models.
- Benchmark utilities for SWE-bench-style exploration and citation scoring.
- Training and serving scripts for reproducing research workflows.

## Installation

FastContext requires Python 3.12 or newer. The repository uses `uv` for environment and package management.

Install from the repository root:

```bash
uv tool install .
```

For development:

```bash
uv sync --all-groups
```

Build a wheel locally:

```bash
uv build
```

The built wheel will be written under `dist/`, for example `dist/fastcontext-0.1.0-py3-none-any.whl`.

## Model Configuration

FastContext expects an OpenAI-compatible chat completions endpoint. Configure the model with environment variables:

```bash
export FASTCONTEXT_BASE_URL="https://your-endpoint.example/v1"
export FASTCONTEXT_MODEL="your-model-name"
export FASTCONTEXT_API_KEY="your-api-key"
```

For compatibility with existing scripts, the CLI also falls back to these variables when the FastContext-specific ones are not set:

```bash
export BASE_URL="https://your-endpoint.example/v1"
export MODEL="your-model-name"
export API_KEY="your-api-key"
```

## CLI Usage

Run FastContext from the repository you want to explore:

```bash
fastcontext \
	--query "Find the files that implement authentication and explain where to make a change" \
	--max-turns 6 \
	--traj .fastcontext/trajectory.jsonl
```

Useful options:

- `--query`, `-q`: the exploration request.
- `--traj`, `-t`: path to a JSONL trajectory file.
- `--max-turns`: maximum agent turns before forcing a final answer.
- `--verbose`: print intermediate messages and runtime information.
- `--citation`: return only the `<final_answer>` block when one is present.

Example citation-only invocation:

```bash
fastcontext \
	--query "Locate the request validation logic" \
	--citation
```

## Expected Output

FastContext answers with a short explanation followed by a machine-readable citation block:

```text
The request validation logic is implemented in the API middleware.

<final_answer>
/path/to/repo/src/api/middleware.py:42-91
/path/to/repo/src/api/validators.py:10-37
</final_answer>
```

Downstream agents can parse the `<final_answer>` block and decide which files to read or edit next.

## Repository Layout

```text
src/fastcontext/
	cli.py                         Command-line entry point
	agent/
		agent.py                     Agent loop
		agent_factory.py             Default FastContext agent construction
		context.py                   Conversation and trajectory storage
		llm.py                       OpenAI-compatible LLM wrapper
		system.md                    Explorer system prompt
		tool/
			read.py                    Read tool
			glob.py                    Glob tool
			grep.py                    Grep tool
			tool.py                    Tool base classes and ToolSet

benchmark/
	environment/                   Docker environment helpers
	evaluation/                    Citation parsing and scoring utilities
	swebench/                      SWE-bench-style runner scripts

prompts/                        Prompt YAML files for benchmark agents

training/
	fastcontext-sft/               Supervised fine-tuning scripts and data utilities
	fastcontext-rl/                Reinforcement-learning scripts and reward utilities

serving/                         Example serving manifests and API checks
tests/                           Unit and integration-style tests
```

## Programmatic Use

The default factory constructs an agent with the `Read`, `Glob`, and `Grep` tools:

```python
import asyncio

from fastcontext.agent.agent_factory import make_fastcontext_agent


async def main() -> None:
		agent = make_fastcontext_agent(
				trajectory_file=".fastcontext/trajectory.jsonl",
				work_dir="/path/to/repo",
		)
		answer = await agent.run(
				prompt="Find where database migrations are defined",
				max_turns=6,
				citation=True,
		)
		print(answer)


asyncio.run(main())
```

## Benchmarking

The `benchmark/swebench/` directory contains scripts for running FastContext in SWE-bench-style Docker environments. A typical workflow is:

1. Build the package wheel with `uv build`.
2. Provide benchmark instances that include subagent queries.
3. Run `benchmark/swebench/bench_fastcontext.py` with the selected dataset and output path.
4. Score returned citations with `benchmark/evaluation/run_score.py`.

Example scoring command:

```bash
uv run --group benchmark python benchmark/evaluation/run_score.py swebench-verified predictions.jsonl
```

For end-to-end SWE-bench runs with mini-swe-agent as the main agent and FastContext as the repository exploration helper, use the runner in `benchmark/evaluation/`:

```bash
git submodule update --init --recursive
uv build
uv run --group benchmark python benchmark/evaluation/bench_mini_swe_agent.py \
	--bench swebench-multilingual \
	--agent-config prompts/gpt-multi-fc.yaml \
	--config benchmark/evaluation/configs/example.env \
	--output preds.json \
	--logs-dir logs
```

The `prompts/` directory contains mini-swe-agent YAML configs for different model families and benchmark settings. The runner uses the `third_party/mini-swe-agent` submodule, installs the local FastContext wheel inside each SWE-bench Docker container, and writes SWE-bench-style `model_patch` predictions.

## Training and Serving

The `training/` directory contains scripts used for SFT and RL experiments. These scripts assume a research training environment with external model checkpoints, datasets, and training frameworks. Treat paths and cluster settings in these scripts as examples to adapt to your own infrastructure.

The `serving/` directory contains example manifests and API checks for serving FastContext-compatible models behind an OpenAI-compatible endpoint.

## Development Checks

Run linting:

```bash
uv run ruff check .
```

Run tests:

```bash
uv run pytest -q
```

Build the package:

```bash
uv build
```

## Notes

- FastContext is intended for repository exploration, not code modification.
- Tool outputs are capped to keep interactions responsive.
- The default CLI records trajectories under `.fastcontext/` unless `--traj` is provided.
- For best results, write specific exploration queries that name the behavior, subsystem, error, or files you are trying to locate.
