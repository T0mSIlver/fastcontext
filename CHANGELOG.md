# Changelog

Notable changes to FastContext. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/).

## [Unreleased]

### Changed
- The repository is now standalone — detached from the fork network, with docs, license, and
  packaging metadata describing the project in its own right.
- `RequestyAPIError` renamed to `LLMAPIError` (#35).

### Removed
- The paper-era SWE-bench benchmark rig — `benchmark/`, `prompts/`, `figures/`, and the
  `mini-swe-agent` submodule — none of it referenced by the harness (#35).
- Unused `azure-core`/`azure-identity` dependencies and the `benchmark` dependency group (#35).
- A broken `*claude*` model-routing path that imported a module that does not exist in this
  repository (#35).

### Added
- CI: ruff + pytest for the harness and the eval project on every PR and push to `main`.
- This changelog.

## [0.2.0] — 2026-07-17

The first versioned release: everything built on top of the research explorer.

### Added
- Live TUI run inspector (`--tui`) — streaming collapsible rows for reasoning, tool calls, and
  results, with a docked token-usage bar (#16).
- Context-budget management: the agent finalizes before the conversation outgrows the model's
  window (`--max-context`, #17), with provider auto-detection of the real window — including
  llama.cpp-swapper launch args, where usable context is `ctx-size ÷ parallel` (#18, #33).
- Per-turn and per-result tool-output caps, in tokens (`--max-turn-output-tokens`,
  `--max-result-output-tokens`) (#29, #31).
- Citation validation with a self-correction loop for line ranges the model never opened (#13),
  plus a citation cap and "candidate evidence" framing (#25).
- Layered TOML configuration (user / project / `--config`) so the endpoint is set once (#19), and
  `fastcontext init` to scaffold a config file (#20).
- Eval harness (`eval/`): run branches from isolated worktrees against (repo, query) tasks and
  compare turns, tool calls, citation quality, and tokens (#15); its measurements back the
  `--max-turns 12` default (#26, #30).
- An agent skill (`skills/fastcontext/`) teaching coding agents to delegate exploration (#22, #32).

### Fixed
- Honest exit codes: a failed run exits `1` with the error on stderr instead of returning the
  error text as an answer (#24, #28).
- `Grep`: `count` output mode wired up, `head_limit` values above 100 honored, `content` made the
  explicit default, no forced `-C 3` context, a subprocess timeout, and accurate docs (#1–#8).
- `Read`: long lines truncated at 500 chars as documented; negative offsets count from the end of
  the file (#9, #10).
- `Glob`: results actually sorted by modification time (#5).
- Mangled or relative tool-call paths resolved against the workspace, with absolute paths primed
  in the prompt (#12, #27).
- The rotted agent/LLM/toolset test suite repaired (#21).

### Changed
- `--max-tokens` split into `--max-completion-tokens` (one response) and `--max-context` (the
  window) — conflating them could disable exploration entirely (#33).
- Tool-output caps renamed after what they bound and converted from characters to tokens, ending a
  ~4× over-reservation on ASCII source (#29, #31).
- Standalone-repo papercuts: stale dependencies and Microsoft process files dropped (#23); the
  wheel name can no longer go stale silently (#34).

## 0.1.0

The upstream research release — the FastContext explorer harness as published with the paper
([arXiv:2606.14066](https://arxiv.org/abs/2606.14066)).
