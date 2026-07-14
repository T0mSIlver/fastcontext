"""Render the eval summary as a single self-contained HTML dashboard."""

from __future__ import annotations

import html
import json

# (key, label, "lower is better" flag) -- the flag drives highlighting.
_METRICS = [
    ("turns", "Turns", True),
    ("tool_calls_total", "Tool calls", True),
    ("failed_tool_calls", "Failed calls", True),
    ("duplicate_tool_calls", "Duplicate calls", True),
    ("corrections", "Self-corrections", True),
    ("citations_total", "Citations", False),
    ("unverified_citations", "Unverified cites", True),
    ("citations_missing_file", "Missing-file cites", True),
    ("total_tokens", "Tokens", True),
]

_CSS = """
:root { color-scheme: light dark; }
body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
       margin: 24px; line-height: 1.4; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 28px 0 8px; }
.sub { color: #888; font-size: 13px; margin-bottom: 16px; }
table { border-collapse: collapse; font-size: 13px; margin-bottom: 8px; width: 100%; }
th, td { border: 1px solid #8883; padding: 5px 9px; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { background: #8881; position: sticky; top: 0; }
tbody tr:hover { background: #8881; }
.good { color: #1a7f37; font-weight: 600; }
.bad  { color: #c0392b; font-weight: 600; }
.tag  { font-size: 11px; padding: 1px 6px; border-radius: 8px; background: #8882; }
.err  { color: #c0392b; }
.q    { color: #888; font-size: 12px; max-width: 460px; white-space: normal; text-align: left; }
.mono { font-variant-numeric: tabular-nums; }
.best { background: #1a7f3722; }
"""


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, int):
        return f"{v:,}"
    return html.escape(str(v)) if v is not None else ""


def _aggregate_table(agg: dict) -> str:
    branches = list(agg.keys())
    head = "".join(f"<th>{html.escape(b)}</th>" for b in branches)
    rows = []
    # meta rows
    for key, label in [("runs", "Runs"), ("scored_runs", "Scored runs (in means)"),
                       ("reached_final_answer", "Reached final answer")]:
        cells = "".join(f"<td>{_fmt(agg[b].get(key))}</td>" for b in branches)
        rows.append(f"<tr><td>{label}</td>{cells}</tr>")
    # Errors are called out as "N/M" so a failed run can never read as a real zero.
    cells = ""
    for b in branches:
        errs, runs = agg[b].get("errored_runs", 0), agg[b].get("runs", 0)
        cls = " class='err'" if errs else ""
        cells += f"<td{cls}>{errs}/{runs}</td>"
    rows.append(f"<tr><td>Errored runs (excluded)</td>{cells}</tr>")
    # metric means, highlight the best branch per row
    for key, label, lower_better in _METRICS:
        vals = {b: agg[b].get(key, 0) for b in branches}
        best = min(vals.values()) if lower_better else max(vals.values())
        cells = ""
        for b in branches:
            cls = " class='best'" if len(branches) > 1 and vals[b] == best else ""
            cells += f"<td{cls}>{_fmt(vals[b])}</td>"
        rows.append(f"<tr><td>{html.escape(label)} <span class='tag'>mean</span></td>{cells}</tr>")
    return f"<table><thead><tr><th>Metric</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _runs_table(rows: list[dict]) -> str:
    cols = [("branch", "Branch"), ("task", "Task")]
    metric_cols = [(k, lbl) for k, lbl, _ in _METRICS]
    head = "".join(f"<th>{lbl}</th>" for _, lbl in cols + metric_cols) + "<th>Tools</th><th>Final?</th><th>Wall(s)</th>"
    body = []
    for r in sorted(rows, key=lambda x: (x.get("task", ""), x.get("branch", ""))):
        tds = []
        for k, _ in cols:
            tds.append(f"<td>{html.escape(str(r.get(k, '')))}</td>")
        err = r.get("run_error") or r.get("error")
        if err or "turns" not in r:
            msg = f"ERRORED (excluded from means): {err}" if err else "not scored"
            body.append(f"<tr>{''.join(tds)}<td colspan='13' class='err'>{html.escape(str(msg))}</td></tr>")
            continue
        for k, _ in metric_cols:
            v = r.get(k, 0)
            cls = ""
            if k in ("failed_tool_calls", "duplicate_tool_calls", "unverified_citations",
                     "citations_missing_file", "corrections") and v:
                cls = " class='bad'"
            tds.append(f"<td{cls}>{_fmt(v)}</td>")
        usage = r.get("tool_usage", {})
        usage_str = " ".join(f"{k}:{v}" for k, v in sorted(usage.items())) if usage else ""
        final = "<span class='good'>yes</span>" if r.get("reached_final_answer") else "<span class='bad'>no</span>"
        wall = r.get("wall_seconds")
        body.append(
            f"<tr>{''.join(tds)}<td class='mono'>{usage_str}</td><td>{final}</td><td>{_fmt(wall)}</td></tr>"
        )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _per_task_section(rows: list[dict]) -> str:
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r.get("task", "?"), []).append(r)
    out = []
    for task, trows in sorted(by_task.items()):
        if len(trows) < 2:
            continue
        q = next((t.get("query") for t in trows if t.get("query")), "")
        out.append(f"<h2>Task: {html.escape(task)}</h2>")
        if q:
            out.append(f"<div class='q'>{html.escape(q)}</div>")
        out.append(_runs_table(trows))
    return "".join(out)


def render_dashboard(summary: dict) -> str:
    agg = summary.get("aggregates", {})
    rows = summary.get("rows", [])
    n_runs = len(rows)
    n_errors = sum(1 for r in rows if r.get("run_error") or r.get("error"))
    n_branches = len(summary.get("branches", []))
    err_note = f"<span class='err'>{n_errors} errored run(s) excluded from means.</span> " if n_errors else ""
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>FastContext Eval</title>",
        f"<style>{_CSS}</style></head><body>",
        "<h1>FastContext Eval Dashboard</h1>",
        f"<div class='sub'>{n_runs} runs across {n_branches} branch(es). {err_note}"
        "Lower is better for everything except Citations. "
        "<span class='best tag'>green</span> = best branch per metric.</div>",
        "<h2>Branch comparison</h2>",
        _aggregate_table(agg) if agg else "<p>No data.</p>",
        _per_task_section(rows),
        "<h2>All runs</h2>",
        _runs_table(rows),
        f"<script>window.__summary__ = {json.dumps(summary)};</script>",
        "</body></html>",
    ]
    return "".join(parts)


def write_dashboard(summary: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_dashboard(summary))
