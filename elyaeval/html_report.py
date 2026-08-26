"""
Shared HTML rendering + per-metric averaging, used by both report.py
(flat e2e CSVs) and tracing_report.py (span-aware component CSVs).

Deliberately reads the CSV back off disk rather than taking rows directly
from the pytest process's memory: evaluate_golden()/TracedRunner already
append one row (or a few rows) to CSV per golden, one golden at a time, as
the parametrized suite runs. Re-reading the file at the end is what lets a
single function work for "generate the HTML for a run that just finished"
(the plugin's pytest_sessionfinish hook) AND "regenerate the HTML for some
run's CSV that already got uploaded to blob storage last week"
(`elyaeval report`, see cli.py) without keeping any separate in-process
state that only the first case could ever have.

Score/threshold/success come back from csv.DictReader as strings — every
function below that touches them numerically goes through _coerce_row()
first so a blank score (e.g. an errored metric) doesn't crash averaging,
it's just excluded from that metric's mean.
"""

from __future__ import annotations

import csv
import html as _html
from pathlib import Path
from typing import Optional


# Populated by report.append_csv_rows / tracing_report.append_traced_csv_rows
# as they write — every CSV path either module has EVER appended a row to
# during this process, paired with the group_keys its rows should be
# averaged by. plugin.py reads this at pytest_sessionfinish to know which
# CSVs to render HTML for and print averages for, without needing the
# generated test file itself to call anything new. A plain module-level
# dict (not a fixture) because both writer functions are called from
# ordinary functions, not fixtures, and can run under multiple test files
# in one pytest session — last-write-wins per path is fine since group_keys
# for a given csv_path never changes between calls in practice.
_SESSION_REPORTS: dict[Path, tuple[str, ...]] = {}


def register_report(csv_path: str | Path, group_keys: tuple[str, ...] = ("metric_name",)) -> None:
    _SESSION_REPORTS[Path(csv_path)] = group_keys


def session_reports() -> dict[Path, tuple[str, ...]]:
    """Snapshot of every CSV path registered so far in this process, e.g.
    for plugin.py's pytest_sessionfinish hook to iterate over."""
    return dict(_SESSION_REPORTS)


def read_csv_rows(csv_path: str | Path) -> list[dict]:
    """Read a report CSV back into a list of dicts, coercing score/threshold/
    evaluation_cost to float and success to bool where possible. Rows are
    returned in file order (i.e. the order goldens actually ran in), which
    is also the order the detail table renders in — no re-sorting happens
    later on the (mistaken) assumption that CSV order is arbitrary."""
    path = Path(csv_path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["score"] = _to_float(row.get("score"))
        row["threshold"] = _to_float(row.get("threshold"))
        row["success"] = _to_bool(row.get("success"))
        # Older CSVs written before cost tracking was added simply have no
        # evaluation_cost/evaluation_model column at all — csv.DictReader
        # then leaves them missing from the dict entirely rather than "",
        # so .get(...) (not [...]) everywhere downstream, and this coerces
        # a genuinely absent column the same way as a blank one: None.
        row["evaluation_cost"] = _to_float(row.get("evaluation_cost"))
    return rows


def _to_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    return str(value).strip().lower() == "true"


def metric_averages(rows: list[dict], group_keys: tuple[str, ...] = ("metric_name",)) -> list[dict]:
    """
    Group `rows` by `group_keys` (default: just metric_name — pass
    ("span_name", "metric_name") for the component CSV, where the same
    metric name can appear under different spans, e.g. Faithfulness scored
    on both "generate_answer" and some other span) and compute, per group:

      - n:            how many rows landed in this group (goldens tested)
      - n_scored:     how many of those had a numeric score (excludes
                      rows where the metric errored and score is blank)
      - avg_score:    mean of the numeric scores, or None if n_scored == 0
      - min_score / max_score: same population
      - pass_count / pass_rate: from the `success` column (True/False),
                      independent of avg_score — a metric can average
                      above its threshold while still failing goldens if
                      the failures are offset by very high scores elsewhere,
                      so both numbers are kept rather than inferring one
                      from the other
      - threshold:    the threshold value, IF every row in the group shares
                      the same one (the normal case — a metric's threshold
                      is fixed per suite); None if goldens in this group
                      used different thresholds, so a caller can't silently
                      display one that isn't actually representative
      - total_cost:   sum of evaluation_cost (USD) across every row in the
                      group that had a cost value. None (not 0) if NO row
                      in the group had a cost — distinguishes "this judge
                      model has no known pricing, cost is unknown" from
                      "this judge model is free" (0.0), the same way
                      DeepEval's own terminal summary shows "token cost:
                      None" rather than $0 for an unpriced/local model.
      - cost_n:       how many rows actually contributed a cost value —
                      lets a caller tell "$0.0031 from 4 metrics" apart
                      from "$0.0031 from 1 metric, other 3 unpriced"
      - evaluation_model: the judge model name, IF every row in the group
                      shares the same one; "multiple" if they differ (e.g.
                      you changed the judge mid-run); None if absent

    Rows with an `error` value are still counted in n and pass_count/n
    (success is already False for them from evaluate_golden), just
    excluded from avg/min/max if their score is blank — an errored metric
    with no score shouldn't silently pull an average toward 0.

    Returns groups in first-seen order (matches CSV/golden run order), not
    sorted alphabetically — keeps e.g. metrics in the same order the suite
    declared them (RAG_METRICS list order) rather than shuffling by name.
    """
    order: list[tuple] = []
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(row)

    summaries = []
    for key in order:
        group_rows = buckets[key]
        scores = [r["score"] for r in group_rows if r.get("score") is not None]
        successes = [r["success"] for r in group_rows if r.get("success") is not None]
        thresholds = {r["threshold"] for r in group_rows if r.get("threshold") is not None}
        costs = [r["evaluation_cost"] for r in group_rows if r.get("evaluation_cost") is not None]
        models = {r["evaluation_model"] for r in group_rows if r.get("evaluation_model")}

        summary = dict(zip(group_keys, key))
        summary["n"] = len(group_rows)
        summary["n_scored"] = len(scores)
        summary["avg_score"] = (sum(scores) / len(scores)) if scores else None
        summary["min_score"] = min(scores) if scores else None
        summary["max_score"] = max(scores) if scores else None
        summary["pass_count"] = sum(1 for s in successes if s)
        summary["pass_total"] = len(successes)
        summary["pass_rate"] = (summary["pass_count"] / summary["pass_total"]) if successes else None
        summary["threshold"] = thresholds.pop() if len(thresholds) == 1 else None
        summary["total_cost"] = sum(costs) if costs else None
        summary["cost_n"] = len(costs)
        summary["evaluation_model"] = (
            models.pop() if len(models) == 1 else ("multiple" if len(models) > 1 else None)
        )
        summaries.append(summary)
    return summaries


def total_cost(rows: list[dict]) -> Optional[float]:
    """Sum of evaluation_cost (USD) across ALL rows, independent of
    grouping — the single "what did this whole run cost the judge model"
    number for the HTML header / terminal summary. None if no row in the
    run has a cost value at all (e.g. a local/unpriced judge model, same
    case DeepEval's own terminal summary shows as "token cost: None"
    rather than $0), not conflating "unknown" with "free"."""
    costs = [r.get("evaluation_cost") for r in rows if r.get("evaluation_cost") is not None]
    return sum(costs) if costs else None


_CSS = """
:root {
  color-scheme: light;
  --pass: #1a7f37;
  --fail: #cf222e;
  --muted: #6e7781;
  --border: #d0d7de;
  --bg-alt: #f6f8fa;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
  margin: 0;
  padding: 2rem;
  color: #1f2328;
  background: #ffffff;
}
h1 { font-size: 1.4rem; margin: 0 0 0.15rem; }
h2 { font-size: 1.1rem; margin: 2rem 0 0.75rem; }
.meta { color: var(--muted); font-size: 0.85rem; margin-bottom: 1.5rem; }
.meta span { margin-right: 1.25rem; }
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 0.88rem;
}
caption { caption-side: top; text-align: left; }
th, td {
  border: 1px solid var(--border);
  padding: 0.45rem 0.65rem;
  text-align: left;
  vertical-align: top;
}
th {
  background: var(--bg-alt);
  font-weight: 600;
  position: sticky;
  top: 0;
}
tbody tr:nth-child(even) { background: #fbfcfd; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.pass { color: var(--pass); font-weight: 600; }
.fail { color: var(--fail); font-weight: 600; }
.reason { color: var(--muted); font-size: 0.82rem; max-width: 32rem; }
.bar-cell { display: flex; align-items: center; gap: 0.5rem; }
.bar-track { flex: 1; background: var(--bg-alt); border-radius: 3px; height: 8px; min-width: 60px; }
.bar-fill { background: var(--pass); border-radius: 3px; height: 8px; }
.bar-fill.low { background: var(--fail); }
.summary-table td, .summary-table th { white-space: nowrap; }
.detail-table td.reason, .detail-table td.input { white-space: normal; }
.error-cell { color: var(--fail); font-size: 0.82rem; }
"""


def _esc(value) -> str:
    return _html.escape("" if value is None else str(value))


def _fmt_score(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.2f}"


def _fmt_pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _fmt_cost(value: Optional[float]) -> str:
    """USD cost values from LLM judges are often fractions of a cent, so a
    flat 2-decimal format would show "$0.00" for almost everything that
    isn't a large batch. Uses more precision for small-but-nonzero values,
    2 decimals once the amount is large enough for that to be meaningful."""
    if value is None:
        return "—"
    if value == 0:
        return "$0.00"
    if abs(value) < 0.01:
        return f"${value:.6f}"
    return f"${value:.4f}"


def _score_bar(value: Optional[float], threshold: Optional[float]) -> str:
    if value is None:
        return "—"
    pct = max(0.0, min(1.0, value))
    low = threshold is not None and value < threshold
    return (
        f'<div class="bar-cell"><span class="num">{_fmt_score(value)}</span>'
        f'<div class="bar-track"><div class="bar-fill{" low" if low else ""}" '
        f'style="width:{pct * 100:.0f}%"></div></div></div>'
    )


def _summary_table(rows: list[dict], group_keys: tuple[str, ...]) -> str:
    summaries = metric_averages(rows, group_keys=group_keys)
    if not summaries:
        return "<p><em>No scored metrics in this report.</em></p>"

    headers = [k.replace("_", " ").title() for k in group_keys] + [
        "Goldens",
        "Avg score",
        "Min",
        "Max",
        "Pass rate",
        "Threshold",
        "Judge model",
        "Total cost",
    ]
    head_html = "".join(f"<th>{_esc(h)}</th>" for h in headers)

    body_rows = []
    for s in summaries:
        key_cells = "".join(f"<td>{_esc(s[k])}</td>" for k in group_keys)
        cost_note = f" ({s['cost_n']}/{s['n']} priced)" if s["cost_n"] and s["cost_n"] < s["n"] else ""
        body_rows.append(
            "<tr>"
            f"{key_cells}"
            f'<td class="num">{s["n"]}</td>'
            f'<td>{_score_bar(s["avg_score"], s["threshold"])}</td>'
            f'<td class="num">{_fmt_score(s["min_score"])}</td>'
            f'<td class="num">{_fmt_score(s["max_score"])}</td>'
            f'<td class="num">{_fmt_pct(s["pass_rate"])} '
            f'<span class="reason">({s["pass_count"]}/{s["pass_total"]})</span></td>'
            f'<td class="num">{_fmt_score(s["threshold"])}</td>'
            f'<td>{_esc(s["evaluation_model"] or "—")}</td>'
            f'<td class="num">{_fmt_cost(s["total_cost"])}'
            f'<span class="reason">{cost_note}</span></td>'
            "</tr>"
        )

    return (
        '<table class="summary-table">'
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _detail_table(rows: list[dict], extra_columns: tuple[str, ...] = ()) -> str:
    if not rows:
        return "<p><em>No rows in this report.</em></p>"

    base_cols = ["golden_id", "priority"] + list(extra_columns) + ["input", "metric_name"]
    headers = [c.replace("_", " ").title() for c in base_cols] + [
        "Score",
        "Threshold",
        "Result",
        "Judge model",
        "Cost",
        "Reason / error",
    ]
    head_html = "".join(f"<th>{_esc(h)}</th>" for h in headers)

    body_rows = []
    for r in rows:
        cells = "".join(f"<td>{_esc(r.get(c, ''))}</td>" for c in base_cols[:-2])
        input_val = _esc(r.get("input", ""))
        metric_val = _esc(r.get("metric_name", ""))
        success = r.get("success")
        result_html = (
            '<span class="pass">pass</span>' if success is True
            else '<span class="fail">fail</span>' if success is False
            else "—"
        )
        reason = r.get("error") or r.get("reason") or ""
        reason_class = "error-cell" if r.get("error") else "reason"
        body_rows.append(
            "<tr>"
            f"{cells}"
            f'<td class="input">{input_val}</td>'
            f"<td>{metric_val}</td>"
            f'<td class="num">{_fmt_score(r.get("score"))}</td>'
            f'<td class="num">{_fmt_score(r.get("threshold"))}</td>'
            f"<td>{result_html}</td>"
            f'<td>{_esc(r.get("evaluation_model") or "—")}</td>'
            f'<td class="num">{_fmt_cost(r.get("evaluation_cost"))}</td>'
            f'<td class="{reason_class}">{_esc(reason)}</td>'
            "</tr>"
        )

    return (
        '<table class="detail-table">'
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def render_html_report(
    csv_path: str | Path,
    html_path: Optional[str | Path] = None,
    title: Optional[str] = None,
    group_keys: tuple[str, ...] = ("metric_name",),
) -> Path:
    """
    Build a single self-contained HTML file (inline CSS, no external
    assets/CDN calls — safe to open straight from a blob-storage download
    with no server behind it) next to `csv_path`, showing:

      1. a per-metric (or per-`group_keys`) summary table: average score,
         min/max, pass rate, and the threshold, computed once across every
         golden that ran in this suite;
      2. the full per-golden/per-metric detail table the CSV already has,
         with pass/fail colored and a score bar per row.

    `html_path` defaults to the same filename as `csv_path` with a .html
    extension, so an uploaded CSV and its HTML sibling are always easy to
    pair up by name (report-results in tasks.yaml relies on exactly this).
    `group_keys` should be ("span_name", "metric_name") for a component
    CSV (tracing_report.py) so metrics repeated under different spans get
    their own summary row instead of being averaged together.
    """
    csv_path = Path(csv_path)
    out_path = Path(html_path) if html_path else csv_path.with_suffix(".html")
    rows = read_csv_rows(csv_path)

    extra_columns = tuple(k for k in group_keys if k != "metric_name")
    report_title = title or csv_path.stem
    run_cost = total_cost(rows)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(report_title)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{_esc(report_title)}</h1>
<div class="meta">
  <span>Source: {_esc(csv_path.name)}</span>
  <span>Rows: {len(rows)}</span>
  <span>Total judge cost: {_fmt_cost(run_cost)}</span>
</div>
<h2>Per-metric summary</h2>
{_summary_table(rows, group_keys)}
<h2>Per-golden detail</h2>
{_detail_table(rows, extra_columns)}
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path


def print_metric_averages(
    rows: list[dict],
    group_keys: tuple[str, ...] = ("metric_name",),
    write=print,
) -> None:
    """Plain-text version of the summary table. `write` defaults to the
    builtin print(); pass a pytest TerminalReporter's write_line (or any
    other str -> None callable) to route it through pytest's own output
    machinery instead — plugin.py does this so the summary shows up
    reliably even under output capturing that a bare print() at
    sessionfinish time could otherwise get swallowed by."""
    summaries = metric_averages(rows, group_keys=group_keys)
    if not summaries:
        return
    label_width = max(
        (len(" / ".join(str(s[k]) for k in group_keys)) for s in summaries), default=10
    )
    label_width = max(label_width, len("metric"))
    write(
        f"{'metric':<{label_width}}  {'avg':>6}  {'min':>6}  {'max':>6}  "
        f"{'pass':>10}  {'cost':>10}  n"
    )
    write("-" * (label_width + 52))
    for s in summaries:
        label = " / ".join(str(s[k]) for k in group_keys)
        pass_str = f"{s['pass_count']}/{s['pass_total']}" if s["pass_total"] else "—"
        write(
            f"{label:<{label_width}}  {_fmt_score(s['avg_score']):>6}  "
            f"{_fmt_score(s['min_score']):>6}  {_fmt_score(s['max_score']):>6}  "
            f"{pass_str:>10}  {_fmt_cost(s['total_cost']):>10}  {s['n']}"
        )
    write(f"\nTotal judge cost for this run: {_fmt_cost(total_cost(rows))}")
