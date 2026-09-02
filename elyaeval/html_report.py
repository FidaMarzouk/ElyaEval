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
    evaluation_cost to float, input_tokens/output_tokens to int, and
    success to bool where possible. Rows are returned in file order (i.e.
    the order goldens actually ran in), which is also the order the detail
    table renders in — no re-sorting happens later on the (mistaken)
    assumption that CSV order is arbitrary."""
    path = Path(csv_path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["score"] = _to_float(row.get("score"))
        row["threshold"] = _to_float(row.get("threshold"))
        row["success"] = _to_bool(row.get("success"))
        # Older CSVs written before cost/token tracking was added simply
        # have no evaluation_cost/evaluation_model/input_tokens/
        # output_tokens column at all — csv.DictReader then leaves them
        # missing from the dict entirely rather than "", so .get(...) (not
        # [...]) everywhere downstream, and this coerces a genuinely
        # absent column the same way as a blank one: None.
        row["evaluation_cost"] = _to_float(row.get("evaluation_cost"))
        row["input_tokens"] = _to_int(row.get("input_tokens"))
        row["output_tokens"] = _to_int(row.get("output_tokens"))
    return rows


def _to_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
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
      - total_input_tokens / total_output_tokens: sum of input_tokens/
                      output_tokens across every row in the group that had
                      a value. Same None-means-unknown convention as
                      total_cost — DeepEval only started reporting these at
                      all in 4.2, and even then only for judges whose
                      provider actually returns token usage, so an older
                      DeepEval version or an unpriced/local judge both
                      leave this None rather than 0.
      - tokens_n:     how many rows actually contributed a token count —
                      same purpose as cost_n, for the same reason

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
        input_tokens_vals = [r["input_tokens"] for r in group_rows if r.get("input_tokens") is not None]
        output_tokens_vals = [r["output_tokens"] for r in group_rows if r.get("output_tokens") is not None]

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
        summary["total_input_tokens"] = sum(input_tokens_vals) if input_tokens_vals else None
        summary["total_output_tokens"] = sum(output_tokens_vals) if output_tokens_vals else None
        summary["tokens_n"] = len(input_tokens_vals) or len(output_tokens_vals)
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


def total_tokens(rows: list[dict]) -> tuple[Optional[int], Optional[int]]:
    """(total_input_tokens, total_output_tokens) across ALL rows,
    independent of grouping — same purpose as total_cost() but for raw
    token counts (DeepEval 4.2+ only — see test_result_to_csv_rows'
    docstring in report.py). Each side is None independently if no row
    in the run reported that value, same None-means-unknown convention
    used throughout — an older DeepEval version or an unpriced/local
    judge leaves both None, not 0."""
    inputs = [r.get("input_tokens") for r in rows if r.get("input_tokens") is not None]
    outputs = [r.get("output_tokens") for r in rows if r.get("output_tokens") is not None]
    return (sum(inputs) if inputs else None, sum(outputs) if outputs else None)


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


def _fmt_tokens(value: Optional[int]) -> str:
    """Thousands-separated integer, "—" if None (unknown/unavailable —
    see total_tokens()'s docstring), never "0" standing in for unknown."""
    return "—" if value is None else f"{value:,}"


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
        "Input tokens",
        "Output tokens",
    ]
    head_html = "".join(f"<th>{_esc(h)}</th>" for h in headers)

    body_rows = []
    for s in summaries:
        key_cells = "".join(f"<td>{_esc(s[k])}</td>" for k in group_keys)
        cost_note = f" ({s['cost_n']}/{s['n']} priced)" if s["cost_n"] and s["cost_n"] < s["n"] else ""
        tokens_note = f" ({s['tokens_n']}/{s['n']} reported)" if s["tokens_n"] and s["tokens_n"] < s["n"] else ""
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
            f'<td class="num">{_fmt_tokens(s["total_input_tokens"])}'
            f'<span class="reason">{tokens_note}</span></td>'
            f'<td class="num">{_fmt_tokens(s["total_output_tokens"])}</td>'
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
        "In tokens",
        "Out tokens",
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
            f'<td class="num">{_fmt_tokens(r.get("input_tokens"))}</td>'
            f'<td class="num">{_fmt_tokens(r.get("output_tokens"))}</td>'
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
    run_input_tokens, run_output_tokens = total_tokens(rows)

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
  <span>Total tokens: {_fmt_tokens(run_input_tokens)} in / {_fmt_tokens(run_output_tokens)} out</span>
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
        f"{'pass':>10}  {'cost':>10}  {'in tok':>10}  {'out tok':>10}  n"
    )
    write("-" * (label_width + 74))
    for s in summaries:
        label = " / ".join(str(s[k]) for k in group_keys)
        pass_str = f"{s['pass_count']}/{s['pass_total']}" if s["pass_total"] else "—"
        write(
            f"{label:<{label_width}}  {_fmt_score(s['avg_score']):>6}  "
            f"{_fmt_score(s['min_score']):>6}  {_fmt_score(s['max_score']):>6}  "
            f"{pass_str:>10}  {_fmt_cost(s['total_cost']):>10}  "
            f"{_fmt_tokens(s['total_input_tokens']):>10}  {_fmt_tokens(s['total_output_tokens']):>10}  {s['n']}"
        )
    run_input_tokens, run_output_tokens = total_tokens(rows)
    write(f"\nTotal judge cost for this run: {_fmt_cost(total_cost(rows))}")
    write(f"Total tokens for this run: {_fmt_tokens(run_input_tokens)} in / {_fmt_tokens(run_output_tokens)} out")


# ---------------------------------------------------------------------------
# Regression comparison — diffing two runs' metric_averages() against each
# other. Works purely off elyaeval's own CSVs, so it needs no Confident AI
# account.
# ---------------------------------------------------------------------------

DEFAULT_REGRESSION_TOLERANCE = 0.02  # absolute avg_score drop that counts as a regression


def compare_runs(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    group_keys: tuple[str, ...] = ("metric_name",),
    tolerance: float = DEFAULT_REGRESSION_TOLERANCE,
) -> list[dict]:
    """
    Align baseline vs candidate metric_averages() by group_keys and compute,
    per group that appears in EITHER run:

      - baseline_avg / candidate_avg / delta (candidate - baseline; None if
        either side is missing an avg — e.g. a metric that errored out
        entirely on one side, or a metric only one of the two runs has)
      - baseline_pass_rate / candidate_pass_rate / pass_rate_delta, same
        None-if-missing rule
      - regressed: True if the group exists in BOTH runs and
        candidate_avg is more than `tolerance` below baseline_avg. A group
        that's brand new in candidate (not in baseline at all) or that
        disappeared (was in baseline, not in candidate) is flagged
        separately via `status`, not silently marked as regressed — a
        metric that's simply new isn't a regression, and one that's gone
        deserves its own callout (did the SUT drop a pipeline stage? did
        the suite change?) rather than being invisible.
      - status: "regressed" | "improved" | "unchanged" | "new" | "removed"
        "improved"/"unchanged" split at the same `tolerance` band as
        regression, so a +0.001 wobble isn't reported as "improved" any
        more than a -0.001 wobble is reported as "regressed."

    tolerance is an absolute score difference (both scores are 0..1), not a
    percentage — default 0.02 matches a reasonable "noise floor" for an
    LLM-judge metric re-run on unchanged inputs, but pass your own if a
    metric's judge is noisier or stricter than that.
    """
    baseline_by_key = {tuple(s[k] for k in group_keys): s for s in metric_averages(baseline_rows, group_keys)}
    candidate_by_key = {tuple(s[k] for k in group_keys): s for s in metric_averages(candidate_rows, group_keys)}

    order: list[tuple] = []
    for key in list(baseline_by_key) + list(candidate_by_key):
        if key not in order:
            order.append(key)

    comparisons = []
    for key in order:
        b = baseline_by_key.get(key)
        c = candidate_by_key.get(key)

        b_avg = b["avg_score"] if b else None
        c_avg = c["avg_score"] if c else None
        b_pass = b["pass_rate"] if b else None
        c_pass = c["pass_rate"] if c else None

        if b is None:
            status = "new"
        elif c is None:
            status = "removed"
        elif b_avg is None or c_avg is None:
            status = "unchanged"  # can't compare numerically; don't guess
        else:
            # Round before comparing to tolerance — plain float subtraction
            # can put a value that's conceptually exactly AT the tolerance
            # boundary a hair past it (e.g. 0.88 - 0.90 == -0.020000000000000018,
            # not -0.02), which would otherwise flip an "unchanged" result
            # to "regressed" purely from binary floating-point representation
            # error, not a real score difference.
            delta = round(c_avg - b_avg, 9)
            if delta < -tolerance:
                status = "regressed"
            elif delta > tolerance:
                status = "improved"
            else:
                status = "unchanged"

        comparisons.append({
            **dict(zip(group_keys, key)),
            "status": status,
            "baseline_avg": b_avg,
            "candidate_avg": c_avg,
            "delta": (c_avg - b_avg) if (b_avg is not None and c_avg is not None) else None,
            "baseline_pass_rate": b_pass,
            "candidate_pass_rate": c_pass,
            "pass_rate_delta": (c_pass - b_pass) if (b_pass is not None and c_pass is not None) else None,
            "baseline_cost": b["total_cost"] if b else None,
            "candidate_cost": c["total_cost"] if c else None,
        })
    return comparisons


def print_comparison(
    comparisons: list[dict],
    group_keys: tuple[str, ...] = ("metric_name",),
    write=print,
) -> None:
    """Plain-text regression table — what `elyaeval compare` prints."""
    if not comparisons:
        write("No metrics in either run.")
        return
    label_width = max(
        (len(" / ".join(str(c[k]) for k in group_keys)) for c in comparisons), default=10
    )
    label_width = max(label_width, len("metric"))
    write(f"{'metric':<{label_width}}  {'baseline':>9}  {'candidate':>9}  {'delta':>8}  status")
    write("-" * (label_width + 45))
    for c in comparisons:
        label = " / ".join(str(c[k]) for k in group_keys)
        delta_str = "—" if c["delta"] is None else f"{c['delta']:+.3f}"
        marker = {
            "regressed": "▼ REGRESSED",
            "improved": "▲ improved",
            "unchanged": "= unchanged",
            "new": "+ new",
            "removed": "- removed",
        }[c["status"]]
        write(
            f"{label:<{label_width}}  {_fmt_score(c['baseline_avg']):>9}  "
            f"{_fmt_score(c['candidate_avg']):>9}  {delta_str:>8}  {marker}"
        )
    n_regressed = sum(1 for c in comparisons if c["status"] == "regressed")
    if n_regressed:
        write(f"\n{n_regressed} metric(s) regressed beyond tolerance.")
    else:
        write("\nNo regressions beyond tolerance.")


def render_comparison_html(
    comparisons: list[dict],
    group_keys: tuple[str, ...] = ("metric_name",),
    html_path: Optional[str | Path] = None,
    title: str = "Regression comparison",
) -> Optional[Path]:
    """HTML version of print_comparison. Returns None (writes nothing) if
    no html_path given — pass one explicitly, there's no CSV to infer a
    path from here since this compares two of them."""
    status_class = {
        "regressed": "fail",
        "improved": "pass",
        "unchanged": "reason",
        "new": "reason",
        "removed": "error-cell",
    }
    status_label = {
        "regressed": "▼ regressed",
        "improved": "▲ improved",
        "unchanged": "= unchanged",
        "new": "+ new",
        "removed": "− removed",
    }

    headers = [k.replace("_", " ").title() for k in group_keys] + [
        "Baseline avg", "Candidate avg", "Delta", "Baseline pass", "Candidate pass", "Status",
    ]
    head_html = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body_rows = []
    for c in comparisons:
        key_cells = "".join(f"<td>{_esc(c[k])}</td>" for k in group_keys)
        delta_str = "—" if c["delta"] is None else f"{c['delta']:+.3f}"
        body_rows.append(
            "<tr>"
            f"{key_cells}"
            f'<td class="num">{_fmt_score(c["baseline_avg"])}</td>'
            f'<td class="num">{_fmt_score(c["candidate_avg"])}</td>'
            f'<td class="num">{delta_str}</td>'
            f'<td class="num">{_fmt_pct(c["baseline_pass_rate"])}</td>'
            f'<td class="num">{_fmt_pct(c["candidate_pass_rate"])}</td>'
            f'<td class="{status_class[c["status"]]}">{status_label[c["status"]]}</td>'
            "</tr>"
        )
    table_html = (
        '<table class="detail-table">'
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )

    n_regressed = sum(1 for c in comparisons if c["status"] == "regressed")
    verdict = (
        f'<p style="color:var(--fail); font-weight:600;">{n_regressed} metric(s) regressed beyond tolerance.</p>'
        if n_regressed
        else '<p style="color:var(--pass); font-weight:600;">No regressions beyond tolerance.</p>'
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{_esc(title)}</h1>
{verdict}
<h2>Per-metric comparison</h2>
{table_html}
</body>
</html>
"""
    if html_path is None:
        return None
    out_path = Path(html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    return out_path