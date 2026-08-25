"""
Per-golden evaluation + CSV reporting.

Replaces the assert_test()-based pattern (junit.xml only, no structured
scores) with deepeval's evaluate(), called ONCE PER GOLDEN — never with the
whole GOLDENS list in one call. Batching all goldens into a single evaluate()
call is exactly what this module exists to avoid: it would still gate/report
correctly, but every row in the CSV would belong to the same opaque batch
run, and there would be no way to tell which golden produced which score.

evaluate_golden() below is meant to replace assert_test() one-for-one inside
a parametrized pytest test: same one-test-per-golden shape, same
input/expected_output/context/retrieval_context test case, but the result is
now a structured TestResult (metrics_data: list[MetricData], each with
name/score/threshold/success/reason) instead of a raised AssertionError.
That structured result is what gets written to CSV — one row per
(golden, metric) pair — and is also what the caller asserts on for the
pytest pass/fail gate, so junit.xml keeps working unchanged in the meantime.

"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.evaluate.types import TestResult
from deepeval.test_case import LLMTestCase
from deepeval.dataset import Golden
from deepeval.metrics import BaseMetric

from .html_report import register_report

CSV_FIELDS = [
    "golden_id",
    "priority",
    "input",
    "metric_name",
    "score",
    "threshold",
    "success",
    "reason",
    "error",
]


def new_report_path(task_type: str, report_dir: str | Path = "report") -> Path:
    """Build a fresh, timestamped CSV path: report/results_<task_type>_<UTC
    timestamp>.csv. Call this ONCE, at module level in a generated test file
    (so it's evaluated once per pytest session, not once per golden) — every
    golden tested in that session then appends to the same file via
    append_csv_rows(), while the next session gets its own new filename
    automatically. Nothing to delete, nothing to clean up between runs."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return report_dir / f"results_{task_type}_{timestamp}.csv"


def _golden_id(golden: Golden, fallback_index: Optional[int] = None) -> str:
    """Same identifier shape already used for pytest ids in the templates
    (priority + index) — reused here so a CSV row and a pytest test id
    point at the same golden without needing a separate id field."""
    meta = golden.additional_metadata or {}
    priority = meta.get("priority", "")
    idx = fallback_index if fallback_index is not None else ""
    return f"{priority}_{idx}" if (priority or idx != "") else golden.input[:40]


def append_csv_rows(csv_path: str | Path, rows: list[dict]) -> None:
    """Append `rows` to csv_path, writing the header first if the file is
    new or empty. Safe to call once per golden across a serial pytest run —
    not safe under parallel workers (xdist) writing the same path.

    Also registers csv_path with the html_report module (group_keys=
    ("metric_name",)) so elyaeval's pytest plugin knows to render an HTML
    sibling + print per-metric averages for it at session end — this is
    the only place that registration needs to happen for the e2e/flat CSV
    shape, since every row this module ever writes goes through here."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    register_report(path, group_keys=("metric_name",))


def test_result_to_csv_rows(golden_id: str, priority: str, test_result: TestResult) -> list[dict]:
    """One row per metric in test_result.metrics_data — this is what makes
    the CSV queryable per-metric (e.g. filter to just Faithfulness failures)
    instead of collapsing a golden's several metric scores into one line."""
    rows = []
    for md in test_result.metrics_data or []:
        rows.append({
            "golden_id": golden_id,
            "priority": priority,
            "input": test_result.input,
            "metric_name": md.name,
            "score": md.score,
            "threshold": md.threshold,
            "success": md.success,
            "reason": md.reason,
            "error": md.error,
        })
    return rows


def evaluate_golden(
    golden: Golden,
    test_case: LLMTestCase,
    metrics: list[BaseMetric],
    csv_path: str | Path,
    golden_id: Optional[str] = None,
    fallback_index: Optional[int] = None,
) -> TestResult:
    """
    Drop-in replacement for `assert_test(test_case, metrics)`:

        result = evaluate_golden(golden, test_case, RAG_METRICS, csv_path=REPORT_CSV)
        assert result.success, result.name

    Runs `metrics` against `test_case` via evaluate() with test_cases=[test_case]
    (a batch of exactly one), appends one CSV row per metric to csv_path, and
    returns the TestResult so the caller decides how to fail the pytest test.
    print_results/show_indicator are disabled
    """
    gid = golden_id or _golden_id(golden, fallback_index)
    priority = (golden.additional_metadata or {}).get("priority", "")

    result = evaluate(
        test_cases=[test_case],
        metrics=metrics,
        display_config=DisplayConfig(show_indicator=False, print_results=False),
        async_config=AsyncConfig(run_async=False),
        error_config=ErrorConfig(ignore_errors=True),
    )
    test_result = result.test_results[0]

    rows = test_result_to_csv_rows(gid, priority, test_result)
    append_csv_rows(csv_path, rows)

    return test_result