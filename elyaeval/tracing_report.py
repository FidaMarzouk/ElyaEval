"""
Component-level (span-aware) evaluation + CSV reporting.

Sibling to report.py — NOT a rewrite of it. report.py drives DeepEval's
end-to-end evaluate() over a single LLMTestCase per golden; this module
drives DeepEval's evals_iterator() over a single-golden EvaluationDataset,
so that per-span metrics attached via @observe(metrics=...) +
update_current_span(test_case=...) inside the SUT are actually collected.
That only works for an IN-PROCESS call into the SUT (spans are scoped to
the call stack) — see the component test template's docstring for why the
existing e2e/HTTP suite (test_elyaeval_<task_type>.py) can never see them.

evals_iterator() is a generator: driving it to exhaustion is what runs the
golden through the SUT and produces its EvaluationResult, which is the
generator's *return* value (StopIteration.value) — a plain
`for golden in ds.evals_iterator(): ...` loop discards that. TracedRunner
below does the manual next()/StopIteration dance so callers don't have to.

For each golden, DeepEval's iterator appends MULTIPLE TestResults to that
EvaluationResult (confirmed against the installed deepeval package's
execute_agentic_test_cases_from_loop):
  - one trace-level result (name=golden.name — usually blank here, since
    this SUT declares no metrics=[...] on run_for_eval itself, only on the
    retrieve/generate_answer spans below it, so this row carries no
    metrics_data and produces no CSV rows)
  - one result per span that had metrics attached, named after that span
    (test_result.name == the @observe'd function's name, e.g. "retrieve",
    "generate_answer")

TestResult has no parent_span field on the public API (only the internal
TraceApi/span tree does), so we surface span_name (== test_result.name) as
a plain CSV column instead of reconstructing a call tree. If you need a
literal parent-span column later, that has to be derived on the SUT side,
since this package intentionally doesn't hardcode your SUT's function
names.
"""

import csv
from pathlib import Path
from typing import Callable, Optional

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.evaluate.types import EvaluationResult, TestResult
from deepeval.metrics import BaseMetric

from .html_report import register_report

TRACED_CSV_FIELDS = [
    "golden_id",
    "priority",
    "span_name",
    "input",
    "metric_name",
    "score",
    "threshold",
    "success",
    "reason",
    "error",
    "evaluation_model",
    "evaluation_cost",
    "input_tokens",
    "output_tokens",
]


def golden_id(golden: Golden, fallback_index: Optional[int] = None) -> str:
    """Same shape as report.py's (private) _golden_id, duplicated — not
    imported — so this module stays independent of report.py's internals.
    Public here (unlike report.py's version) because, unlike
    evaluate_golden(), TracedRunner doesn't write CSV rows itself — the
    component template computes the id itself and needs this exported.
    Mirrors the priority + index identifier already used for pytest ids in
    the templates, so a CSV row and a pytest test id point at the same
    golden."""
    meta = golden.additional_metadata or {}
    priority = meta.get("priority", "")
    idx = fallback_index if fallback_index is not None else ""
    return f"{priority}_{idx}" if (priority or idx != "") else golden.input[:40]


def append_traced_csv_rows(csv_path: str | Path, rows: list[dict]) -> None:
    """Same append-with-header-on-first-write behavior as
    report.append_csv_rows, duplicated (not imported) because the two CSVs
    have different columns (span_name) and are meant to stay independent
    files/schemas — safe to call once per golden across a serial pytest
    run, not safe under parallel workers (xdist) writing the same path.

    Registers csv_path with group_keys=("span_name", "metric_name")
    instead of report.py's ("metric_name",) — the same metric name can be
    scored on more than one span here (e.g. Faithfulness on both a
    "retrieve" and a "generate_answer" span), so averaging by metric name
    alone would silently blend two different spans' scores together."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRACED_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    register_report(path, group_keys=("span_name", "metric_name"))


def traced_test_result_to_csv_rows(golden_id: str, priority: str, test_result: TestResult) -> list[dict]:
    """One row per metric in test_result.metrics_data, same convention as
    report.test_result_to_csv_rows (including evaluation_model/
    evaluation_cost/input_tokens/output_tokens — see that function's
    docstring for what DeepEval does and doesn't expose there, and why the
    two token fields are read via getattr(..., None) rather than direct
    attribute access), plus span_name (test_result.name — the @observe'd
    function name for a span-level result). A result with no metrics_data
    (the trace-level result, here) simply produces no rows — nothing was
    scored, so there's nothing to write."""
    rows = []
    for md in test_result.metrics_data or []:
        rows.append({
            "golden_id": golden_id,
            "priority": priority,
            "span_name": test_result.name,
            "input": test_result.input,
            "metric_name": md.name,
            "score": md.score,
            "threshold": md.threshold,
            "success": md.success,
            "reason": md.reason,
            "error": md.error,
            "evaluation_model": md.evaluation_model,
            "evaluation_cost": md.evaluation_cost,
            "input_tokens": getattr(md, "input_tokens", None),
            "output_tokens": getattr(md, "output_tokens", None),
        })
    return rows


class TracedRunner:
    """
    Drives DeepEval's evals_iterator() for a SINGLE golden, wrapped around
    an in-process call into the SUT, and exposes the resulting
    EvaluationResult once that golden has run.

        runner = TracedRunner(golden)
        for _golden in runner:      # exactly one iteration for one golden
            run_for_eval(_golden.input, expected_output=_golden.expected_output)
        result = runner.result      # EvaluationResult, now populated
        test_results = result.test_results   # [trace-level, *span-level]

    Or, equivalently, via the .run() convenience wrapper:

        result = TracedRunner(golden).run(
            lambda g: run_for_eval(g.input, expected_output=g.expected_output)
        )

    `metrics` here means TRACE-level metrics (evals_iterator(metrics=...)),
    NOT the span-level metrics — those are already declared on the SUT
    side, at the @observe(metrics=...) call sites in retriever.py/
    generator.py, and don't need to be passed in again here. Leave
    metrics=None unless you also want a metric judged against the whole
    trace (e.g. an end-to-end correctness check spanning retrieve+generate)
    — in which case run_for_eval() must also call update_current_trace(...)
    with input/output for that metric to have something to score.

    error_config defaults to ignore_errors=True (mirrors evaluate_golden in
    report.py) so one golden's SUT exception doesn't abort the whole
    parametrized suite — it surfaces as an errored span/trace in the
    result instead of raising out of this call.
    """

    def __init__(
        self,
        golden: Golden,
        metrics: Optional[list[BaseMetric]] = None,
        display_config: Optional[DisplayConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        error_config: Optional[ErrorConfig] = None,
    ):
        self.golden = golden
        self.metrics = metrics
        self.display_config = display_config or DisplayConfig(show_indicator=False, print_results=False)
        self.cache_config = cache_config or CacheConfig()
        self.error_config = error_config or ErrorConfig(ignore_errors=True)
        self.result: Optional[EvaluationResult] = None

        self._dataset = EvaluationDataset(goldens=[golden])
        self._iterator = self._dataset.evals_iterator(
            metrics=self.metrics,
            display_config=self.display_config,
            cache_config=self.cache_config,
            error_config=self.error_config,
            async_config=AsyncConfig(run_async=False),
        )

    def __iter__(self):
        return self

    def __next__(self) -> Golden:
        try:
            return next(self._iterator)
        except StopIteration as e:
            # e.value is the generator's `return EvaluationResult(...)` —
            # the only place it's available; a plain for-loop discards it.
            self.result = e.value
            raise

    def run(self, call_sut: Callable[[Golden], None]) -> EvaluationResult:
        """Convenience for the common case: no extra logic needed around
        the SUT call itself — drive the single-golden iterator, call
        call_sut(golden) for its one iteration, and return the result."""
        for golden in self:
            call_sut(golden)
        assert self.result is not None, "evals_iterator exhausted without producing a result"
        return self.result