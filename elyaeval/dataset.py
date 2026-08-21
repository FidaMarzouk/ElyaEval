"""
Loads the standard dataset (goldens.jsonl, versioned and shipped inside this
package and returns deepeval Golden objects, filtered by task_type and/or ci_stage.

Pass your own `dataset_path` to evaluate against a goldens file written
for YOUR app's real corpus instead of the bundled generic one — this is
the recommended input for the default `elyaeval init` template.

Same JSONL shape as the standard dataset; `context` is optional and not
required by any task_type — no metric preset in metrics.py (RAG_METRICS,
RETRIEVAL_METRICS, GENERATION_METRICS) reads it. Leave it out.

GOLDEN_SCHEMA below is the single source of truth for what a row must
contain, per task_type.
"""

import json
from importlib import resources
from pathlib import Path
from typing import Optional

from deepeval.dataset import Golden

_PACKAGE_DATASET = "standard_dataset/goldens.jsonl"

# Every row needs these regardless of task_type — enforced on EVERY row in
# the file, even ones that will be filtered out by task_type/ci_stage,
# because a row that's missing these can never be matched by ANY filter

BASELINE_REQUIRED_FIELDS = ["input"]
BASELINE_REQUIRED_METADATA_FIELDS = ["task_type", "ci_stage"]

# Per-task-type requirements, layered on top of the baseline above.
# Keys here are the only valid `task_type` values.

GOLDEN_SCHEMA = {
    "rag_qa": {
        "required_fields": ["expected_output"],
        "why": (
            "expected_output is what's actually scored — read by "
            "ContextualPrecision/Recall (against live retrieval_context) "
            "and by Faithfulness's reference check. "
        ),
    },
    "retrieval": {
        "required_fields": ["expected_output"],
        "why": (
            "Same as rag_qa: expected_output is read by "
            "ContextualPrecision/Recall (against live retrieval_context); "
            "ContextualRelevancy needs neither expected_output nor context, "
            "just input + retrieval_context. "
        ),
    },
    "summarization": {
        "required_fields": [],
        "why": "SummarizationMetric is reference-free, only input + your app's actual_output are scored.",
    },
    "qa_correctness": {
        "required_fields": ["expected_output"],
        "why": "The Correctness G-Eval rubric explicitly compares actual_output against expected_output.",
    },
    "agentic_action": {
        "required_fields": [],
        "required_metadata_fields": [],
        "why": (
            "ToolCorrectness/ArgumentCorrectness score run_app()'s actual "
            "tools_called. additional_metadata.expected_tools is read if "
            "present but is opt-in (score against ground truth) — see "
            "elyaeval/templates/test_template_agentic.py.tmpl."
        ),
    },
}


class GoldenValidationError(ValueError):
    """Raised once per load_standard_dataset() call, listing every problem
    found across the whole file — not just the first one — so a goldens
    file with several bad rows can be fixed in one pass instead of
    one failed run at a time."""

    def __init__(self, path: Path, problems: list[str]):
        self.path = path
        self.problems = problems
        header = f"{len(problems)} problem(s) in goldens file {path}:"
        super().__init__("\n".join([header] + [f"  - {p}" for p in problems]))


def _default_dataset_path() -> Path:
    return resources.files("elyaeval") / _PACKAGE_DATASET


def _validate_row(line_no: int, row: dict) -> list[str]:
    """Returns a list of problem strings for this one row (empty if none).
    Never raises — the caller collects problems across every row before
    deciding whether to raise, so validation is whole-file-upfront."""
    problems = []

    for field in BASELINE_REQUIRED_FIELDS:
        if not row.get(field):
            problems.append(f"line {line_no}: missing required field '{field}'")

    meta = row.get("additional_metadata")
    if not isinstance(meta, dict):
        problems.append(
            f"line {line_no}: 'additional_metadata' is missing or not an object "
            f"(needs at least: {BASELINE_REQUIRED_METADATA_FIELDS})"
        )
        return problems  # nothing below is checkable without metadata

    for field in BASELINE_REQUIRED_METADATA_FIELDS:
        if not meta.get(field):
            problems.append(f"line {line_no}: missing required additional_metadata.'{field}'")

    row_task_type = meta.get("task_type")
    if row_task_type is not None and row_task_type not in GOLDEN_SCHEMA:
        problems.append(
            f"line {line_no}: unknown additional_metadata.task_type '{row_task_type}' "
            f"— expected one of {sorted(GOLDEN_SCHEMA)}"
        )
        return problems  # can't check task-specific fields against an unknown schema

    if row_task_type in GOLDEN_SCHEMA:
        schema = GOLDEN_SCHEMA[row_task_type]
        for field in schema.get("required_fields", []):
            if not row.get(field):
                problems.append(
                    f"line {line_no}: task_type='{row_task_type}' requires '{field}' "
                    f"({schema['why']})"
                )
        for field in schema.get("required_metadata_fields", []):
            if not meta.get(field):
                problems.append(
                    f"line {line_no}: task_type='{row_task_type}' requires "
                    f"additional_metadata.'{field}' ({schema['why']})"
                )

    return problems


def load_standard_dataset(
    task_type: Optional[str] = None,
    ci_stage: Optional[str] = None,
    dataset_path: Optional[str] = None,
) -> list[Golden]:
    """
    Parameters
    ----------
    task_type: filter to e.g. "rag_qa", "summarization", "qa_correctness".
               None returns all task types mixed
    ci_stage:  filter to e.g. "per_commit", "merge", "nightly", "release".
               Controls run frequency.
    dataset_path: override the package-bundled dataset (e.g. to point at a
               pinned version, or a path fetched via the version tag.
               Defaults to the dataset shipped inside elyaeval.

    Validates the WHOLE file upfront against GOLDEN_SCHEMA before returning
    anything — every row is checked (regardless of task_type/ci_stage
    filters, since a row that fails baseline checks would otherwise just
    vanish from every filter silently)
    """
    path = Path(dataset_path) if dataset_path else _default_dataset_path()

    parsed_rows: list[tuple[int, dict]] = []
    problems: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"line {line_no}: invalid JSON ({e})")
                continue
            problems.extend(_validate_row(line_no, row))
            parsed_rows.append((line_no, row))

    if problems:
        raise GoldenValidationError(path, problems)

    goldens: list[Golden] = []
    for _, row in parsed_rows:
        meta = row.get("additional_metadata", {})
        if task_type is not None and meta.get("task_type") != task_type:
            continue
        if ci_stage is not None and meta.get("ci_stage") != ci_stage:
            continue

        goldens.append(
            Golden(
                input=row["input"],
                expected_output=row.get("expected_output"),
                context=row.get("context"),
                # retrieval_context deliberately NOT passed through —
                # see module docstring.
                additional_metadata=meta,
                comments=row.get("comments"),
            )
        )
    return goldens