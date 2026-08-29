"""
Run-level metadata — what SUT config produced a given CSV, e.g. which
generator model, which prompt version — captured once per test session so
`elyaeval compare` (html_report.py) can later explain WHY two runs' scores
differ, not just THAT they differ.

Two things happen when you call log_run_metadata(), deliberately kept
independent of each other:

  1. DeepEval's own deepeval.log_hyperparameters() gets called with the same
     dict, so it shows up in DeepEval's own terminal output and (if
     DEEPEVAL_RESULTS_FOLDER is set) its own test_run_<timestamp>.json.
     This is genuinely local-only for plain str/int/float values — DeepEval
     only talks to Confident AI here if a value is a deepeval.prompt.Prompt
     object (see deepeval/test_run/hyperparameters.py:process_hyperparameters
     — the is_confident() branch only fires for Prompt values, plain values
     go straight to str(value)). Don't pass Prompt objects in here unless
     you have a Confident AI key and want that push to happen.

  2. A sidecar JSON is written next to your CSV: same basename, .meta.json
     instead of .csv. This is what elyaeval.compare_runs() / `elyaeval
     compare` actually reads — kept independent of DEEPEVAL_RESULTS_FOLDER
     (which you may not have set) and independent of DeepEval's own JSON
     schema (which isn't guaranteed stable across DeepEval versions), so
     regression comparisons work purely off elyaeval's own artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Union


def log_run_metadata(csv_path: Union[str, Path], hyperparameters: dict) -> Path:
    """
    Call once per test session (module level in your generated test file,
    right after REPORT_CSV is assigned — see the TODO block in the
    template) with whatever identifies THIS run's configuration, e.g.:

        log_run_metadata(REPORT_CSV, {
            "generator_model": os.environ.get("GENERATOR_MODEL", "unknown"),
            "prompt_version": os.environ.get("PROMPT_VERSION", "unknown"),
        })

    Values must be strings, ints, or floats — same restriction DeepEval's
    own log_hyperparameters enforces, and for the same reason: anything
    else can't be written to the sidecar JSON or shown in a comparison
    table without additional handling this function doesn't do.

    Safe to call even if deepeval.log_hyperparameters raises for an
    unrelated reason (e.g. a DeepEval version mismatch) — the sidecar JSON
    still gets written either way, since that's the one `elyaeval compare`
    actually depends on.
    """
    for key, value in hyperparameters.items():
        if not isinstance(key, str):
            raise TypeError(f"hyperparameter key {key!r} must be a string")
        if not isinstance(value, (str, int, float)):
            raise TypeError(
                f"hyperparameter value for {key!r} must be a string, int, or float "
                f"(got {type(value).__name__}) — log_run_metadata does not support "
                f"DeepEval's Prompt objects, only plain values"
            )

    try:
        import deepeval

        deepeval.log_hyperparameters(lambda: dict(hyperparameters))
    except Exception as exc:  # noqa: BLE001 — metadata logging must never fail the run
        print(f"[elyaeval] deepeval.log_hyperparameters failed (continuing): {exc}")

    csv_path = Path(csv_path)
    meta_path = csv_path.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "csv": csv_path.name,
                "logged_at": datetime.now(timezone.utc).isoformat(),
                "hyperparameters": hyperparameters,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return meta_path


def read_run_metadata(csv_path: Union[str, Path]) -> dict:
    """Read back the sidecar JSON for a CSV, or {} if log_run_metadata was
    never called for that run (e.g. an older run, or a template that
    hasn't added the TODO block yet) — never raises for a missing file,
    since not every run is expected to have logged metadata."""
    meta_path = Path(csv_path).with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
