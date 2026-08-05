"""
`elyaeval init` — generates a filled-in pytest file for a chosen task type,
using the standard dataset + the matching metric preset.
"""

import argparse
from importlib import resources
from pathlib import Path

from elyaeval.dataset import load_standard_dataset

_TASK_TYPE_TO_METRICS_CONSTANT = {
    "rag_qa": "RAG_METRICS",
    "retrieval": "RETRIEVAL_METRICS",
    "summarization": "SUMMARIZATION_METRICS",
    "qa_correctness": "QA_CORRECTNESS_METRICS",
}


_CORPUS_MODE_TO_TEMPLATE = {
    "existing": "test_template.py.tmpl",
    "seeded": "test_template_seeded.py.tmpl",
}


def _load_template(corpus_mode: str) -> str:
    filename = _CORPUS_MODE_TO_TEMPLATE[corpus_mode]
    return (resources.files("elyaeval") / "templates" / filename).read_text()


def init(task_type: str, ci_stage: str, output: str, corpus_mode: str = "existing") -> Path:
    if task_type not in _TASK_TYPE_TO_METRICS_CONSTANT:
        valid = ", ".join(_TASK_TYPE_TO_METRICS_CONSTANT)
        raise SystemExit(f"Unknown --task-type '{task_type}'. Valid options: {valid}")

    if corpus_mode not in _CORPUS_MODE_TO_TEMPLATE:
        valid = ", ".join(_CORPUS_MODE_TO_TEMPLATE)
        raise SystemExit(f"Unknown --corpus-mode '{corpus_mode}'. Valid options: {valid}")

    n_matched = len(load_standard_dataset(task_type=task_type, ci_stage=ci_stage))
    if n_matched == 0:
        print(
            f"Warning: 0 goldens in the standard dataset match "
            f"task_type='{task_type}' and ci_stage='{ci_stage}'. "
            f"The generated file will collect but every test will be skipped by pytest. "
            f"Check `elyaeval.dataset.load_standard_dataset` filters or pass a different --ci-stage."
        )

    if corpus_mode == "existing":
        print(
            "Note: generating in existing-corpus mode — GOLDENS defaults to the shared "
            "standard dataset, which is a generic benchmark, not written against your "
            "app's corpus. Point GOLDENS at your own goldens file (see the comment in "
            "the generated file) for retrieval-dependent metrics to mean anything."
        )

    metrics_constant = _TASK_TYPE_TO_METRICS_CONSTANT[task_type]
    rendered = _load_template(corpus_mode).format(
        task_type=task_type,
        ci_stage=ci_stage,
        metrics_constant=metrics_constant,
    )

    out_path = Path(output)
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists — refusing to overwrite. Remove it or pass --output.")

    out_path.write_text(rendered)
    return out_path


def main():
    parser = argparse.ArgumentParser(prog="elyaeval")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Generate a starter pytest file for a task type")
    p_init.add_argument(
        "--task-type",
        required=True,
        choices=list(_TASK_TYPE_TO_METRICS_CONSTANT),
        help="Which use case to generate a test file for.",
    )
    p_init.add_argument(
        "--ci-stage",
        default="nightly",
        help="Which ci_stage's goldens to pull (per_commit / merge / nightly / release). Default: nightly.",
    )
    p_init.add_argument(
        "--output",
        default=None,
        help="Path to write the generated test file. Default: test_elyaeval_<task_type>.py",
    )
    p_init.add_argument(
        "--corpus-mode",
        choices=list(_CORPUS_MODE_TO_TEMPLATE),
        default="existing",
        help=(
            "'existing' (default): run_app() calls your app and it retrieves from "
            "whatever corpus it already has — nothing is inserted or torn down. Use "
            "this with your own goldens file written against your real corpus. "
            "'seeded': temporarily inserts each golden's context passages into your "
            "app's storage before the suite and removes them after — for running the "
            "shared standard dataset against an app whose corpus doesn't already "
            "cover it. Requires filling in _ingest_passage/_teardown_passages for "
            "your storage backend."
        ),
    )

    args = parser.parse_args()

    if args.command == "init":
        output = args.output or f"test_elyaeval_{args.task_type}.py"
        out_path = init(args.task_type, args.ci_stage, output, args.corpus_mode)
        print(f"Wrote {out_path}")
        print("Next steps:")
        if args.corpus_mode == "existing":
            print(f"  1. Open {out_path} and fill in TODO 1 and TODO 2 (import + run_app()).")
            print(f"  2. Point GOLDENS at your own goldens file (dataset_path=...).")
            print(f"  3. Run: deepeval test run {out_path}")
        else:
            print(f"  1. Open {out_path} and fill in TODO 1, TODO 2, and TODO 3 (ingest/teardown).")
            print(f"  2. Run: deepeval test run {out_path}")


if __name__ == "__main__":
    main()
