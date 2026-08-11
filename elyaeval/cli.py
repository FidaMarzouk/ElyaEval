"""
`elyaeval init` — generates a filled-in pytest file for a chosen task type,
using the standard dataset + the matching metric preset.

`elyaeval init-ci` — generates a per-repo Tekton PipelineRun stub that wires
this repo into the shared elyaeval-rag-recipe Pipeline. The Pipeline/Task
definitions themselves are NOT generated here — they're centralized and
applied to the cluster independently (see recipes/tekton/ in the elyaeval
repo). Only what's specific to this repo (repo URL, revision, where results
get published) is rendered into the caller's repo.
"""

import argparse
from importlib import resources
from importlib.metadata import version as _pkg_version
from pathlib import Path

from elyaeval.dataset import load_standard_dataset

_TASK_TYPE_TO_METRICS_CONSTANT = {
    "rag_qa": "RAG_METRICS",
    "retrieval": "RETRIEVAL_METRICS",
    "summarization": "SUMMARIZATION_METRICS",
    "qa_correctness": "QA_CORRECTNESS_METRICS",
    "agentic_action": "AGENTIC_ACTION_METRICS",
}


_CORPUS_MODE_TO_TEMPLATE = {
    "existing": "test_template.py.tmpl",
    "seeded": "test_template_seeded.py.tmpl",
}

# Task types with no retrieval leg at all — --corpus-mode doesn't apply
# (there's no corpus to point at "existing" vs "seed"), so each gets its
# own dedicated template instead of going through _CORPUS_MODE_TO_TEMPLATE.
_TASK_TYPES_WITHOUT_CORPUS_MODE = {
    "agentic_action": "test_template_agentic.py.tmpl",
}

_CI_TEMPLATE = "pipelinerun_template.yaml.tmpl"


def _load_template(filename: str) -> str:
    return (resources.files("elyaeval") / "templates" / filename).read_text()


def _template_filename(task_type: str, corpus_mode: str) -> str:
    if task_type in _TASK_TYPES_WITHOUT_CORPUS_MODE:
        return _TASK_TYPES_WITHOUT_CORPUS_MODE[task_type]
    return _CORPUS_MODE_TO_TEMPLATE[corpus_mode]


def _default_test_filename(task_type: str) -> str:
    """The naming convention every generated test file follows — the single
    place that convention is defined, so init() and init_ci() can't drift
    out of sync on what a task type's default filename looks like."""
    return f"test_elyaeval_{task_type}.py"


def _load_ci_template() -> str:
    return (resources.files("elyaeval") / "templates" / _CI_TEMPLATE).read_text()


def init(task_type: str, ci_stage: str, output: str, corpus_mode: str = "existing") -> Path:
    if task_type not in _TASK_TYPE_TO_METRICS_CONSTANT:
        valid = ", ".join(_TASK_TYPE_TO_METRICS_CONSTANT)
        raise SystemExit(f"Unknown --task-type '{task_type}'. Valid options: {valid}")

    if corpus_mode not in _CORPUS_MODE_TO_TEMPLATE:
        valid = ", ".join(_CORPUS_MODE_TO_TEMPLATE)
        raise SystemExit(f"Unknown --corpus-mode '{corpus_mode}'. Valid options: {valid}")

    if task_type in _TASK_TYPES_WITHOUT_CORPUS_MODE and corpus_mode != "existing":
        raise SystemExit(
            f"--corpus-mode is not applicable to task_type='{task_type}' "
            f"(no retrieval corpus involved) — omit --corpus-mode."
        )

    n_matched = len(load_standard_dataset(task_type=task_type, ci_stage=ci_stage))
    if n_matched == 0:
        print(
            f"Warning: 0 goldens in the standard dataset match "
            f"task_type='{task_type}' and ci_stage='{ci_stage}'. "
            f"The generated file will collect but every test will be skipped by pytest. "
            f"Check `elyaeval.dataset.load_standard_dataset` filters or pass a different --ci-stage."
        )

    if corpus_mode == "existing" and task_type not in _TASK_TYPES_WITHOUT_CORPUS_MODE:
        print(
            "Note: generating in existing-corpus mode — GOLDENS defaults to the shared "
            "standard dataset, which is a generic benchmark, not written against your "
            "app's corpus. Point GOLDENS at your own goldens file (see the comment in "
            "the generated file) for retrieval-dependent metrics to mean anything."
        )

    metrics_constant = _TASK_TYPE_TO_METRICS_CONSTANT[task_type]
    template_filename = _template_filename(task_type, corpus_mode)
    rendered = _load_template(template_filename).format(
        task_type=task_type,
        ci_stage=ci_stage,
        metrics_constant=metrics_constant,
    )

    out_path = Path(output)
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists — refusing to overwrite. Remove it or pass --output.")

    out_path.write_text(rendered)
    return out_path


def init_ci(
    repo_url: str,
    revision: str,
    results_repo_url: str,
    results_branch: str,
    output: str,
    task_type: str = "rag_qa",
    test_file: str = None,
) -> Path:
    try:
        elyaeval_version = _pkg_version("elyaeval")
    except Exception:
        elyaeval_version = "unknown"

    test_file = test_file or _default_test_filename(task_type)

    rendered = _load_ci_template().format(
        repo_url=repo_url,
        revision=revision,
        results_repo_url=results_repo_url,
        results_branch=results_branch,
        elyaeval_version=elyaeval_version,
        test_file=test_file,
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
            "your storage backend. Not applicable to task types with no retrieval "
            "leg (currently: agentic_action) — omit this flag for those."
        ),
    )

    p_init_ci = sub.add_parser(
        "init-ci",
        help="Generate a per-repo Tekton PipelineRun stub (does NOT generate the shared Pipeline/Task recipe)",
    )
    p_init_ci.add_argument(
        "--repo-url",
        required=True,
        help="HTTPS URL of THIS repo (the one fetch-source will clone), e.g. https://github.com/org/repo.git",
    )
    p_init_ci.add_argument(
        "--revision",
        default="main",
        help="Branch/tag/commit of --repo-url to run the suite against. Default: main.",
    )
    p_init_ci.add_argument(
        "--results-repo-url",
        default=None,
        help="HTTPS repo URL to publish CI results into. Default: same as --repo-url.",
    )
    p_init_ci.add_argument(
        "--results-branch",
        default="ci-results",
        help="Branch of --results-repo-url that CI result artifacts get committed onto. Default: ci-results.",
    )
    p_init_ci.add_argument(
        "--task-type",
        default="rag_qa",
        help=(
            "Which task type's generated test file this PipelineRun should execute. "
            "Only used to derive the default --test-file value (test_elyaeval_<task_type>.py) "
            "— has no effect if --test-file is passed explicitly. Default: rag_qa, since "
            "that's the only task type currently wired into the shared recipe."
        ),
    )
    p_init_ci.add_argument(
        "--test-file",
        default=None,
        help=(
            "Path (relative to --repo-url's root) of the generated test file to run in CI. "
            "Default: test_elyaeval_<task_type>.py, matching elyaeval init's default naming. "
            "Pass this explicitly if you generated your test file with a custom --output."
        ),
    )
    p_init_ci.add_argument(
        "--output",
        default="pipelinerun.yaml",
        help="Path to write the generated PipelineRun. Default: pipelinerun.yaml",
    )

    args = parser.parse_args()

    if args.command == "init":
        output = args.output or _default_test_filename(args.task_type)
        out_path = init(args.task_type, args.ci_stage, output, args.corpus_mode)
        print(f"Wrote {out_path}")
        print("Next steps:")
        if args.task_type == "agentic_action":
            print(f"  1. Open {out_path} and fill in TODO 1, TODO 2 (import + run_app()), "
                  f"and TODO 3 (expected tools) if you're scoring against ground truth.")
            print(f"  2. Point GOLDENS at your own goldens file (dataset_path=...).")
            print(f"  3. Run: deepeval test run {out_path}")
        elif args.corpus_mode == "existing":
            print(f"  1. Open {out_path} and fill in TODO 1 and TODO 2 (import + run_app()).")
            print(f"  2. Point GOLDENS at your own goldens file (dataset_path=...).")
            print(f"  3. Run: deepeval test run {out_path}")
        else:
            print(f"  1. Open {out_path} and fill in TODO 1, TODO 2, and TODO 3 (ingest/teardown).")
            print(f"  2. Run: deepeval test run {out_path}")

    elif args.command == "init-ci":
        results_repo_url = args.results_repo_url or args.repo_url
        test_file = args.test_file or _default_test_filename(args.task_type)
        out_path = init_ci(
            repo_url=args.repo_url,
            revision=args.revision,
            results_repo_url=results_repo_url,
            results_branch=args.results_branch,
            output=args.output,
            task_type=args.task_type,
            test_file=args.test_file,
        )
        print(f"Wrote {out_path}")
        print("Next steps:")
        print("  1. Confirm the elyaeval-rag-recipe Pipeline/Task set is applied to your target cluster")
        print("     (from the elyaeval repo's recipes/tekton/ — this file does not generate those).")
        print(f"  2. Confirm {test_file} exists in this repo at the path set for test-file in {out_path}")
        print(f"     (it must match whatever `elyaeval init` actually wrote — rename either side if not).")
        print(f"  3. Commit {out_path} to this repo.")
        print("  4. Set $SUT_URL / $JUDGE_MODEL_NAME / $JUDGE_BASE_URL and apply — see comments in the file.")


if __name__ == "__main__":
    main()