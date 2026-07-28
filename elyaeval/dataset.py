"""
Loads the standard dataset (goldens.jsonl, versioned and shipped inside this
package and returns deepeval Golden objects, filtered by task_type and/or ci_stage.

IMPORTANT — retrieval_context is intentionally dropped here, even if a row
in goldens.jsonl happens to contain one. The standard dataset is shared
across every user's app; retrieval_context is what a specific app's
retriever actually returned for a given input, which differs per app and
can only be known by calling that app. It is filled in by the generated
test file (see templates/test_template.py.tmpl), never by this loader.
`context` (the ideal/ground-truth passages) is the one retrieval-shaped
field that *is* safe to ship pre-filled, since it does not depend on any
particular app's retriever.
"""

import json
from importlib import resources
from pathlib import Path
from typing import Optional

from deepeval.dataset import Golden

_PACKAGE_DATASET = "standard_dataset/goldens.jsonl"


def _default_dataset_path() -> Path:
    return resources.files("elyaeval") / _PACKAGE_DATASET


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
    """
    path = Path(dataset_path) if dataset_path else _default_dataset_path()

    goldens: list[Golden] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed row {line_no} in {path}: {e}")
                continue

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
