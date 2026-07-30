"""
Generic golden-context ingestion fixture.

Retrieval-dependent metrics (ContextualPrecision/Recall/Relevancy,
Faithfulness) can only score meaningfully if the app under test has
something relevant to retrieve. The standard dataset ships each golden's
ground-truth `context` passages precisely so any app can seed itself with
them before running — see load_standard_dataset()'s module docstring.

This module owns everything that is the SAME for every consuming app:
looping goldens, deduping context passages, and the pytest fixture
lifecycle (ingest before the session, teardown after). It knows nothing
about HOW to ingest or tear down, or what your storage looks like —
every app's backend is different (pgvector, Chroma, FAISS, an HTTP
endpoint, ...) — so that part is supplied by the app as two callables.

Contract: ingest_fn(passage) inserts one passage and returns whatever
handle your storage needs to delete it again later — a SQL row id, a
Chroma document id, a vector index position, a dict key, anything your
backend can address. teardown_fn(handles) receives the exact list of
handles that were returned and removes exactly those, nothing else.

This deliberately does NOT assume your storage has a queryable metadata
field (e.g. a "source" column) to tag and filter on — some backends
don't offer one in a comparable shape. Identifier-based delete is a
weaker, more general assumption: almost any store can address and
remove what it was just asked to insert.
"""
from typing import Any, Callable, Optional

import pytest
from deepeval.dataset import Golden

#factory function: you feed it your goldens and your app's specific insert/delete logic,
#  and it hands you back a ready-made pytest fixture shaped for your app
def make_golden_context_fixture(
    goldens: list[Golden],
    ingest_fn: Callable[[str], Any],
    teardown_fn: Optional[Callable[[list[Any]], None]] = None,
    #how long the fixture lives for (default: once per test session)
    scope: str = "session",
):
    """
    Returns a pytest fixture (autouse, given `scope`) that:
      1. Collects every unique `context` passage across `goldens`.
      2. Calls `ingest_fn(passage)` once per unique passage before the suite
         runs, collecting each returned handle.
      3. Calls `teardown_fn(handles)` once after the suite finishes, passing
         the full list of handles collected in step 2 — if provided.

    ingest_fn / teardown_fn are the only app-specific pieces you write.

    Skip this fixture only if the app under test does NOT retrieve from a
    corpus to produce actual_output (e.g. a stateless single-shot LLM call
    with no external knowledge base). For any RAG app, this fixture matters
    regardless of which metrics you're scoring with: a domain mismatch
    between the goldens and the app's corpus produces a garbage
    actual_output at generation time, and every metric that reads
    actual_output — including ones that never touch retrieval_context, like
    AnswerRelevancy or a summarization/correctness GEval — inherits that
    garbage. The metric preset tells you what's being SCORED; it says
    nothing about whether the app's own retrieval step needs seeding.
    """
    #set automatically drops duplicates
    seen: set[str] = set()
    for golden in goldens:
        for passage in golden.context or []:
            seen.add(passage)

    @pytest.fixture(scope=scope, autouse=True)
    def _golden_context_fixture():
        handles = [ingest_fn(passage) for passage in seen]
        yield
        if teardown_fn is not None:
            teardown_fn(handles)

    return _golden_context_fixture