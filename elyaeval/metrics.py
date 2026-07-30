"""
Shared metric presets, one list per use case.

Editing thresholds or swapping a metric in/out means editing
this list.

Every metric below defaults to whatever judge model DeepEval is configured
to use (env var DEEPEVAL_MODEL / --model on the CLI, or the ollama model
set via `deepeval set-local-model` during the current build phase). Do not
hardcode a model= here — Phase 6 swaps the judge model project-wide, and a
hardcoded model per preset would need to be edited in N places instead of one.
"""

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    SummarizationMetric,
    GEval,
)
from deepeval.test_case import SingleTurnParams

# --- rag_qa -----------------------------------------------------------
# End-to-end RAG: generation quality (answer_relevancy, faithfulness) +
# retriever quality (the three contextual metrics). All five read
# retrieval_context, which callers must populate themselves at test time —
# see dataset.py docstring for why it is never pre-filled in the standard
# dataset.
RAG_METRICS = [
    AnswerRelevancyMetric(threshold=0.7),
    FaithfulnessMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualRelevancyMetric(threshold=0.7),
]

# --- retrieval-only -----------------------------------------------------
# Component-level: scores the retriever in isolation, assuming
# no actual_output is available yet — only input + retrieval_context.
RETRIEVAL_METRICS = [
    ContextualPrecisionMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualRelevancyMetric(threshold=0.7),
]

# --- generation-only --------------------------------------------------
# Component-level: scores the generator in isolation, given whatever
# retrieval_context it was actually handed.
GENERATION_METRICS = [
    AnswerRelevancyMetric(threshold=0.7),
    FaithfulnessMetric(threshold=0.7),
]

# --- summarization --------------------------------------------------------
SUMMARIZATION_METRICS = [
    SummarizationMetric(threshold=0.7),
]

# --- qa_correctness ---------------------------------------------------
# Plain QA (no retrieval leg to score) — correctness via G-Eval, referencing
# expected_output as the rubric's ground truth.
QA_CORRECTNESS_METRICS = [
    GEval(
        name="Correctness",
        criteria="Determine whether the actual output is factually correct and "
                 "consistent with the expected output. Minor differences in "
                 "phrasing are acceptable; differences in facts are not.",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.7,
    ),
]
