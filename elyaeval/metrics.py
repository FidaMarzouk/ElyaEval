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
    ToolCorrectnessMetric,
    ArgumentCorrectnessMetric,
    TaskCompletionMetric,
    StepEfficiencyMetric,
    PlanQualityMetric,
    PlanAdherenceMetric,
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

# --- agentic: action layer (component-level) ---------------------------
# Test-case-based, same as every list above — takes a single LLMTestCase
# with tools_called (and optionally expected_tools/available_tools) and
# runs through the existing evaluate_golden() / evaluate(test_cases=[...])
# path unchanged. Wired to task_type="agentic_action" in cli.py and
# elyaeval/templates/test_template_agentic.py.tmpl.
# See elyaeval.run_contracts.AgenticRunResult for what run_app() must return.
AGENTIC_ACTION_METRICS = [
    ToolCorrectnessMetric(threshold=0.7),
    ArgumentCorrectnessMetric(threshold=0.7),
]

# --- agentic: trajectory layer (trace-level) ----------------------------
# NOT wired to any template or CLI task_type yet, and cannot be dropped
# into evaluate_golden() as-is. All four are trace-only metrics: DeepEval
# requires them to run against a full @observe trace via evals_iterator or
# an @observe(metrics=[...]) decorator — they do not accept a plain
# LLMTestCase, so there is no golden-by-golden "run_app() returns a tuple"
# story for them the way every other preset in this file has. Defined here
# so the metric choice and thresholds live in one place when the tracing
# runner gets built; do not reference this constant from cli.py or a
# template until that exists.
AGENTIC_TRAJECTORY_METRICS = [
    TaskCompletionMetric(threshold=0.7),
    StepEfficiencyMetric(threshold=0.7),
    PlanQualityMetric(threshold=0.7),
    PlanAdherenceMetric(threshold=0.7),
]