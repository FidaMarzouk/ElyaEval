# elyaeval — starter kit

A DeepEval-based testing framework for LLM apps. It gives you:
1. **A standard dataset** (`elyaeval/standard_dataset/goldens.jsonl`) — a generic, versioned cross-app benchmark.
2. **Metric presets per use case** (`elyaeval/metrics.py`) — plain lists, no reference-app knowledge baked in.
3. **`elyaeval init`** — generates a filled-in pytest suite (e2e or component) that calls DeepEval's own API directly.
4. **`elyaeval init-ci`** — generates the per-repo Tekton `PipelineRun` that wires your suite into the shared CI recipe.

## Install

```bash
pip install -e .
# or, in another repo: pip install "elyaeval @ git+https://github.com/FidaMarzouk/ElyaEval.git@v0.1.1"
```

## Configure a judge model (once)

Metric objects (`RAG_METRICS`, etc.) initialize a judge model the moment they're constructed —
so this has to happen before you *run* a suite, though not before you *generate* one. Never
hardcode a model in `elyaeval/metrics.py`; it always reads whatever's configured below.

**Local** (Ollama, vLLM, LM Studio, or any OpenAI-compatible server — current default):
```bash
deepeval set-local-model --model-name=llama3.1 --base-url="http://localhost:11434/v1" --api-key="ollama"
```

**Anthropic / OpenAI** (env vars — same ones the Tekton `run-suite` Task sets per `judge-provider`, see `recipes/tekton/tasks.yaml`):
```bash
# Anthropic
export USE_ANTHROPIC_MODEL=1
export ANTHROPIC_MODEL_NAME=claude-sonnet-4-6
export ANTHROPIC_API_KEY=...

# OpenAI
export OPENAI_MODEL_NAME=gpt-4.1
export OPENAI_API_KEY=...
```

## Generate a test file

```bash
elyaeval init --task-type rag_qa
```

`--task-type`: `rag_qa`, `retrieval`, `summarization`, `qa_correctness`, `agentic_action`.
`--ci-stage per_commit|merge|nightly|release` filters which goldens are pulled in (default: `nightly`).
If the combination matches 0 goldens, `init` warns immediately instead of generating a silently-empty suite.

### `--eval-mode`: `e2e` (default) vs `component`

- **`e2e`**: `run_app()` calls your app over HTTP and produces one flat result per golden
  (`elyaeval.report`). Answers "is the final output still good?"
- **`component`**: imports your app's `@observe`'d entrypoint **in-process** and scores each
  traced span separately (`elyaeval.tracing_report`). Answers "which stage regressed?" — requires
  each span you want scored to declare its own `metrics=[...]` and call `update_current_span(test_case=...)`
  on the SUT side (see `retriever.py`/`generator.py` in the rag-demo case study for an example).
  Not applicable to `agentic_action` or `--corpus-mode` — both suites are meant to run side by side in CI,
  not replace one another.

```bash
elyaeval init --task-type rag_qa --eval-mode component
```

### `--corpus-mode`: `existing` (default) vs `seeded`

`existing` assumes your app's corpus already has content relevant to the goldens you're testing
with — the generated suite just calls your app and lets it retrieve for real; nothing is inserted
into or deleted from your storage. Point `GOLDENS` at your own goldens file — see
[Goldens file schema](#goldens-file-schema) below; the bundled standard dataset is a generic
cross-app benchmark and won't match your content.

`seeded` is for the opposite case: you want to run the *shared* standard dataset against an app
whose corpus doesn't already cover it. The generated suite temporarily inserts each golden's
`context` passages before the run and removes them after (`elyaeval.fixtures.make_golden_context_fixture`)
— you fill in `_ingest_passage`/`_teardown_passages` for your storage backend. Reach for this only
when you actually need the shared dataset. Not applicable to `agentic_action` or `--eval-mode component`.

```bash
elyaeval init --task-type rag_qa --corpus-mode seeded
```

## Goldens file schema

One JSON object per line (`.jsonl`). Every row, regardless of task type, needs:

| Field                          | Required | Notes |
|---                             |---       |---    |
| `input`                        | always   | the question/prompt |
| `additional_metadata.task_type`| always   | one of: `rag_qa`, `retrieval`, `summarization`, `qa_correctness`, `agentic_action` |
| `additional_metadata.ci_stage` | always   | `per_commit` / `merge` / `nightly` / `release` — controls which `--ci-stage` picks it up |

On top of that baseline, each `task_type` needs its own fields for its metrics to score anything
meaningful instead of erroring against an empty reference:

| task_type | Also requires | Why |
|---|---|---|
| `rag_qa` | `expected_output` | read by ContextualPrecision/Recall (against live `retrieval_context`) and Faithfulness's reference check |
| `retrieval` | `expected_output` | same as `rag_qa` — drives ContextualPrecision/Recall; ContextualRelevancy needs neither |
| `summarization` | *(none)* | SummarizationMetric is reference-free |
| `qa_correctness` | `expected_output` | compared against `actual_output` by the Correctness G-Eval rubric |
| `agentic_action` | *(none — `additional_metadata.expected_tools` is optional, opt-in)* | see `elyaeval/templates/test_template_agentic.py.tmpl` |

`context` is optional and not required by any task_type — no metric preset here reads it. The one
place it IS used is `--corpus-mode seeded`, above: `make_golden_context_fixture` ingests each
golden's `context` passages into your storage before the run. If you're using seeded mode, your
goldens need `context`; for `existing` mode (the default), leave it out. It's also never handed to
your generator: `generate_answer()` is always called with the **live** `retrieval_context` your app
just retrieved, never with a goldens-file field. That's intentional — `GENERATION_METRICS`
(Faithfulness + AnswerRelevancy) is meant to catch hallucination/off-topic answers under real
retrieval conditions, good or bad, not to score the generator against a hand-picked "ideal" chunk set.

`retrieval_context` is never pre-filled by ElyaEval either — only your own retriever, at test
time, knows what it fetched.

This table mirrors `elyaeval/dataset.py`'s `GOLDEN_SCHEMA`, the single source of truth.
`load_standard_dataset()` validates your **whole file upfront** against it: every row is checked,
every problem found (missing field, unknown `task_type`, invalid JSON) is collected, and a single
`GoldenValidationError` listing all of them is raised — not just the first one.

## Fill in the TODOs

Open the generated file and search for `TODO`:
1. Import your app's (or agent's) entrypoint.
2. Implement `run_app()` — HTTP or in-process — to call it and return the typed result
   (`RAGRunResult` / `AgenticRunResult`, see `elyaeval/run_contracts.py`), or `call_sut()` for
   `--eval-mode component`.
3. `--corpus-mode seeded` only: implement `_ingest_passage`/`_teardown_passages` for your storage.
   `agentic_action` only: adjust `_expected_tools_from_golden()` if your goldens encode ground-truth tools differently.

## Run it

```bash
deepeval test run test_elyaeval_rag_qa.py
```

Each golden runs through one `evaluate()` call (never the whole `GOLDENS` list batched together),
so pass/fail and per-metric scores stay traceable to a single golden.

## Reading the results

Two outputs, both written per golden as the suite runs (not just at the end):
- **`junit.xml`** — plain pass/fail, what CI gates on.
- **`report/results_<task_type>[_component].csv`** — one row per `(golden, metric)` (plus
  `span_name` for component mode): `golden_id, priority, input, metric_name, score, threshold,
  success, reason, error`. Filter/sort this to find e.g. every Faithfulness failure without
  reading raw pytest output.

Set `DEEPEVAL_RESULTS_FOLDER` to also get DeepEval's own structured JSON (score/reason/cost per
metric per test case) — useful for cost-budget tracking later; not required to run the suite.

## CI/CD (Tekton)

`elyaeval init-ci` generates the per-repo `PipelineRun` stub that runs your generated suite against
the shared `elyaeval-rag-recipe` Pipeline:

```bash
elyaeval init-ci --repo-url https://github.com/you/your-sut.git --blob-container your-project
```

The Pipeline/Task definitions themselves are centralized, not generated per repo — see
[`recipes/tekton/README.md`](recipes/tekton/README.md) for cluster setup, secrets, and how to apply them.