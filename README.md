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

Three outputs, written per golden as the suite runs (not just at the end), plus one written once
at the very end:

- **`junit.xml`** — plain pass/fail, what CI gates on.
- **`report/results_<task_type>_<UTC timestamp>[_component].csv`** — one row per `(golden, metric)`
  (plus `span_name` for component mode): `golden_id, priority, input, metric_name, score, threshold,
  success, reason, error, evaluation_model, evaluation_cost`. Filter/sort this to find e.g. every
  Faithfulness failure without reading raw pytest output. `new_report_path()` builds the filename once
  per session, so every run gets its own timestamped file automatically — nothing to clean up between
  runs, and nothing from an old run gets silently appended to or overwritten by a new one.
- **`report/results_<task_type>_<UTC timestamp>[_component].html`** — same data as the CSV, rendered
  as a self-contained HTML file (inline CSS, no CDN calls — opens standalone, including straight off
  a blob-storage download). Two tables: a **per-metric summary** (average/min/max score, pass rate,
  threshold, computed across every golden that ran) and the full **per-golden detail** table, with
  pass/fail colored and a score bar per row. This is what answers "what did Faithfulness average
  across the whole run, not just per input" — see [Per-metric averages](#per-metric-averages) below.
  Written automatically at the end of the pytest session by elyaeval's own pytest plugin (auto-registered
  via `pyproject.toml`'s `pytest11` entry point the moment `elyaeval` is installed) — no change needed
  in any generated test file to get it.
- **`DEEPEVAL_RESULTS_FOLDER`** (opt-in, set the env var) — DeepEval's own structured JSON
  (score/reason/cost per metric per test case), useful for cost-budget tracking later; not required
  to run the suite.

The plugin also prints the same per-metric average table to the terminal right after pytest's own
summary, so you see it locally without opening the CSV or HTML at all:

```
==================== elyaeval report ====================
[elyaeval] report/results_rag_qa_20260825T130000Z.csv -> results_rag_qa_20260825T130000Z.html
metric              avg     min     max        pass  n
-------------------------------------------------------
Faithfulness       0.74    0.55    0.92         1/2  2
AnswerRelevancy    0.84    0.81    0.88         2/2  2
```

### Per-metric averages

`elyaeval.metric_averages(rows)` (also importable as `from elyaeval import metric_averages`) groups
CSV rows by `metric_name` (or pass `group_keys=("span_name", "metric_name")` for a component CSV, so
the same metric scored on two different spans doesn't get blended into one number) and returns, per
group: `n`, `n_scored`, `avg_score`, `min_score`, `max_score`, `pass_count`, `pass_total`,
`pass_rate`, `threshold` (only populated if every row in the group shares one), `total_cost`,
`cost_n`, and `evaluation_model` (see [Judge cost tracking](#judge-cost-tracking) below). Rows with a
blank score — a metric that errored rather than scored — are excluded from the average rather than
counted as 0, but are still counted in `n`/`pass_total`.

Use it directly if you want the numbers in code (e.g. to fail a build on a metric's *average*
dropping below some bar, not just individual golden failures):

```python
from elyaeval import read_csv_rows, metric_averages

rows = read_csv_rows("report/results_rag_qa_20260825T130000Z.csv")
for summary in metric_averages(rows):
    print(summary["metric_name"], summary["avg_score"], summary["pass_rate"])
```

### Judge cost tracking

Every CSV/HTML row also carries `evaluation_model` and `evaluation_cost`, straight off DeepEval's own
`MetricData` (the same numbers DeepEval sums into its own terminal output's `token cost: $X USD`
line). `evaluation_cost` is a USD amount, not a raw token count — DeepEval only tracks per-token cost
on the public API, and only for judge models it has pricing built in for (OpenAI, Anthropic, etc.). A
local/custom judge with no pricing config leaves this blank, same as DeepEval's own terminal output
shows `token cost: None` rather than `$0` — the HTML/CSV/terminal summary here all follow the same
convention: **blank/`—` means "unknown," not "free."** Nothing to configure — this is populated
automatically the moment you point `elyaeval init` at a judge model DeepEval has pricing for; a run
under your current local judge just shows `—` everywhere cost would go, and the per-metric/per-run
totals will start populating with no further changes the moment that switches.

The per-metric summary table (HTML and terminal) shows a `total_cost` per metric group, and
`elyaeval.total_cost(rows)` gives the single across-the-whole-run number (same `—`-if-unknown rule):

```python
from elyaeval import read_csv_rows, total_cost

rows = read_csv_rows("report/results_rag_qa_20260825T130000Z.csv")
print(total_cost(rows))  # None if every metric's judge is unpriced, else summed USD
```

### Regenerating a report from an existing CSV

For a CSV from before this feature existed (e.g. one already sitting in blob storage), or to
regenerate on demand after downloading one locally:

```bash
elyaeval report --csv report/results_rag_qa_20260825T130000Z.csv
# component CSVs need span_name in the grouping:
elyaeval report --csv report/results_rag_qa_component_20260825T130500Z.csv --group-by span_name,metric_name
```

Writes `<csv-stem>.html` next to the CSV (or pass `--html` for a different path) and prints the same
averages table.

## Regression testing (across model/prompt changes)

Compares two runs' scores against each other, using nothing but two CSVs elyaeval already produces —
no Confident AI account needed.

### `elyaeval compare`

```bash
elyaeval compare --baseline report/results_rag_qa_20260820T090000Z.csv \
                  --candidate report/results_rag_qa_20260826T090000Z.csv
```

Diffs per-metric averages (`metric_averages()` under the hood) between the two CSVs and prints a
table:

```
Baseline:  report/results_rag_qa_20260820T090000Z.csv
Candidate: report/results_rag_qa_20260826T090000Z.csv
Tolerance: ±0.02

metric            baseline  candidate     delta  status
------------------------------------------------------------
Faithfulness          0.88       0.65    -0.225  ▼ REGRESSED
AnswerRelevancy       0.86       0.86    +0.000  = unchanged

1 metric(s) regressed beyond tolerance.
```

- `--tolerance` (default `0.02`): absolute avg-score drop that counts as a regression. A metric
  landing within tolerance is `unchanged`, not `improved`/`regressed` — a ±0.001 wobble from LLM-judge
  noise shouldn't read as either.
- `--group-by` (default `metric_name`; use `span_name,metric_name` for component CSVs): same grouping
  `metric_averages()`/`elyaeval report` use.
- A metric only present in one of the two runs is flagged `new` or `removed`, not `regressed` —
  something appearing or disappearing between runs isn't the same claim as an existing metric
  scoring worse, and deserves its own callout (did a pipeline stage get added/dropped? did the suite
  change?) rather than being silently absorbed into the regression count.
- `--html <path>` also writes an HTML regression report — the same comparison table as a page you
  can open directly.
- **Exit code is 0 if nothing regressed beyond tolerance, 1 if anything did** — this is what
  `check-regression` (below) gates the pipeline on directly.

## CI/CD (Tekton)

`elyaeval init-ci` generates the per-repo `PipelineRun` stub that runs your generated suite against
the shared `elyaeval-rag-recipe` Pipeline:

```bash
elyaeval init-ci --repo-url https://github.com/you/your-sut.git --blob-container your-project
```

The Pipeline/Task definitions themselves are centralized, not generated per repo — see
[`recipes/tekton/README.md`](recipes/tekton/README.md) for cluster setup, secrets, and how to apply them.

`report-results` uploads every `results_*.csv` under `report/` as its own blob, named
`<run-id>_<suffix>.csv` (suffix = the CSV filename minus its `results_` prefix, so the UTC timestamp
from `new_report_path()` carries straight through into the blob name). It uploads each CSV's
`results_*.html` sibling the same way, as `<run-id>_<suffix>.html`, if the HTML file exists —
older runs with no HTML sibling are skipped silently, not treated as an error.

### Two independent gates

The pipeline fails a run for either of two different reasons, enforced by two different `finally`
tasks:

- **`check-results`** — gates on each individual metric's threshold, same run, no history involved.
  Already existed: DeepEval marks a test `failed` when a score misses its threshold, `check-results`
  reads that out of `junit.xml`.
- **`check-regression`** (new) — gates on this run's per-metric *averages* against a **baseline**
  from a previous run, via `elyaeval compare` under the hood. A run can clear every metric's
  threshold and still fail this gate if it scored meaningfully worse than the baseline it's compared
  against — that's the "regression across model/prompt changes" case threshold-gating alone can't
  catch (nothing dropped below threshold, but everything got worse).

Both are ordinary `finally` tasks — either one failing fails the `PipelineRun` overall, independent
of the other.

### How the baseline works

`check-regression` compares against a **stable, non-timestamped blob**, `baseline_<task-type>.csv`
(e.g. `baseline_rag_qa.csv`, `baseline_rag_qa_component.csv` — one per test-file "shape", derived
automatically from the CSV filename, same container as everything else). This is deliberately
**self-gating, no separate on/off switch needed**:

- No baseline blob yet → every run just logs `No baseline blob 'baseline_rag_qa.csv' yet` and passes.
  Adding this Task to an existing pipeline doesn't fail anyone's very first run after upgrading.
- A baseline exists → every subsequent run compares against it automatically.
- Nothing promotes (overwrites) the baseline unless a run explicitly asks to, via the
  `promote-baseline` param (default `"false"`) — see below.

Set `promote-baseline: "true"` only on `PipelineRun`s you deliberately want to become the new
known-good state (a main-branch run, a nightly run against a stable SUT deployment — whatever your
pipeline trigger setup treats as "this passed, trust it"). Leave it `"false"` for ordinary/PR runs,
so a single PR can never silently move the baseline every other PR compares against. **Promotion only
happens if the comparison itself passed** — `check-regression`'s three steps
(`download-baseline` → `compare` → `promote-baseline`) run in that fixed order within the same Task,
and Tekton skips later steps once one fails, so a regressed run never reaches the promote step
regardless of what `promote-baseline` is set to.

```yaml
# in your pipelinerun.yaml, or via --param on a manual `tkn pipeline start`:
- name: regression-tolerance
  value: "0.02"      # default — override if a metric's judge is noisier/stricter than that
- name: promote-baseline
  value: "false"     # "true" only for runs that should become the new baseline
```

The very first time you want a baseline to exist at all, run once with `promote-baseline: "true"` —
`check-regression` bootstraps it from that run's CSV since there's nothing to compare against yet.