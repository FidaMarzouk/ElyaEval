# elyaeval — starter kit

Not a wrapper framework. This package gives you three things:
1. The standard dataset (`elyaeval/standard_dataset/goldens.jsonl`), versioned and bundled.
2. Metric presets per use case, as plain lists (`elyaeval/metrics.py`).
3. `elyaeval init`, which generates a filled-in pytest file that calls DeepEval's real API directly.

## Install

```bash
pip install -e .
```

## Configure a judge model (once)

Metric objects (`RAG_METRICS`, etc.) initialize a judge model the moment they're constructed —
so this has to happen before you *run* a suite, though not before you *generate* one.

Local (current build/test phase):
```bash
deepeval set-local-model --model-name=llama3.1 --base-url="http://localhost:11434/v1" --api-key="ollama"
```

Hosted (Anthropic, going forward):
```bash
export OPENAI_API_KEY=... 
```

## Generate a test file

```bash
elyaeval init --task-type rag_qa
```

Task types available: `rag_qa`, `retrieval`, `summarization`, `qa_correctness`.
Add `--ci-stage per_commit|merge|nightly|release` as a run-frequency filtering
(default: `nightly`). If the combination matches 0 goldens, `init` warns you immediately instead
of generating a file that pytest silently skips.

### Corpus mode: `existing` (default) vs `seeded`

`--corpus-mode existing` (the default) assumes your app's corpus already has
content relevant to the goldens you're testing with. The generated suite just
calls your app and lets it retrieve for real — nothing is inserted into or
deleted from your storage. Point `GOLDENS` at your own goldens file (same
JSONL shape as the standard dataset — see `elyaeval/dataset.py`) written
against what's actually in your corpus; the bundled standard dataset is a
generic cross-app benchmark and won't match your content.

`--corpus-mode seeded` is for the opposite situation: you want to run the
*shared* standard dataset against an app whose corpus doesn't already cover
it. The generated suite temporarily inserts each golden's `context` passages
before the run and removes them after (`elyaeval.fixtures.make_golden_context_fixture`)
— you fill in `_ingest_passage`/`_teardown_passages` for your storage backend.
Reach for this only when you actually need the shared dataset; it's extra
moving parts (mutating storage, cleanup) that existing-mode doesn't need.

```bash
elyaeval init --task-type rag_qa --corpus-mode seeded
```

## Fill in the TODOs

Open the generated file and:
1. Import your app's entrypoint.
2. Implement `run_app(user_input) -> (actual_output, retrieval_context)` to call it.
3. (seeded mode only) Implement `_ingest_passage` / `_teardown_passages` for your storage.

`retrieval_context` is never pre-filled by ElyaEval — it's what *your* retriever actually
fetched, which only your app knows. See `elyaeval/dataset.py` for why.

## Run it

```bash
deepeval test run test_elyaeval_rag_qa.py
```
