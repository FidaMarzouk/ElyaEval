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

Local (current build/test phase, per Section 8):
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

## Fill in the two TODOs

Open the generated file and:
1. Import your app's entrypoint.
2. Implement `run_app(user_input) -> (actual_output, retrieval_context)` to call it.

`retrieval_context` is never pre-filled by ElyaEval — it's what *your* retriever actually
fetched, which only your app knows. See `elyaeval/dataset.py` for why.

## Run it

```bash
deepeval test run test_elyaeval_rag_qa.py
```
