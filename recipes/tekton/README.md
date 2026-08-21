# Tekton CI recipe

Shared, centralized `Pipeline` + `Task` set (`pipeline.yaml`, `tasks.yaml`) that any SUT repo can
run its `elyaeval`-generated suite against. Applied to the cluster **once**, independently of any
one SUT repo — a repo only needs its own `PipelineRun` (see [Per-repo setup](#per-repo-setup) below).

## What the pipeline does

`fetch-source` → `setup-env` → `run-suite`, then, always (`finally`): `check-results` + `report-results`.

- **fetch-source**: clones the SUT repo (the one containing the generated `test_elyaeval_*.py`).
- **setup-env**: `pip install --target` the `elyaeval` package + pytest/deepeval/requests into the
  shared workspace — deliberately *not* the SUT's own `requirements.txt` (the test process only
  talks to the SUT over HTTP, so it never needs the SUT's dependencies).
- **run-suite**: runs the test file over HTTP against `sut-url`, wiring judge-model env vars for
  whichever `judge-provider` was chosen (`local` / `anthropic` / `openai` — see
  [Judge model](#judge-model) below).
- **check-results**: parses `junit.xml`; fails the `PipelineRun` on any test failure/error. This is
  the actual gate.
- **report-results**: uploads every `report/results_*.csv` produced to an Azure Blob Storage
  container, one blob per CSV, named `<run-id>_<suffix>.csv`. Runs regardless of whether
  `check-results` failed, so a failing run's scores are still uploaded.

## One-time cluster setup

```bash
kind create cluster --name elyaeval-ci
kubectl create namespace elyaeval-ci
kubectl apply -f pipeline.yaml -f tasks.yaml -n elyaeval-ci
```

Create the two shared Secrets (contents differ per judge provider / storage backend, but the
Secret names/keys stay fixed — see the per-field comments in `tasks.yaml` for exactly what each is used for):

```bash
kubectl create secret generic judge-model-credentials \
  --from-literal=api-key=$ANTHROPIC_API_KEY -n elyaeval-ci

kubectl create secret generic blob-storage-credentials \
  --from-literal=connection-string=$AZURE_STORAGE_CONNECTION_STRING -n elyaeval-ci
```

For local testing without a real Storage Account, run `docker compose up -d` in this repo's root
to start an Azurite emulator and point `blob-storage-credentials` at its connection string instead.

## Per-repo setup

Each SUT repo generates and commits its own `PipelineRun` — it is *not* part of this shared recipe:

```bash
elyaeval init-ci --repo-url https://github.com/you/your-sut.git --blob-container your-project
```

Then, before applying it:
1. Confirm the generated `test-file` value matches whatever `elyaeval init` actually wrote in that repo.
2. Set the judge/SUT env vars the generated file's own header comment lists (`SUT_URL`,
   `JUDGE_PROVIDER`, `JUDGE_MODEL_NAME`, `JUDGE_BASE_URL`).
3. Apply it:
   ```bash
   envsubst < pipelinerun.yaml | kubectl create -n elyaeval-ci -f -
   ```

## Judge model

`judge-provider` picks which env vars `run-suite` exports — see the `case` block in `tasks.yaml`'s
`run-pytest` step for the exact mapping. Summary:

| `judge-provider` | Needs |
|---|---|
| `local` (default) | `judge-model-name`, `judge-model-base-url`, `judge-model-format` — any OpenAI-compatible server (Ollama, vLLM, LM Studio, Groq's OpenAI-compat endpoint) |
| `anthropic` | `judge-model-name` only (e.g. `claude-sonnet-4-6`); `judge-model-base-url` is ignored |
| `openai` | `judge-model-name` only (e.g. `gpt-4.1`); `judge-model-base-url` is ignored |

`judge-api-key-secret-name`/`-key` are reused across all three — same Secret, only its contents
need to be a real key for `anthropic`/`openai` (for `local`/Ollama it just needs to exist).

## Reachability

`sut-url` and the local judge's base URL must be reachable **from inside the cluster**, not from
your host:
- Docker Desktop: `http://host.docker.internal:<port>`
- Plain Linux `kind` (no Docker Desktop): the kind bridge gateway IP —
  `docker network inspect kind -f '{{(index .IPAM.Config 0).Gateway}}'`

## Adding a new task type to this recipe

`test-file` defaults to `test_elyaeval_rag_qa.py` because `rag_qa` is the only task type currently
wired into this Pipeline. To run another task type (`agentic_action`, etc.), pass a different
`test-file` value on that repo's `PipelineRun` — no change needed to `pipeline.yaml`/`tasks.yaml`.
