# Step classification annotation

LLM coding of the policy-finder search: what each rule does and why the notes say it does it. Spec: `docs/plans/step-classification-pipeline.md`. Config: `configs/annotation/default.yml` (merge overrides with `--config`; found relative to the package in a source tree, or named by `ANNOT_CONFIG`; a missing default config is an error). Prompts: `prompts/*.md`.

The package needs only the standard library, `pydantic` v2, `pyyaml` and `httpx`, and imports nothing else from `aimanager`, so it runs inside a vLLM container.

## Environment on the Mac

```bash
uv venv .venv-annot --python 3.9
VIRTUAL_ENV=.venv-annot uv pip install "pydantic>=2" pyyaml httpx pytest
```

Run everything with `PYTHONPATH=src .venv-annot/bin/python ...`; tests: `PYTHONPATH=src .venv-annot/bin/python -m pytest src/aimanager/tests/test_annot_*.py`.

## Commands

All write under `<runs_root>/<run_id>/` (`--runs-root` or `ANNOT_RUNS_ROOT`, default `runs/annotation`). LLM steps take `--base-url`/`ANNOT_BASE_URL`, `--model`/`ANNOT_MODEL` (default: first id of `GET /models`), `--backend fake` for a deterministic offline fake, and every step takes `--limit N`.

```bash
A="python -m aimanager.annotation"
$A corpus --run R                       # 0: versions.jsonl, split.json from git refs
$A preflight --run R --base-url URL     # one request per schema; exit 1 if any fails
$A stage A --run R --base-url URL       # 1a open-rules, 1b open-notes, 2a what-codebook,
                                        # 2b coverage, 3 why-codebook
$A review status --run R                # current codebook revisions and their hashes
$A review begin --kind what --run R     # copy the latest revision to the next one
$A stage B --run R --base-url URL       # 5a apply-rules, 5b apply-notes
$A expert-check export --run R          # expert_check/sheets/<stem>.yml
$A expert-check ingest --run R          # reads expert_check/filled/*.yml
$A join --run R                         # analysis/version_codes.csv, quality.json,
                                        # agreement.csv
```

Each step can also be run alone (`$A coverage --run R ...`). Every validated LLM reply (and every `length` reply) is cached under `cache/` keyed on model, messages, schema and sampling parameters, so a rerun repeats only missing or failed calls; invalid replies are never cached. Every call is logged in `logs/calls.jsonl`.

## Failures

A step that cannot produce valid outputs exits non-zero and writes nothing:

- **Failure budget** (`failure_budget` in the config): the share of calls of one pass kind (`open_rules`, `open_notes`, `assign`, `propose`, `judge`) that may fail in a step. Default 0 for proposals and judge calls, a few percent for assignments and per-document extraction. Beyond it the step raises before writing. Within it, failed items are never evidence: a code is dropped only if it would stay below `min_members` even with every failed item, a candidate is rejected as never implemented only if no judge call failed, and in `records_coded` a failed assignment has `what_code_status: failed` (versus `none`).
- **Dead or hung server**: 5xx, 429, connection errors and malformed 200 replies are retried with `backoff`; a read timeout (`read_timeout`) at most `timeout_retries` times. `max_consecutive_server_errors` failed attempts in a row abort the step and cancel the queued calls; so do `max_client_errors` HTTP 4xx replies, and any exception in a worker.
- **Thinking**: the request body under `llm.extra_body` is chosen by case-insensitive match on the model id and logged once per step. An empty reply with reasoning text aborts with a message naming that key. A leading `<think>...</think>` block is stripped before parsing and counted (`think_stripped` in the call log).

## Preflight

`preflight` sends one structured request per schema the pipeline uses (free codes, notes record, codebook proposal, assignment, rule judgement) with a short built-in sample, bypassing the cache, and prints per schema whether the reply parsed, the latency and whether reasoning text leaked. A cluster job runs it before the real stage; it exits 1 if any schema fails.

## Manifests and revisions

Every step writes `manifests/<step>.json` last: its inputs and outputs with sha256 hashes, expected and written counts, failed items, the codebook revisions it used (file, revision, sha256), the model and a timestamp. A step reads its inputs only through the producing step's manifest and fails if that step never finished, if one of its outputs changed since, or if one of its inputs changed since (an upstream rerun made it stale). Records are loaded exactly as listed in the manifest; files a rerun did not write are deleted.

A reviewer writes the next codebook revision next to the current one plus `review/comments.md`. `review begin --kind what` copies the latest revision to `what.v<k+1>.yml` with `based_on: {revision: k, sha256: ...}`; edit that file in place. Every revision above v1 must carry `based_on` matching the current previous revision, and v1 must be the file its step wrote, so a review of a v1 that a coverage rerun has since replaced is rejected. Stage B uses the highest revision. `join` fails unless apply-rules and apply-notes used the latest what and why revisions, and agreement pairs rater and judge rows only within one codebook revision. Code names `none` and `failed` are reserved.

## Prompts

Prompts are shared instructions (then the codebook, if any) followed by a per-document part: a context section with the campaign's game setting (`settings` in the config, copied into `corpus/versions.jsonl` as `setting` and recorded in the open-notes manifest) and, for notes, whether the version is the first of its lineage, then the document. The shared part is a byte-identical prefix across items. Examples in the prompts come from an unrelated domain (library rules for overdue books) so the first pass stays bottom-up. Reasons are assigned to why codes with their mechanism and stance as a labelled context line that is not coded.

## Run directory

- `corpus/versions.jsonl`, `corpus/split.json`
- `open/rule_free_codes.jsonl`, `records/<stem>.json`
- `codebook/what.draft.yml` (from rules), `what.v1.yml` (plus implemented notes candidates), `why.v1.yml`, `never_implemented.yml` (same shape as a codebook; notes ideas no non-held-out rule implements), `*.members.yml` (item ids per code, for auditing the clustering), `what_saturation.csv`
- `coverage/` (notes mechanisms against the draft, candidate codes and their judgements on the rules outside the expert check, `summary.json`)
- `codes/rule_codes.jsonl` (LLM judge and expert rows: `present`, `switchable_off`, `quote`, `coder`, `codebook_revision`), `records_coded/<stem>.json` (notes mechanisms coded against the latest what codebook plus the never-implemented codes)
- `analysis/version_codes.csv` (one row per version and code, never-implemented codes included with `code_source`), `analysis/quality.json`, `analysis/agreement.csv`
- `manifests/<step>.json`
