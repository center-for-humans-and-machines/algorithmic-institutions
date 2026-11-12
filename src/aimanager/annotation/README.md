# Step classification annotation

LLM coding of the policy-finder search: what each rule does and why the notes say it does it. Spec: `docs/plans/step-classification-pipeline.md`. Config: `configs/annotation/default.yml` (merge overrides with `--config`). Prompts: `prompts/*.md`.

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
$A stage A --run R --base-url URL       # 1a open-rules, 1b open-notes, 2a what-codebook,
                                        # 2b coverage, 3 why-codebook
$A review status --run R                # current codebook revisions
$A stage B --run R --base-url URL       # 5a apply-rules, 5b apply-notes
$A expert-check export --run R          # expert_check/sheets/<stem>.yml
$A expert-check ingest --run R          # reads expert_check/filled/*.yml
$A join --run R                         # analysis/version_codes.csv, quality.json
```

Each step can also be run alone (`$A coverage --run R ...`). Steps are idempotent: every LLM reply is cached under `cache/` keyed on model, messages, schema and sampling parameters, so a rerun repeats only missing or failed calls. Every call is logged in `logs/calls.jsonl`.

## Review

A reviewer writes the next codebook revision next to the current one (`codebook/what.v2.yml`, same format) plus `review/comments.md`. Stage B always uses the highest revision of each codebook.

## Run directory

- `corpus/versions.jsonl`, `corpus/split.json`
- `open/rule_free_codes.jsonl`, `records/<stem>.json`
- `codebook/what.draft.yml` (from rules), `what.v1.yml` (plus implemented notes candidates), `why.v1.yml`, `never_implemented.yml`, `*.members.yml` (item ids per code, for auditing the clustering), `what_saturation.csv`
- `coverage/` (notes mechanisms against the draft, candidate codes and their rule judgements)
- `codes/rule_codes.jsonl` (LLM judge and expert rows, column `coder`), `records_coded/<stem>.json`
- `analysis/version_codes.csv`, `analysis/quality.json`, `analysis/agreement.csv`
