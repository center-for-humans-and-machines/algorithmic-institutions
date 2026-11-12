# Step classification annotation

LLM coding of the policy-finder search: what each rule does and why the notes say it does it. Spec: `docs/plans/step-classification-pipeline.md`. Config: `configs/annotation/default.yml` (merge overrides with `--config`; found relative to the package in a source tree, or named by `ANNOT_CONFIG`; a missing default config is an error). Prompts: `prompts/*.md`.

The package needs only the standard library, `pydantic` v2, `pyyaml` and `httpx`, and imports nothing else from `aimanager`, so it runs inside a vLLM container.

## Environment on the Mac

```bash
uv venv .venv-annot --python 3.9
VIRTUAL_ENV=.venv-annot uv pip install "pydantic>=2" pyyaml httpx pytest
# for `analyse` only (Mac): imported lazily, never needed in the container
VIRTUAL_ENV=.venv-annot uv pip install pandas matplotlib
```

Run everything with `PYTHONPATH=src .venv-annot/bin/python ...`; tests: `PYTHONPATH=src .venv-annot/bin/python -m pytest src/aimanager/tests/test_annot_*.py`.

## Commands

All write under `<runs_root>/<run_id>/` (`--runs-root` or `ANNOT_RUNS_ROOT`, default `runs/annotation`). LLM steps take `--base-url`/`ANNOT_BASE_URL`, `--model`/`ANNOT_MODEL` (default: first id of `GET /models`), `--backend fake` for a deterministic offline fake, and every step takes `--limit N`.

```bash
A="python -m aimanager.annotation"
$A corpus --run R                       # 0: versions.jsonl, split.json from git refs
$A preflight --run R --base-url URL     # one request per schema; exit 1 if any fails
$A stage A --run R --base-url URL       # 1a open-rules, 1b open-notes, 2a what-codebook,
                                        # 2b coverage, 2c consolidate-what,
                                        # 3a why-codebook, 3b consolidate-why,
                                        # 3c basis-codebook, 3d consolidate-basis
$A review status --run R                # current codebook revisions and their hashes
$A review begin --kind what --run R     # copy the latest revision to the next one
                                        # (--kind what|why|basis|never_implemented)
$A stage B --run R --base-url URL       # 5a apply-rules (judge repeats), 5b apply-notes
$A expert-check export --run R          # expert_check/sheets/<stem>.yml
$A expert-check ingest --run R          # reads expert_check/filled/*.yml
$A join --run R                         # analysis/version_codes.csv, quality.json,
                                        # agreement.csv, theory_mentions.csv
$A analyse --run R                      # 7 (Mac): analysis tables, figures/,
                                        # summary.json, REPORT.md
```

Proposal prompts hold at most `clustering.max_items_in_proposal` items and only as many as fit the context length (`llm.max_model_len`, or `ANNOT_MAX_MODEL_LEN` from the job) next to `max_tokens.propose.retry`; the rest are sampled out with a seed and the step logs how many. Each step can also be run alone (`$A coverage --run R ...`). Every validated LLM reply (and every `length` reply) is cached under `cache/` keyed on model, messages, schema and sampling parameters, so a rerun repeats only missing or failed calls; invalid replies are never cached. Every call is logged in `logs/calls.jsonl`.

## Codebooks

- **Positive ideas.** Each notes mechanism carries `idea`, the mechanism stated positively; clustering, coverage and assignment use it, never the description, which may be negated for a rejected idea. Proposed what codes (mechanisms, including coverage candidates) whose name, label or definition starts with a negation (do not, avoid, ignore, never, no, ...), or whose name centres on one, are rejected and logged (`rejected_negated` in the history files). There is no such guard for why and basis codes: reasons and kinds of support are often legitimately negative ("late punishment has no future benefit", "no data available"). What codes are defined by the lever (what the punishment depends on), not by the direction or size of the effect.
- **Level and size.** A what code is a lever a manager can use, general enough that several rules could implement it; a why code is a kind of justification that could be given for several mechanisms; a basis code is a kind of support. The prompts ask for about `codebook.<kind>.target_codes` codes in total (settings: what 30, why 25, basis 10; coverage candidates 15), merging rather than adding; refinement rounds are told how many codes exist and may only add codes for ideas recurring in at least `max(min_members, 2)` leftover items. The schema cap of 60 codes per reply is a safety net only.
- **Generation sampling.** Extraction, assignment and judge calls run at temperature 0; proposals and consolidation at `llm.temperature_generate` (0.6) with `llm.seed_generate`, both sent with the request (so in the cache key) and recorded in the manifests (`generation_sampling`). Repeated names in one proposal reply are dropped (first kept, logged); a reply that repeats a name more than `clustering.max_name_repeats` times or fills the cap with repeats is re-asked once with seed + 1 (`degenerate` in the history files).
- **Coverage.** Notes ideas that match no draft code are clustered into candidates, which are judged in one judge call with the draft codes on the rules outside the expert check. A candidate is accepted if at least one and at most `coverage.max_prevalence` of the judged rules implement it; above that it goes to `codebook/non_discriminating.yml` with its prevalence, at zero to the never-implemented list.
- **Consolidation and families.** `consolidate-what` (draft + accepted + never-implemented codes) and `consolidate-why` make one call each that merges codes that are one code and puts every surviving code into a family derived from the codes (about 5-10, mutually exclusive; for why, by kind of justification). A merge is allowed only when the codes describe the same thing: anything that fits one necessarily fits the other and the other way round; each states its `reason`. Unknown or already merged names are skipped. For what codes, a merge that involves a negated or excluding code (`exclude_group_size_from_punishment`) is not applied, and a negated rewritten definition keeps the old one. Families should have two or more codes, and no catch-all family is allowed. Every code must be placed: codes left out are re-asked once against the existing families (placement schema), and only then get a family of their own. All of it, with the reasons, is in `codebook/<kind>.merges.yml`; family sizes are in `quality.json`. If more than 1.5 times the target survive, a second pass merges codes within families at the code level (families kept, merges audited with `pass: 2`). The assignment is rerun so `*.v1.members.yml` follows the merges, and the items no code covers are written with their text to `codebook/<kind>.v1.unassigned.yml`. Prompts show codes under their families.
- **Never-implemented list.** It has revisions like the codebooks: `consolidate-what` writes `codebook/never_implemented.v1.yml`, a reviewer writes `.v2.yml` with `review begin --kind never_implemented`, and stage B and join use the highest revision. A run made before this (with only `codebook/never_implemented.yml`) is migrated on first use: v1 is a byte-identical copy, accepted against the hash consolidate-what recorded for the old name.
- **What the coders see.** Prompts show a codebook as family headings (with the family definition) over each code's name, label, definition and examples, each example its own list item. Theory candidates, theory links, review notes and sources are never shown to the judge or the assigners; they are carried to `analyse` only.
- **Basis.** `basis-codebook` clusters the reasons' `evidence` phrases (the pointer is in `evidence_ref`) into `basis.draft.yml`; `consolidate-basis` merges codes and gives families, which are the few kinds of support a reason rests on (codes about the content of a finding merge into the kind of support). Every basis item carries a labelled context line with the reason it supports (and its stance) and its `evidence_ref`, as reasons carry their mechanism. apply-notes assigns `basis_code`; join carries `basis_codes` and `basis_families`.

## Rule judge

Per code the judge returns `evidence` (verbatim line or empty), `present`, `switchable_off`, in that order. `present: true` with empty evidence is re-asked once (schema restricted to those codes); if it stays empty, the answer is `present: false, unsupported: true`. The judge runs `judge.repeats` times (default 2), repeat 1 in codebook order and the others in seeded orders (families shuffled as blocks); the final answer is the unanimous one, or that of a tie-break run in a further order. `analysis/judge_reliability.csv` (apply-rules) and `coverage/judge_reliability.csv` hold the test-retest agreement and kappa of repeats 1 and 2 per code and overall; `codes/judge_repeats.jsonl` every repeat. `quality.json` lists per code the share of `present` answers by repeat (one-sided splits show order effects) and `unreliable_codes` below `judge.min_agreement` (0.8); `version_codes.csv` marks events involving them (`event_unreliable_code`). The client submits the longest prompts of a pass first.

## Failures

A step that cannot produce valid outputs exits non-zero and writes nothing:

- **Failure budget** (`failure_budget` in the config): the share of calls of one pass kind (`open_rules`, `open_notes`, `assign`, `propose`, `judge`) that may fail in a step (also `consolidate`, default 0). Default 0 for proposals and judge calls, a few percent for assignments and per-document extraction. Beyond it the step raises before writing. Within it, failed items are never evidence: a code is dropped only if it would stay below `min_members` even with every failed item, a candidate is rejected as never implemented only if no judge call failed, and in `records_coded` a failed assignment has `what_code_status: failed` (versus `none`).
- **Dead or hung server**: 5xx, 429, connection errors and malformed 200 replies are retried with `backoff`; a read timeout (`read_timeout`) at most `timeout_retries` times. `max_consecutive_server_errors` failed attempts in a row abort the step and cancel the queued calls; so do `max_client_errors` HTTP 4xx replies, and any exception in a worker.
- **Thinking**: the request body under `llm.extra_body` is chosen by case-insensitive match on the model id and logged once per step. An empty reply with reasoning text aborts with a message naming that key. A leading `<think>...</think>` block is stripped before parsing and counted (`think_stripped` in the call log).

## Preflight

`preflight` sends one structured request per schema the pipeline uses (free codes, notes record, codebook proposal, assignment, rule judgement, consolidation) with a short built-in sample, bypassing the cache, and prints per schema whether the reply parsed, the latency and whether reasoning text leaked. A cluster job runs it before its command (unless `CFG_PIPELINE_PREFLIGHT=0`) and runs the command only if it passes; it exits 1 if any schema fails.

## Manifests and revisions

Every step writes `manifests/<step>.json` last: its inputs and outputs with sha256 hashes, expected and written counts, failed items, the codebook revisions it used (file, revision, sha256), the model and a timestamp. A step reads its inputs only through the producing step's manifest and fails if that step never finished, if one of its outputs changed since, or if one of its inputs changed since (an upstream rerun made it stale). Records are loaded exactly as listed in the manifest; files a rerun did not write are deleted.

A reviewer writes the next codebook revision next to the current one plus `review/comments.md`. `review begin --kind what` copies the latest revision to `what.v<k+1>.yml` with `based_on: {revision: k, sha256: ...}`; edit that file in place. Every revision above v1 must carry `based_on` matching the current previous revision, and v1 must be the file its step wrote, so a review of a v1 that a coverage rerun has since replaced is rejected. Stage B uses the highest revision. `join` fails unless apply-rules and apply-notes used the latest what, why and basis revisions, and agreement pairs rater and judge rows only within one codebook revision. Code names `none` and `failed` are reserved.

## Prompts

Prompts are shared instructions (then the codebook, if any) followed by a per-document part: a context section with the campaign's game setting (`settings` in the config, copied into `corpus/versions.jsonl` as `setting` and recorded in the open-notes manifest) and, for notes, whether the version is the first of its lineage, then the document. The shared part is a byte-identical prefix across items. Examples in the prompts come from an unrelated domain (library rules for overdue books) so the first pass stays bottom-up. Reasons are assigned to why codes with their mechanism and stance as a labelled context line that is not coded.

## Run directory

- `corpus/versions.jsonl`, `corpus/split.json`
- `open/rule_free_codes.jsonl`, `records/<stem>.json`
- `codebook/what.draft.yml` (from rules), `what.premerge.yml` and `never_implemented.premerge.yml` (coverage), `what.v1.yml` and `never_implemented.yml` (consolidated, with families), `non_discriminating.yml`, `why.draft.yml`, `why.v1.yml`, `basis.draft.yml`, `basis.v1.yml`, `*.merges.yml` (consolidation audit), `*.members.yml` (item ids per code, for auditing), `*.history.json` (proposal rounds, rejected negated codes), `what_saturation.csv`
- `coverage/` (notes ideas against the draft, candidate codes, the judgements of draft and candidates together, `judge_repeats.jsonl`, `judge_reliability.csv`, `summary.json` with prevalence)
- `codes/rule_codes.jsonl` (LLM judge and expert rows: `present`, `switchable_off`, `quote`, `coder`, `codebook_revision`; judge rows also `unsupported`, `repeats`, `tiebreak`), `codes/judge_repeats.jsonl`, `records_coded/<stem>.json` (mechanisms coded against the latest what codebook plus the never-implemented codes, reasons with why and basis codes)
- `analysis/version_codes.csv` (one row per version and code, never-implemented codes included with `code_source`), `analysis/quality.json`, `analysis/agreement.csv`, `analysis/judge_reliability.csv`, `analysis/theory_mentions.csv` (join)
- `unassigned/<kind>.v<rev>.stage_b.yml` (items the codebooks stage B used do not cover)
- from `analyse`: `analysis/idea_events.csv`, `idea_flow_by_generation.csv`, `idea_first_seen.csv`, `idea_value.csv`, `why_by_what.csv`, `basis_by_generation.csv`, `theory.csv` (+ `theory_by_generation.csv`, `theory_codes.csv`, `theory_mechanisms.csv`), `quality.csv`, `summary.json`, `REPORT.md`, `figures/*.png`

## Analyse (step 7, Mac only)

`analyse --run R` needs a finished join. It reads the join outputs, the coded records, the highest codebook revisions with their families, the quality files and `codebook/literature.yml` if present (per code a list of publications with `reference`, `doi` or `url`, `connection` and `verified`; either a mapping `code: [publications]` or `codes: [{code, publications}]`). Every event gets an `idea_state` from the rule and the notes: introduced, kept, dropped (rule), tested_and_refused, refused_untested, considered (notes), absent; plus say-do flags. Idea value is the change in within-campaign standardised score from a scored parent to a scored child when a code is introduced or dropped, without codes below `judge.min_agreement` (raw numbers in `*_unfiltered`). Figures: prevalence per campaign, one idea-flow grid per lineage with at least 4 versions, a funnel per what family, idea value, why by what, basis and theory by version index, judge reliability (Okabe-Ito colours, PNG at 200 dpi). `summary.json` holds every headline number and figure caption; `REPORT.md` is generated from it.
- `manifests/<step>.json`
