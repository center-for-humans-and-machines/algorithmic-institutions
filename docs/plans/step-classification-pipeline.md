# [ACTIVE] Step classification pipeline for the policy-finder search

Branch: `annot/step-classification` (from `yannik/handover`). Design page: https://claude.ai/artifact/XjzM7AfruzpWFaSBprrYaC (tabs Pipeline and Data formats).

## Goal

Code every version of the policy-finder search for *what* its rule does and *why* the agent says it does it, so that two questions can be answered:

1. **Idea flow.** At which step was which idea tested, introduced, kept, dropped or refused? An evolutionary account of how ideas move through the search.
2. **Theory.** Which ideas belong to which theory, at which step are they implemented, and how does the agent rationalise them? Is political and game-theoretic theory used explicitly (the notes name it) or implicitly (the mechanism matches it without being named)? The codebook links each idea to concrete publications.

This first run is a full dry run without stops. The two human steps (codebook review, expert check of rule codes) are carried out by Claude in the same file formats a person would fill in. A second run replaces them with real experts.

## Corpus (verified)

- 110 versions on 36 branches `origin/policy-finder/pf-2026110{4,5,6}-<n>`. A lineage branch holds all its versions. Stem: `<name>` for v1, `<name>-v<k>` after.
- Rule: `configs/managers/rule_based/<stem>.yml` (keys `params`, `sweep_config`, optional `constraints`, `code`). Notes: `notes/policy_finder/<stem>.md`, always with the headings Explorations, Key findings, Hypothesis, Justification.
- Lineage branches also carry copies of other lineages' rules. Keep only files matching `^<name>(-v\d+)?\.(yml|md)$` for the branch's own `<name>`.
- Score: `plots/simulation/policy_finder/<stem>_sweep/final/sweep.json`, the entry of `points[]` whose `name == best_name`, field `pool` (also `pool_se`). Missing for `pf-20261104-1-v2`, `pf-20261104-11-v2` (never run) and `pf-20261104-7-v7` (failed). Units differ by campaign, so scores are only compared within a campaign.
- Parent: the previous version on the same lineage. Not recorded anywhere; derived from the stem. Notes may name an older version as the source of an idea; that is extracted, not assumed.
- Size: a rule is at most about 1k tokens, notes at most about 4.5k tokens.

## Data formats

The four formats of the design page, with the additions the two goals need. Everything is JSON Lines or YAML under one run directory.

- `corpus/versions.jsonl`: stem, campaign, lineage, version, parent, rule text, notes text, score, score_se, score status.
- `open/rule_free_codes.jsonl`: version, description, quote.
- `records/<stem>.json` (notes record): version; `mechanisms[]` with description, quote, `status` (adopted, rejected, considered), `tested` (did an exploration check it), `source_version` (earlier version the notes credit, if any), `what_code` (added later), `reasons[]` with description, quote, `why_code` (added later); `theory_mentions[]` (theories, concepts or literature the notes name explicitly, with quote); `unlinked_findings[]`.
- `codebook/what.yml`, `codebook/why.yml`: name, definition, examples (fits, does not fit), theory_link. Each revision is kept (`what.v1.yml`, ...).
- `codebook/literature.yml`: per code, publications with full reference and DOI or URL, one sentence on the connection, and whether the link was verified against the source.
- `codes/rule_codes.jsonl`: version, what_code, present, quote, coder.
- `analysis/`: joined tables and figures.

## Steps

| # | Step | Where | Notes |
|---|---|---|---|
| 0 | `corpus` | Mac | Reads the 36 refs with `git show`. No checkout. |
| 1a | `open-rules` | cluster | Free coding of a stratified sample of rules (config: `free_coding.sample`). Reports new codes per batch as a saturation check. |
| 1b | `open-notes` | cluster | Extraction of a notes record for every version. |
| 2a | `what-codebook` | cluster | Cluster the rule free codes into mechanisms. |
| 2b | `coverage` | cluster | Assign what codes to the mechanisms in the notes records. Cluster the unmatched ones into candidate codes, run the rule judge for those candidates on all rules, add a candidate to the codebook only if some rule implements it. The rest go to `codebook/never_implemented.yml`. |
| 3 | `why-codebook` | cluster | Cluster the extracted reasons. No scheme is given. |
| 4a | `literature` | Mac, Claude | Link codes to publications. Web search, references verified. Not run on the cluster model, which has no web access. |
| 4b | `review` | Mac, Claude now, experts later | Reads `codebook/*.v1.yml`, writes `*.v2.yml` plus `review/comments.md`. |
| 5a | `apply-rules` | cluster | One judge call per rule against the whole what codebook. Yes or no per code with a quoted line. |
| 5b | `apply-notes` | cluster | Final what and why codes on the records. |
| 6 | `expert-check` | Mac, Claude now, experts later | `export` writes a coding sheet for held-out rules (not in the free-coding sample). `ingest` reads filled sheets and writes rows to `rule_codes.jsonl` with the rater as coder. Agreement per code and overall. |
| 7 | `analyse` | Mac | Joins and figures, see below. |

Stage A is steps 1a to 3 in one cluster job against one resident server. Stage B is steps 5a and 5b in a second job. Between them sit literature and review.

## Analysis outputs

- **Idea events per version and code**, from rule codes of a version and its parent: introduced, kept, dropped. From the notes record: tested and rejected, considered only.
- **Idea flow** by generation and campaign, and per lineage.
- **Idea value**: change in score within campaign when a code is introduced or dropped.
- **Reasons per mechanism**: what code by why code.
- **Theory**: codes by theory; share of adopted mechanisms whose notes name a theory explicitly versus those that match one only through the codebook; publications per code.
- **Quality**: coverage of the what codebook on notes mechanisms, say-do gap (notes say adopted, rule judge says absent, and the reverse), agreement with the expert check, parse failures.

## Engineering

- Package `src/aimanager/annotation/`, CLI `python -m aimanager.annotation <step> --run <run_id> [--config ...]`. It must import without torch or PyG, since it runs on the Mac and inside the inference container.
- One LLM client for every step: OpenAI-compatible endpoint, top-level `response_format` JSON schema from pydantic models, thinking off, explicit `max_tokens`, bounded window of outstanding requests that is refilled continuously, timeouts, backoff on 5xx, fail fast on 400, never retry `finish_reason == "length"` unchanged.
- Exact-result cache keyed on model, messages, schema and sampling parameters, so steps resume and reruns are free. Every call is logged with tokens and latency.
- Prompt order: system, shared instructions and codebook, then the document, so the shared part is a byte-identical prefix. One fixed schema per pass.
- Steps are idempotent and judged by rows written, not exit code.
- A `--backend fake` mode with canned responses makes the whole pipeline testable locally with `pytest`.
- Config in `configs/annotation/`. Sample sizes, model names and concurrency live there, not in code.

## Cluster

- **viper (development):** AMD MI300A nodes, `--constraint=apu --gres=gpu:2`. ROCm vLLM 0.30.0 container, Qwen3-30B-A3B-Instruct-2507, both already on `/ptmp`. Base: `machine-cultural-evolution/containers/gpu_efficiency_probe.sbatch`. Never load the host `rocm` module.
- **dais (production):** B200 nodes. Base: `llm-strategic-tuning-rlobj/projects/coopbot/simulation/interp/jobs/annotate.sbatch` and `common.sh`. Model `deepseek-ai/DeepSeek-V4.1-Flash-1017` (510 GB), second judge `zai-org/GLM-5.3`. A pinned vLLM image that supports the model, checked at job start.
- **Storage:** weights and containers on `/dais/fs/scratch/levinb` and `/ptmp/levinb`, never in home. `HF_HOME` on scratch, `HF_HUB_OFFLINE=1` in jobs, downloads on the login node.
- **Efficiency:** the workload is small (about 110 documents, a handful of passes, a few million tokens), so model load time dominates. Run all passes of a stage against one server. Request only as many GPUs as weights plus a modest KV cache need, not the whole node, if the model allows that tensor-parallel size. Set `--max-model-len` to what the longest prompt needs, `--max-num-seqs`, prefix caching, `--gpu-memory-utilization` and the KV dtype explicitly. Record `/metrics` so KV use and throughput are visible afterwards.
- **Data:** run directories live on cluster scratch during a job and are copied to `s3://chm-algorithmic-institutions/policy_finder/step_classification/<run_id>/`. Credentials come from `.env` (`AWS_*`, `MINIO_*`), never from code or job scripts.

## Order of work

1. Core package, corpus loader, client, fake backend, tests. In parallel: cluster scripts, viper smoke test, start the production weight download.
2. Steps and prompts, end to end on the fake backend, then on viper with the small model.
3. Review of code and of the viper outputs. Refine prompts.
4. Production run on dais: stage A, literature and review, stage B, expert check, analysis.
5. Results into the design page.
