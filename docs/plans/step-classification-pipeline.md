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

- `corpus/versions.jsonl`: stem, campaign, lineage, version, parent, rule text, notes text, score, score_se, score status, `setting` (the campaign's game setting from `settings` in the config, verified against the agents' prompts, `docs/reports/basics*.md` and the campaign files on the branches).
- `open/rule_free_codes.jsonl`: version, description, quote.
- `records/<stem>.json` (notes record): version; `mechanisms[]` with description (how the notes present it), quote, `idea` (the mechanism stated positively as something a manager does, whatever the decision; clustering, coverage and assignment use it), `status` (adopted, rejected, considered), `tested` (did an exploration check it), `relation_to_previous` (new, kept, modified, removed, not_stated: how the notes present it relative to the previous version; always not_stated for a first version), `source_version` (earlier version the notes credit, if any), `what_code` and `what_code_status` (coded, none, failed; added later), `reasons[]` with description, quote, `stance` (for or against the mechanism), `evidence` (a short free-text phrase in the agent's terms, taken from the notes, for what the reason rests on, or null), `evidence_ref` (what it points to: an exploration number of these notes, an earlier version's stem, "none", or null), `why_code` and `why_code_status`, `basis_code` and `basis_code_status` (coded, none, failed, no_evidence) (added later); `theory_mentions[]` (theories, concepts or literature the notes name explicitly, with `kind`, free text such as concept, author, field, paper, quote, and `name_ok`: the name occurs in the notes); `unlinked_findings[]`.
- `codebook/what.yml`, `codebook/why.yml`, `codebook/basis.yml`: name, label, definition, examples (fits, does not fit), theory_link, source, and for what and why a `family` (a broader grouping derived from the codes by `consolidate`; the file lists `families[]` with name and one-sentence definition) and `merged_from` (codes merged into it). No what code may be defined as an absence (rejected in code at proposal time; why and basis codes may be negative claims) and what codes are defined by the lever, not by the direction of the effect. Each revision is kept (`what.v1.yml`, ...); every revision above v1 carries `based_on` (revision and sha256 of the revision it reviewed) and is rejected if that base has changed. `codebook/never_implemented.yml` has the same shape. `codebook/non_discriminating.yml`: candidates from the notes that the judge found in more than `coverage.max_prevalence` of the judged rules, with their `prevalence`. `codebook/<kind>.v1.unassigned.yml`: items no code covers, with their text. Target sizes per codebook are settings (`codebook.<kind>.target_codes`: what 30, why 25, basis 10), not limits; a second consolidation pass merges within families when more than 1.5 times the target survive. `codebook/<kind>.merges.yml`: audit of the consolidation (merges applied or skipped, each with the model's `reason`; families; codes placed on the re-ask or given a family of their own). Every code has a family; merges only join codes that imply each other both ways, and never involve a negated or excluding what code. Code names `none` and `failed` are reserved.
- `codebook/literature.yml`: per code, publications with full reference and DOI or URL, one sentence on the connection, and whether the link was verified against the source.
- `codes/rule_codes.jsonl`: version, what_code, present, `switchable_off` (a parameter value inside the declared sweep range disables the mechanism entirely), quote (the judge's verbatim evidence), coder, codebook_revision; for judge rows also `unsupported` (said present without evidence twice, recorded absent), `repeats` (present per repeat) and `tiebreak`. `codes/judge_repeats.jsonl`: every repeat's answer (evidence, present, switchable_off, reasked, unsupported) and the tie-break answers.
- `analysis/`: joined tables and figures. `version_codes.csv` has one row per version and code, including never-implemented codes (`code_source`, `in_rule` false); notes status and relation reduced by precedence (adopted > rejected > considered; new > modified > removed > kept > not_stated) with the full lists in `*_all` columns and `no_record` / `no_mechanism` kept apart; why codes overall, for and against; the reasons' evidence phrases, evidence refs, basis codes and basis families; `switchable_off`; `event_unreliable_code` (the event involves a code whose judge agreement is below `judge.min_agreement`). `agreement.csv` pairs rater and judge rows within one codebook revision. `judge_reliability.csv`: test-retest agreement and Cohen's kappa of judge repeats 1 and 2, per code and overall. `theory_mentions.csv`: per version the mention names found in the notes, their kinds, and names not in the notes. `quality.json` adds code prevalence with codes above `coverage.max_prevalence` flagged (not removed), codes per rule, judge tie-breaks and unsupported answers, the reliability summary (overall, the five least reliable codes, `unreliable_codes`), the share of present answers per code by judge repeat, and family sizes of the three codebooks.
- `manifests/<step>.json`: written last by every step: inputs and outputs with sha256, expected and written counts, failed items, codebook revisions used, model, timestamp. Consumers check it and fail on a missing, stale or modified upstream.

## Steps

| # | Step | Where | Notes |
|---|---|---|---|
| 0 | `corpus` | Mac | Reads the 36 refs with `git show`. No checkout. |
| 1a | `open-rules` | cluster | Free coding of a stratified sample of rules (config: `free_coding.sample`). Reports new codes per batch as a saturation check. |
| 1b | `open-notes` | cluster | Extraction of a notes record for every version. |
| 2a | `what-codebook` | cluster | Cluster the rule free codes into mechanisms. |
| 2b | `coverage` | cluster | Assign draft codes to the notes mechanisms (their `idea`). Cluster the unmatched ones into candidate codes and judge them in one call with the draft codes on the rules outside the expert check. A candidate is added only if at least one and at most `coverage.max_prevalence` (0.9) of the judged rules implement it; more go to `non_discriminating.yml`, none to the never-implemented list. |
| 2c | `consolidate-what` | cluster | One call over draft, accepted and never-implemented codes: merges of codes that are one code (each implies the other) and a family for every surviving code (about 5-10, derived from the codes). Writes `what.v1.yml`, `never_implemented.yml`, `what.merges.yml`; the assignment of free codes and notes ideas is rerun so members follow the merges. |
| 3a | `why-codebook` | cluster | Cluster the extracted reasons into `why.draft.yml`. No scheme is given. |
| 3b | `consolidate-why` | cluster | As 2c for the why codes; families group by kind of justification. Writes `why.v1.yml`, `why.merges.yml`. |
| 3c | `basis-codebook` | cluster | Cluster the reasons' evidence phrases into `basis.draft.yml` (what reasons rest on). No scheme is given. |
| 3d | `consolidate-basis` | cluster | As 2c; families are the few kinds of support a reason rests on, and codes about the content of a finding merge into the kind of support. Writes `basis.v1.yml`, `basis.merges.yml`. |
| 4a | `literature` | Mac, Claude | Link codes to publications. Web search, references verified. Not run on the cluster model, which has no web access. |
| 4b | `review` | Mac, Claude now, experts later | Reads `codebook/*.v1.yml`, writes `*.v2.yml` plus `review/comments.md`. |
| 5a | `apply-rules` | cluster | Judge calls per rule against the whole what codebook (codes under their families): per code the verbatim evidence first, then present and switchable_off. `judge.repeats` (2) runs in seeded code orders; present without evidence is re-asked once; disagreements go to one tie-break run. Writes the judge's test-retest reliability. |
| 5b | `apply-notes` | cluster | Final what, why and basis codes on the records; mechanism ideas against the what codebook plus the never-implemented codes, reasons with their mechanism and stance as uncoded context, evidence phrases against the basis codebook. |
| 6 | `expert-check` | Mac, Claude now, experts later | `export` writes a coding sheet for held-out rules (not in the free-coding sample). `ingest` reads filled sheets and writes rows to `rule_codes.jsonl` with the rater as coder. Agreement per code and overall. |
| 7 | `analyse` | Mac | Tables, figures, `summary.json` and a generated `REPORT.md` from the join outputs, see below and the package README. |

Stage A is steps 1a to 3d in one cluster job against one resident server. Stage B is steps 5a and 5b in a second job. Between them sit literature and review.

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
- Prompt order: system, shared instructions and codebook, then the per-document part (context section with the campaign's game setting and, for notes, the version's position in its lineage; then the document), so the shared part is a byte-identical prefix. One fixed schema per pass. Prompt examples come from an unrelated domain so the first pass stays bottom-up.
- Steps are idempotent. A step whose failed calls exceed its failure budget (config `failure_budget`, 0 for proposals and judge calls) exits non-zero and writes nothing; failed calls never count as evidence. A dead server aborts the step after a few consecutive failed attempts. `preflight` checks every schema against the endpoint before a stage.
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
