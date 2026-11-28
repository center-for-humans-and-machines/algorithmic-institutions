# [ACTIVE] Policy finder on open-weight models: OpenCode agents, developed on OpenRouter, final runs on DAIS

## Goal

Run the single-idea policy-finder campaign (PR #246: 5 chains, the seed `punishment = 0` plus 20 one-idea changes each) with an open-weight model as the agent, so that the run no longer depends on Claude Code, a Claude plan, or a hosted model's safeguards. Development uses the same model through OpenRouter; the final runs serve the weights ourselves on DAIS.

## Success (Levin, 2026-11-27)

The work is a success when an open-weight model, with OpenCode, on DAIS, has done one run of the old (free) setting for **experiment 1** and one for **experiment 2** (12 first rules, the best 4 lineages iterated, 36 versions each, as in the Opus campaigns), and each run shows a similar cumulative improvement and a similar final performance to its Opus reference. A bit worse and a slower improvement are acceptable.

Measured as: the best-so-far curve rises over the generations, and the final best rule reaches **at least 70% of the Opus run's improvement**, where the improvement is the gain of the final best rule over the best first-generation rule. Both sides are scored with the same simulator: the Opus reference rules are rescored on DAIS. The model is the strongest open-weight model that serves reliably on DAIS: Qwen3-Coder-Next first, to get the pipeline working, then the frontier candidates. The single-idea campaign comes after this.

The bars, from re-evaluating all versions of both Opus campaigns on DAIS with fresh common seeds (30,000 episodes per version for exp2, 90,000 for exp1; paired differences; `docs/reports/policy_finder_reference_curves.md` on `policy-finder-dais-pipeline`, PR #250):

| | Best first-generation rule | Final best rule | Improvement | 70% bar |
|---|---|---|---|---|
| Experiment 2 (`pf-20261104`) | 69.84 ± 0.29 | 73.39 ± 0.29 | 3.55 ± 0.09 | 2.49 |
| Experiment 1 (`pf-20261105`) | 649.61 ± 0.20 | 655.37 ± 0.19 | 5.76 ± 0.09 | 4.03 |

An open-model campaign is put through the same tool (`infra/dais/reevaluate.py`): its improvement is its final best rule minus its best first-generation rule, both selected by the campaign's own recorded scores and re-evaluated on the same fresh seeds. Its final best rule's absolute score is reported next to the Opus one, since "similar final performance" is part of the goal as well as the improvement.

Standing authorisations for unattended work, from the same day:

- **DAIS:** up to two full nodes at a time, each up to the 24-hour job limit, resubmitted as needed. Runs happen mainly on DAIS once the basic setup works there; OpenRouter is for development only.
- **OpenRouter:** the life-hub key (Levin's private one, a stopgap), capped at 20 EUR above the baseline of 2026-11-27. A CHM key is the long-term route (the request process is not documented in Confluence; it goes through CHM IT).
- **Hetzner:** the server may be deleted without asking again once the pipeline has passed its smoke run end to end on DAIS and state files, transcripts and results are copied off and the secrets removed.
- **Blocked on Levin:** a dropped SSH connection to DAIS (only he can reopen it), empty OpenRouter credit, or a decision outside this list. Notify on Slack and continue with whatever does not depend on it.

## Run plan (Levin, 2026-11-28)

Agreed after the first Qwen campaigns. It refines the Success section above; the bars stay as they are.

- **Models:** DeepSeek-V4.1-Flash, GLM-5.3 and Tencent Hy4 (FP8), each on one DAIS node; then Kimi K3.1 Pro as the full official model on two nodes (it does not fit one; a pruned third-party build was rejected, since a weak result could then come from the pruning). Qwen3-Coder-Next stays the development model and baseline. One setup for all: a model is one profile file in `infra/dais/model_profiles/` plus its OpenCode limits.
- **Step 1, per model, both experiments in one job:** a free-mode run in the spirit of the Opus campaigns, with 12 first rules and 7 generations, without pruning (all 12 lineages iterated, 84 versions per experiment). Reported twice from the same run: the Opus-comparable result over the 4 lineages that were best after generation 1 (what the bars were computed for), and the result over all 12.
- **Step 2, by default for every model unless step 1 looks totally off:** single-idea mode (the seed `punishment = 0`, one justified change per generation), 20 chains of 20 changes. Format slips in the change note are accepted; its reference only has to indicate a strand of research (a title with the lead or senior author is enough).
- **Agent context:** avoid compaction within a version where possible: the model's full context is served, and the agent's copies of the parent's sweep results are rounded and sorted best first. An overflow is a failed attempt, retried.
- **Time limit:** a cap stays against stalls, but must not distort results: a load stage halts the job when any run hits the time or output limit, and a version lost with such a stop among its attempts is rerun with a higher cap.
- **Kimi K3.1 Pro:** everything that needs no two-node reservation is prepared now; a single run of both experiments follows once the other three models' step-1 runs are in.

## Status (2026-11-28, midday)

The pipeline runs end to end on one DAIS node, and the first real campaigns with Qwen3-Coder-Next are in. Experiment 1 passes its bar on improvement but ends below Opus; experiment 2 does not pass.

Full-node job 989450 (8 B200, `daisg206`, code `7273bb7d`) took 1 h 27 min for everything: both smoke campaigns, the kill-and-resume test, the load stage with 24 agents at once (2 campaigns of 16 versions, all scored), and the two real 36-version campaigns (37 and 50 minutes) with their re-evaluations. The model server answered 4,679 requests with no error.

| | Best first-generation rule | Final best rule | Improvement | 70% bar | Opus improvement | Opus final best |
|---|---|---|---|---|---|---|
| Experiment 1 (`pfq-exp1-20261127`) | 637.05 ± 0.24 | 644.78 ± 0.21 | 7.73 ± 0.09 (cross-fit 7.73 ± 0.09) | 4.03 | 5.76 | 655.37 |
| Experiment 2 (`pfq-exp2-20261127`) | 66.49 ± 0.28 | 67.69 ± 0.29 | 1.21 ± 0.13 (cross-fit 1.21 ± 0.14) | 2.49 | 3.55 | 73.39 |

- **Experiment 1:** the best-so-far curve rises in 5 of the 6 iteration generations (gain 0.96, 2.49, 4.66, 5.90, 5.90, 7.73). The improvement is above the bar, but the run starts 12.6 points below Opus's first generation and ends 10.6 below Opus's final rule, so "similar final performance" is not met.
- **Experiment 2:** the best rule comes in generation 2 and nothing later beats it. The final rule is 5.7 points below Opus's.
- **Agents:** 35 of 36 versions scored in experiment 2, 34 of 36 in experiment 1. Failures: one rule rejected by `validate-rule` in the Raven image, one by `check_instance.sh`, one version whose agent failed three times. A typical agent run takes 70 to 90 seconds.
- **Next:** the frontier open-weight models on the same pipeline, since a campaign pair costs under an hour of one node.
- **Results on DAIS:** `/dais/fs/scratch/levinb/policy-finder-agents/pf_runs_node/` (`campaign/`, `reevaluate/dais-qwen-exp{1,2}/`), log `logs/pf_campaign.989450.out`.

| Piece | Pull request | State |
|---|---|---|
| DAIS measurements, vLLM serving scripts, the shell network filter | #248 | merged after three reviews |
| Simulations on DAIS CPUs (local cluster backend), the re-evaluation tool, the Opus reference curves | #250 | merged after three reviews |
| Follow-ups: sims must import their snapshot's code, lost-job handling at job end | #252 | merged after one review |
| OpenCode harness beside Claude Code | #247 | merged after four reviews |
| Agents, model server and simulations in one DAIS job | #251 | merged after three reviews |
| Resume robustness and the stages files for a restart | #253, #254, #255 | merged |

- **The Claude path** stays paused (Opus safeguards). The three small OpenRouter campaigns on the Hetzner host were stopped; the host is still up for the two Raven equivalence jobs and is deleted after them.
- **Open for Levin:** how to treat the single-idea mode's format failures with open models (2 of 8 passed the `## Change` block's exact format: clarify the wording, loosen the check, or accept the failures as data); a CHM OpenRouter key with a limit if OpenRouter is used again; whether both ways of running the pipeline (Docker and Raven with Claude; one DAIS node with open weights) go to `main`, to be agreed with Yannik.
- **Known differences from the Opus reference runs, to state with the results:** the reference `per_round.parquet` files the agents read were recreated on DAIS CPUs (same config and seed, different draws); simulations run on CPUs (equivalence with Raven GPUs: no difference found at about 1 point; the full-power comparison waits for two Raven jobs); the agents' shell commands run under a network filter and a transcript audit, not in a Docker sandbox.

## Why

- **The Claude path is blocked.** On 2026-11-27 the smoke run's agent version failed twice at its first response: "Opus 5.5's safeguards flagged this message", detail `reasoning_extraction`. Yannik saw the same on 2026-11-13 with the older agent definition. The agent definition asks for a running first-person record of the agent's thinking, which is the purpose of this mode; we do not reword it to get past a safeguard.
- **Reproducibility.** Pinned weights and a pinned server version give a run that can be repeated later; a hosted model can change or refuse from one day to the next.
- **The reasoning becomes data.** With open weights the model's own reasoning tokens can be logged legitimately, next to the notes.

## What stays and what changes

| Part | Today | Target |
|---|---|---|
| Rule format, `validate-rule`, sweep design, GP fit, final run, scoring | `src/aimanager/manager/rule.py`, `scripts/policy_finder/{generate_sim_config,fit_surrogate}.py` | Unchanged |
| Orchestration | `pipeline.py`, `campaign.py`, state files, one git branch per instance | Unchanged in logic |
| Single-idea mode | seed, `base_version`, `## Change` checks (`check_instance.sh`) | Unchanged |
| Agent harness | Claude Code: `claude -p ... --agent policy-finder-host`, a relay plus the `policy-finder` subagent | OpenCode: `opencode run` with one agent definition; no relay |
| Model | Opus 5.5 on a Claude plan | One open-weight model: OpenRouter in development, vLLM on DAIS for the final runs |
| Agent sandbox | Docker container from the spawner, egress only to `api.anthropic.com` | Development: the same container, egress only to `openrouter.ai`. DAIS: Apptainer with the same mounts; network rule decided in phase 0 |
| Cluster access | HTTP to the command server, which holds an SSH connection to Raven (`cluster_proxy.py`) | DAIS: a local backend (`sbatch`, `squeue`, files) behind the same seven functions |
| Simulations | Raven, `algoinst.sif` (torch 1.11, CUDA 11.3, A100) | Open: DAIS or Raven, see Decisions |
| Transcripts, usage | Claude Code session files, plan usage | OpenCode session export, tokens from the endpoint |

## Facts this plan rests on

Checked on 2026-11-27 unless a source is named.

- **The pipeline works end to end without an agent.** The seed version ran all 14 steps from the Hetzner host through Raven and scored 55.3 pool at 20 episodes. Only the agent step fails.
- **DAIS** (from `../understanding/scripts/slurm/evo_dais.slurm.sh` and its plan): NVIDIA H200 and B200 GPUs with more than 128 GB each, partition `gpu`, account `mpib_inst`, QoS `g000N` for N GPUs, 1 day maximum job time, scratch `/dais/fs/scratch/$USER`, Apptainer 1.5.2, a group-readable `vllm.sif` at `/dais/fs/scratch/bsup/container/vllm.sif` (vLLM v0.25.1 in August). A compute node there reached a remote database (their job 409790), so compute nodes have outbound network.
- **Raven**, for comparison: 4 A100 with 40 GB each per node. Our own `vllm.sif` and Qwen3 weights up to 32B are on `/ptmp/levinb`.
- **vLLM and Qwen3 are known ground here.** `../understanding` serves Qwen3-32B on both clusters; this repo's LLM-manager work (PR #222) served Qwen3-8B and 32B on Raven.
- **OpenCode** takes any OpenAI-compatible endpoint as a custom provider (`@ai-sdk/openai-compatible`, a `baseURL` ending in `/v1`, a placeholder key for vLLM). Not yet checked against the official docs: the headless `opencode run`, agent files, permission settings, session export, and running without network access to its own services.
- **OpenRouter keys** exist in three sister projects (`life-hub`, `photo-rating`, `polyvox`).
- **The spawner** runs a command list it is given and passes one token variable (`token_env`); the Claude command is built in `agent_runner.command()`. Whether the spawner adds Claude-specific flags is a phase 0 check.

## Which model

Three roles, decided by Levin on 2026-11-27: a cheap model to debug the harness, the Qwen coder model to develop against, and frontier open-weight models for the actual runs. Prices are OpenRouter's on 2026-11-27 in USD per million tokens (input, cached input, output); sizes are from the Hugging Face API (published weights, mostly FP8 for the large ones).

| Role | Model | OpenRouter | Weights | Licence | Note |
|---|---|---|---|---|---|
| Debugging the harness | DeepSeek-V4.1-Flash | 0.05, 0.001, 0.20 | 763B, about 765 GB | MIT | Cached input is nearly free, and an agent loop resends its context every turn. |
| Development | Qwen3-Coder-Next | 0.12, 0.07, 0.80 | 79.7B, 159 GB in BF16 | Apache 2.0 | Built for agent loops with tool calls, 256k context; one or two DAIS GPUs. |
| Actual runs, candidate | DeepSeek-V4.1-Flash | as above | as above | MIT | Needs most of a full DAIS node. |
| Actual runs, candidate | GLM-5.3 | 0.07, 0.069, 1.40 | 753B, about 755 GB | other, to read | Needs most of a full node. |
| Actual runs, candidate | GLM-5.3-Flash | 0.15, 0.03, 0.50 | 321B, about 328 GB | MIT | About 3 GPUs. |
| Actual runs, candidate | MiniMax-M3.1-Flash | 0.15, 0.015, 0.60 | 220B, about 224 GB | other, to read | About 2 GPUs. |
| Too large for one node | DeepSeek-V4-Pro (1.6T), Kimi-K3.1-Pro (2.8T), Qwen3.8-2.4T | | 1.6 to 2.8 TB | | Not candidates for DAIS. |

Which frontier models go into the actual runs is decided after phase 1, from a bake-off on our own task (version 2 on top of the scored seed, several repeats per model, judged by the acceptance check, the rule's validity in the sim image, and a read of the notes). Whether a model fits depends on the number of GPUs in a DAIS node, which phase 0 reads.

An agent version costs cents on OpenRouter at these prices (an estimate: about 1.5 million cumulative input tokens, mostly cached, and 20 thousand output tokens), so the whole campaign would cost tens of dollars there. DAIS is chosen for pinned weights and independence, not to save money.

What to expect: an open model will fail the acceptance check more often than Opus and cite papers from memory less reliably. The campaign keeps failed versions in the chain, and citations are checked afterwards. The result describes how that model explores the policy space, which should be said wherever results are compared with the Opus reference run `pf-20261104`.

## Decisions (Levin, 2026-11-27)

1. **Everything on DAIS, on one node, simulations included; the simulations run on the node's CPUs** (revised 2026-11-27 after measuring). The existing sim image (torch 1.11) has no code for H200 or B200 GPUs, but on CPUs it is fast when run as many single-threaded processes, one per hyperthread, on a sweep split into slices (`generate_sim_config.py --n-parts`): a real sweep of 128,000 episodes in well under a minute on 16 cores, by extrapolation from measured slices (`docs/reports/dais_feasibility.md`, PR #248). The model server gets the GPUs. With CPU sims the GPU type does not matter for the simulations, so jobs can take H200 or B200, whichever is free. A GPU sim image (torch 2.5 or 2.7) was started and dropped by Levin on 2026-11-27: not needed. Open: the equivalence of CPU results on DAIS with GPU results on Raven, tested with enough power to see a 1-point difference.
2. **Development on the existing Hetzner host**, with the simulations on Raven as today. The host stays up for phase 1.
3. **OpenRouter:** the key of `../life-hub/.env`, with a cap of 20 EUR for now. The key has no limit of its own and cannot create sub-keys, so the cap is enforced by reading the key's usage before each run (baseline 0.31 USD on 2026-11-27) and stopping at 21 USD above it. The account behind it had 1.76 USD of credit left on 2026-11-27 and needs a top-up before the cap can be reached.
4. **Network rule on DAIS:** restricting the agent's tools and auditing the transcripts is accepted if the container's network cannot be cut. Measured since: DAIS gives users no network namespace, but a seccomp filter that denies internet sockets works without root, with the model reachable through a Unix socket. That stricter rule is the target; the accepted fallback stays if it fails with OpenCode in the container.
5. **Harness: OpenCode.** It is independent of the model vendor, runs headless, defines agents in markdown files much like the current definition, and has per-tool permissions. Fallback if tool calling proves unreliable with a model: mini-swe-agent, which drives the model through shell commands only and needs no tool-call API.

## Principle: reuse the existing setup (Levin, 2026-11-27)

Everything new is the smallest adapter beside what exists, selected by a setting; nothing that works is rewritten.

| Existing piece | On DAIS |
|---|---|
| `pipeline.py`, `campaign.py`, state files, instance branches, `check_instance.sh`, the rule checks, sweep design, fit and scoring | Unchanged |
| `export_instance.py`: the agent's read-only copy and its write folders | Unchanged; the same folders become Apptainer binds |
| The agent image with OpenCode (phase 1) | Converted to Apptainer's format, not rebuilt |
| The OpenCode configuration and agent definitions of phase 1 | Unchanged; only the provider's base URL points at the local server |
| `agent_runner.py`: `start`, `finish`, the slot limit, the result and usage records | Same functions; a second backend that launches Apptainer instead of asking the spawner. If DAIS offers a rootless runtime with a Docker-compatible socket, the spawner itself is tried first. |
| `cluster_proxy.py`: `snapshot`, `run`, `validate_rule`, `submit`, `queue`, `wait`, `fetch` | Same functions; a second backend that runs on the node |
| The sim image `algoinst.sif` and `python -m aimanager simulate` | Unchanged if the image runs on H200; a rebuild only if it does not |
| vLLM | The group's existing image on DAIS and the sister project's launch conventions (`../understanding/scripts/slurm/`) |
| `preflight.py`, `raven_setup.sh` | The same checks, extended for DAIS |
| The command server, `slurm_tools` | Not used on DAIS: they exist to reach Raven from outside |

The branch for all of this is `policy-finder-open-weights`; the agents work on `policy-finder-opencode` and `policy-finder-dais` and are merged into it after review. Yannik's external repositories are not changed.

## Phases

### Phase 0: feasibility checks (about 1 day)

- OpenCode from its official docs: headless run with a prompt and an agent, agent definition format, permissions, JSON output and session export, how to stop it calling home (updates, model lists, language servers).
- The spawner: does it accept an arbitrary command and token variable, and a different egress host.
- DAIS: Levin's account, QoS and scratch; whether a job can submit jobs; whether Apptainer there can cut a container's network; the vLLM image version against what the candidate models need; outbound network from a compute node to `openrouter.ai` and to the model hub.
- The sim image on a DAIS GPU: one reference sim, compared with its Raven result.
- Model cards of the candidates: sizes, licences, serve commands.

Done when: each item has an answer, and decisions 1 and 4 have a recommendation.

### Phase 1: OpenCode agent on OpenRouter, in the existing pipeline (about 2 to 3 days)

- An OpenCode configuration with one provider (OpenRouter, the chosen model pinned, including the upstream provider where OpenRouter allows it) and the agent definition ported from `.claude/agents/policy-finder-single-idea.md`. The text stays the same; only tool names and the harness-specific parts change. The relay agent goes: it existed to keep `CLAUDE.md` out of the agent's context, and the export does not contain OpenCode's equivalent.
- `agent_runner.py`: a harness setting (`claude` or `opencode`) that selects the command, the result parsing, the transcript location and the usage record. The Claude path stays as it is.
- The agent image gets OpenCode; a spawner profile with egress to `openrouter.ai` only and the OpenRouter key as its token.
- `pipeline.py`: transcripts and usage for the new harness; the campaign file names the harness and the model.
- Tests for the new parsing and command building in `scripts/tests`.
- A bake-off: each candidate model does the same task several times (version 2 on top of the scored seed), judged by the acceptance check, the validity of the rule in the Raven image, and a read of the notes.

Done when: a one-chain smoke run (seed plus 2 agent versions) reaches `scored` with an open model, and the bake-off table names the model.

### Phase 2: the model on DAIS (about 1 to 2 days)

- Download the chosen weights to DAIS scratch, pinned to a revision. Serve with vLLM in a Slurm job with tool calling enabled; record the image version and the serve arguments in the repo.
- A throughput test with 5 concurrent agent-like sessions; choose the number of GPUs.
- Point the phase 1 agent at this endpoint for one version, to confirm it behaves as on OpenRouter.

Done when: one agent version completes against the DAIS endpoint.

### Phase 3: the pipeline on DAIS (about 3 to 4 days)

- **Agent runner:** a Slurm and Apptainer backend beside the spawner one: the same read-only export and write folders as mounts, the network rule from decision 4, a time limit, and a cap on parallel agents.
- **Cluster backend:** `cluster_proxy.py`'s seven functions (`snapshot`, `run`, `validate_rule`, `submit`, `queue`, `wait`, `fetch`) implemented as a local process pool on the node: a snapshot is a directory, a sim is a process pinned to one GPU, the queue is the pool.
- **One campaign job on a full node:** vLLM, the campaign driver, the agents and the simulations. The job resubmits itself before the 1-day limit, and the campaign resumes from its state files.
- **Setup:** a `dais_setup.sh` with a check mode and a preflight, in the style of `infra/raven/raven_setup.sh` and `preflight.py`.

Done when: the one-chain smoke run reaches `scored` entirely on the cluster side, with no service outside MPCDF.

### Phase 4: the campaign

- The full single-idea campaign on DAIS, then the results, state files, transcripts and reasoning logs back into the repo.
- A short report: the chains, how often versions failed and why, and what differs from the Opus reference run.

## Effort and cost

Roughly 8 to 12 working days for phases 0 to 3, with the widest uncertainty in phase 3 (the sandbox and the cross-cluster question). Opus agents execute the phases and Fable reviews each, as in the Hetzner plan. Money: OpenRouter usage in development (to be capped), the Hetzner host at about 5 EUR per day while phase 1 runs, and nothing per token on DAIS.

## Risks

| Risk | What we do |
|---|---|
| No candidate follows the agent definition well enough (too many failed versions) | The bake-off shows it before any cluster work. Then: a larger model, or a shorter step list in the definition, decided with Levin. |
| Tool calling is unreliable with OpenCode and vLLM for the chosen model | Tested in phase 2 before the pipeline is ported; the parser and the vLLM version are pinned. |
| OpenRouter serves a quantised or different build than the weights we host | Pin the upstream provider; phase 2 repeats one version on DAIS and compares. |
| The sim image does not run on DAIS GPUs | Phase 0 shows it; then option B, or a rebuilt image validated against the reference sims. |
| The agents cannot be cut off from the network on DAIS | Decision 4; the weaker rule is written into the report if it is used. |
| OpenCode needs network access to its own services | Phase 0; pin a version and vendor what it downloads into the image. |
| DAIS queue times and the 1-day limit stretch the campaign | The campaign job resubmits and resumes; chains can be split over several jobs. |
| Scratch file systems delete old files | Weights and results are copied to permanent storage; the setup check verifies the weights before each run. |
