# [DRAFT] Policy finder on open-weight models: OpenCode agents, developed on OpenRouter, final runs on DAIS

## Goal

Run the single-idea policy-finder campaign (PR #246: 5 chains, the seed `punishment = 0` plus 20 one-idea changes each) with an open-weight model as the agent, so that the run no longer depends on Claude Code, a Claude plan, or a hosted model's safeguards. Development uses the same model through OpenRouter; the final runs serve the weights ourselves on DAIS.

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

No model is chosen yet. The numbers below come from model cards as reported by third parties and from blog rankings that disagree with each other; phase 1 measures the candidates on our own task.

| Candidate | What speaks for it | What to check |
|---|---|---|
| **Qwen3-Coder-Next** (first to try) | Built for agentic coding with tool calls; mixture of experts with few active parameters, so fast enough for 5 parallel agents; 256k context; the model card gives a vLLM serve command with a tool-call parser; reported 70.6% on SWE-bench Verified | Its size and whether one H200 holds it at full precision; reports of tool-call parser trouble in vLLM; that OpenRouter serves the unquantised weights |
| **A larger frontier open model** (MiniMax M2.5 or its successor, GLM-5 family) | The highest open-weight agentic coding scores in the rankings found (roughly 73 to 80% on SWE-bench Verified) | Exact names, sizes and licences from the model cards; whether 2 to 4 H200 hold it; vLLM support |
| **Devstral Small 2 (24B)** | Dense and small (one GPU), reported 68% on SWE-bench Verified, simple to serve | Whether it follows the long agent definition and the notes format |
| **Qwen3-32B** (floor) | Already on both clusters and known to work with our vLLM images | A general model, not trained for agent loops; expected to be the weakest |

Selection criteria, in this order: (1) the same weights are on OpenRouter and downloadable under a licence that allows research use; (2) tool calls work reliably with OpenCode and vLLM; (3) it fits a DAIS allocation with at least 64k context while serving 5 agents; (4) it produces accepted versions on our task (phase 1); (5) speed.

What to expect: a smaller open model will fail the acceptance check more often than Opus, write weaker analyses, and cite papers from memory less reliably. The campaign already keeps failed versions in the chain, and citations are checked afterwards. The result describes how that model explores the policy space, which should be said wherever results are compared with the Opus reference run `pf-20261104`.

## Decisions for Levin

1. **Where the simulations run in the final setup.** The sim image is torch 1.11 with CUDA 11.3 and was built for A100s; it may not run on H200 or B200. Option A: everything on DAIS, with a rebuilt sim image if needed, and a check that it reproduces the reference sims. Option B: agents and model on DAIS, simulations stay on Raven, which needs a way to submit Raven jobs from DAIS. Phase 0 tests the image on DAIS; the recommendation follows from that test.
2. **Where development happens.** Recommended: on the existing Hetzner host. Everything but the agent already works there, so swapping the agent is a small change and gives real pipeline runs (with sims on Raven) within days. The alternative is to build the DAIS runner first and point it at OpenRouter, which avoids one throwaway piece (the spawner profile) but delays the first real agent run. The host costs about 5 EUR per day.
3. **OpenRouter key and budget.** Which key to use, and a spending cap for development (a model bake-off and smoke runs are tens of agent runs).
4. **Network rule for the agents on DAIS.** Today's guarantee is "no network but the model". On DAIS that needs Apptainer's network isolation, which may need privileges we do not have. The fallback is tool-level restriction (no web tools, an allowlist for shell commands) plus an audit of the transcripts, which is weaker and should be accepted knowingly.
5. **Whether the Claude campaign plan (`single-idea-run-on-hetzner.md`) is dropped or kept paused** in case the safeguard is resolved.

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

### Phase 3: the pipeline on DAIS (about 3 to 5 days)

- **Agent runner:** a Slurm and Apptainer backend beside the spawner one: the same read-only export and write folders as mounts, the network rule from decision 4, a time limit, and a cap on parallel agents.
- **Cluster backend:** `cluster_proxy.py`'s seven functions (`snapshot`, `run`, `validate_rule`, `submit`, `queue`, `wait`, `fetch`) implemented with local commands for DAIS, or kept as a bridge to Raven under option B of decision 1.
- **One campaign job:** vLLM, the campaign driver and the agents in one Slurm job; the simulations as separate jobs. The job resubmits itself before the 1-day limit, and the campaign resumes from its state files.
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
