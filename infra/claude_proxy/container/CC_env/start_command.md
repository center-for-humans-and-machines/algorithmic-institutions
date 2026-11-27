# Starting the proxy

Run from `infra/claude_proxy` in the algorithmic-institutions checkout, after
`container/CC_env/.env` exists (copy `.env.example`; `RAVEN_USER` is required)
and `host_check.sh` passes.

```bash
# first start on a new host only: the spawner copies this image when it starts
docker compose -p algoinst -f container/CC_env/docker-compose.yml build claude

docker compose -p algoinst -f container/CC_env/docker-compose.yml up -d cc_command_server container_spawner --build && docker compose -p algoinst -f container/CC_env/docker-compose.yml exec cc_command_server bash -ic 'ssh -fN raven && sleep 0.5'
docker compose -p algoinst -f container/CC_env/docker-compose.yml run --build --rm claude
```

On an Apple Silicon Mac, add `-f container/CC_env/docker-compose.amd64.yml`
after each `-f container/CC_env/docker-compose.yml`: the proxy image needs
amd64 (no Linux ARM wheels for PyG). On any Mac, the proxy logs in with
`CLAUDE_CODE_OAUTH_TOKEN` from `~/secrets/.cc_env`
(`docs/handover/handover.md`, On a Mac).

Extra mounts the policy finder does not need (`algo-inst-otree`,
`slurm_tools`): copy `docker-compose.extra.example.yml` to
`docker-compose.extra.yml` and add `-f container/CC_env/docker-compose.extra.yml`
after each `-f container/CC_env/docker-compose.yml`.

## OpenCode agents (harness opencode)

Policy-finder agents on an open-weight model (`harness: opencode` in a
campaign file, `scripts/policy_finder/agent_runner.py`) run through a second
spawner, `container_spawner_oc` (project `algoinst_oc`, egress `openrouter.ai`
only), whose token is the OpenRouter key in `~/secrets/openrouter_token` (one
line, mode 600). Once, after the `claude` image is built:

```bash
mkdir -p "${CC_STATE_DIR:-$HOME/tmp}/container_spawner/oc"
docker build -t algoinst_claude:opencode -f container/CC_env/Dockerfile.opencode .
docker compose -p algoinst -f container/CC_env/docker-compose.yml --profile opencode up -d container_spawner_oc
```

**The key must be a dedicated OpenRouter key with a credit limit set at
OpenRouter for any unattended use.** That limit is the only control on what
the key can spend. What else there is, and what it does not do:

- `PF_OPENROUTER_MAX_USAGE` (the key's total usage in USD; in `CC_env/.env`,
  passed to the proxy by the compose file, or `openrouter_max_usage` in the
  campaign file) stops agents *before they start* once the key reaches it,
  and the campaign then stops. It does not stop a running agent.
  `PF_OPENROUTER_REQUIRE_LIMIT=1` also refuses to start an agent whose key has
  no credit limit; without it the runner only warns (in the pipeline log).
- The runner (`scripts/policy_finder/opencode_run.py`) redacts *accidental
  copies of the literal key* from the transcripts, OpenCode's logs, its
  summary and the agent's notes and scripts; the pipeline scans the
  transcripts it copies once more (for a run whose runner was killed). It is
  an exact match: a base64 copy, or one split by a space, is not found. Notes
  are committed and published with the snapshots. A process the agent
  detached into its own session survives the kill of OpenCode's process
  group; the runner kills such strays only in a container of its own (the
  spawner's), not under Apptainer, whose processes share the node's.
- Nothing stops an agent from printing its environment, key included, into
  the model's context. And `openrouter.ai`, the only egress, is a general
  gateway: an agent with the key and a shell can call any model there,
  past the provider pin and the cap above.

## Raven image

Raven jobs run `algoinst.sif`, built from `algoinst.def` (torch 1.11, CUDA
11.3). Every user runs the shared copy in `/raven/ptmp/ykeller/cc/containers/apptainer/`
(`SIF_OWNER` in `CC_command_server/constants.py`);
`infra/raven/raven_setup.sh` checks that your account can read it. To build a new one (needs Apptainer with `--fakeroot`), from
`infra/claude_proxy`:

```bash
apptainer build --fakeroot algoinst.sif container/CC_env/algoinst.def
# then the image owner copies it to /raven/ptmp/ykeller/cc/containers/apptainer/algoinst.sif
```
