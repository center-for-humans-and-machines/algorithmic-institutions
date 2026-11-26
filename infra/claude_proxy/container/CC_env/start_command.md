# Starting the proxy

Run from `infra/claude_proxy` in the algorithmic-institutions checkout, after
`container/CC_env/.env` exists (copy `.env.example`; `RAVEN_USER` is required)
and `host_check.sh` passes.

```bash
C="docker compose -p algoinst -f container/CC_env/docker-compose.yml"

# 1. the command server and the spawner
$C up -d cc_command_server container_spawner --build

# 2. open the Raven connection (asks for your MPCDF password and OTP; keep it alive)
$C exec cc_command_server bash -ic 'ssh -fN raven && sleep 0.5'

# 3. the proxy (CPU image); this is the Claude session that runs the pipeline
$C run --build --rm claude
```

Extra mounts the policy finder does not need (`algo-inst-otree`,
`slurm_tools`): copy `docker-compose.extra.example.yml` to
`docker-compose.extra.yml` and add `-f container/CC_env/docker-compose.extra.yml`
to `C`.

## With a GPU

On a machine with an RTX 4090. `claude-gpu` is behind the `gpu` profile and is
not started by the commands above.

```bash
$C up -d cc_command_server container_spawner --build
$C exec cc_command_server bash -ic 'ssh -fN raven && sleep 0.5'
$C --profile gpu run --build --rm claude-gpu
```

Neither Claude service has Raven SSH keys. Cluster access is
`http://cc_command_server:8765/` with `CC_PROJECT=algoinst`.

## Raven image

Raven jobs run `algoinst.sif`, built from `algoinst.def` (torch 1.11, CUDA
11.3). A shared copy exists; `infra/raven/raven_setup.sh` links it into your
Raven folders. To build a new one (needs Apptainer with `--fakeroot`), from
`infra/claude_proxy`:

```bash
apptainer build --fakeroot algoinst.sif container/CC_env/algoinst.def
# then copy it to /raven/ptmp/<raven user>/cc/containers/apptainer/algoinst.sif
```
