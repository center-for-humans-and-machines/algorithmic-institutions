# Starting the proxy

Run from `infra/claude_proxy` in the algorithmic-institutions checkout, after
`container/CC_env/.env` exists (copy `.env.example`; `RAVEN_USER` is required)
and `host_check.sh` passes.

```bash
docker compose -p algoinst -f container/CC_env/docker-compose.yml up -d cc_command_server container_spawner --build && docker compose -p algoinst -f container/CC_env/docker-compose.yml exec cc_command_server bash -ic 'ssh -fN raven && sleep 0.5'
docker compose -p algoinst -f container/CC_env/docker-compose.yml run --build --rm claude
```

Extra mounts the policy finder does not need (`algo-inst-otree`,
`slurm_tools`): copy `docker-compose.extra.example.yml` to
`docker-compose.extra.yml` and add `-f container/CC_env/docker-compose.extra.yml`
to `C`.

## Raven image

Raven jobs run `algoinst.sif`, built from `algoinst.def` (torch 1.11, CUDA
11.3). A shared copy exists; `infra/raven/raven_setup.sh` links it into your
Raven folders. To build a new one (needs Apptainer with `--fakeroot`), from
`infra/claude_proxy`:

```bash
apptainer build --fakeroot algoinst.sif container/CC_env/algoinst.def
# then copy it to /raven/ptmp/<raven user>/cc/containers/apptainer/algoinst.sif
```
