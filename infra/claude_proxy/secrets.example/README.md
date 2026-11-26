# Secrets layout

Nothing in this folder is a secret: it shows what goes into `$SECRETS_DIR`
(default `~/secrets`, set in `container/CC_env/.env`). Compose refuses to
start if a required path is missing.

```
$SECRETS_DIR/
├── claude_setup_token        required  spawner
├── cc_ssh_keys/              required  command server (/root/.ssh)
│   ├── config                          MPCDF config: see cc_ssh_keys/config.example
│   ├── id_ed25519(.pub)                optional: only if your MPCDF login uses a key
│   ├── known_hosts                     written on the first connection
│   └── master-*                        control sockets, written at run time
├── S3/.s3cfg                 required  command server (/root/.s3cfg)
├── algoinst_deploy_key/      required  proxy (/home/claude/.ssh); may be empty
└── .cc_env                   optional  proxy env file
```

| Path | What | How to make it |
|---|---|---|
| `claude_setup_token` | OAuth token of the Claude plan the policy-finder agents run on. Usage counts against that plan's 5-hour and weekly limits. | `claude setup-token`, paste the token as the only line. A `.credentials.json` also works but expires. |
| `cc_ssh_keys/` | The only Raven credentials. The command server runs `ssh raven` with them. | `config` from `config.example` (the MPCDF FAQ config), plus whatever your MPCDF login uses (a key, if you have one registered; otherwise you type password and OTP when you open the connection after start-up). Keep the folder writable: the control sockets live there. OpenSSH runs as root in the container; if it reports `Bad owner or permissions`, run `sudo chown -R root:root cc_ssh_keys && sudo chmod 700 cc_ssh_keys && sudo chmod 600 cc_ssh_keys/*`. |
| `S3/.s3cfg` | s3cmd config for `s3://chm-algorithmic-institutions` (data files, AH models). | `s3cmd --configure`, or a CHM-provided `.s3cfg`. |
| `algoinst_deploy_key/` | SSH key the proxy uses for `git push` to GitHub (instance branches). The pipeline itself does not push, so an empty folder works. | An SSH key with write access to `center-for-humans-and-machines/algorithmic-institutions` (a deploy key or your own), plus `known_hosts` with github.com. Owned by uid 1000. |
| `.cc_env` | Extra env for the proxy (e.g. Slack). Not used by the policy finder. | Leave it out. |

On Raven, one more secret: `~/private/cc_secrets` holding `WANDB_API_KEY=...`.
Every Raven job loads it. `infra/raven/raven_setup.sh` creates it.
