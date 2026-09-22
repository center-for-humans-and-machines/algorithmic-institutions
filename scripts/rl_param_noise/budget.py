"""Count the environment episodes a training config actually consumes.

The contract's budget is equal *environment episodes*, not update steps, so
the number has to be counted rather than asserted. Counted by monkeypatching
`ArtificialHumanEnv.step` on a real, shortened `train_manager` call and
dividing by `n_rounds`.

Counting `reset` instead would be wrong: `ArtificialHumanEnv.__init__` resets
once without playing an episode, so resets overstate the budget by exactly one
rollout per run.

The short run is then extrapolated linearly in `n_update_steps`, which is
exact because the loop runs one behaviour rollout per update step and one
evaluation rollout every `eval_period`.

Usage (Raven, GPU):
    python scripts/rl_param_noise/budget.py \\
        configs/training/rl_manager/rl_pnoise_s42.yml \\
        --steps 6 --eval-period 2 --batch-size 8 \\
        --out plots/data_analysis/evaluation/rl_manager_param_noise/budget.json
"""

import argparse
import json
import os
import sys
import tempfile

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aimanager.manager.environment import ArtificialHumanEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--eval-period", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None, help="override the config's device")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as fh:
        full = yaml.safe_load(fh)

    counts = {"step": 0, "reset": 0}
    real_step, real_reset = ArtificialHumanEnv.step, ArtificialHumanEnv.reset

    def counted_step(self, *a, **k):
        counts["step"] += 1
        return real_step(self, *a, **k)

    def counted_reset(self, *a, **k):
        counts["reset"] += 1
        return real_reset(self, *a, **k)

    ArtificialHumanEnv.step = counted_step
    ArtificialHumanEnv.reset = counted_reset

    from aimanager.rl_manager import train_manager  # noqa: E402

    short = dict(full)
    short["n_update_steps"] = args.steps
    short["eval_period"] = args.eval_period
    short["env_args"] = {**full["env_args"], "batch_size": args.batch_size}
    short["job_id"] = "budget_probe"
    if args.device:
        short["device"] = args.device
    with tempfile.TemporaryDirectory() as tmp:
        short["output_dir"] = tmp
        train_manager(short)

    ArtificialHumanEnv.step, ArtificialHumanEnv.reset = real_step, real_reset

    n_rounds = full["env_args"]["n_rounds"]
    measured_rollouts = counts["step"] / n_rounds
    expected_rollouts = args.steps + len(range(0, args.steps, args.eval_period))
    assert measured_rollouts == expected_rollouts, (
        f"{counts['step']} steps / {n_rounds} rounds = {measured_rollouts} "
        f"rollouts, expected {expected_rollouts}"
    )

    n_steps = full["n_update_steps"]
    period = full["eval_period"]
    batch = full["env_args"]["batch_size"]
    n_eval = len(range(0, n_steps, period))
    result = {
        "config": args.config,
        "short_run": {
            "n_update_steps": args.steps,
            "eval_period": args.eval_period,
            "env_step_calls": counts["step"],
            "env_reset_calls": counts["reset"],
            "rollouts_from_steps": measured_rollouts,
            "rollouts_from_resets": counts["reset"],
            "note": (
                "resets overstate by one: the env constructor resets without "
                "playing an episode"
            ),
        },
        "full_run": {
            "n_update_steps": n_steps,
            "eval_period": period,
            "env_batch_size": batch,
            "n_rounds": n_rounds,
            "behaviour_rollouts": n_steps,
            "evaluation_rollouts": n_eval,
            "behaviour_episodes": n_steps * batch,
            "evaluation_episodes": n_eval * batch,
            "total_episodes": (n_steps + n_eval) * batch,
            "total_agent_rounds": (n_steps + n_eval) * batch * n_rounds,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
