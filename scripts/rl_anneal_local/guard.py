"""Pre-launch guards for the annealed, local epsilon-greedy arm.

Four questions, four subcommands. The first three are asked of the real
training code path with the real config, because all three failures are
silent; the fourth is a follow-up on what the first three produced.

budget
    How many environment episodes does a 4000-step run consume? The arms are
    budgeted in episodes, not update steps, so the number has to be measured
    rather than assumed. `ArtificialHumanEnv.reset` and `.step` are counted
    over a real (shortened, CPU, small-batch) `train_manager` call. `step` is
    the honest counter: the constructor resets once without playing anything,
    so rounds over `n_rounds` is the rollout count. It is affine in
    `n_update_steps`, so it extrapolates exactly.

gap
    The thing this arm exists to fix. Reads a finished pilot's metrics parquet
    and reports mean `punishment` under the behaviour rollout (`sampling ==
    "eps-greedy"`) against the evaluation rollout (`sampling == "greedy"`) at
    the first and the last logged update step. Run it on both pilots: the
    control's gap should sit still, this arm's should close.

shape
    The primary outcome. Mean punishment per contribution bin, on the
    evaluation suite's own RPA bins (`RPA_EDGES` / `RPA_LABELS`, imported, not
    re-declared), with the row count per bin so a reader can see it is not
    noise, and with the human and clone columns beside it. Reported for both
    the evaluated policy and the behaviour policy, so a reader can see how far
    the sampled action distribution sits from the evaluated one at each
    contribution level.

    Note what this is and is not. It describes *actions that were sampled*,
    not the policy that gets learned. DQN is off-policy and bootstraps toward
    the max over actions, so a flatter action distribution in the buffer is
    exploration working rather than failing, and no conclusion about the
    learned contingency follows from this table alone. See `state`.

    The clone column is taken from the same rollout (group 1 is the linear
    punisher throughout), so the reference is not carried over from a
    different run. The human column comes through
    `evaluation_suite.convert.load_human`, which drops the flip duplicates and
    keeps a manager timeout as NaN rather than 0.

state
    The follow-up the off-policy objection demands. `gap` and `shape` measure
    actions, and a behaviour policy whose actions differ from the target's is
    what off-policy learning is for. What off-policy correction cannot supply
    is *states* the behaviour policy never visited, and here the state
    distribution is endogenous: contributors are recurrent and group
    membership responds to punishment. This compares a few state summaries
    between the two rollouts. It is deliberately weak -- marginal means, no
    joint, no trajectories -- and is a first cut, not a test.

Usage. `budget` and `shape` need Raven -- torch_geometric is required to
unpickle the GNNs -- but `gap` is pure pandas and runs locally, which is where
this project does its analysis, so the torch imports are deferred into the two
subcommands that need them rather than taken at module level.

    python scripts/rl_anneal_local/guard.py budget CONFIG --out OUT.json
    python scripts/rl_anneal_local/guard.py gap PARQUET [PARQUET ...] --out OUT.md
    python scripts/rl_anneal_local/guard.py state PARQUET [...] --out OUT.csv
    python scripts/rl_anneal_local/guard.py summary PARQUET [...] --spread
    python scripts/rl_anneal_local/guard.py shape CONFIG MANAGER.pt --out OUT.csv
"""

import argparse
import json
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def load_cfg(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# budget
# --------------------------------------------------------------------------- #
def cmd_budget(args):
    """Count env rounds over a real, shortened training run and extrapolate."""
    import aimanager.rl_manager as rl
    from aimanager.manager.environment import ArtificialHumanEnv

    # The probe is a measurement, not a run: keep it out of wandb.
    os.environ.pop("WANDB_API_KEY", None)
    cfg = load_cfg(args.config)
    full_steps = cfg["n_update_steps"]
    full_eval_period = cfg["eval_period"]
    full_batch = cfg["env_args"]["batch_size"]

    counts = {"reset": 0, "step": 0}
    orig_reset, orig_step = ArtificialHumanEnv.reset, ArtificialHumanEnv.step

    def counted_reset(self, *a, **kw):
        counts["reset"] += 1
        return orig_reset(self, *a, **kw)

    def counted_step(self, *a, **kw):
        counts["step"] += 1
        return orig_step(self, *a, **kw)

    ArtificialHumanEnv.reset = counted_reset
    ArtificialHumanEnv.step = counted_step
    try:
        probe = dict(cfg)
        probe["n_update_steps"] = args.steps
        probe["eval_period"] = args.eval_period
        probe["device"] = "cpu"
        probe["env_args"] = dict(cfg["env_args"])
        probe["env_args"]["batch_size"] = args.batch_size
        probe["output_dir"] = args.workdir
        probe["job_id"] = "budget_probe"
        rl.train_manager(probe)
    finally:
        ArtificialHumanEnv.reset = orig_reset
        ArtificialHumanEnv.step = orig_step

    n_rounds = cfg["env_args"]["n_rounds"]
    n_agents = cfg["env_args"]["n_agents"]
    n_eval = len(range(0, args.steps, args.eval_period))
    expected = args.steps + n_eval

    # `env.step` is the honest counter, not `env.reset`: the constructor
    # resets once without playing an episode, so resets run one ahead of
    # rollouts. Rounds divided by n_rounds recovers the rollout count and
    # agrees with the arithmetic, which is what makes the extrapolation safe.
    rollouts_from_steps = counts["step"] / n_rounds

    def full(steps, eval_period, batch):
        evals = len(range(0, steps, eval_period))
        return {
            "n_update_steps": steps,
            "eval_period": eval_period,
            "batch_size": batch,
            "behaviour_rollouts": steps,
            "eval_rollouts": evals,
            "behaviour_episodes": steps * batch,
            "eval_episodes": evals * batch,
            "total_episodes": (steps + evals) * batch,
            "total_episode_rounds": (steps + evals) * batch * n_rounds,
            "total_agent_rounds": (steps + evals) * batch * n_rounds * n_agents,
        }

    report = {
        "config": args.config,
        "probe": {
            "n_update_steps": args.steps,
            "eval_period": args.eval_period,
            "batch_size": args.batch_size,
            "env_resets": counts["reset"],
            "env_rounds": counts["step"],
            "constructor_reset": counts["reset"] - rollouts_from_steps,
            "rollouts_measured": rollouts_from_steps,
            "rollouts_expected": expected,
            "rollouts_match": rollouts_from_steps == expected,
            "episodes": rollouts_from_steps * args.batch_size,
            "rollouts_per_update_step": rollouts_from_steps / args.steps,
        },
        "full_run": full(full_steps, full_eval_period, full_batch),
    }
    emit(report, args.out)
    return 0


# --------------------------------------------------------------------------- #
# gap
# --------------------------------------------------------------------------- #
BEHAVIOUR, EVALUATED = "eps-greedy", "greedy"

# State the manager did not pick directly: what the contributors did and how
# the group ended up. `punishment` is carried alongside as the action, so a
# reader can see the action shift and the state shift in one table.
STATE_METRICS = [
    "punishment",
    "contribution",
    "rl_end_group_size",
    "common_good",
    "next_reward",
]


def gap_table(path):
    df = pd.read_parquet(path)
    p = df[df["metric"] == "punishment"]
    wide = (
        p.groupby(["update_step", "sampling"])["value"]
        .mean()
        .unstack("sampling")
        .sort_index()
    )
    missing = [c for c in (BEHAVIOUR, EVALUATED) if c not in wide.columns]
    if missing:
        raise SystemExit(f"{path}: no rows tagged {missing} in `sampling`")
    wide = wide.dropna(subset=[BEHAVIOUR, EVALUATED])
    wide["gap"] = wide[BEHAVIOUR] - wide[EVALUATED]
    wide["ratio"] = wide[BEHAVIOUR] / wide[EVALUATED].replace(0.0, float("nan"))
    return wide


def cmd_gap(args):
    lines = [
        "# Behaviour versus evaluated punishment",
        "",
        "Mean punishment per member per round over the RL manager's own "
        "group. `behaviour` is the rollout that fills the replay buffer, "
        "`evaluated` the fully deterministic rollout with every exploration "
        "mechanism disabled. `gap` is the punishment the behaviour policy "
        "adds that the evaluated policy never asked for.",
        "",
        "| run | update_step | behaviour | evaluated | gap | ratio |",
        "|---|---|---|---|---|---|",
    ]
    report = {}
    for path in args.parquet:
        name = os.path.basename(path).replace(".parquet", "")
        w = gap_table(path)
        first, last = w.index[0], w.index[-1]
        for step in (first, last):
            r = w.loc[step]
            ratio = "n/a" if pd.isna(r["ratio"]) else f"{r['ratio']:.2f}"
            lines.append(
                f"| {name} | {int(step)} | {r[BEHAVIOUR]:.4f} | "
                f"{r[EVALUATED]:.4f} | {r['gap']:.4f} | {ratio} |"
            )
        report[name] = {
            "first_step": int(first),
            "last_step": int(last),
            "first": w.loc[first].to_dict(),
            "last": w.loc[last].to_dict(),
            "n_eval_points": int(len(w)),
        }
    lines += [
        "",
        "The arm passes if its `gap` at the last step is far below its own "
        "gap at the first step and below the control's gap at the last step. "
        "It says nothing about whether the arm improves the policy.",
        "",
        "```json",
        json.dumps(report, indent=2, sort_keys=True),
        "```",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text)
    return 0


# --------------------------------------------------------------------------- #
# state
# --------------------------------------------------------------------------- #
def cmd_state(args):
    """Does the behaviour rollout visit a different world than the evaluated
    one?

    The gap and shape guards measure *actions*. DQN is off-policy, so a
    different action distribution in the buffer is what the algorithm is for
    and is not by itself a problem. What off-policy correction cannot supply
    is states the behaviour policy never visited -- and in this environment
    the state distribution is endogenous to the manager, because the
    contributors are recurrent and group membership responds to punishment.

    This costs no GPU: the metrics parquet already logs both rollouts at the
    same update steps and carries state beside the action. It is a weak probe
    -- four marginal means, not a distribution, and nothing about trajectories
    -- and the caller should read it as one.
    """
    rows = []
    for path in args.parquet:
        name = os.path.basename(path).replace(".parquet", "")
        df = pd.read_parquet(path)
        steps = sorted(df["update_step"].unique())
        for step in (steps[0], steps[-1]):
            sub = df[(df["update_step"] == step) & df["metric"].isin(STATE_METRICS)]
            w = sub.groupby(["metric", "sampling"])["value"].mean().unstack("sampling")
            for metric in STATE_METRICS:
                if metric not in w.index:
                    continue
                b, e = w.loc[metric, BEHAVIOUR], w.loc[metric, EVALUATED]
                rows.append(
                    {
                        "run": name,
                        "update_step": int(step),
                        "metric": metric,
                        "is_action": metric == "punishment",
                        "behaviour": b,
                        "evaluated": e,
                        "shift": b - e,
                        "pct": 100 * (b - e) / e if e else float("nan"),
                    }
                )
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out.to_csv(args.out, index=False)
    return 0


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
FINAL_METRICS = [
    "punishment",
    "rl_end_group_size",
    "contribution",
    "common_good",
    "next_reward",
]


def cmd_summary(args):
    """Final-window means of the *evaluated* policy, per run.

    The window is the last `--window` evaluation points (default 10, i.e. the
    final 200 update steps at eval_period 20), averaged over rounds and over
    those points, so a single unlucky evaluation does not set the number.
    Only `sampling == greedy` rows are used: this is the policy that would be
    deployed, not the one that filled the buffer.
    """
    rows = []
    for path in args.parquet:
        name = os.path.basename(path).replace(".parquet", "")
        df = pd.read_parquet(path)
        ev = df[df["sampling"] == EVALUATED]
        steps = sorted(ev["update_step"].unique())[-args.window :]
        sub = ev[ev["update_step"].isin(steps) & ev["metric"].isin(FINAL_METRICS)]
        row = {"run": name, "n_eval_points": len(steps), "first_step": steps[0]}
        # end-of-episode group size is a final-round quantity, so it is taken
        # at the last round rather than averaged across rounds.
        last_round = sub["round_number"].max()
        for m in FINAL_METRICS:
            d = sub[sub["metric"] == m]
            if m.endswith("end_group_size"):
                d = d[d["round_number"] == last_round]
            row[m] = float(d["value"].mean()) if len(d) else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows).set_index("run")
    print(out.to_string())

    if args.spread:
        # Spread across the seeds of each arm; the arm is read off the job id
        # prefix, which is how the two families are named.
        out = out.copy()
        out["arm"] = [
            "arm" if i.startswith("rl_anneal_local_s") else "control" for i in out.index
        ]
        agg = out.groupby("arm")[FINAL_METRICS].agg(["mean", "std", "min", "max"])
        print()
        print(agg.to_string())
        if args.out:
            agg.to_csv(args.out.replace(".csv", "_spread.csv"))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out.to_csv(args.out)
    return 0


# --------------------------------------------------------------------------- #
# shape
# --------------------------------------------------------------------------- #
def build_env(cfg, device, batch_size=None):
    from aimanager.artificial_humans import AH_MODELS
    from aimanager.manager.environment import ArtificialHumanEnv
    from aimanager.manager.linear_opponent import load_opponent

    kind = AH_MODELS[cfg["artificial_humans_model"]]
    basedir = cfg["basedir"]

    def load(key):
        return kind.load(os.path.join(basedir, cfg[key]), device=device).to(device)

    env_args = dict(cfg["env_args"])
    rl_group_id = env_args.pop("rl_group_id", 0)
    env_args.pop("reward_formula", None)
    if batch_size is not None:
        env_args["batch_size"] = batch_size
    env = ArtificialHumanEnv(
        artifical_humans=load("artificial_humans"),
        artifical_humans_valid=load("artificial_humans_valid"),
        artifical_humans_switch=(
            load("switch_model") if "switch_model" in cfg else None
        ),
        device=device,
        **env_args,
    )
    opponent = None
    if "opponent_manager" in cfg:
        opponent = load_opponent(
            os.path.join(basedir, cfg["opponent_manager"]),
            n_groups=env_args.get("n_groups", 1),
            device=device,
        ).to(device)
    return env, opponent, rl_group_id


def rollout_cells(env, manager, opponent, rl_group_id, greedy, update_step, seed):
    """One episode batch; returns the (contribution, punishment) cells for the
    RL manager's own group and, separately, for the opponent clone's group.

    Cells where the player gave no input are dropped -- that is where the
    canonical human frame carries NaN, so keeping them would deflate every
    bin the same way pooling would.
    """
    import random

    import numpy as np
    import torch as th

    th.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env.reset()
    state = env.served_state()
    out = {
        "rl": {"contribution": [], "punishment": [], "sw_c": [], "sw_left": []},
        "clone": {"contribution": [], "punishment": [], "sw_c": [], "sw_left": []},
    }
    for rnd in range(env.n_rounds):
        action, _ = manager.get_action(
            state, first=rnd == 0, greedy=greedy, update_step=update_step
        )
        if opponent is not None:
            opp, _ = opponent.predict(
                state, reset_rnn=rnd == 0, edge_index=env.batch_edge_index
            )
            rl_mask = (env.agent_groups.squeeze(-1) == rl_group_id).unsqueeze(-1)
            final = th.where(rl_mask, action, opp)
        else:
            rl_mask = th.ones_like(action, dtype=th.bool)
            final = action
        env.punish(final)

        # Pre-step groups: who actually received this round's punishment.
        valid = state["contribution_valid"].squeeze(-1)
        contrib = state["contribution"].squeeze(-1)
        groups_before = env.agent_groups.squeeze(-1).clone()
        masks = {
            "rl": rl_mask.squeeze(-1) & valid,
            "clone": ~rl_mask.squeeze(-1) & valid,
        }
        for who, mask in masks.items():
            out[who]["contribution"].append(contrib[mask].cpu())
            out[who]["punishment"].append(final.squeeze(-1)[mask].cpu())

        _, _, done = env.step()
        groups_after = env.agent_groups.squeeze(-1)

        # The leaver/stayer split, taken from the realised membership change
        # rather than by re-running the switch predictor -- a second forward
        # pass would disturb its RNN state and change the rollout. A round
        # counts only if some agent actually moved, which is exactly the
        # arrival rounds and needs no hardcoded switch_every.
        moved = groups_before != groups_after
        if bool(moved.any()):
            for who, gid in (("rl", rl_group_id), ("clone", 1 - rl_group_id)):
                here = (groups_before == gid) & valid
                out[who]["sw_c"].append(contrib[here].cpu())
                out[who]["sw_left"].append(moved[here].cpu())

        state = env.served_state()
        if done:
            break

    def pack(d):
        return {
            k: th.cat(v).to(th.float).numpy() if v else th.zeros(0).numpy()
            for k, v in d.items()
        }

    return {who: pack(d) for who, d in out.items()}


def targeting_row(cells, label):
    """Leavers minus stayers, in contribution, at the rounds where membership
    actually changed.

    A correctly targeted manager drives out the free-riders, so its leavers
    contributed *less* than its stayers and the difference is negative. An
    inverted manager drives out the contributors it punishes and the sign
    flips. It reads off a rollout with no counterfactual, which is what makes
    it worth running on every seed.
    """
    c, left = cells["sw_c"], cells["sw_left"].astype(bool)
    if c.size == 0:
        return {"manager": label}
    leavers, stayers = c[left], c[~left]
    return {
        "manager": label,
        "n_decisions": int(c.size),
        "n_leavers": int(left.sum()),
        "leave_rate": float(left.mean()),
        "leaver_contribution": float(leavers.mean()) if leavers.size else float("nan"),
        "stayer_contribution": float(stayers.mean()) if stayers.size else float("nan"),
        "leaver_minus_stayer": (
            float(leavers.mean() - stayers.mean())
            if leavers.size and stayers.size
            else float("nan")
        ),
    }


def human_targeting_row():
    """The same statistic on the human managers, through the evaluation
    suite's own frame so a decision round is the suite's decision round."""
    from aimanager.evaluation_suite.convert import HUMAN_DATA_FILE, load_human

    df = load_human(os.path.join(ROOT, HUMAN_DATA_FILE))
    d = df[df["switch_valid"]].dropna(subset=["contribution"])
    leavers = d[d["does_switch"]]["contribution"]
    stayers = d[~d["does_switch"]]["contribution"]
    return {
        "manager": "human managers",
        "n_decisions": int(len(d)),
        "n_leavers": int(len(leavers)),
        "leave_rate": float(d["does_switch"].mean()),
        "leaver_contribution": float(leavers.mean()),
        "stayer_contribution": float(stayers.mean()),
        "leaver_minus_stayer": float(leavers.mean() - stayers.mean()),
    }


def bin_shape(contribution, punishment, label):
    """Exactly the evaluation suite's RPA binning, plus the row counts."""
    from aimanager.evaluation_suite.metrics import RPA_EDGES, RPA_LABELS

    df = pd.DataFrame({"contribution": contribution, "punishment": punishment})
    bins = pd.cut(df["contribution"], RPA_EDGES, labels=RPA_LABELS).astype(str)
    grouped = df.groupby(bins)["punishment"]
    out = pd.DataFrame(
        {
            f"{label}_mean": grouped.mean(),
            f"{label}_n": grouped.size(),
        }
    ).reindex(RPA_LABELS)
    out.index.name = "contribution_bin"
    return out


def human_shape():
    """The human managers on the same bins, through the evaluation suite's own
    loader so a manager timeout stays a NaN punishment rather than a 0."""
    from aimanager.evaluation_suite.convert import HUMAN_DATA_FILE, load_human

    df = load_human(os.path.join(ROOT, HUMAN_DATA_FILE))
    valid = df.dropna(subset=["punishment", "contribution"])
    return bin_shape(
        valid["contribution"].to_numpy(), valid["punishment"].to_numpy(), "human"
    )


def cmd_shape(args):
    import torch as th

    from aimanager.manager.exploration import Exploration
    from aimanager.manager.manager import ArtificalManager

    cfg = load_cfg(args.config)
    device = th.device(args.device)
    env, opponent, rl_group_id = build_env(cfg, device, args.batch_size)
    # `ArtificalManager.load` assigns the unpickled model straight through
    # without moving it, and `save` puts it on the CPU first, so a load onto
    # cuda comes back with CPU weights and a cuda `self.device`. Its only
    # other caller (api_manager.RLManager) loads on the CPU and never hits
    # this. Moved here rather than in the manager: that is a fix for its own
    # branch, not for an exploration arm.
    manager = ArtificalManager.load(args.manager, device=device)
    manager.policy_model = manager.policy_model.to(device)
    manager.policy_model.eval()

    margs = cfg["manager_args"]
    manager.exploration = Exploration(
        eps=margs["eps"],
        n_actions=env.n_punishments,
        device=device,
        eps_final=margs.get("eps_final"),
        eps_anneal_steps=margs.get("eps_anneal_steps"),
        sigma=margs.get("explore_sigma"),
    )
    end_step = cfg["n_update_steps"] - 1

    frames = [human_shape()]
    targeting = []
    for label, greedy in (("evaluated", True), ("behaviour", False)):
        cells = rollout_cells(
            env, manager, opponent, rl_group_id, greedy, end_step, args.seed
        )
        rl = cells["rl"]
        frames.append(bin_shape(rl["contribution"], rl["punishment"], label))
        if label == "evaluated":
            targeting.append(targeting_row(rl, f"{cfg['job_id']} ({label})"))
            if opponent is not None:
                # The clone under the identical rollout, so the reference
                # column is not carried over from a different run.
                clone = cells["clone"]
                frames.append(
                    bin_shape(clone["contribution"], clone["punishment"], "clone")
                )
                targeting.append(targeting_row(cells["clone"], "clone (same rollout)"))
    shape = pd.concat(frames, axis=1)
    shape.insert(0, "job_id", cfg["job_id"])
    shape.insert(1, "behaviour_eps", manager.exploration.epsilon(end_step))
    shape.insert(2, "explore_sigma", margs.get("explore_sigma"))

    print(shape.to_string())
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        shape.to_csv(args.out)

    if args.targeting_out:
        tgt = pd.DataFrame(targeting + [human_targeting_row()]).set_index("manager")
        print()
        print(tgt.to_string())
        os.makedirs(os.path.dirname(args.targeting_out), exist_ok=True)
        tgt.to_csv(args.targeting_out)
    return 0


def emit(report, out):
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            fh.write(text + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("budget")
    b.add_argument("config")
    b.add_argument("--steps", type=int, default=6)
    b.add_argument("--eval-period", type=int, default=2)
    b.add_argument("--batch-size", type=int, default=4)
    b.add_argument("--workdir", default="temp/budget_probe")
    b.add_argument("--out", default=None)
    b.set_defaults(func=cmd_budget)

    g = sub.add_parser("gap")
    g.add_argument("parquet", nargs="+")
    g.add_argument("--out", default=None)
    g.set_defaults(func=cmd_gap)

    st = sub.add_parser("state")
    st.add_argument("parquet", nargs="+")
    st.add_argument("--out", default=None)
    st.set_defaults(func=cmd_state)

    sm = sub.add_parser("summary")
    sm.add_argument("parquet", nargs="+")
    sm.add_argument("--window", type=int, default=10)
    sm.add_argument("--spread", action="store_true")
    sm.add_argument("--out", default=None)
    sm.set_defaults(func=cmd_summary)

    s = sub.add_parser("shape")
    s.add_argument("config")
    s.add_argument("manager")
    s.add_argument("--batch-size", type=int, default=None)
    s.add_argument("--device", default="cpu")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default=None)
    s.add_argument("--targeting-out", default=None)
    s.set_defaults(func=cmd_shape)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
