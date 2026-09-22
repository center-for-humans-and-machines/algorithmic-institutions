"""Pre-launch guards and cost pilot for the evolution-strategies arm.

Everything here is asked of the REAL training world -- the real artifacts, the
real environment, the real opponent -- built from the real config, because
each failure it looks for is silent.

GUARDS
  1. PAIRED START. The initial parameter vector this arm optimises must be
     bit-identical to the one the DQN arm starts from under the same seed. If
     it is not, the five-seed comparison is matched but not paired and every
     per-seed difference carries an initialisation difference with it.
  2. THE EVALUATION IS THE OTHER ARMS' EVALUATION. A single-member population
     rollout must reproduce `rl_manager.run_batch(..., on_policy=True)` value
     for value at the real batch size.
  3. NO ACTION NOISE. The behaviour policy must be a deterministic function of
     its parameters and the state. Checked with teeth: the same check is run
     against the DQN arm's epsilon-greedy selection, which must FAIL it.

MEASUREMENTS
  cost   wall clock per generation at the real batch size, against the cost of
         one DQN update step measured in the same session, for a range of
         population sizes. This is the only honest way to price this arm: its
         work profile is P small manager forwards per round where the DQN arm
         has one large one, and the six-hour figure for the DQN arm does not
         transfer.
  noise  the quantity that decides whether this method can learn anything
         here: the spread of fitness ACROSS members against the standard
         error of each member's own fitness. If the second swamps the first,
         the ranking is noise and no amount of budget helps. Swept over sigma.
  shape  mean punishment per contribution bin for every population member at
         generation 0, before any selection has happened. Binned with the
         evaluation suite's own RPA edges, so it sits beside
         plots/data_analysis/evaluation/rl_manager_two_worlds/policy_shape.csv.

Usage (Raven, GPU node for `cost`):
    python scripts/rl_es/pilot.py configs/training/rl_manager/rl_es_pilot.yml \\
        --mode all --out plots/data_analysis/evaluation/rl_manager_es
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch as th
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aimanager.es_manager import (  # noqa: E402
    PopulationRollout,
    build_members,
    build_world,
    centered_ranks,
    draw_perturbations,
    flat_params,
    member_kwargs_for,
    set_flat_params_,
)
from aimanager.evaluation_suite.metrics import RPA_LABELS  # noqa: E402
from aimanager.manager.environment import ArtificialHumanEnv  # noqa: E402
from aimanager.manager.linear_opponent import load_opponent  # noqa: E402
from aimanager.manager.manager import ArtificalManager  # noqa: E402
from aimanager.manager.memory import Memory  # noqa: E402
from aimanager.artificial_humans import AH_MODELS  # noqa: E402
import aimanager.rl_manager as rl_manager  # noqa: E402

SHARED_METRICS = [
    "punishment",
    "opp_punishment",
    "contribution",
    "common_good",
    "contributor_payoff",
    "group_payoff",
    "group_payoff_sum",
    "opp_sum_payoff",
    "next_reward",
    "rl_end_group_size",
    "opp_end_group_size",
]


def seed_all(seed):
    th.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_world_dqn(config, device, env_overrides=None):
    """The DQN arm's construction sequence, transcribed from
    `rl_manager.train_manager`. Only guard 1 uses it; it exists so the guard
    compares two independently written sequences rather than one code path
    with itself."""
    basedir = config["basedir"]
    kind = AH_MODELS[config["artificial_humans_model"]]
    ah = kind.load(
        os.path.join(basedir, config["artificial_humans"]), device=device
    ).to(device)
    ahv = kind.load(
        os.path.join(basedir, config["artificial_humans_valid"]), device=device
    ).to(device)
    opponent = load_opponent(
        os.path.join(basedir, config["opponent_manager"]),
        n_groups=config["env_args"].get("n_groups", 1),
        device=device,
    ).to(device)
    switch = kind.load(os.path.join(basedir, config["switch_model"]), device=device).to(
        device
    )
    env_args = dict(config["env_args"])
    rl_group_id = env_args.pop("rl_group_id", 0)
    env_args.update(env_overrides or {})
    env = ArtificialHumanEnv(
        artifical_humans=ah,
        artifical_humans_valid=ahv,
        artifical_humans_switch=switch,
        device=device,
        **env_args,
    )
    manager = ArtificalManager(
        n_contributions=env.n_contributions,
        n_punishments=env.n_punishments,
        n_groups=env.n_groups,
        default_values=ah.default_values,
        device=device,
        **config["manager_args"],
    )
    return env, opponent, manager, rl_group_id


def load_theta(base, path, device):
    """Overwrite the untrained parameters with a saved manager's.

    The interesting sigma is not the one at the initialisation but the one at
    the policy the run actually reaches. Evolution strategies here reads a
    DISCRETE argmax over 31 ordinal actions, so once the Q-gap between action
    0 and action 1 grows past what a perturbation can bridge, the whole
    population implements the same policy and the ranking is pure episode
    noise. That failure cannot be seen from the initialisation; it can be seen
    from a checkpoint.
    """
    saved = ArtificalManager.load(path, device=device)
    theta = flat_params(saved.policy_model.to(device))
    set_flat_params_(base.policy_model, theta)
    return theta.clone()


# -- guards ----------------------------------------------------------


def guard_paired_start(config, device, seed):
    seed_all(seed)
    _, _, dqn, _ = build_world_dqn(config, device, {"batch_size": 8, "n_rounds": 4})
    seed_all(seed)
    _, _, es, _ = build_world(config, device, {"batch_size": 8, "n_rounds": 4})
    a, b = flat_params(dqn.policy_model), flat_params(es.policy_model)
    return {
        "n_params": int(a.numel()),
        "max_abs_difference": float((a - b).abs().max()),
        "identical": bool(th.equal(a, b)),
    }


def guard_evaluation_matches_run_batch(config, device, seed, batch_size):
    seed_all(seed)
    env, opponent, manager, rl_group_id = build_world(
        config, device, {"batch_size": batch_size}
    )
    seed_all(seed + 1)
    reference = rl_manager.run_batch(
        manager,
        env,
        replay_mem=None,
        on_policy=True,
        update_step=0,
        opponent_manager=opponent,
        rl_group_id=rl_group_id,
    )
    rollout = PopulationRollout(env, opponent, rl_group_id, device)
    seed_all(seed + 1)
    got, _, _ = rollout.run([manager], update_step=0, sampling="greedy")
    worst = max(
        abs(g[k] - r[k]) for g, r in zip(got, reference) for k in SHARED_METRICS
    )
    return {
        "batch_size": batch_size,
        "n_rounds": len(got),
        "n_metrics_compared": len(SHARED_METRICS) * len(got),
        "max_abs_difference": float(worst),
        "identical": worst == 0.0,
    }


def guard_no_action_noise(config, device, seed, batch_size):
    """A deterministic policy gives the same action twice. The DQN arm's
    epsilon-greedy selection must not."""
    seed_all(seed)
    env, _, manager, _ = build_world(config, device, {"batch_size": batch_size})
    state = env.served_state()
    a1, _ = manager.get_action(state, first=True, greedy=True)
    a2, _ = manager.get_action(state, first=True, greedy=True)
    e1, _ = manager.get_action(state, first=True, greedy=False)
    e2, _ = manager.get_action(state, first=True, greedy=False)
    return {
        "deterministic_repeat_identical": bool(th.equal(a1, a2)),
        "eps_greedy_repeat_identical": bool(th.equal(e1, e2)),
        "eps_greedy_cells_differing_from_greedy": int((e1 != a1).sum()),
        "greedy_mean_punishment": float(a1.to(th.float).mean()),
        "eps_greedy_mean_punishment": float(e1.to(th.float).mean()),
    }


# -- cost ------------------------------------------------------------


def measure_cost(config, device, seed, population_sizes, n_generations):
    rows = []
    # The DQN arm's per-step cost, measured here rather than quoted, so the
    # comparison is same-machine and same-session.
    seed_all(seed)
    env, opponent, manager, rl_group_id = build_world(config, device)
    rl_manager.replay_keys = list(
        {
            n["name"]
            for n in config["manager_args"]["model_args"]["x_encoding"]
            + config["manager_args"]["model_args"]["b_encoding"]
        }
        | {"punishment", "agent_group"}
    )
    replay = Memory(
        n_episode_steps=env.n_rounds, device=th.device("cpu"), n_episodes=100
    )
    for step in range(n_generations + 1):
        if step == 1:
            _sync(device)
            t0 = time.time()
        rl_manager.run_batch(
            manager,
            env,
            replay,
            on_policy=False,
            update_step=step,
            opponent_manager=opponent,
            rl_group_id=rl_group_id,
        )
        replay.next_episode(step)
        sample = replay.get_random(device=device, n_episodes=1)
        if sample is not None:
            manager.update(
                step,
                **sample,
                batch=env.batch,
                edge_index=env.batch_edge_index,
                agent_group_mask=env.agent_group_mask,
                rl_group_id=rl_group_id,
            )
    _sync(device)
    rows.append(
        {
            "arm": "dqn",
            "population_size": 1,
            "episodes_per_step": env.batch_size,
            "seconds_per_step": (time.time() - t0) / n_generations,
        }
    )

    for n_members in population_sizes:
        seed_all(seed)
        env, opponent, base, rl_group_id = build_world(config, device)
        members = build_members(
            base.policy_model, member_kwargs_for(env, base), n_members, device
        )
        rollout = PopulationRollout(env, opponent, rl_group_id, device)
        for step in range(n_generations + 1):
            if step == 1:
                _sync(device)
                t0 = time.time()
            rollout.run(members, step, "es-population")
        _sync(device)
        rows.append(
            {
                "arm": "es",
                "population_size": n_members,
                "episodes_per_step": env.batch_size,
                "seconds_per_step": (time.time() - t0) / n_generations,
            }
        )
        print(rows[-1])
    return rows


def _sync(device):
    if device.type == "cuda":
        th.cuda.synchronize()


def measure_budget(config, device, n_generations):
    """Count the environment episodes a real training call actually consumes.

    Counted by instrumenting `ArtificialHumanEnv.step` and dividing by
    `n_rounds`, NOT by counting `reset`: the env's constructor calls `reset`
    once without playing an episode, so a reset count overstates the budget by
    one rollout per run.

    The point of the measurement for THIS arm is the claim that a generation
    costs `batch_size` behaviour episodes whatever the population size --
    because the population is scored inside one rollout rather than one
    rollout each. If that is wrong, the arm would have to trade generations
    against population size to hit the same budget, and the comparison with
    the other three arms would be unfair in a way nobody would see.
    """
    import aimanager.es_manager as es_manager

    counts = {"behaviour": 0, "evaluation": 0, "current": None}
    original_step = ArtificialHumanEnv.step
    original_run = es_manager.PopulationRollout.run

    def counted_step(self):
        counts[counts["current"]] += 1
        return original_step(self)

    def tagged_run(self, members, update_step, sampling, collect_shape=False):
        counts["current"] = (
            "evaluation" if sampling == es_manager.SAMPLING_EVAL else "behaviour"
        )
        try:
            return original_run(self, members, update_step, sampling, collect_shape)
        finally:
            counts["current"] = None

    ArtificialHumanEnv.step = counted_step
    es_manager.PopulationRollout.run = tagged_run
    try:
        short = dict(config)
        short["n_generations"] = n_generations
        short["eval_period"] = max(1, n_generations // 2)
        short["job_id"] = "rl_es_budget_probe"
        with tempfile.TemporaryDirectory() as tmp:
            es_manager.train_manager_es(short, data_dir=tmp)
    finally:
        ArtificialHumanEnv.step = original_step
        es_manager.PopulationRollout.run = original_run

    batch = config["env_args"]["batch_size"]
    rounds = config["env_args"]["n_rounds"]
    population = config["es_args"]["population_size"]
    n_evals = -(-n_generations // short["eval_period"])
    behaviour = counts["behaviour"] // rounds * batch
    evaluation = counts["evaluation"] // rounds * batch
    full_evals = -(-config["n_generations"] // config["eval_period"])
    return {
        "generations_run": n_generations,
        "population_size": population,
        "step_calls_behaviour": counts["behaviour"],
        "step_calls_evaluation": counts["evaluation"],
        "behaviour_rollouts": counts["behaviour"] // rounds,
        "evaluation_rollouts": counts["evaluation"] // rounds,
        "behaviour_episodes": behaviour,
        "evaluation_episodes": evaluation,
        "behaviour_episodes_per_generation": behaviour // n_generations,
        # The claim under test: one rollout per generation, not one per member.
        "rollouts_per_generation": counts["behaviour"] / rounds / n_generations,
        "independent_of_population_size": (
            counts["behaviour"] // rounds == n_generations
        ),
        "evaluation_rollouts_expected": n_evals,
        "evaluation_rollouts_correct": counts["evaluation"] // rounds == n_evals,
        "projected_behaviour_episodes_full_run": (
            behaviour // n_generations * config["n_generations"]
        ),
        "projected_evaluation_episodes_full_run": full_evals * batch,
        "projected_total_episodes_full_run": (
            behaviour // n_generations * config["n_generations"] + full_evals * batch
        ),
    }


# -- noise -----------------------------------------------------------


def measure_noise(config, device, seed, n_members, sigmas, repeats, theta_from=None):
    """Between-member fitness spread against within-member standard error.

    The ratio is what decides whether the ranking carries signal. A member's
    own fitness is the mean of `batch_size / P` episode returns, and this
    environment's episode return varies a lot on its own (the switch
    predictor moves members between groups, so the group the manager is paid
    on is between 0 and 8 players wide).
    """
    rows = []
    seed_all(seed)
    env, opponent, base, rl_group_id = build_world(config, device)
    theta = (
        load_theta(base, theta_from, device)
        if theta_from
        else flat_params(base.policy_model).clone()
    )
    members = build_members(
        base.policy_model, member_kwargs_for(env, base), n_members, device
    )
    rollout = PopulationRollout(env, opponent, rl_group_id, device)
    for sigma in sigmas:
        for rep in range(repeats):
            g = th.Generator(device=device)
            g.manual_seed(seed + 10007 + rep)
            eps = draw_perturbations(n_members, theta.numel(), g, device)
            for p, m in enumerate(members):
                set_flat_params_(m.policy_model, theta + sigma * eps[p])
            seed_all(seed + 1000 * rep)
            _, episode_return, shape = rollout.run(
                members, 0, "es-population", collect_shape=True
            )
            per_member = episode_return.reshape(n_members, -1)
            n_episodes = per_member.shape[1]
            within_se = float(per_member.std(dim=1).mean() / np.sqrt(n_episodes))
            between = float(per_member.mean(dim=1).std())
            member_punishment = _member_mean_punishment(shape)
            rows.append(
                {
                    "sigma": sigma,
                    "repeat": rep,
                    "population_size": n_members,
                    "episodes_per_member": n_episodes,
                    "fitness_mean": float(per_member.mean()),
                    "between_member_sd": between,
                    "within_member_se": within_se,
                    # > 1 means the ordering of the members carries more than
                    # the sampling noise in their scores.
                    "signal_to_noise": between / within_se if within_se else np.nan,
                    "member_punishment_sd": float(np.std(member_punishment)),
                    "member_punishment_mean": float(np.mean(member_punishment)),
                    "identical_member_policies": bool(
                        np.allclose(member_punishment, member_punishment[0])
                    ),
                    # The plateau detector. A population where nobody punishes
                    # has nothing to rank, so the update is a random walk.
                    "members_punishing_nothing": int(
                        np.sum(np.asarray(member_punishment) == 0.0)
                    ),
                    "rank_spearman_vs_punishment": _spearman(
                        centered_ranks(per_member.mean(dim=1)).cpu().numpy(),
                        np.asarray(member_punishment),
                    ),
                }
            )
            print(rows[-1])
    return rows


def _member_mean_punishment(shape):
    num = shape.sum.sum(dim=1).cpu().numpy()
    den = shape.n.sum(dim=1).cpu().numpy()
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def _spearman(a, b):
    a, b = pd.Series(a), pd.Series(b)
    return float(a.corr(b, method="spearman"))


# -- shape -----------------------------------------------------------


def measure_shape(config, device, seed, n_members, sigma, theta_from=None):
    """Generation 0: what policy shape does each member already implement?

    The DQN arm's finished seeds disagree about the SIGN of the
    contribution -> punishment contingency: two of three punish the full
    contributor harder than the free rider, where human managers do the
    opposite. Whether untrained population members already disagree the same
    way says how much of that spread is decided before any selection.
    """
    seed_all(seed)
    env, opponent, base, rl_group_id = build_world(config, device)
    theta = (
        load_theta(base, theta_from, device)
        if theta_from
        else flat_params(base.policy_model).clone()
    )
    members = build_members(
        base.policy_model, member_kwargs_for(env, base), n_members, device
    )
    g = th.Generator(device=device)
    g.manual_seed(seed + 10007)
    eps = draw_perturbations(n_members, theta.numel(), g, device)
    for p, m in enumerate(members):
        set_flat_params_(m.policy_model, theta + sigma * eps[p])
    rollout = PopulationRollout(env, opponent, rl_group_id, device)
    seed_all(seed + 1)
    _, _, shape = rollout.run(members, 0, "es-population", collect_shape=True)
    rows = [_label(r) for r in shape.rows("es-population", 0, per_member=True)]

    # The mean parameter vector, deterministic, over the whole batch -- this
    # arm's evaluation rollout.
    with th.no_grad():
        set_flat_params_(base.policy_model, theta)
    seed_all(seed + 1)
    _, _, mean_shape = rollout.run([base], 0, "greedy", collect_shape=True)
    rows += [_label(r) for r in mean_shape.rows("greedy", 0, per_member=False)]
    return rows


def _label(row):
    if row["sampling"] == "greedy":
        row["label"] = "theta (evaluated)"
    elif row["member"] < 0:
        row["label"] = "population pooled"
    else:
        row["label"] = f"member {row['member']:02d}"
    return row


def shape_table(rows, subset="all"):
    df = pd.DataFrame(rows)
    df = df[df["subset"] == subset]
    return df.pivot_table(
        index="contribution_bin", columns="label", values="mean_punishment"
    ).reindex(RPA_LABELS)


def sign_of_contingency(table):
    """Slope of mean punishment across the six bins, per column.

    Human managers are monotone DECREASING (4.76 at contribution 0 down to
    0.27 at 20, measured on experiments/2group_8agent_50ep.csv). A negative
    slope is therefore the human sign and a positive one the inverted sign
    that two of the three finished DQN seeds came out with.
    """
    x = np.arange(len(RPA_LABELS), dtype=float)
    out = {}
    for column in table.columns:
        y = table[column].to_numpy(dtype=float)
        ok = ~np.isnan(y)
        out[column] = (
            float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() >= 2 else float("nan")
        )
    return out


# -- entry point -----------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--mode", default="all")
    ap.add_argument("--out", default="plots/data_analysis/evaluation/rl_manager_es")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guard-batch-size", type=int, default=100)
    ap.add_argument("--cost-generations", type=int, default=3)
    ap.add_argument("--budget-generations", type=int, default=4)
    ap.add_argument("--population-sizes", default="10,20,40,100")
    ap.add_argument("--sigmas", default="0.005,0.01,0.02,0.05")
    ap.add_argument("--noise-population", type=int, default=40)
    ap.add_argument("--noise-repeats", type=int, default=2)
    ap.add_argument(
        "--theta-from",
        default=None,
        help="a saved <job>_manager.pt whose parameters replace the "
        "initialisation for the noise and shape measurements",
    )
    # Names the output files, so a second measurement at a checkpoint sits
    # beside the initialisation one instead of overwriting it.
    ap.add_argument("--suffix", default="_generation0")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = th.device(config["device"] if th.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    modes = (
        {"guards", "budget", "cost", "noise", "shape"}
        if args.mode == "all"
        else set(args.mode.split(","))
    )
    print(f"device={device} modes={sorted(modes)}")

    if "guards" in modes:
        guards = {
            "paired_start": guard_paired_start(config, device, args.seed),
            "evaluation_matches_run_batch": guard_evaluation_matches_run_batch(
                config, device, args.seed, args.guard_batch_size
            ),
            "no_action_noise": guard_no_action_noise(
                config, device, args.seed, args.guard_batch_size
            ),
        }
        path = os.path.join(args.out, "guards.json")
        with open(path, "w") as f:
            json.dump(guards, f, indent=2)
        print(json.dumps(guards, indent=2))
        print(f"wrote {path}")

    if "budget" in modes:
        seed_all(args.seed)
        budget = measure_budget(config, device, args.budget_generations)
        path = os.path.join(args.out, "budget.json")
        with open(path, "w") as f:
            json.dump(budget, f, indent=2)
        print(json.dumps(budget, indent=2))
        print(f"wrote {path}")

    if "cost" in modes:
        rows = measure_cost(
            config,
            device,
            args.seed,
            [int(p) for p in args.population_sizes.split(",")],
            args.cost_generations,
        )
        df = pd.DataFrame(rows)
        dqn = df[df["arm"] == "dqn"]["seconds_per_step"].iloc[0]
        df["relative_to_dqn_step"] = df["seconds_per_step"] / dqn
        df["hours_for_4000"] = df["seconds_per_step"] * 4000 / 3600
        path = os.path.join(args.out, "cost.csv")
        df.to_csv(path, index=False)
        print(df.to_string(index=False))
        print(f"wrote {path}")

    if "noise" in modes:
        suffix = "" if args.suffix == "_generation0" else args.suffix
        rows = measure_noise(
            config,
            device,
            args.seed,
            args.noise_population,
            [float(s) for s in args.sigmas.split(",")],
            args.noise_repeats,
            theta_from=args.theta_from,
        )
        path = os.path.join(args.out, f"noise{suffix}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(pd.DataFrame(rows).to_string(index=False))
        print(f"wrote {path}")

    if "shape" in modes:
        sigma = float(config["es_args"]["sigma"])
        rows = measure_shape(
            config,
            device,
            args.seed,
            config["es_args"]["population_size"],
            sigma,
            theta_from=args.theta_from,
        )
        pd.DataFrame(rows).to_csv(
            os.path.join(args.out, f"shape{args.suffix}_rows.csv"), index=False
        )
        for subset in ("all", "valid"):
            table = shape_table(rows, subset)
            table.to_csv(os.path.join(args.out, f"shape{args.suffix}_{subset}.csv"))
            slopes = sign_of_contingency(table)
            members = {k: v for k, v in slopes.items() if k.startswith("member")}
            summary = {
                "subset": subset,
                "sigma": sigma,
                "n_members": len(members),
                "theta_slope": slopes.get("theta (evaluated)"),
                "population_pooled_slope": slopes.get("population pooled"),
                "members_with_human_sign_negative": int(
                    sum(1 for v in members.values() if v < 0)
                ),
                "members_with_inverted_sign_positive": int(
                    sum(1 for v in members.values() if v > 0)
                ),
                "members_flat_or_undefined": int(
                    sum(1 for v in members.values() if not (v < 0 or v > 0))
                ),
                "slope_min": float(np.nanmin(list(members.values()))),
                "slope_max": float(np.nanmax(list(members.values()))),
            }
            with open(
                os.path.join(args.out, f"shape{args.suffix}_{subset}.json"), "w"
            ) as f:
                json.dump(summary, f, indent=2)
            print(f"\n--- policy shape{args.suffix} ({subset}) ---")
            print(table.to_string())
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
