"""Pre-launch measurements for the parameter-space-noise exploration arm.

Everything here is asked of the **real** stack built from the real config --
the same four artifacts, the same env, the same opponent -- because every
question below is about magnitudes, and a toy network would answer a different
question.

Five measurements, written to one JSON:

A. THE EPSILON-GREEDY REFERENCE. What epsilon-greedy at 0.1 actually injects,
   measured rather than derived from the uniform formula: the mean
   |a_behaviour - a_greedy| in punishment levels over real states. This is the
   number the adaptation target is set to, so that the arm changes the
   *consistency* of the exploration and not its size -- one variable at a
   time. Plappert sets his DQN threshold the same way, as the distance an
   epsilon-greedy policy would have.

B. THE ORDINAL QUESTION. Two perturbations of the real Q tensor that are
   very different for this experiment and must not look alike:
     shift   -- every punishment moved up one level; the contingency on
                contribution is preserved exactly;
     invert  -- the action axis reversed, so a policy monotone in contribution
                becomes monotone the other way; this is the failure mode two
                of the three finished seeds actually landed in.
   Reported under all three divergence measures. If the unweighted measure
   gives the two the same number, it cannot be the adaptation target.

C. THE SIGMA SWEEP. For a grid of noise scales: the three divergences, and
   the policy's shape -- mean punishment per RPA bin, on the evaluation
   suite's own bins -- so the scale the run starts from is chosen from a
   measurement. The epsilon-greedy profile is printed on the same axis: the
   arm's whole claim is that uniform action noise flattens the contingency
   (its draw is independent of the contribution it is aimed at) while weight
   noise moves the contingency without destroying it.

D. THE INVARIANTS. The opponent's, the target network's and the policy
   network's parameters, hashed before and after a full behaviour episode.

E. THE EPISODE BUDGET. Counted from the rollouts this config would run, not
   asserted: n_update_steps behaviour rollouts plus one evaluation rollout
   every eval_period, each of batch_size episodes.

Usage (Raven; torch_geometric is needed to unpickle the GNNs):
    python scripts/rl_param_noise/probe.py \\
        configs/training/rl_manager/rl_new_clones_s42.yml \\
        --batch-size 64 --device cpu \\
        --out plots/data_analysis/evaluation/rl_manager_param_noise/probe.json
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch as th
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aimanager.artificial_humans import AH_MODELS  # noqa: E402
from aimanager.evaluation_suite.metrics import RPA_EDGES, RPA_LABELS  # noqa: E402
from aimanager.manager.environment import ArtificialHumanEnv  # noqa: E402
from aimanager.manager.linear_opponent import load_opponent  # noqa: E402
from aimanager.manager.manager import ArtificalManager  # noqa: E402
from aimanager.manager.param_noise import (  # noqa: E402
    ParameterNoise,
    divergence_l2,
    divergence_mad,
    divergence_w1,
)

SIGMAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def build(cfg, batch_size, device):
    basedir = cfg["basedir"]
    kind = AH_MODELS[cfg["artificial_humans_model"]]

    def load(key):
        return kind.load(os.path.join(basedir, cfg[key]), device=device).to(device)

    env_args = dict(cfg["env_args"])
    rl_group_id = env_args.pop("rl_group_id", 0)
    env_args.pop("reward_formula", None)
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
    manager_args = dict(cfg["manager_args"])
    manager_args.pop("param_noise", None)
    manager = ArtificalManager(
        n_contributions=env.n_contributions,
        n_punishments=env.n_punishments,
        n_groups=env.n_groups,
        default_values=env.artifical_humans.default_values,
        device=device,
        **manager_args,
    )
    return env, manager, opponent, rl_group_id


def collect_states(env, manager, opponent, rl_group_id, n_rounds):
    """One greedy rollout, keeping every round's served state and the env's
    own contribution / validity / group tensors. The state pool every
    candidate policy below is then scored on, so they are all compared on the
    same states rather than each on its own trajectory."""
    env.reset()
    state = env.served_state()
    pool = []
    for r in range(n_rounds):
        served = {k: v.clone() for k, v in state.items()}
        action, _ = manager.get_action(state, first=r == 0, greedy=True)
        if opponent is not None:
            opp_action, _ = opponent.predict(
                state, reset_rnn=r == 0, edge_index=env.batch_edge_index
            )
            mask = (env.agent_groups.squeeze(-1) == rl_group_id).unsqueeze(-1)
            action = th.where(mask, action, opp_action)
        env.punish(action)
        pool.append(
            {
                "state": served,
                "contribution": env.state["contribution"].squeeze(-1).clone(),
                "valid": env.state["contribution_valid"].squeeze(-1).clone(),
                "groups": env.agent_groups.squeeze(-1).clone(),
            }
        )
        _, _, done = env.step()
        if done:
            break
        state = env.served_state()
    return pool


def rpa_profile(actions, pool, rl_group_id):
    """Mean intended punishment per contribution bin, over the whole pool.

    The policy's shape, on the evaluation suite's own RPA bins, weighted by
    the number of agent-rounds in each bin exactly as pooling the raw rows
    would be. Cells where the contributor gave no input are dropped.
    """
    edges = th.tensor(RPA_EDGES, dtype=th.float)
    tot = np.zeros(len(RPA_LABELS))
    cnt = np.zeros(len(RPA_LABELS))
    for a, item in zip(actions, pool):
        c = item["contribution"].to(th.float).cpu()
        keep = item["valid"].cpu() & (item["groups"].cpu() == rl_group_id)
        idx = th.bucketize(c, edges) - 1
        p = a.squeeze(-1).to(th.float).cpu()
        for b in range(len(RPA_LABELS)):
            m = keep & (idx == b)
            cnt[b] += float(m.sum())
            tot[b] += float((p * m).sum())
    mean = [float(tot[b] / cnt[b]) if cnt[b] else float("nan") for b in range(len(cnt))]
    return {
        "bins": list(RPA_LABELS),
        "mean_punishment": mean,
        "n": [int(x) for x in cnt],
        # The single number the inverted-policy result turns on: human
        # managers sit at +4.49 (4.76 at contribution 0, 0.27 at 20), the two
        # inverted seeds at -4.92 and -1.77.
        "contrast_0_minus_20": (
            mean[0] - mean[-1]
            if mean[0] == mean[0] and mean[-1] == mean[-1]
            else float("nan")
        ),
    }


def act_all(pool, forward):
    """Run a per-round action function over the pool, RNN reset at round 0."""
    return [forward(item["state"], r == 0) for r, item in enumerate(pool)]


def q_of(model, manager, state, first):
    n_batch, n_agents, n_rounds = list(state.values())[0].shape
    exp = manager.expand_obs_for_groups(state, manager.n_groups)
    with th.no_grad():
        q = model(model.encode(exp, edge_index=None), reset_rnn=first)
    return q.reshape(n_batch, manager.n_groups, n_agents, n_rounds, -1)


def gather(q, state):
    return q.argmax(-1).gather(1, state["agent_group"].unsqueeze(1)).squeeze(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    device = th.device(args.device)
    env, manager, opponent, rl_group_id = build(cfg, args.batch_size, device)
    pool = collect_states(env, manager, opponent, rl_group_id, env.n_rounds)
    policy = manager.policy_model

    result = {
        "config": args.config,
        "batch_size": args.batch_size,
        "device": args.device,
        "n_rounds_pooled": len(pool),
    }

    # -- greedy reference --------------------------------------------
    qs = act_all(pool, lambda s, f: q_of(policy, manager, s, f))
    greedy = [gather(q, item["state"]) for q, item in zip(qs, pool)]
    result["greedy"] = rpa_profile(greedy, pool, rl_group_id)

    # -- A. what epsilon-greedy injects ------------------------------
    eps = cfg["manager_args"]["eps"]
    eps_actions, disp = [], []
    for g in greedy:
        rnd = th.randint(0, env.n_punishments, size=g.shape, device=g.device)
        sel = th.rand(size=g.shape, device=g.device) < eps
        a = th.where(sel, rnd, g)
        eps_actions.append(a)
        disp.append((a - g).abs().to(th.float).mean().item())
    result["eps_greedy"] = {
        "eps": eps,
        "mean_abs_action_displacement": float(np.mean(disp)),
        "profile": rpa_profile(eps_actions, pool, rl_group_id),
    }

    # -- B. the ordinal question -------------------------------------
    shift, invert = [], []
    for q, item in zip(qs, pool):
        shift.append(
            {
                "mad": divergence_mad(q, q.roll(1, dims=-1)),
                "l2": divergence_l2(q, q.roll(1, dims=-1)),
                "w1": divergence_w1(q, q.roll(1, dims=-1)),
            }
        )
        rev = q.flip(-1)
        invert.append(
            {
                "mad": divergence_mad(q, rev),
                "l2": divergence_l2(q, rev),
                "w1": divergence_w1(q, rev),
            }
        )

    def avg(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    shift_actions = [gather(q.roll(1, dims=-1), i["state"]) for q, i in zip(qs, pool)]
    invert_actions = [gather(q.flip(-1), i["state"]) for q, i in zip(qs, pool)]
    result["ordinal_question"] = {
        "shift_one_level": {
            "divergence": avg(shift),
            "profile": rpa_profile(shift_actions, pool, rl_group_id),
        },
        "inverted_contingency": {
            "divergence": avg(invert),
            "profile": rpa_profile(invert_actions, pool, rl_group_id),
        },
    }

    # -- C. the sigma sweep ------------------------------------------
    sweep = []
    for sigma in SIGMAS:
        th.manual_seed(args.seed)
        pn = ParameterNoise(policy, scale=sigma, target=1.0, adapt=False)
        acts, rows = [], []
        for r, item in enumerate(pool):
            pq = q_of(pn.perturbed, manager, item["state"], r == 0)
            rows.append(
                {
                    "mad": divergence_mad(qs[r], pq),
                    "l2": divergence_l2(qs[r], pq),
                    "w1": divergence_w1(qs[r], pq),
                }
            )
            acts.append(gather(pq, item["state"]))
        sweep.append(
            {
                "scale": sigma,
                "divergence": avg(rows),
                "profile": rpa_profile(acts, pool, rl_group_id),
            }
        )
    result["sigma_sweep"] = sweep

    target = result["eps_greedy"]["mean_abs_action_displacement"]
    reachable = [s for s in sweep if s["divergence"]["mad"] >= target]
    result["recommended"] = {
        "target_mad": target,
        "init_scale": reachable[0]["scale"] if reachable else SIGMAS[-1],
        "init_scale_is_a_ceiling": not reachable,
    }

    # -- D. the invariants -------------------------------------------
    def digest(mod):
        if mod is None:
            return None
        return {
            k: float(v.detach().to(th.float).sum().item())
            for k, v in mod.state_dict().items()
        }

    def opp_digest():
        """The linear opponent is an sklearn estimator, not a module: its
        weights are the multinomial's coefficients and intercept."""
        if opponent is None:
            return None
        est = opponent.estimator
        return [
            float(np.asarray(est.coef_).sum()),
            float(np.asarray(est.intercept_).sum()),
        ]

    before = {
        "policy": digest(policy),
        "target": digest(manager.target_model),
        "opponent": opp_digest(),
    }
    pn = ParameterNoise(policy, scale=0.1, target=target, adapt=True)
    for r, item in enumerate(pool):
        pq = q_of(pn.perturbed, manager, item["state"], r == 0)
        pn.observe(qs[r], pq)
    pn.finish()
    after = {
        "policy": digest(policy),
        "target": digest(manager.target_model),
        "opponent": opp_digest(),
    }
    result["invariants"] = {
        k: (before[k] == after[k] if before[k] is not None else "absent")
        for k in before
    }
    assert all(
        v is True or v == "absent" for v in result["invariants"].values()
    ), f"parameter noise wrote outside the acting copy: {result['invariants']}"

    # -- E. the episode budget ---------------------------------------
    n_steps = cfg["n_update_steps"]
    period = cfg["eval_period"]
    batch = cfg["env_args"]["batch_size"]
    n_eval = len(range(0, n_steps, period))
    result["episode_budget"] = {
        "n_update_steps": n_steps,
        "eval_period": period,
        "env_batch_size": batch,
        "behaviour_rollouts": n_steps,
        "evaluation_rollouts": n_eval,
        "behaviour_episodes": n_steps * batch,
        "evaluation_episodes": n_eval * batch,
        "total_episodes": (n_steps + n_eval) * batch,
        "env_rounds_per_episode": cfg["env_args"]["n_rounds"],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "sigma_sweep"}, indent=2))
    print("\nsigma sweep")
    for s in sweep:
        d = s["divergence"]
        print(
            f"  scale {s['scale']:<6} mad {d['mad']:7.3f}  l2 {d['l2']:.5f}"
            f"  w1 {d['w1']:7.3f}  contrast {s['profile']['contrast_0_minus_20']:+7.3f}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
