"""The pre-launch guards for the new-clones RL runs.

All of them are asked of the **real** training environment built from the real
config -- the same models, the same env, the same opponent -- because every
failure they look for is silent: a run trained on the wrong reward, or with the
free punishment lever still open, looks entirely normal until somebody checks.

GUARD 1 -- the reward really is what the config says it is.
    Dispatches on `env_args.reward_mode`, and for every group and round
    `env.reward` must equal that mode's quantity:

      * `common_pool` -- 1.6 * sum(contributions) - sum(punishments);
      * `common_pool_per_capita` -- that same pool divided by the number of
        players who gave an input in the group (`count_valid_per_group`).

    Checked several ways so the answer does not rest on this file
    reimplementing the env:
      a. against the quantity recomputed here from the env's own state
         tensors -- for the per-capita mode, this file does its own division;
      b. against the `common_good` state field, which is produced by a
         different method (`update_common_good`) and never by the reward
         path, so agreement is an independent confirmation rather than a
         tautology. `common_good` *is* the per-capita share, so the pool mode
         compares against `common_good * n_valid` and the per-capita mode
         against `common_good` itself;
      c. against the same rollout under `reward_mode='sum'`, which must NOT
         match -- if it does, the mode never changed and guard 1 is vacuous.

GUARD 2 -- the free punishment lever is closed.
    A punishment aimed at a player who gave no input costs the manager nothing
    (the accounting zeroes it) but used to still reach every artificial human
    through `punishment` / `prev_punishment`. Here the manager is forced to
    punish the maximum on every cell -- the strongest possible probe -- and the
    guard reports every punishment value the contribution, validity and switch
    models were actually served at a timed-out cell. All of them must be 0.

GUARD 3 -- the per-capita reward was divided, not relabelled.
    Only for `reward_mode='common_pool_per_capita'`, and the whole point of
    that arm: it must be a genuinely different reward from `common_pool`, not
    the same number under a new name.
      a. Against a `common_pool` rollout of the same config and seed, the two
         rewards must differ by a margin far larger than any constant could
         be hiding -- and the two rollouts are first shown to be the same
         trajectory (identical pools), so the difference is the reward and
         not the run.
      b. `reward * n_valid` must return the pool exactly. That is the
         algebraic inverse of the division, and it pins the divisor.
      c. The equal-headcount leg. `pool` and `pool / n` can only ever be
         numerically equal at n = 1, so "they agree" is checked in the only
         form it can hold: on a controlled rollout where every group holds
         the same number of members and nobody times out, the ratio
         `common_pool / per_capita` must be a **single** value, exactly the
         integer headcount. A relabelling would give a ratio of 1; a division
         by anything other than the headcount would give something that is
         not the headcount; and a divisor that is not really per group would
         not, in the free rollout of (a), take the several distinct values it
         does. This leg forces the equal headcount by dropping the switch
         model (membership cannot move) and forcing every player valid.

Usage (Raven; torch_geometric is needed to unpickle the GNNs):
    python scripts/rl_two_worlds/launch_guards.py \\
        configs/training/rl_manager/rl_percapita_s42.yml \\
        --batch-size 64 --out plots/data_analysis/<dir>/guards.json
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch as th
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aimanager.artificial_humans import AH_MODELS  # noqa: E402
from aimanager.manager.environment import ArtificialHumanEnv  # noqa: E402
from aimanager.manager.linear_opponent import load_opponent  # noqa: E402

POOL_RATE = 1.6


def build_env(
    cfg,
    batch_size,
    device,
    reward_mode=None,
    drop_switch=False,
    force_all_valid=False,
):
    """The real training env from the real config.

    `drop_switch` and `force_all_valid` are guard 3c's two knobs and are off
    everywhere else. Together they pin every group to the membership the
    config starts it with and every player to valid, which is the controlled
    equal-headcount rollout that leg needs. They are probes, not a second
    configuration of the experiment: nothing is trained under them.
    """
    basedir = cfg["basedir"]
    kind = AH_MODELS[cfg["artificial_humans_model"]]

    def load(key):
        return kind.load(os.path.join(basedir, cfg[key]), device=device)

    env_args = dict(cfg["env_args"])
    env_args.pop("rl_group_id", 0)
    env_args.pop("reward_formula", None)
    env_args["batch_size"] = batch_size
    if reward_mode is not None:
        env_args["reward_mode"] = reward_mode
    switch = None
    if "switch_model" in cfg and not drop_switch:
        switch = load("switch_model")
    env = ArtificialHumanEnv(
        artifical_humans=load("artificial_humans"),
        artifical_humans_valid=load("artificial_humans_valid"),
        artifical_humans_switch=switch,
        device=device,
        **env_args,
    )
    if force_all_valid:
        inner_valid = env.artifical_humans_valid.predict

        def always_valid(state, _inner=inner_valid, **kw):
            out, extra = _inner(state, **kw)
            return th.ones_like(out), extra

        env.artifical_humans_valid.predict = always_valid
    opponent = None
    if "opponent_manager" in cfg:
        opponent = load_opponent(
            os.path.join(basedir, cfg["opponent_manager"]),
            n_groups=env_args.get("n_groups", 1),
            device=device,
        ).to(device)
    return env, opponent


def pool_from_state(env):
    """1.6 * sum(c) - sum(p) per group, from the env's own state tensors.

    Invalid cells are zeroed first: a player who gave no input contributed
    nothing and, once the lever is closed, was punished nothing. That zeroing
    is what makes the identity hold on the human data.
    """
    valid = env.state["contribution_valid"]
    c = th.where(valid, env.state["contribution"], th.zeros_like(valid.long()))
    p = th.where(valid, env.state["punishment"], th.zeros_like(valid.long()))
    mask = env.agent_group_mask
    sum_c = (c.to(th.float).unsqueeze(-2) * mask).sum(dim=1)
    sum_p = (p.to(th.float).unsqueeze(-2) * mask).sum(dim=1)
    return POOL_RATE * sum_c - sum_p


def raw_pool_from_state(env):
    """The same without zeroing invalid cells. Equal to `pool_from_state` only
    if nothing was ever charged at a timed-out cell -- a second read on
    guard 2, from the accounting side rather than the serving side."""
    mask = env.agent_group_mask
    c = env.state["contribution"].to(th.float).unsqueeze(-2) * mask
    p = env.state["punishment"].to(th.float).unsqueeze(-2) * mask
    return POOL_RATE * c.sum(dim=1) - p.sum(dim=1)


def run_rollout(env, opponent, n_rounds, watch_punishment=False, seed=0):
    """Drive the env with a maximum-punishment policy and collect the guards.

    Maximum punishment everywhere is deliberate: it is the strongest possible
    test of guard 2 and it keeps the reward well away from 0, so guard 1 is not
    satisfied trivially.
    """
    th.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    served_at_timeout = Counter()  # punishment values the AHs saw there
    served_prev_at_timeout = Counter()
    watched = {"calls": 0, "timeout_cells": 0, "agent_cells": 0}

    if watch_punishment:
        for model in (
            env.artifical_humans,
            env.artifical_humans_valid,
            env.artifical_humans_switch,
        ):
            if model is None:
                continue
            inner = model.predict

            def probed(state, _inner=inner, **kw):
                ok = state["contribution_valid"].cpu().reshape(-1).numpy()
                ok = ok.astype(bool)
                watched["calls"] += 1
                watched["agent_cells"] += len(ok)
                watched["timeout_cells"] += int((~ok).sum())
                # Each channel is read against the validity of the round whose
                # punishment it carries. `punishment` is this round's, so it
                # takes this round's mask. `prev_punishment` is round t-1's,
                # so it takes round t-1's: a player who gave input at t-1 was
                # punishable then, that punishment was charged, and it is
                # right for it to still be visible at t even though they have
                # since timed out. Masking it with round t's validity would
                # flag correct behaviour as a defect.
                prev_ok = state.get("prev_contribution_valid")
                if prev_ok is not None:
                    prev_ok = prev_ok.cpu().reshape(-1).numpy().astype(bool)
                    prev_ok = prev_ok | state["is_first"].cpu().reshape(-1).numpy()
                for key, sink, mask in (
                    ("punishment", served_at_timeout, ok),
                    ("prev_punishment", served_prev_at_timeout, prev_ok),
                ):
                    if key not in state or mask is None:
                        continue
                    v = state[key].detach().cpu().reshape(-1).numpy()
                    for x in v[~mask]:
                        sink[float(x)] += 1
                return _inner(state, **kw)

            model.predict = probed

    env.reset()
    rows = []
    for rnd in range(n_rounds):
        state = env.served_state()
        action = th.full_like(env.state["punishment"], env.n_punishments - 1)
        if opponent is not None:
            opp, _ = opponent.predict(
                state, reset_rnn=rnd == 0, edge_index=env.batch_edge_index
            )
            rl_mask = (env.agent_groups.squeeze(-1) == 0).unsqueeze(-1)
            action = th.where(rl_mask, action, opp)
        env.punish(action)

        n_valid = env.count_valid_per_group(env.state["contribution_valid"])
        members = env.count_members_per_group()
        cg_per_agent = env.state["common_good"]
        cg_per_group = (cg_per_agent.unsqueeze(-2) * env.agent_group_mask).sum(
            dim=1
        ) / members.clamp(min=1)
        pool = pool_from_state(env)
        raw_pool = raw_pool_from_state(env)
        # This file's own division, so guard 1 under the per-capita mode does
        # not rest on re-running the env's expression. A group with nobody
        # valid has an all-zero pool, so the clamp cannot invent a share.
        divisor = n_valid.to(th.float).clamp(min=1)
        rows.append(
            {
                "round": rnd,
                "reward": env.reward.detach().clone(),
                "pool": pool,
                "raw_pool": raw_pool,
                "cg_times_nvalid": cg_per_group * n_valid,
                "share": pool / divisor,
                "raw_share": raw_pool / divisor,
                "common_good_state": cg_per_group,
                "n_valid": n_valid.to(th.float),
                "members": members.to(th.float),
                "reward_times_nvalid": env.reward.detach().clone()
                * n_valid.to(th.float),
            }
        )
        _, _, done = env.step()
        if done:
            break
    return rows, served_at_timeout, served_prev_at_timeout, watched


def residuals(rows, key, ref="reward"):
    d = th.cat([(r[ref] - r[key]).abs().reshape(-1) for r in rows])
    return {"max": float(d.max()), "mean": float(d.mean())}


def cross_residuals(rows_a, rows_b, key_a, key_b=None):
    """The same column, or two named columns, across two rollouts."""
    key_b = key_a if key_b is None else key_b
    d = th.cat(
        [(a[key_a] - b[key_b]).abs().reshape(-1) for a, b in zip(rows_a, rows_b)]
    )
    return {"max": float(d.max()), "mean": float(d.mean())}


def _cat(rows, key):
    return th.cat([r[key].reshape(-1) for r in rows])


def ratio_stats(rows_num, rows_den, eps=1e-4):
    """`common_pool` reward over `common_pool_per_capita` reward.

    Group-rounds where the per-capita reward is ~0 are dropped: an empty or
    fully timed-out group has a zero pool and there is no ratio to take.
    """
    num = _cat(rows_num, "reward")
    den = _cat(rows_den, "reward")
    keep = den.abs() > eps
    r = num[keep] / den[keep]
    nearest = r.round()
    return {
        "n_group_rounds": int(keep.sum()),
        "min": float(r.min()),
        "max": float(r.max()),
        "distinct_nearest_integers": sorted({int(x) for x in nearest.tolist()}),
        "max_abs_dev_from_nearest_integer": float((r - nearest).abs().max()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    device = th.device("cpu")
    mode = cfg["env_args"]["reward_mode"]
    per_capita = mode == "common_pool_per_capita"
    report = {"config": args.config, "reward_mode": mode}

    # ---- guard 1 + guard 2, under the configured mode --------------------- #
    env, opponent = build_env(cfg, args.batch_size, device)
    rows, served, served_prev, watched = run_rollout(
        env, opponent, args.rounds, watch_punishment=True, seed=cfg["seed"]
    )
    if per_capita:
        report["guard1"] = {
            "expected": (
                "1.6*sum(c) - sum(p), divided by count_valid_per_group -- the "
                "players who gave an input, which is the divisor "
                "share_pool_per_group uses and the game's own rule"
            ),
            "vs_recomputed_share": residuals(rows, "share"),
            "vs_raw_share_no_zeroing": residuals(rows, "raw_share"),
            "vs_common_good_state_field": residuals(rows, "common_good_state"),
            "mean_reward": float(_cat(rows, "reward").mean()),
        }
    else:
        report["guard1"] = {
            "expected": "1.6*sum(c) - sum(p), undivided",
            "vs_recomputed_pool": residuals(rows, "pool"),
            "vs_raw_pool_no_zeroing": residuals(rows, "raw_pool"),
            "vs_common_good_times_n_valid": residuals(rows, "cg_times_nvalid"),
            "mean_reward": float(_cat(rows, "reward").mean()),
        }
    report["guard2"] = {
        "punishment_values_served_at_timeout": {
            str(k): v for k, v in sorted(served.items())
        },
        # keyed on round t-1's validity: see the mask note in `probed`
        "prev_punishment_values_served_at_prev_timeout": {
            str(k): v for k, v in sorted(served_prev.items())
        },
        "timeout_cells_observed": watched["timeout_cells"],
        "agent_cells_observed": watched["agent_cells"],
        "timeout_rate": (
            watched["timeout_cells"] / watched["agent_cells"]
            if watched["agent_cells"]
            else None
        ),
    }

    # ---- guard 1c: the same rollout under `sum` must NOT match the pool --- #
    env_sum, opp_sum = build_env(cfg, args.batch_size, device, reward_mode="sum")
    rows_sum, _, _, _ = run_rollout(
        env_sum, opp_sum, args.rounds, watch_punishment=False, seed=cfg["seed"]
    )
    report["guard1c_sum_contrast"] = {
        "vs_recomputed_pool": residuals(rows_sum, "pool"),
        "mean_reward": float(_cat(rows_sum, "reward").mean()),
    }

    served_nonzero = {k: v for k, v in served.items() if k != 0.0}
    prev_nonzero = {k: v for k, v in served_prev.items() if k != 0.0}
    guard1_key = (
        "guard1_reward_is_the_pool_over_n_valid"
        if per_capita
        else "guard1_reward_is_common_pool"
    )
    own_key = "vs_recomputed_share" if per_capita else "vs_recomputed_pool"
    cg_key = (
        "vs_common_good_state_field" if per_capita else "vs_common_good_times_n_valid"
    )
    report["verdict"] = {
        guard1_key: (
            report["guard1"][own_key]["max"] < 1e-3
            and report["guard1"][cg_key]["max"] < 1e-3
        ),
        "guard1c_sum_differs": (
            report["guard1c_sum_contrast"]["vs_recomputed_pool"]["max"] > 1.0
        ),
        "guard2_no_punishment_at_timeout": (not served_nonzero and not prev_nonzero),
        "guard2_saw_timeouts": watched["timeout_cells"] > 0,
    }

    # ---- guard 3: divided, not relabelled --------------------------------- #
    if per_capita:
        env_pool, opp_pool = build_env(
            cfg, args.batch_size, device, reward_mode="common_pool"
        )
        rows_pool, _, _, _ = run_rollout(
            env_pool, opp_pool, args.rounds, watch_punishment=False, seed=cfg["seed"]
        )
        # (c) the controlled equal-headcount rollout, both modes on it
        eq = {}
        for tag, rm in (("per_capita", None), ("common_pool", "common_pool")):
            e, o = build_env(
                cfg,
                args.batch_size,
                device,
                reward_mode=rm,
                drop_switch=True,
                force_all_valid=True,
            )
            eq[tag], _, _, _ = run_rollout(
                e, o, args.rounds, watch_punishment=False, seed=cfg["seed"]
            )
        eq_members = _cat(eq["per_capita"], "members")
        eq_valid = _cat(eq["per_capita"], "n_valid")
        headcount = float(eq_members.min())
        eq_ratio = ratio_stats(eq["common_pool"], eq["per_capita"])

        report["guard3_divided_not_relabelled"] = {
            "a_same_trajectory": cross_residuals(rows_pool, rows, "pool"),
            "a_reward_gap_vs_common_pool": cross_residuals(rows_pool, rows, "reward"),
            "a_mean_reward_common_pool": float(_cat(rows_pool, "reward").mean()),
            "a_mean_reward_per_capita": float(_cat(rows, "reward").mean()),
            "a_ratio_free_rollout": ratio_stats(rows_pool, rows),
            "b_reward_times_n_valid_vs_pool": residuals(
                rows, "pool", ref="reward_times_nvalid"
            ),
            "c_equal_headcount": {
                "members_min": float(eq_members.min()),
                "members_max": float(eq_members.max()),
                "n_valid_min": float(eq_valid.min()),
                "n_valid_max": float(eq_valid.max()),
                "headcount": headcount,
                "ratio": eq_ratio,
                "reward_times_n_vs_common_pool": cross_residuals(
                    eq["per_capita"],
                    eq["common_pool"],
                    "reward_times_nvalid",
                    "reward",
                ),
            },
        }
        g3 = report["guard3_divided_not_relabelled"]
        report["verdict"].update(
            {
                "guard3a_same_trajectory": g3["a_same_trajectory"]["max"] < 1e-3,
                # a renamed constant would show a gap of 0, and a shifted one a
                # constant gap; this is a per-group-round gap of many points
                "guard3a_differs_from_common_pool": (
                    g3["a_reward_gap_vs_common_pool"]["max"] > 10.0
                    and g3["a_reward_gap_vs_common_pool"]["mean"] > 1.0
                ),
                "guard3a_divisor_varies_per_group": (
                    len(g3["a_ratio_free_rollout"]["distinct_nearest_integers"]) > 1
                ),
                "guard3b_reward_times_n_valid_is_the_pool": (
                    g3["b_reward_times_n_valid_vs_pool"]["max"] < 1e-3
                ),
                "guard3c_headcount_really_equal": (
                    g3["c_equal_headcount"]["members_min"]
                    == g3["c_equal_headcount"]["members_max"]
                    == g3["c_equal_headcount"]["n_valid_min"]
                    == g3["c_equal_headcount"]["n_valid_max"]
                ),
                # the one form in which pool and pool/n can "agree": a single
                # ratio, exactly the headcount. A relabelling would give 1.
                "guard3c_ratio_is_exactly_the_headcount": (
                    eq_ratio["distinct_nearest_integers"] == [int(headcount)]
                    and eq_ratio["max_abs_dev_from_nearest_integer"] < 1e-3
                    and headcount > 1
                ),
                "guard3c_exact_agreement_after_scaling": (
                    g3["c_equal_headcount"]["reward_times_n_vs_common_pool"]["max"]
                    < 1e-3
                ),
            }
        )

    report["verdict"]["ALL_PASS"] = all(
        v for k, v in report["verdict"].items() if k != "ALL_PASS"
    )

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    return 0 if report["verdict"]["ALL_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
