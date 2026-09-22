"""Run the standard battery on a manager, with its baselines, and score it.

This is the entry point the language-model manager will be handed to. It was
written and validated against a stub that returns fixed punishments, so the
harness was finished and checked before the model existed -- which is the
only way the first language-model number can mean anything.

**The objective is the group's undivided common pool**, so every table
leads with it and the power calculation is sized on it. Total contribution
and pool per member stay as diagnostics beside it and are never averaged
with it: two managers can reach the same pool by opposite routes, one by
raising contributions and paying for them and one by doing nothing, and
only the contribution column tells them apart.

Three things it does in one invocation, because none of them is safe to
quote from elsewhere:

  * the battery, for the manager under test and for the clone, `thr9_p10`,
    never-punish and #219's capped sigmoid rule, all against the same rival
    on the same stack from the same seeds;
  * the symmetric controls, both seats holding the same manager, which is
    the noise floor every effect has to be read against;
  * the minimum detectable difference across 50 to 3,000 episodes, and the
    episode count at which each already-measured effect becomes detectable.

`--validate-217` adds PR #217's four inverted rules and prints this
harness's numbers beside that arm's published ones. If they disagree, the
harness is wrong until proven otherwise.

Usage:
    PYTHONPATH=src python scripts/llm_manager/run_battery.py \\
        --out plots/data_analysis/llm_manager_eval/validation \\
        --episodes 100 --seeds 42,43,44 --validate-217
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from aimanager.llm_manager import battery as bat
from aimanager.llm_manager.harness import (
    contingency_frame,
    load_models,
    run_battery,
)
from aimanager.llm_manager.stub import THR9_P10_TABLE, StubManager


#: The plan's baseline set, plus the two stubs the harness was built
#: against. `stub_zero` must come out identical to `never` and
#: `stub_thr9_p10` identical to `thr9_p10`, cell for cell -- that is the
#: check that the stub really is a manager and not a special case.
def base_arms(with_stub=True):
    arms = {
        "clone": ("clone", "clone"),
        "never": ("never", "clone"),
        "thr9_p10": ("thr9_p10", "clone"),
        "capped_sigmoid": ("capped_sigmoid", "clone"),
        # three symmetric controls, not one: the noise floor is a property
        # of the world a manager makes, and a silent world is a quieter one
        # (#217 measured -0.33 members for `never_vs_never` against +0.08
        # for the clone's own control). `clone` above is the third, and is
        # both the clone baseline and its own symmetric control.
        "never_vs_never": ("never", "never"),
        "thr9_vs_thr9": ("thr9_p10", "thr9_p10"),
    }
    if with_stub:
        arms["stub_zero"] = (StubManager(punishment=0), "clone")
        arms["stub_thr9_p10"] = (
            StubManager(
                table=THR9_P10_TABLE,
                prompt_tokens_per_call=900,
                completion_tokens_per_call=12,
            ),
            "clone",
        )
    return arms


#: PR #217's four inverted mirrors, through the very class that arm ran.
ARMS_217 = {
    "inv_thr11_p10": (
        {"kind": "rule", "rule": "inv_threshold", "threshold": 11, "amount": 10},
        "clone",
    ),
    "inv_thr7_p10": (
        {"kind": "rule", "rule": "inv_threshold", "threshold": 7, "amount": 10},
        "clone",
    ),
    "band16_p10": (
        {"kind": "rule", "rule": "inv_threshold", "threshold": 16, "amount": 10},
        "clone",
    ),
    "band16_p20": (
        {"kind": "rule", "rule": "inv_threshold", "threshold": 16, "amount": 20},
        "clone",
    ),
}

#: What PR #217 published, focal seat against the clone, 3 seeds x 100
#: episodes. `share` is that arm's "per member" column
#: (`rule_vs_clone_paired_report.share_corr`), NOT pool / members.
#: `mean_p` is its `mean_punishment_valid`. Source: the PR body, Result 1,
#: Result 3 and the mechanism table.
PUBLISHED_217 = {
    "never": {"members": 4.57, "pool": 62.22, "share": 12.82, "c_gap": -1.20},
    "thr9_p10": {
        "members": 3.92,
        "pool": 62.32,
        "share": 14.02,
        "mean_p": 2.80,
        "c_gap": -3.51,
    },
    "inv_thr11_p10": {
        "members": 3.55,
        "pool": 29.51,
        "share": 7.96,
        "mean_p": 2.17,
        "c_gap": 1.69,
    },
    "inv_thr7_p10": {
        "members": 2.78,
        "pool": 20.75,
        "share": 6.96,
        "mean_p": 4.49,
        "c_gap": 1.83,
    },
    "band16_p10": {
        "members": 4.04,
        "pool": 40.51,
        "share": 9.68,
        "mean_p": 1.07,
        "c_gap": 0.89,
    },
    "band16_p20": {"mean_p": 1.62},
    "clone": {"c_gap": -2.35},
}

#: PR #219's own cross-check of `paired_rollout` against #217: one rollout,
#: 1024 episodes, seed 42 (log section 3.1). This is the tightest target
#: available, because it is the same rollout code on the same four
#: artifacts; only the RNG stream differs, since #219 ran every anchor
#: inside one batched rollout and this harness gives each arm its own.
#: `mean_p` here is per member-round, which is what that table's column is.
PUBLISHED_219_CROSSCHECK = {
    "never": {
        "members": 4.62,
        "pool": 62.64,
        "mean_p_member": 0.0,
        "mean_c": 8.64,
        "c_gap": -1.45,
    },
    "thr9_p10": {
        "members": 3.91,
        "pool": 60.69,
        "mean_p_member": 2.90,
        "mean_c": 11.74,
        "c_gap": -3.64,
    },
    "clone": {
        "members": 3.99,
        "mean_p_member": 1.85,
        "mean_c": 10.03,
        "c_gap": -2.26,
    },
}

#: PR #219's held-out validation table, seeds 45/46/47 at 2,048 episodes
#: each (6,144 per rule), focal seat against the clone. Run this harness at
#: those seeds to compare like with like. `capped_sigmoid` is that table's
#: `best_cap10_pool`, which is the P_max-capped rule, not the
#: severity-constrained one.
PUBLISHED_219_VALIDATION = {
    "never": {"contribution": 37.30, "pool": 59.68, "members": 4.58},
    "thr9_p10": {
        "contribution": 44.64,
        "pool": 60.19,
        "members": 3.87,
        "mean_p_member": 2.90,
    },
    "clone": {
        "contribution": 40.09,
        "pool": 56.85,
        "members": 4.04,
        "mean_p_member": 1.80,
    },
    "capped_sigmoid": {
        "contribution": 43.35,
        "pool": 65.19,
        "members": 4.41,
        "mean_p_member": 0.94,
    },
}

MEASURED = {
    "members": "focal_members",
    "pool": "focal_pool",
    "contribution": "focal_contribution",
    "share": "focal_share_roundmean",
    "mean_p": "focal_mean_punishment_valid",
    "mean_p_member": "focal_mean_punishment",
    "mean_c": "focal_mean_contribution",
    "c_gap": "c_gap",
}


def validation_table(battery, published, source):
    b = battery.set_index("arm")
    rows = []
    for arm, ref in published.items():
        if arm not in b.index:
            continue
        for key, want in ref.items():
            got = float(b.loc[arm, MEASURED[key]])
            se = b.loc[arm].get(MEASURED[key] + "_se", np.nan)
            rows.append(
                {
                    "source": source,
                    "arm": arm,
                    "quantity": key,
                    "published": want,
                    "measured": got,
                    "delta": got - want,
                    "se_measured": float(se) if se == se else np.nan,
                    "sd_away": (
                        abs(got - want) / float(se) if se == se and se else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def stub_identity_table(battery):
    """The stub must be indistinguishable from the rule it is wearing."""
    b = battery.set_index("arm")
    rows = []
    for stub, rule in (("stub_zero", "never"), ("stub_thr9_p10", "thr9_p10")):
        if stub not in b.index or rule not in b.index:
            continue
        for q in bat.HEADLINE:
            rows.append(
                {
                    "stub": stub,
                    "rule": rule,
                    "quantity": q,
                    "stub_value": float(b.loc[stub, f"focal_{q}"]),
                    "rule_value": float(b.loc[rule, f"focal_{q}"]),
                    "delta": float(
                        b.loc[stub, f"focal_{q}"] - b.loc[rule, f"focal_{q}"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--validate-217", action="store_true")
    ap.add_argument("--no-stub", action="store_true")
    ap.add_argument(
        "--mdd-episodes",
        default=",".join(str(n) for n in bat.EPISODE_BUDGETS),
        help="budgets the minimum detectable difference is published at",
    )
    args = ap.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    mdd_at = tuple(int(s) for s in args.mdd_episodes.split(","))
    os.makedirs(args.out, exist_ok=True)

    arms = base_arms(with_stub=not args.no_stub)
    if args.validate_217:
        arms.update(ARMS_217)

    models = load_models(device=args.device)
    with open(os.path.join(args.out, "run_args.json"), "w") as f:
        json.dump(
            {**vars(args), "stack": models["_stack"], "arms": list(arms)}, f, indent=2
        )

    print(
        f"\n{len(arms)} arms x {len(seeds)} seeds x {args.episodes} episodes "
        f"= {len(arms) * len(seeds) * args.episodes} episodes\n"
    )
    out = run_battery(
        arms,
        models,
        episodes=args.episodes,
        chunk=args.chunk,
        seeds=seeds,
        device=args.device,
    )
    battery, episodes = out["battery"], out["episodes"]

    battery.to_csv(os.path.join(args.out, "battery.csv"), index=False)
    episodes.to_parquet(os.path.join(args.out, "episodes.parquet"), index=False)
    contingency_frame(out["contingency"]).to_parquet(
        os.path.join(args.out, "contingency.parquet"), index=False
    )

    shape = pd.concat(
        [
            bat.policy_shape(g).assign(arm=name)
            for name, g in episodes.groupby("arm", sort=False)
        ],
        ignore_index=True,
    )
    shape.to_csv(os.path.join(args.out, "policy_shape.csv"), index=False)

    targeting_cols = [
        "arm",
        "rho",
        "rho_floor",
        "rho_rel",
        "magnitude",
        "noise_gate",
        "tie_frac_c",
        "tie_frac_p",
        "zero_share",
        "punish_rate",
        "mean_p_valid",
        "mean_p_given_positive",
        "n_decisions",
    ]
    battery[targeting_cols].to_csv(os.path.join(args.out, "targeting.csv"), index=False)
    bat.leaver_ordering(battery).to_csv(
        os.path.join(args.out, "leaver_ordering.csv"), index=False
    )

    # the noise floor, from every symmetric control in the run
    sym = [a for a, sp in arms.items() if isinstance(sp, tuple) and sp[0] == sp[1]]
    floors = pd.concat(
        [bat.noise_floor(episodes[episodes["arm"] == a], arm=a) for a in sym],
        ignore_index=True,
    )
    floors.to_csv(os.path.join(args.out, "noise_floor.csv"), index=False)
    mdd = pd.concat(
        [
            bat.mdd_table(floors[floors["arm"] == a], mdd_at).assign(control=a)
            for a in floors["arm"].unique()
        ],
        ignore_index=True,
    )
    mdd.to_csv(os.path.join(args.out, "mdd.csv"), index=False)
    detect = pd.concat(
        [
            bat.detectability(floors[floors["arm"] == a]).assign(control=a)
            for a in floors["arm"].unique()
        ],
        ignore_index=True,
    )
    detect.to_csv(os.path.join(args.out, "detectability.csv"), index=False)

    val = pd.concat(
        [
            validation_table(battery, PUBLISHED_217, "PR #217"),
            validation_table(battery, PUBLISHED_219_CROSSCHECK, "PR #219 cross-check"),
            validation_table(battery, PUBLISHED_219_VALIDATION, "PR #219 validation"),
        ],
        ignore_index=True,
    )
    val.to_csv(os.path.join(args.out, "validation.csv"), index=False)
    stub_id = stub_identity_table(battery)
    stub_id.to_csv(os.path.join(args.out, "stub_identity.csv"), index=False)

    # each arm against the two bars, pool first: never-punishing (which
    # `thr9_p10` is indistinguishable from on the pool) and the incumbent
    # rule (which the capped one beats by +5.00). The contribution column
    # beside the pool is what says by which route an arm got there.
    con = pd.concat(
        [
            bat.contrasts(episodes, ref)
            for ref in ("never", "thr9_p10")
            if ref in set(episodes["arm"])
        ],
        ignore_index=True,
    )
    con.to_csv(os.path.join(args.out, "contrasts.csv"), index=False)

    fmt = dict(index=False, float_format=lambda v: f"{v:.3f}")
    battery_cols = ["arm", "rival", "n_episodes"] + [f"focal_{q}" for q in bat.HEADLINE]
    print("\n=== the battery, pool first (the objective) ===")
    print(
        battery[battery_cols]
        .sort_values("focal_pool", ascending=False)
        .to_string(**fmt)
    )
    print("\n=== against the bars: pool first, contribution beside it ===")
    print(con[con["quantity"].isin(("pool", "contribution"))].to_string(**fmt))
    print("\n=== policy shape, evaluation-suite bins, valid cells ===")
    print(
        shape.pivot(index="arm", columns="bin", values="mean_punishment").to_string(
            float_format=lambda v: f"{v:.2f}"
        )
    )
    print("\n=== targeting: three statistics, and the tie structure ===")
    print(battery[targeting_cols].to_string(**fmt))
    print("\n=== the leaver diagnostic, as an ordering ===")
    print(bat.leaver_ordering(battery).to_string(**fmt))
    print("\n=== noise floor: both seats the same manager ===")
    print(floors.to_string(**fmt))
    print("\n=== minimum detectable difference, on the objective first ===")
    print(mdd[mdd["quantity"] == bat.PRIMARY_OBJECTIVE].to_string(**fmt))
    print("\n--- the rest ---")
    print(mdd[mdd["quantity"] != bat.PRIMARY_OBJECTIVE].to_string(**fmt))
    print("\n=== episodes needed for an effect this project has measured ===")
    print(detect.to_string(**fmt))
    if len(val):
        print("\n=== against the published tables ===")
        print(val.to_string(**fmt))
    if len(stub_id):
        print("\n=== the stub against the rule it wears ===")
        print(stub_id.to_string(**fmt))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
