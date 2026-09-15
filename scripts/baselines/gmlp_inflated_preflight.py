"""Pre-sim diagnostic for the inflated contributor: a SCORED, schedule-matched
closed-loop proxy of the real simulation. Step 8 of
notes/autoresearch_log/contribution-inflated-gmlp.md.

What it is
----------
Two arms -- the parent's stamped incumbent
(`contribution_gaussian_mlp_v2_group_copula.joblib`) and this branch's stamped
candidate (`contribution_gaussian_mlp_inflated_group_copula.joblib`) -- are
rolled through the REAL punisher bundle along the PARENT SIM'S OWN per-episode
regrouping schedule, read agent-by-agent-by-round off its `per_round.parquet`.
The env round order is reproduced exactly (contribution -> punishment ->
per-group per-capita common good -> `prev_*` shift), 100 episodes x 24 rounds,
torch seed 42, every agent valid. Each arm is written as a per-round frame,
loaded through the frozen suite's `convert.load_sim` and scored against
`convert.load_human` with `scoring.score_all(n_repeats=500, seed=42)`. The
evaluation suite is imported READ-ONLY; nothing under it is written.

What it is NOT
--------------
* **It is never a gate.** Amendment A of the orchestrator's validation: the
  atom set (`prev,0,20`) and the dose (0.048443521435665396) are frozen before
  this runs and may not be revisited on the strength of what it prints. The
  simulation runs whatever this says.
* **It is a proxy, and its error bar is the point.** The incumbent arm does not
  reproduce the real parent's scores; the per-row gap (`offset` in the table)
  runs 0.1-0.3, which is the same size as the distance to a band edge. A delta
  smaller than a row's own offset predicts nothing.
* **It cannot see the switch slot.** The regrouping schedule is FIXED at the
  parent's, so the switch model never reacts to the changed contributions.
  SA / SB / SC read the switch sequence alone and are therefore BIT-IDENTICAL
  in the two arms -- their zero deltas are structural, not predictions.
  PR #175 moved SC by 0.16 through the per-capita common-good channel; nothing
  here forecasts that. RSA is not identical (it conditions the fixed switch
  decisions on received-punishment bins, which do move with contributions) but
  it is the least trustworthy row in the table: its incumbent-arm offset is
  1.39, ten times the typical one, because real switches respond to punishment
  and the proxy's do not.
* It also carries no validity model (the real sim's `valid_model` can void a
  contribution), which is part of why the C rows are offset.

Usage
-----
    PYTHONPATH=$PWD/src .venv/bin/python scripts/baselines/gmlp_inflated_preflight.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch as th  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "baselines"))

from aimanager.evaluation_suite.convert import load_human, load_sim  # noqa: E402
from aimanager.evaluation_suite.metrics import GROUPS  # noqa: E402
from aimanager.evaluation_suite.scoring import score_all  # noqa: E402
from aimanager.simulation.linear_ah import LinearAHAdapter  # noqa: E402

PARENT_DIR = (
    "plots/simulation/23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_"
    "gnn_joint_exodus_k_onehot_switch"
)
INCUMBENT = "artifacts/baselines/contribution_gaussian_mlp_v2_group_copula.joblib"
CANDIDATE = "artifacts/baselines/contribution_gaussian_mlp_inflated_group_copula.joblib"
PUNISHER = "artifacts/baselines/punishment_multinomial_severity_copula.joblib"
HUMAN = "experiments/2group_8agent_50ep.csv"

N_EPISODES, N_ROUNDS, N_AGENTS, N_LEVELS = 100, 24, 8, 21
SEED = 42

# Declaration §1: the gate-2 ceiling and the two declared rows' parent scores.
GATE2_CEILING = 1.3449481666413449
DECLARED = {"RCA": 3.50675152179351, "CG": 2.079263618744627}


def schedule(parquet):
    """{episode: [A, T] post-arrival group ids} off the parent's parquet."""
    df = pd.read_parquet(parquet)
    df = df.assign(aid=df.participant_code.str.split("_").str[0].astype(int))
    out = {}
    for ep, d in df.groupby("episode"):
        piv = d.pivot(index="aid", columns="round_number", values="agent_group")
        out[int(ep)] = piv.values.astype(int)
    return out


def env_state(t, prev, groups, prev_groups):
    """The subset of the env state the contribution adapter records: prev_*
    measures of round t-1, membership post-arrival at t and pre-arrival at
    t-1."""

    def col(x, dtype):
        return th.tensor(np.asarray(x).reshape(1, -1, 1), dtype=dtype)

    return {
        "round_number": col(np.full(N_AGENTS, t), th.int64),
        "prev_contribution": col(prev["contribution"], th.int64),
        "prev_punishment": col(prev["punishment"], th.int64),
        "prev_common_good": col(prev["common_good"], th.float),
        "prev_agent_group": col(prev_groups, th.int64),
        "agent_group": col(groups, th.int64),
    }


def rollout(contribution_bundle, punisher_bundle, sched, label):
    """One arm: closed loop over the fixed schedule, per-round frame out."""
    ah = LinearAHAdapter(
        contribution_bundle, n_agents=N_AGENTS, n_contributions=N_LEVELS
    )
    pm = LinearAHAdapter(punisher_bundle, n_agents=N_AGENTS)
    dv = ah.default_values
    th.manual_seed(SEED)
    rows = []
    for episode in range(N_EPISODES):
        G = sched[episode]
        prev = {
            "contribution": np.full(N_AGENTS, dv["contribution"]),
            "punishment": np.full(N_AGENTS, dv["punishment"]),
            "common_good": np.full(N_AGENTS, dv["common_good"]),
        }
        rounds = []
        for t in range(N_ROUNDS):
            groups = G[:, t]
            prev_groups = G[:, t - 1] if t else groups
            pred, _ = ah.predict(
                env_state(t, prev, groups, prev_groups), reset_rnn=(t == 0)
            )
            c = pred.reshape(-1).numpy().astype(np.int64)
            rd = {
                "contribution": c.tolist(),
                "contribution_valid": [True] * N_AGENTS,
                "punishment": [None] * N_AGENTS,
                "punishment_valid": [False] * N_AGENTS,
                "agent_group": groups.tolist(),
                "round": t,
            }
            p = pm.get_punishments(rounds + [rd]).numpy().astype(np.int64)
            rounds.append(
                {**rd, "punishment": p.tolist(), "punishment_valid": [True] * N_AGENTS}
            )
            cg = np.empty(N_AGENTS)
            for g in np.unique(groups):
                sel = groups == g
                cg[sel] = (1.6 * c[sel].sum() - p[sel].sum()) / sel.sum()
            rows.append(
                pd.DataFrame(
                    dict(
                        episode=episode,
                        participant_code=[f"{a}_{episode}" for a in range(N_AGENTS)],
                        round_number=t,
                        punishment=p,
                        common_good=cg,
                        contribution=c,
                        agent_group=groups,
                        payoff=0.0,
                        group_id=groups,
                        run=label,
                    )
                )
            )
            prev = {"contribution": c, "punishment": p, "common_good": cg}
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- U1-U4 ---
def _with_prev(canon):
    d = canon.sort_values(["episode_id", "participant_code", "round_number"]).copy()
    d["prev"] = d.groupby(["episode_id", "participant_code"])["contribution"].shift(1)
    return d


def u1(canon):
    """Closed-loop repetition: P(c_t = c_{t-1}) and P(20 | prev 20)."""
    d = _with_prev(canon).dropna(subset=["contribution", "prev"])
    at20 = d[d["prev"] == 20]
    return {
        "P(c=prev)": float((d["contribution"] == d["prev"]).mean()),
        "P(20|prev20)": float((at20["contribution"] == 20).mean()),
        "n_prev20": int(len(at20)),
    }


def u2(human, arms):
    """RCA's per-round-type EMD, full-sample (no resampling), human-weighted
    exactly as `MetricGroup.d` weights it -- so the share of the weighted
    reduction carried by `no_switch_allowed` is readable."""
    grp = GROUPS["R"]
    w = grp.weights("RCA", human)
    a = grp.rca(human)
    a_strata = dict(iter(a.groupby(level=0)))
    out = {}
    for label, canon in arms.items():
        b_strata = dict(iter(grp.rca(canon).groupby(level=0)))
        emd = pd.Series(
            {s: wasserstein_distance(a_strata[s], b_strata[s]) for s in w.index}
        )
        out[label] = emd
    return w, pd.DataFrame(out)


def u3(canon):
    """The CG signature: group-mean tails and the two SDs behind the ratio."""
    d = canon.dropna(subset=["contribution"])
    cells = d.groupby(["episode_id", "round_number", "group_id"])["contribution"]
    gm = cells.mean()
    sd_g = float(gm.std(ddof=0))
    sd_i = float(d["contribution"].std(ddof=0))
    return {
        "P(gmean>=18)": float((gm >= 18).mean()),
        "P(gmean<=2)": float((gm <= 2).mean()),
        "sd_group_means": sd_g,
        "sd_individual": sd_i,
        "ratio": sd_g / sd_i,
        "n_cells": int(len(gm)),
    }


def u4(canon):
    c = canon["contribution"].dropna()
    return {
        "P(c=0)": float((c == 0).mean()),
        "P(c=20)": float((c == 20).mean()),
        "mean": float(c.mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(_ROOT))
    ap.add_argument("--out-dir", default=None, help="where to write arm parquets")
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "data" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    sched = schedule(root / PARENT_DIR / "per_round.parquet")
    punisher = joblib.load(root / PUNISHER)
    arms = {
        "incumbent": joblib.load(root / INCUMBENT),
        "candidate": joblib.load(root / CANDIDATE),
    }
    print("arms (stamped bundles as committed):")
    for label, b in arms.items():
        print(
            f"  {label:10s} model={b['model']:24s} "
            f"rho_p={b.get('copula_rho_p')} rho_t={b.get('copula_rho_t')} "
            f"atoms={b.get('atoms')}"
        )

    human = load_human(root / HUMAN)
    frames, canon = {}, {}
    for label, bundle in arms.items():
        t0 = time.time()
        fr = rollout(bundle, punisher, sched, label)
        path = out_dir / f"preflight_{label}.parquet"
        fr.to_parquet(path, index=False)
        canon[label] = list(load_sim(path).values())[0]
        frames[label] = canon[label]
        print(f"rollout {label}: {time.time() - t0:.0f}s -> {path}")

    t0 = time.time()
    scores = score_all(human, frames, n_repeats=500, seed=SEED)
    print(f"scoring: {time.time() - t0:.0f}s")

    tab = scores.pivot(index="metric", columns="run", values="score")
    real = pd.read_csv(root / PARENT_DIR / "evaluation" / "scores.csv").set_index(
        "metric"
    )["score"]
    order = [m for g in GROUPS.values() for m in g.KINDS]
    tab = tab.reindex(order)
    tab.insert(0, "real parent", real.reindex(order))
    tab["delta"] = tab["candidate"] - tab["incumbent"]
    tab["offset"] = tab["incumbent"] - tab["real parent"]
    tab["predicted real"] = tab["real parent"] + tab["delta"]
    tab["|delta|>|offset|"] = tab["delta"].abs() > tab["offset"].abs()

    pd.set_option("display.width", 200)
    print("\n=== 21-row proxy table (exact, one draw, no re-runs) ===")
    print(tab.to_string())
    print("\n--- arm means (21 rows) ---")
    for col in ("real parent", "incumbent", "candidate", "predicted real"):
        print(f"  {col:16s} {tab[col].mean()!r}")
    print(f"  gate-2 ceiling   {GATE2_CEILING!r}")
    print(
        "  predicted real mean vs ceiling: "
        f"{tab['predicted real'].mean() - GATE2_CEILING!r}"
    )
    print("\n--- rows <= 1 (context, not a criterion) ---")
    print(
        {
            c: int((tab[c] <= 1).sum())
            for c in ("real parent", "incumbent", "candidate", "predicted real")
        }
    )
    print("\n--- declared rows ---")
    for row, parent in DECLARED.items():
        print(
            f"  {row}: real {parent!r} | incumbent {tab.loc[row, 'incumbent']!r} "
            f"| candidate {tab.loc[row, 'candidate']!r} | delta "
            f"{tab.loc[row, 'delta']!r} | offset {tab.loc[row, 'offset']!r} "
            f"| predicted real {tab.loc[row, 'predicted real']!r}"
        )

    print("\n=== U1-U4, closed-loop on the rollout (NOT the step-4 ===")
    print("=== teacher-forced likelihood diagnostics of Note 20)  ===")
    hu = {"U1": u1(human), "U3": u3(human), "U4": u4(human)}
    print("U1 exact repetition:")
    print(f"  human     {hu['U1']}")
    for label in arms:
        print(f"  {label:9s} {u1(canon[label])}")
    w, emd = u2(human, canon)
    print("U2 RCA per-round-type EMD (full sample, human weights):")
    print(pd.concat([w.rename("weight"), emd], axis=1).to_string())
    num = (emd.mul(w, axis=0).sum() / w.sum()).rename("weighted RCA d")
    print(num.to_string())
    drop_total = num["incumbent"] - num["candidate"]
    ns = "no_switch_allowed"
    drop_ns = (emd.loc[ns, "incumbent"] - emd.loc[ns, "candidate"]) * w[ns] / w.sum()
    print(f"  total weighted reduction {drop_total!r}")
    print(f"  {ns} contribution {drop_ns!r} -> share {drop_ns / drop_total!r}")
    print("U3 CG signature:")
    print(f"  human     {hu['U3']}")
    for label in arms:
        print(f"  {label:9s} {u3(canon[label])}")
    print("U4 overshoot watch:")
    print(f"  human     {hu['U4']}")
    for label in arms:
        print(f"  {label:9s} {u4(canon[label])}")

    print(
        "\nREMINDER: diagnostic only (Amendment A). SA/SB/SC are identical "
        "across arms by construction -- the regrouping schedule is the "
        "parent's and the switch model never reacts; RSA moves only through "
        "punishment bins and carries the table's largest offset. A row whose "
        "|delta| is below its own |offset| predicts nothing."
    )


if __name__ == "__main__":
    main()
