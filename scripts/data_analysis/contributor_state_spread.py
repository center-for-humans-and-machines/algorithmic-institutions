"""The state-spread diagnostic, with the noise machinery disabled.

The protocol asks every contributor-slot change for one judgement that the
22 rows cannot give: how much of the human variance in contributions the
model's own conditional expectation still explains once the simulation feeds
on its own output. PR #186 established the decomposition on a free-running
copula-off simulation,

    Var(c) = Var(E[c | history]) + Var(residual),

and measured `Var(E[c | history])` = 27.93 on human histories against 18.88
in the closed loop for the categorical stimulus-skip trunk -- the loop keeps
the per-round noise and loses the state spread. This script reproduces that
measurement for two trunks that differ only in `prev_contribution_max`, each
teacher-forced over (a) the 50 single-copy human games and (b) its OWN
copula-off simulation's realised states.

The human pass under the baseline trunk is the calibration: it must return
PR #186's 27.93. If it does, this tree measures the same thing PR #186 did.

Two stages, because the trunk unpickles torch_geometric modules:

  Raven:  .venv/bin/python scripts/data_analysis/contributor_state_spread.py \
              --teacher-force            # writes tf_*.parquet
  local:  python scripts/data_analysis/contributor_state_spread.py \
              --analyse                  # the table

Outputs under plots/data_analysis/evaluation/contributor_ceiling_indicator/.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "artificial_humans"))
sys.path.insert(0, str(ROOT / "scripts" / "data_analysis"))

OUT = ROOT / "plots/data_analysis/evaluation/contributor_ceiling_indicator"
SIM = ROOT / "plots/simulation"
AH = ROOT / "artifacts/artificial_humans"
PT = "model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
STACK = "_self_gnncopar1_contr_gnn_switch_ceiling"

# label -> (BARE trunk, its own copula-off simulation)
ARMS = {
    "baseline": (
        AH / "group_switching_contribution_50ep_vnode_stimulus_skip" / PT,
        SIM / f"23_2g8a_contr_skip_copoff{STACK}" / "per_round.parquet",
    ),
    "candidate": (
        AH / "group_switching_contribution_50ep_vnode_stimulus_skip_ceilind" / PT,
        SIM / f"23_2g8a_contr_ceilind_copoff{STACK}" / "per_round.parquet",
    ),
}
PR186_HUMAN_VAR_E = 27.93  # the calibration target for the baseline trunk


def tf_frame(model, data, idx):
    """One row per valid agent-round: observed c, E[c | history], and the
    predictive SD and entropy of the teacher-forced marginal."""
    import contribution_copula_rho as cc

    rows = cc.teacher_forced_rows(model, data, idx)
    P = rows["P"]
    lev = np.arange(P.shape[1], dtype=np.float64)
    e = P @ lev
    var = P @ lev**2 - e**2
    ent = -(P * np.log(np.clip(P, 1e-12, None))).sum(1)
    return pd.DataFrame(
        dict(
            episode=rows["episode"],
            agent=rows["agent"],
            round=rows["round"],
            group=rows["group"],
            c=rows["y"].astype(float),
            e=e,
            sd=np.sqrt(np.clip(var, 0, None)),
            entropy=ent,
        )
    )


def teacher_force():
    import contribution_copula_rho as cc
    import torch as th
    from rcb_teacher_forced import check_sim_defaults, load_sim

    from aimanager.generic.graph import GraphNetwork

    OUT.mkdir(parents=True, exist_ok=True)
    data, _, key_to_idx, defaults = cc.load_full()
    tr = cc.select_split(key_to_idx, cc.TRAIN, cc.N_TRAIN_EP)
    te = cc.select_split(key_to_idx, cc.TEST, cc.N_TEST_EP)
    idx = np.array(sorted(set(tr.tolist()) | set(te.tolist())))

    for arm, (trunk, sim_path) in ARMS.items():
        model = GraphNetwork.load(str(trunk), device="cpu")
        model.eval()
        assert model.copula_rho == 0.0, f"{arm}: teacher-force the BARE trunk"
        print(f"\narm {arm}: {cc.rel(trunk)}")
        print(f"  x_encoding={[e['name'] for e in model.x_encoding]}")
        with th.no_grad():
            df = tf_frame(model, data, idx)
        df["episode"] = idx[df["episode"]]
        df.to_parquet(OUT / f"tf_human_{arm}.parquet", index=False)
        print(f"  human: {len(df)} rows over {len(idx)} single-copy games")

        assert sim_path.exists(), f"{arm}: no copula-off sim at {sim_path}"
        check_sim_defaults(model, defaults)
        sim, _ = load_sim(sim_path, defaults)
        with th.no_grad():
            df = tf_frame(model, sim, np.arange(sim["contribution"].shape[0]))
        df.to_parquet(OUT / f"tf_sim_{arm}.parquet", index=False)
        print(f"  sim:   {len(df)} rows from {sim_path.parent.name}")


def summarise(tf):
    resid = tf["c"] - tf["e"]
    return {
        "n_rows": len(tf),
        "var_c": tf["c"].var(),
        "var_cond_mean": tf["e"].var(),
        "var_resid": resid.var(),
        "pred_sd_mean": tf["sd"].mean(),
        "resid_sd": resid.std(),
        "entropy_mean": tf["entropy"].mean(),
    }


def cg_ratio(tf):
    """SD of the group-mean contribution over SD of individual contributions
    -- the CG ingredient, on exactly the rows the decomposition uses."""
    gm = tf.groupby(["episode", "round", "group"])["c"].mean()
    return gm.std() / tf["c"].std()


def analyse():
    rows = []
    for arm in ARMS:
        for src in ("human", "sim"):
            p = OUT / f"tf_{src}_{arm}.parquet"
            if not p.exists():
                print(f"missing {p} -- run --teacher-force on Raven first")
                continue
            tf = pd.read_parquet(p)
            rows.append(
                {
                    "arm": arm,
                    "states": "human histories" if src == "human" else "closed loop",
                    **summarise(tf),
                    "cg_ratio": cg_ratio(tf),
                }
            )
    tab = pd.DataFrame(rows)
    if tab.empty:
        return
    # retention: each arm's closed-loop Var(E) over its OWN human-history
    # value, so a worse fit is not charged as a contraction (PR #191's rule)
    hv = {
        r["arm"]: r["var_cond_mean"] for r in rows if r["states"] == "human histories"
    }
    tab["retention"] = [
        r["var_cond_mean"] / hv[r["arm"]] if r["arm"] in hv else np.nan for r in rows
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "state_spread.csv", index=False)
    print(tab.to_string(index=False, float_format=lambda v: f"{v: .4f}"))

    base = tab[(tab["arm"] == "baseline") & (tab["states"] == "human histories")]
    if len(base):
        got = float(base["var_cond_mean"].iloc[0])
        gap = abs(got - PR186_HUMAN_VAR_E)
        print(
            f"\nCALIBRATION baseline trunk on human histories: "
            f"Var(E[c|hist]) = {got:.4f} vs PR #186's {PR186_HUMAN_VAR_E} "
            f"-> {'PASS' if gap < 0.05 else 'FAIL'}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-force", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.teacher_force:
        teacher_force()
    if a.analyse:
        analyse()
