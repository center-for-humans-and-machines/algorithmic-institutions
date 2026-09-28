"""Teacher-forced sanity check of the punisher's contribution mechanism.

For each punisher artifact (linear multinomial bundle or GNN), replay the
human games (single copy, 50 episodes) and read the model's predicted
punishment distribution at every valid round -- the model sees the human
history, never its own draws. The same statistics are computed from the
observed human punishments on exactly the same rows, so the columns are
directly comparable (`notes/autoresearch_log/punisher-current-contribution.md`,
stage C):

  * P(p>0 | c_t = 20), P(p>0 | c_t <= 4): predicted P(p>0) averaged over the
    rows in that contribution class (human: the observed rate);
  * E[p | p>0] by band of c_t (0-4 / 5-9 / 10-14 / 15-19 / 20):
    sum E[p] / sum P(p>0) over the band (human: mean positive punishment);
  * the cross-tab P(p>0 | c_t = 20, c_{t-1} <= 4) / P(p>0 | c_t <= 4, c_{t-1} = 20);
  * OLS of the predicted expected punishment on c_t and c_{t-1} over the rows
    with a valid previous contribution (human: the observed punishment);
  * the in-sample NLL over the same rows (context only: 40 of the 50
    episodes are the training split).

Rows: `punishment_valid & contribution_valid` (manager and player both gave
input). Each model's tensors are built with its own stored `default_values`,
as the simulation environment does at round 0.

Linear bundles run anywhere; GNN artifacts import graph.py and therefore run
on Raven only (pass none to skip them):

    python scripts/data_analysis/punisher_mechanism_check.py \
        [--linear NAME=PATH ...] [--gnn NAME=PATH ...] [--out CSV]

Closed-loop (self-play) version: `--sim NAME=PATH[:RUN]` reads a finished
simulation's per_round.parquet (one run; RUN is the pairing name, e.g.
`gnn_self`, default: the parquet's only run) and computes the same statistics
from the realised punishments, exactly as the human row is computed from the
observed ones -- every sim row is valid, and c_{t-1} is the same agent's
contribution in the previous round of the episode. `--sim` alone (with a bare
`--linear --gnn`, no artifacts) gives the human row next to the sim rows.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch as th  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

from handcrafted_grid import build_feature_pool  # noqa: E402

from aimanager.generic.data import create_torch_data  # noqa: E402

FULL = ROOT / "experiments/2group_8agent_50ep.csv"
EXPERIMENTS = ["ah_group_switching"]
SWITCH_EVERY = 4
N_LEVELS = 31
BANDS = [-0.5, 4.5, 9.5, 14.5, 19.5, 20.5]
BAND_LABELS = ["0-4", "5-9", "10-14", "15-19", "20"]

LINEAR = {
    "lin old (lagged)": "artifacts/baselines/"
    "punishment_multinomial_best_with_contr.joblib",
    "lin new (c_t)": "artifacts/baselines/punishment_multinomial_current_contr.joblib",
}
GNN = {
    "gnn old (lagged)": "artifacts/artificial_humans/punishment_rnn_edge_50ep_doubled/"
    "model/architecture_node+edge+rnn__dataset_50ep_doubled.pt",
    "gnn new (c_t)": "artifacts/artificial_humans/punishment/"
    "rnn_edge_50ep_doubled_current_contr/model/"
    "architecture_node+edge+rnn__dataset_50ep_doubled.pt",
}


def load_single_copy():
    df = pd.read_csv(FULL)
    df = df[df["experiment_name"].isin(EXPERIMENTS)]
    return df[~df["global_group_id"].str.contains("(flipped)", regex=False)]


def tensors(df, default_values=None):
    data, dv, _ = create_torch_data(
        df, default_values=default_values, switch_every=SWITCH_EVERY
    )
    return data, dv


def rows_of(data):
    """Flat observed columns over the valid rows of [G, A, T] tensors."""
    m = (data["punishment_valid"] & data["contribution_valid"]).numpy()
    prev_ok = data["prev_contribution_valid"].numpy() & (
        data["round_number"].numpy() > 0
    )
    return dict(
        mask=m,
        y=data["punishment"].numpy()[m].astype(float),
        c=data["contribution"].numpy()[m].astype(float),
        c_prev=data["prev_contribution"].numpy()[m].astype(float),
        prev_ok=prev_ok[m],
    )


def linear_proba(bundle, data):
    pool = build_feature_pool(data, SWITCH_EVERY)
    m = (data["punishment_valid"] & data["contribution_valid"]).numpy()
    X = np.stack([pool[f][m] for f in bundle["features"]], axis=1)
    est = bundle["estimator"]
    p = np.full((len(X), N_LEVELS), 1e-12)
    p[:, est.classes_] = est.predict_proba(bundle["scaler"].transform(X))
    return p / p.sum(1, keepdims=True)


def gnn_proba(model, data):
    n_ep, n_agents, _ = data["punishment"].shape
    with th.no_grad():
        edge_index = model.create_fully_connected(n_agents, n_batch=n_ep)
        _, proba = model.predict_independent(
            data, sample=False, reset_rnn=True, edge_index=edge_index
        )
    m = (data["punishment_valid"] & data["contribution_valid"]).numpy()
    return proba.double().numpy()[m]


def sim_rows(parquet, run=None):
    """Observed rows of one run of a sim's per_round.parquet, in rows_of format."""
    df = pd.read_parquet(parquet)
    if run is not None:
        df = df[df["run"] == run]
    else:
        assert df["run"].nunique() == 1, f"pick a run: {sorted(df['run'].unique())}"
    df = df.sort_values(["episode", "participant_code", "round_number"])
    g = df.groupby(["episode", "participant_code"])
    c_prev = g["contribution"].shift(1)
    prev_ok = c_prev.notna().to_numpy()
    return dict(
        mask=np.ones(len(df), bool),
        y=df["punishment"].to_numpy(float),
        c=df["contribution"].to_numpy(float),
        c_prev=c_prev.fillna(0).to_numpy(float),
        prev_ok=prev_ok,
    )


def stats(P, r):
    """P: [N, 31] predicted (or one-hot observed) punishment distribution."""
    lev = np.arange(N_LEVELS, dtype=float)
    pos, ep = 1.0 - P[:, 0], P @ lev
    c, cp = r["c"], r["c_prev"]
    o = {
        "n": len(c),
        "P(p>0|c_t=20)": pos[c == 20].mean(),
        "P(p>0|c_t<=4)": pos[c <= 4].mean(),
        "P(p>0|c_t=20,c_t-1<=4)": pos[(c == 20) & (cp <= 4) & r["prev_ok"]].mean(),
        "P(p>0|c_t<=4,c_t-1=20)": pos[(c <= 4) & (cp == 20) & r["prev_ok"]].mean(),
    }
    band = pd.cut(c, BANDS, labels=BAND_LABELS)
    for lab in BAND_LABELS:
        sel = np.asarray(band == lab)
        o[f"E[p|p>0] {lab}"] = ep[sel].sum() / pos[sel].sum()
    k = r["prev_ok"]
    X = np.column_stack([np.ones(k.sum()), c[k], cp[k]])
    b = np.linalg.lstsq(X, ep[k], rcond=None)[0]
    o["OLS c_t"], o["OLS c_t-1"], o["n_ols"] = b[1], b[2], int(k.sum())
    y = r["y"].astype(int)
    o["nll"] = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean()
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--linear", nargs="*", default=None, metavar="NAME=PATH")
    ap.add_argument("--gnn", nargs="*", default=None, metavar="NAME=PATH")
    ap.add_argument("--sim", nargs="*", default=[], metavar="NAME=PATH[:RUN]")
    ap.add_argument("--out", default=None, help="write the table as CSV")
    args = ap.parse_args()
    linear = (
        LINEAR if args.linear is None else dict(s.split("=", 1) for s in args.linear)
    )
    gnn = GNN if args.gnn is None else dict(s.split("=", 1) for s in args.gnn)

    df = load_single_copy()
    data, dv = tensors(df)
    r = rows_of(data)
    print(
        f"human rows={len(r['y'])} (episodes={data['punishment'].shape[0]}), "
        f"data defaults contribution={dv['contribution']} punishment={dv['punishment']}"
    )
    onehot = np.eye(N_LEVELS)[r["y"].astype(int)]
    table = {"human": stats(onehot, r)}

    import joblib

    for name, path in linear.items():
        b = joblib.load(ROOT / path)
        d, _ = tensors(df, default_values=b["default_values"])
        table[name] = stats(linear_proba(b, d), rows_of(d))
        print(f"{name}: {path} features={b['features']}")

    if gnn:
        import torch_geometric.nn.models.meta as meta_module

        sys.modules["torch_geometric.nn.meta"] = meta_module
        from aimanager.generic.graph import GraphNetwork

        for name, path in gnn.items():
            model = GraphNetwork.load(str(ROOT / path), device="cpu")
            model.eval()
            assert model.y_name == "punishment", model.y_name
            d, _ = tensors(df, default_values=model.default_values or dv)
            table[name] = stats(gnn_proba(model, d), rows_of(d))
            print(f"{name}: {path} x={[e['name'] for e in model.x_encoding]}")

    for spec in args.sim:
        name, path = spec.split("=", 1)
        path, _, run = path.partition(":")
        rs = sim_rows(ROOT / path, run or None)
        table[name] = stats(np.eye(N_LEVELS)[rs["y"].astype(int)], rs)
        table[name]["nll"] = np.nan  # realised draws, not a predictive distribution
        print(f"{name}: {path} run={run or '(only)'} rows={len(rs['y'])}")

    T = pd.DataFrame(table).T
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print()
    print(T.to_string(float_format=lambda x: f"{x:.3f}"))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        T.to_csv(args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
