"""Dump the base GNN switch predictor's marginal p(does_switch) per row.

Feeds the herding-copula calibration (plan step 3): the pairwise-likelihood
MLE of `copula_rho` needs the model's own predicted Bernoulli marginals on
the human train split, plus the holdout split as an out-of-sample check.

One full-episode forward per game (`predict_independent(sample=False)`, RNN
reset per game, GRU over the whole round axis) -- the simulation's warm-RNN
semantics. Feature preprocessing mirrors the training config
`configs/training/artificial_humans/switch_predictor/opt_50ep_doubled_reanchored.yml`
(`switch_every: 4`, `default_values` from the doubled 50-episode file).

Game key in the output is (global_group_id, episode_id); `agent_group` is the
PRE-switch membership at that round (log note D7).

Method details and conventions:
notes/autoresearch_log/switch-herding-copula.md.

RAVEN ONLY (unpickles torch_geometric modules, imports torch_scatter):
    uv run python scripts/artificial_humans/dump_switch_probs.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch as th  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aimanager.generic.data import (  # noqa: E402
    create_torch_data_new,
    get_default_values,
    parse_agent_rounds,
)
from aimanager.generic.graph import GraphNetwork  # noqa: E402

ARTIFACT = (
    ROOT
    / "artifacts/artificial_humans/switch_pred_opt_50ep_doubled_reanchored"
    / "model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt"
)
OUT_DIR = ROOT / "artifacts/artificial_humans/switch_pred_herding_copula/calibration"
DOUBLED = ROOT / "experiments/2group_8agent_50ep.csv"
SPLITS = {
    "train": ROOT / "experiments/baseline/2group_8agent_50ep_bline_train.csv",
    "test": ROOT / "experiments/baseline/2group_8agent_50ep_bline_test.csv",
}
MASK = "switch_valid"
TARGET = "does_switch"
SWITCH_EVERY = 4
EXPERIMENTS = ("ah_group_switching",)
COLUMNS = [
    "episode_id",
    "global_group_id",
    "round_number",
    "player_idx",
    "agent_group",
    "does_switch",
    "switch_valid",
    "p_switch",
]

# planner's read-only train-split numbers (log note 5), asserted below
EXP_ELIGIBLE = 1515
EXP_RATE = 0.29372937
EXP_PER_ROUND = {3: 0.4290, 7: 0.3026, 11: 0.2601, 15: 0.2292, 19: 0.2434}
RATE_TOL = 0.03


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def read_human(path):
    """Human csv, single copy per game (drop flipped copies if present)."""
    df = pd.read_csv(path)
    df = df[df["experiment_name"].isin(EXPERIMENTS)]
    df = df[~df["global_group_id"].str.contains("(flipped)", regex=False)]
    return df


def parse_split(path):
    """Parsed agent-round frame, keeping `global_group_id` (parse drops it)."""
    df = read_human(path).copy()
    df["_ggid"] = df["global_group_id"]
    parsed = parse_agent_rounds(df, switch_every=SWITCH_EVERY)
    parsed = parsed.rename(columns={"_ggid": "global_group_id"})
    return parsed


def training_default_values():
    """`default_values` exactly as the training run computed them.

    train.py calls `create_torch_data` on the full doubled file (flipped
    copies included), so feature normalisation here matches training.
    """
    df = pd.read_csv(DOUBLED)
    df = df[df["experiment_name"].isin(EXPERIMENTS)]
    return get_default_values(parse_agent_rounds(df.copy(), SWITCH_EVERY))


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def predict_split(model, parsed, default_values, device):
    """p(does_switch=1) per parsed row, one full-episode forward per game."""
    data, _, _ = create_torch_data_new(parsed, default_values)
    data = {k: v.to(device) for k, v in data.items()}
    n_games, n_agents, n_rounds = data[TARGET].shape
    edge_index = model.create_fully_connected(n_agents, n_batch=1)

    proba = th.zeros((n_games, n_agents, n_rounds), dtype=th.float64)
    with th.no_grad():
        for g in range(n_games):
            game = {k: v[g : g + 1] for k, v in data.items()}
            _, p = model.predict_independent(
                game, sample=False, reset_rnn=True, edge_index=edge_index
            )
            proba[g] = p[0, ..., 1].double().cpu()

    idx = (
        parsed["group_idx"].to_numpy(),
        parsed["player_idx"].to_numpy(),
        parsed["round_number"].to_numpy(),
    )
    out = parsed.loc[:, [c for c in COLUMNS if c != "p_switch"]].copy()
    out["p_switch"] = proba.numpy()[idx]
    out = out.sort_values(
        ["global_group_id", "episode_id", "round_number", "player_idx"]
    )
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def rel(path):
    return os.path.relpath(path, ROOT)


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def report(name, out, expected=None):
    """Print the eligible-row profile next to the log-note-5 constants."""
    eligible = out[out[MASK]]
    y = eligible[TARGET].to_numpy().astype(float)
    p = eligible["p_switch"].to_numpy()
    n, mean_p, rate = len(eligible), float(p.mean()), float(y.mean())

    games = out.groupby(["global_group_id", "episode_id"]).ngroups
    print(f"\n[{name}] games={games} rows={len(out)} eligible={n}")
    print(f"  mean p_switch {mean_p:.8f}   observed rate {rate:.8f}")
    print("  round     mean p    observed      note 5        n")
    for r, grp in eligible.groupby("round_number"):
        pr = float(grp["p_switch"].mean())
        obs = float(grp[TARGET].mean())
        ref = "" if expected is None else f"{expected.get(r, float('nan')):9.4f}"
        print(f"  {r:5d}  {pr:9.4f}  {obs:9.4f}  {ref:>9}  {len(grp):7d}")
    print(f"  log loss (diagnostic) {log_loss(y, p):.8f}")


def check_train(out):
    """Hard sanity checks on the train dump; raise SystemExit on failure."""
    eligible = out[out[MASK]]
    n = len(eligible)
    mean_p = float(eligible["p_switch"].mean())
    rate = float(eligible[TARGET].to_numpy().astype(float).mean())
    fails = []

    if n != EXP_ELIGIBLE:
        fails.append(f"eligible rows {n} != {EXP_ELIGIBLE} (log note 5)")
    if abs(rate - EXP_RATE) > 1e-6:
        fails.append(f"observed rate {rate:.8f} != {EXP_RATE} (log note 5)")
    if abs(mean_p - rate) > RATE_TOL:
        fails.append(
            f"mean p_switch {mean_p:.8f} off the observed rate "
            f"{rate:.8f} by {abs(mean_p - rate):.8f} > {RATE_TOL}"
        )
    rounds = sorted(eligible["round_number"].unique().tolist())
    if rounds != sorted(EXP_PER_ROUND):
        fails.append(f"decision rounds {rounds} != {sorted(EXP_PER_ROUND)}")

    if fails:
        raise SystemExit(
            "SANITY CHECK FAILED on the train dump:\n  - " + "\n  - ".join(fails)
        )
    print(f"\nsanity checks PASS (eligible {n}, mean p {mean_p:.8f})")


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, default=ARTIFACT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--device", default="cpu", help="cpu -- runs on the login node")
    args = ap.parse_args()

    device = th.device(args.device)
    model = GraphNetwork.load(str(args.artifact), device=device).to(device)
    model.eval()
    assert model.y_name == TARGET, f"unexpected y_name {model.y_name}"
    assert model.y_levels == 2, f"unexpected y_levels {model.y_levels}"
    print(f"model     {rel(args.artifact)}")
    print(f"  y_name={model.y_name} y_levels={model.y_levels} device={device}")

    default_values = training_default_values()
    print(f"defaults  from {rel(DOUBLED)} (switch_every={SWITCH_EVERY})")
    print(f"  recomputed {default_values}")
    print(f"  artifact   {model.default_values}")
    for k, v in default_values.items():
        got = model.default_values.get(k, "<missing>")
        if got != v:
            print(f"  WARNING default_values[{k}]: artifact {got} != {v}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dumps = {}
    for name, path in SPLITS.items():
        parsed = parse_split(path)
        out = predict_split(model, parsed, default_values, device)
        dest = args.out_dir / f"switch_probs_{name}.parquet"
        out[COLUMNS].to_parquet(dest, index=False)
        report(name, out, EXP_PER_ROUND if name == "train" else None)
        print(f"  wrote {rel(dest)}")
        dumps[name] = out

    check_train(dumps["train"])
    print(f"total runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
