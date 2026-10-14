"""Interpretability of the learned peer-attention weights (alpha).

Supports the `auto/contribution-peer-attention` autoresearch log
(`notes/autoresearch_log/contribution-peer-attention.md`, step 15). The two
attention variants replace `NodeModel`'s uniform `scatter_mean` over the 7
incoming peer messages with a scored softmax; this script asks what the
softmax actually learned:

a. mean alpha on same-group vs other-group edges (uniform = 1/7 = 0.142857),
   overall and by round-third;
b. Spearman correlation of alpha with the SOURCE peer's previous contribution
   and with its extremeness |src prev contr - room mean prev contr|;
c. mean per-destination attention entropy vs the uniform benchmark ln(7).

Alpha is reproduced exactly as `AttentionMetaLayer.forward` computes it (same
tensors, pre-update x / edge_attr) without touching `graph.py`. Inputs are
teacher-forced and cover the FULL dataset -- the saved artifact is the
full-data fit (`get_cross_validations` yields the no-test fold last), so there
is no clean held-out set for it and these numbers include training data.

Runs on Raven only (torch_scatter / torch_geometric).

Usage:
    python scripts/data_analysis/peer_attention_weights.py
    python scripts/data_analysis/peer_attention_weights.py --artifact <path.pt>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from scipy.stats import spearmanr
from torch_scatter import scatter_add

from aimanager.artificial_humans.train import apply_mask_pattern, create_fully_connected
from aimanager.generic.data import create_torch_data
from aimanager.generic.graph import GraphNetwork

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = (
    ROOT / "artifacts/artificial_humans/contribution_peer_attention_sg/model/"
    "architecture_node+edge+rnn__dataset_50ep__epochs_575__attention_sg.pt"
)
DATA = ROOT / "experiments/2group_8agent_50ep.csv"
OUTDIR = ROOT / "plots/data_analysis"
EXPERIMENT_NAMES = ["ah_group_switching"]
N_PLAYER = 8
Y_NAME = "contribution"
MASK_NAME = "contribution_valid"
NOTE = "full-data fit: alpha computed on all data, includes training data"


def load_alpha(artifact, data_file):
    """Return alpha (E, R) plus the per-edge/per-node context tensors."""
    model = GraphNetwork.load(str(artifact), device="cpu")
    model.eval()

    df = pd.read_csv(data_file)
    df = df[df["experiment_name"].isin(EXPERIMENT_NAMES)]
    data, default_values, _ = create_torch_data(df)

    n_episodes, n_player, n_rounds = data[Y_NAME].shape
    assert n_player == N_PLAYER, f"expected {N_PLAYER} players, got {n_player}"

    # Teacher forcing, exactly as training's non-autoregressive path: a single
    # all-True mask pattern (every agent predicted, its own target masked out).
    pattern = th.ones((1, n_player), dtype=th.bool)
    d = apply_mask_pattern(data, pattern, Y_NAME, MASK_NAME, default_values)

    edge_index = create_fully_connected(n_player, n_groups=n_episodes)
    enc = model.encode(
        d, mask=MASK_NAME, edge_index=edge_index, device=th.device("cpu")
    )

    x, u, batch = enc["x"], enc["u"], enc["batch"]
    edge_attr, edge_index = enc["edge_attr"], enc["edge_index"]
    row, col = edge_index

    with th.no_grad():
        alpha = model.op1.edge_attention(
            x[row], x[col], edge_attr, u, batch[row], col, x.size(0)
        )
    alpha = alpha.squeeze(-1)  # (E, R)

    # sanity: a softmax over each destination's 7 incoming edges
    sums = scatter_add(alpha, col, dim=0, dim_size=x.size(0))
    assert th.allclose(sums, th.ones_like(sums), atol=1e-5), "alpha is not a softmax"

    ag = data["agent_group"].flatten(0, 1)  # (N, R)
    prev_c = data["prev_contribution"].flatten(0, 1).float()  # (N, R)
    prev_valid = data["prev_contribution_valid"].flatten(0, 1)  # (N, R)
    room_mean = data["prev_contribution"].float().mean(dim=1)  # (B, R)

    ctx = {
        "same_group": (ag[row] == ag[col]).numpy(),
        "src_prev_contr": prev_c[row].numpy(),
        "src_prev_valid": prev_valid[row].numpy(),
        "extremeness": (prev_c[row] - room_mean[batch[row]]).abs().numpy(),
        "col": col.numpy(),
        "n_nodes": x.size(0),
        "n_episodes": n_episodes,
        "n_rounds": n_rounds,
    }
    return alpha.numpy(), ctx


def round_thirds(n_rounds):
    """[(label, boolean round mask)] for `all` plus the three round-thirds."""
    edges = np.linspace(0, n_rounds, 4).astype(int)
    r = np.arange(n_rounds)
    out = [("all", np.ones(n_rounds, dtype=bool))]
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append((f"r{lo}-{hi - 1}", (r >= lo) & (r < hi)))
    return out


def analyse(alpha, ctx, variant):
    rows = []

    def rec(metric, subset, third, value, n):
        rows.append(
            {
                "variant": variant,
                "metric": metric,
                "subset": subset,
                "round_third": third,
                "value": float(value),
                "n": int(n),
                "note": NOTE,
            }
        )

    same = ctx["same_group"]
    thirds = round_thirds(ctx["n_rounds"])

    # incoming edges regrouped per destination: every node has exactly 7, so a
    # stable sort on the destination index gives contiguous blocks of 7.
    order = np.argsort(ctx["col"], kind="stable")
    by_dest = alpha[order].reshape(ctx["n_nodes"], 7, ctx["n_rounds"])

    # (a) mean alpha on same-group vs other-group edges
    for third, rmask in thirds:
        a = alpha[:, rmask]
        s = same[:, rmask]
        rec("mean_alpha", "same_group", third, a[s].mean(), s.sum())
        rec("mean_alpha", "other_group", third, a[~s].mean(), (~s).sum())
        rec("mean_alpha", "uniform_1_over_7", third, 1 / 7, 0)

    # concentration: how much weight the top peer of each destination gets
    top = by_dest.max(axis=1)  # (N, R)
    for third, rmask in thirds:
        t = top[:, rmask]
        rec("mean_max_alpha", "all", third, t.mean(), t.size)

    # (b) alpha vs the source peer's previous contribution / its extremeness
    valid = ctx["src_prev_valid"]  # False at round 0 and for no-input peers
    for name, feat in [
        ("spearman_alpha_src_prev_contr", ctx["src_prev_contr"]),
        ("spearman_alpha_src_extremeness", ctx["extremeness"]),
    ]:
        for subset, smask in [
            ("all", np.ones_like(same)),
            ("same_group", same),
            ("other_group", ~same),
        ]:
            m = valid & smask
            rho = spearmanr(alpha[m], feat[m]).correlation
            rec(name, subset, "all", rho, m.sum())

    # (c) per-destination attention entropy vs ln(7)
    ent = -(by_dest * np.log(np.clip(by_dest, 1e-12, None))).sum(axis=1)  # (N, R)
    for third, rmask in thirds:
        e = ent[:, rmask]
        rec("mean_entropy", "all", third, e.mean(), e.size)
        rec("mean_entropy", "uniform_ln7", third, np.log(7), 0)
        rec("entropy_ratio_to_ln7", "all", third, e.mean() / np.log(7), e.size)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--out-dir", type=Path, default=OUTDIR)
    args = parser.parse_args()

    variant = args.variant
    if variant is None:
        stem = args.artifact.stem
        variant = stem.split("__attention_")[-1] if "__attention_" in stem else stem

    alpha, ctx = load_alpha(args.artifact, args.data)
    res = analyse(alpha, ctx, variant)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"peer_attention_weights_{variant}.csv"
    res.to_csv(out, index=False)

    print(f"\n# peer attention weights -- variant {variant}")
    print(f"artifact: {args.artifact}")
    print(
        f"data: {args.data.name}  "
        f"({ctx['n_episodes']} episodes x {N_PLAYER} agents x "
        f"{ctx['n_rounds']} rounds, {alpha.shape[0]} edges)"
    )
    print(f"NOTE: {NOTE}.")
    with pd.option_context("display.width", 100, "display.max_rows", None):
        print(res.drop(columns=["variant", "note"]).to_string(index=False))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
