"""What the trained stimulus gate actually learned.

The gated skip's hypothesis (notes/autoresearch_log/contributor-gated-skip.md)
is that a scalar gate lets the immediate stimulus dominate on the rounds where
something happened to the player and the carried memory dominate otherwise. The
screen says the response got weaker, not stronger, so the question is whether
the gate is an event detector at all.

This captures `g = sigmoid(w . x_skip + b)` on every teacher-forced row of the
human episodes -- one scalar per (episode, agent, round) -- and reports its
distribution overall and split by the stimulus the gate is supposed to react
to: whether the player was punished in the round the gate reads
(`prev_punishment > 0`), and how far that round's own contribution was from
the previous one.

The comparison the numbers answer: a gate that detects events separates those
conditions by much more than its own within-condition spread; a gate that has
collapsed to the fixed ratio it was meant to replace does not.

Measurement only: trains nothing, writes one CSV.

Imports graph.py, so this runs on Raven only:
    .venv/bin/python scripts/data_analysis/gated_skip_gate_stats.py --model PT
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
sys.path.insert(0, str(ROOT / "scripts" / "artificial_humans"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

import contribution_copula_rho as cc  # noqa: E402

from aimanager.generic.graph import GraphNetwork  # noqa: E402

OUT = ROOT / "plots/data_analysis/evaluation/contributor_gated_skip"


def f(x):
    return repr(float(x))


def gate_rows(model, data, idx):
    """One row per valid (episode, agent, round) with the gate's value and the
    stimulus it read. The forward is the same teacher-forced pass the screen
    makes -- `predict_independent`, `reset_rnn=True`, the human history and
    never the model's own draws -- with a hook on the gate module."""
    sub = {k: v[th.as_tensor(idx)] for k, v in data.items()}
    n_ep, n_agents, n_rounds = sub["contribution"].shape

    captured = {}

    def hook(_module, _args, output):
        captured["logit"] = output.detach()

    handle = model.stimulus_gate_module.register_forward_hook(hook)
    try:
        with th.no_grad():
            edge_index = model.create_fully_connected(n_agents, n_batch=n_ep)
            model.predict_independent(
                sub, sample=False, reset_rnn=True, edge_index=edge_index
            )
    finally:
        handle.remove()

    # [n_ep * n_agents, n_rounds, 1] in the flattened node layout op1 uses
    gate = th.sigmoid(captured["logit"]).reshape(n_ep, n_agents, n_rounds)
    gate = gate.double().numpy()

    mask = sub[cc.MASK].numpy().astype(bool)
    g, a, t = np.nonzero(mask)
    prev_p = sub["prev_punishment"].numpy().astype(np.float64)
    prev_c = sub["prev_contribution"].numpy().astype(np.float64)
    contr = sub["contribution"].numpy().astype(np.float64)
    return pd.DataFrame(
        {
            "episode": g,
            "agent": a,
            "round": t,
            "gate": gate[mask],
            "prev_punishment": prev_p[mask],
            "prev_contribution": prev_c[mask],
            "contribution": contr[mask],
        }
    )


def report(df, label):
    print(f"\n================ {label} (n={len(df)}) ================")
    print(
        f"gate: mean {f(df.gate.mean())} sd {f(df.gate.std())} "
        f"min {f(df.gate.min())} max {f(df.gate.max())}"
    )
    qs = [f(v) for v in df.gate.quantile([0.05, 0.25, 0.5, 0.75, 0.95])]
    print(f"  quantiles 5/25/50/75/95: {qs}")

    out = []
    punished = df.prev_punishment > 0
    for name, m in (
        ("punished last round", punished),
        ("not punished last round", ~punished),
        ("punished hard (p >= 5)", df.prev_punishment >= 5),
        ("round 0 (no stimulus yet)", df["round"] == 0),
        ("rounds 1+", df["round"] > 0),
    ):
        sub = df[m]
        out.append(
            dict(
                condition=name,
                n=len(sub),
                gate_mean=float(sub.gate.mean()),
                gate_sd=float(sub.gate.std()),
            )
        )
    tab = pd.DataFrame(out)
    sep = tab.loc[0, "gate_mean"] - tab.loc[1, "gate_mean"]
    pooled = float(df.gate.std())
    tab["diff_vs_unpunished"] = tab.gate_mean - tab.loc[1, "gate_mean"]
    print(tab.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
    print(
        f"\npunished - unpunished separation = {f(sep)}; "
        f"gate's own sd over all rows = {f(pooled)}; "
        f"ratio = {f(sep / pooled)}"
    )
    corr = float(np.corrcoef(df.gate, df.prev_punishment)[0, 1])
    print(f"corr(gate, prev_punishment) = {f(corr)}")
    corr_c = float(np.corrcoef(df.gate, df.prev_contribution)[0, 1])
    print(f"corr(gate, prev_contribution) = {f(corr_c)}")
    return tab


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    model = GraphNetwork.load(args.model, device="cpu")
    model.eval()
    assert model.y_name == "contribution", f"not a contributor: {model.y_name}"
    assert getattr(model, "stimulus_gate", False), "this model has no stimulus gate"
    w = model.stimulus_gate_module.weight.detach()
    b = model.stimulus_gate_module.bias.detach()
    print(f"model  {cc.rel(Path(args.model).resolve())}")
    print(f"  gate weight |w| = {f(w.norm())}  bias = {f(b.item())}")
    print(f"  sigmoid(bias) = {f(th.sigmoid(b).item())}")

    data, _, key_to_idx, _ = cc.load_full()
    tr_idx = cc.select_split(key_to_idx, cc.TRAIN, cc.N_TRAIN_EP)
    te_idx = cc.select_split(key_to_idx, cc.TEST, cc.N_TEST_EP)
    both = np.array(sorted(set(tr_idx.tolist()) | set(te_idx.tolist())))

    df = gate_rows(model, data, both)
    tab = report(df, "train+test, single copy (50 ep)")

    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "gate_stats.csv", index=False)
    df.to_csv(OUT / "gate_rows.csv", index=False)
    print(f"\nwrote {cc.rel(OUT / 'gate_stats.csv')}")


if __name__ == "__main__":
    main()
