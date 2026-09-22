"""Does the one-round punishment response compound, and does it compound the
same way at every contribution level?  A closed-loop companion to
``contributor_punishment_intervention.py``.

The open-loop probe measures one round: ``E[c_{t+1} | do(c, p)] - c``.  A
manager is not paid on one round -- it is paid on the discounted stream of
contributions the dose sets off.  If the one-round response decayed away
immediately at low contributions and persisted at high ones (or the reverse),
the one-round targeting gradient would misstate the gradient a manager's return
actually has.  This script measures the stream.

Design -- OWN-PATH ROLLOUT, a controlled closed loop:

  * the same contexts and the same forced cell at round t* as the open-loop
    probe;
  * from t*+1 on, the focal agent's own contribution is drawn by the model and
    fed back into its own history, round after round -- the closed loop for the
    agent whose response is being measured;
  * everything else is held at the human record: the peers' contributions, and
    every punishment after t*.  They are identical across the treated and the
    control arm, so they cancel in the difference, and freezing them keeps the
    manager from reacting to the perturbation and confounding the persistence
    with a policy response;
  * membership is whatever the human record says, so no switch is triggered by
    the dose either.

SHARED NOISE IS ON HERE, AND THAT IS THE POINT.  The rollout needs draws, not
marginals, so ``sample=True`` runs the model's own herding copula.  The copula
is an inverse-CDF sampler from a latent drawn off the global torch RNG, so
seeding the treated and the control arm identically and keeping the batch
layout identical makes the two arms share every uniform: common random numbers,
and the paired difference carries far less sampling noise than the two arms do
separately.  ``se_one_round_unpaired`` reports the standard error the same
difference would have carried without the shared uniforms, so how much the
pairing bought is measured and not claimed.

Reports per contribution level the discounted and undiscounted stream

    S(c, p) = sum_{k=1..K} gamma^k * (c_{t*+k}(p) - c_{t*+k}(0)),

the one-round term, the implied persistence multiplier S / (one round), and
the manager-facing value 1.6 * S / p -- pool units returned per point of
punishment spent, against the 1 pool unit it costs.

Imports graph.py, so this runs on Raven only.
"""

import argparse
import json
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
sys.path.insert(0, str(ROOT / "scripts" / "artificial_humans"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

import contribution_copula_rho as cc  # noqa: E402

from aimanager.generic.graph import GraphNetwork  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "data_analysis"))
from contributor_punishment_intervention import (  # noqa: E402
    DEFAULT_MODEL,
    _edge_cache,
    _override,
    _recompute_common_good,
    check_model,
    f,
    select_contexts,
)

MPCR = 1.6  # the pool multiplier: one contributed point returns 1.6 to the pool


def _write_back(sub, rows, agents, t, values):
    """Feed the focal agent's drawn round-t contribution back into its own
    history: the round-t slot, the ``prev_``-slot round t+1 reads, the derived
    ceiling flag and the pool."""
    sub["contribution"][rows, agents, t] = values
    sub["contribution_valid"][rows, agents, t] = True
    sub["contribution_max"] = sub["contribution"] == 20
    if t + 1 < sub["contribution"].shape[2]:
        sub["prev_contribution"][rows, agents, t + 1] = values
        sub["prev_contribution_valid"][rows, agents, t + 1] = True
        _recompute_common_good(sub, t)


def roll_arm(model, data, ctx, t_star, c, p, horizon, get_edges, seed, n_draws):
    """Mean drawn focal contribution at each of the K rounds after t*, over
    contexts and draws, for one forced cell.  The RNG is seeded per draw and
    the batch layout is fixed, so two calls that differ only in ``p`` share
    every uniform."""
    device = model.device
    eps = th.as_tensor(ctx["row"].to_numpy())
    ags = th.as_tensor(ctx["agent"].to_numpy(), device=device)
    n_b = len(ctx)
    rows = th.arange(n_b, device=device)
    T = t_star + horizon + 1
    base = {k: v[eps][:, :, :T].to(device) for k, v in data.items()}
    acc = np.zeros((n_draws, horizon), dtype=np.float64)
    for d in range(n_draws):
        sub = {k: v.clone() for k, v in base.items()}
        _override(
            sub,
            rows,
            ags,
            t_star,
            th.full((n_b,), c, dtype=th.int64, device=device),
            th.full((n_b,), p, dtype=th.int64, device=device),
        )
        for k in range(1, horizon + 1):
            th.manual_seed(seed + 100003 * d + 17 * k)
            step = {key: v[:, :, : t_star + k + 1] for key, v in sub.items()}
            with th.no_grad():
                pred, _ = model.predict_independent(
                    step, sample=True, reset_rnn=True, edge_index=get_edges(n_b)
                )
            drawn = pred[rows, ags, t_star + k].to(th.int64)
            acc[d, k - 1] = float(drawn.double().mean())
            _write_back(sub, rows, ags, t_star + k, drawn)
    return acc


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(ROOT / DEFAULT_MODEL))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--rounds", default="5,9,13")
    ap.add_argument("--n-contexts", type=int, default=600)
    ap.add_argument(
        "--levels",
        default="2,7,12,17,20",
        help="forced contribution levels; one per band plus the ceiling",
    )
    ap.add_argument("--dose", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--n-draws", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(
        args.output_dir or ROOT / "plots/data_analysis/contributor_punishment_targeting"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = GraphNetwork.load(str(Path(args.model).resolve()), device=str(device))
    model.eval()
    print(f"model     {cc.rel(args.model)}")
    print(f"  device={device} copula_rho={f(model.copula_rho)}")
    print("  SHARED NOISE ON: sample=True, the model's own copula, paired seeds")
    check_model(model)

    data, _, key_to_idx, _ = cc.load_full()
    tr = cc.select_split(key_to_idx, cc.TRAIN, cc.N_TRAIN_EP)
    te = cc.select_split(key_to_idx, cc.TEST, cc.N_TEST_EP)
    idx = np.array(sorted(set(tr.tolist()) | set(te.tolist())), dtype=np.int64)

    rounds = [int(r) for r in args.rounds.split(",")]
    levels = [int(v) for v in args.levels.split(",")]
    n_rounds = data["contribution"].shape[2]
    assert max(rounds) + args.horizon < n_rounds, "the horizon runs past the game"
    rng = np.random.default_rng(args.seed)
    ctx_all = select_contexts(data, idx, rounds, rng, args.n_contexts)
    get_edges = _edge_cache(model, data["contribution"].shape[1])
    print(
        f"contexts  {len(ctx_all)} over t*={rounds}; levels={levels}; "
        f"dose={args.dose}; horizon={args.horizon}; draws={args.n_draws}"
    )

    disc = args.gamma ** np.arange(1, args.horizon + 1)
    rows, paths = [], []
    for c in levels:
        per_t = {0: [], args.dose: []}
        for t_star, grp in ctx_all.groupby("t_star"):
            grp = grp.reset_index(drop=True)
            for p in (0, args.dose):
                per_t[p].append(
                    roll_arm(
                        model,
                        data,
                        grp,
                        int(t_star),
                        c,
                        p,
                        args.horizon,
                        get_edges,
                        args.seed,
                        args.n_draws,
                    )
                )
        ctrl = np.concatenate(per_t[0], axis=0)  # (draws * t*, horizon)
        trt = np.concatenate(per_t[args.dose], axis=0)
        diff = trt - ctrl
        mean_path = diff.mean(0)
        sd_path = diff.std(0, ddof=1) / np.sqrt(diff.shape[0])
        s_disc = float((mean_path * disc).sum())
        s_raw = float(mean_path.sum())
        one = float(mean_path[0])
        rows.append(
            {
                "contribution": c,
                "dose": args.dose,
                "one_round_effect": one,
                "one_round_slope": one / args.dose,
                "sum_discounted": s_disc,
                "sum_undiscounted": s_raw,
                "multiplier_discounted": s_disc / one if one != 0 else float("nan"),
                "value_per_point_discounted": MPCR * s_disc / args.dose,
                "se_one_round": float(sd_path[0]),
                "se_sum_discounted": float(np.sqrt(((sd_path * disc) ** 2).sum())),
                # the same standard error the two arms would have had if they
                # had NOT shared their uniforms -- how much the pairing bought
                "se_one_round_unpaired": float(
                    np.sqrt(ctrl[:, 0].var(ddof=1) + trt[:, 0].var(ddof=1))
                    / np.sqrt(diff.shape[0])
                ),
            }
        )
        for k in range(args.horizon):
            paths.append(
                {
                    "contribution": c,
                    "k": k + 1,
                    "effect": float(mean_path[k]),
                    "se": float(sd_path[k]),
                }
            )
        mult = s_disc / one if one != 0 else float("nan")
        print(
            f"  c={c:2d}  one-round {f(one)}  discounted sum {f(s_disc)}  "
            f"multiplier {f(mult)}",
            flush=True,
        )

    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "rollout_by_level.csv", index=False)
    pd.DataFrame(paths).to_csv(out_dir / "rollout_paths.csv", index=False)
    print("\n--- the stream, by forced contribution level ---")
    print(tab.to_string(index=False, float_format=lambda v: f"{v: .5f}"))

    lo, hi = tab.iloc[0], tab[tab["contribution"] < 20].iloc[-1]
    summary = {
        "dose": args.dose,
        "gamma": args.gamma,
        "horizon": args.horizon,
        "n_contexts": int(len(ctx_all)),
        "n_draws": args.n_draws,
        "shared_noise": "on (sample=True, the model's own copula, paired seeds)",
        "one_round_gradient": float(lo["one_round_slope"] - hi["one_round_slope"]),
        "discounted_gradient_per_point": float(
            (lo["sum_discounted"] - hi["sum_discounted"]) / args.dose
        ),
        "value_gradient_pool_units_per_point": float(
            MPCR * (lo["sum_discounted"] - hi["sum_discounted"]) / args.dose
        ),
        "levels": levels,
    }
    print("\n=================== the stream's verdict ===================")
    print(
        f"  one-round gradient  (c={int(lo['contribution'])} minus "
        f"c={int(hi['contribution'])})   {f(summary['one_round_gradient'])}"
    )
    print(
        f"  discounted gradient per punishment point   "
        f"{f(summary['discounted_gradient_per_point'])}"
    )
    print(
        f"  in pool units per point (x1.6, cost 1)     "
        f"{f(summary['value_gradient_pool_units_per_point'])}"
    )
    summary["wall_seconds"] = time.time() - t0
    (out_dir / "rollout_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
