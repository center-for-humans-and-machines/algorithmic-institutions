"""Evaluate a design of sigmoid rules in the competing setting, batched.

One rollout carries a shard of design points side by side along the batch
dimension: `shard_size` parameter vectors, `chunk` episodes each. The focal
seat plays the design row, the rival seat plays the clone (or never-punish),
members move every fourth round, and the whole thing is one GPU job.

Seeds are the unit of replication, and they are split by the caller: fit the
surrogate on some, validate the chosen parameters on others. Each (shard,
seed, rep) is seeded deterministically from those three numbers, so a run is
reproducible and the design points inside one rollout face the same initial
draw.

Usage:
    PYTHONPATH=src python scripts/rule_sigmoid/sweep.py \
        --design design.csv --out runs/sweep_clone \
        --seeds 42,43 --episodes 512 --rival clone
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch as th
import yaml

from aimanager.manager.linear_opponent import LinearPunisherOpponent
from aimanager.manager.paired_rollout import (
    contingency,
    make_env,
    rollout,
    summarise,
)
from aimanager.manager.sigmoid_rule import (
    PARAM_NAMES,
    ConstantManager,
    SigmoidRuleBatch,
)

# The frontier stack: identical artifacts to the paired arms this builds on
# (configs/simulation/manager_testing/26_rule_inverted_targeting_s42.yml).
DEFAULT_STACK = {
    "contribution_model": (
        "artifacts/artificial_humans/"
        "group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/"
        "model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
    ),
    "valid_model": (
        "artifacts/artificial_humans/raven_script_22/model/"
        "rnn_False__dataset_full.pt"
    ),
    "switch_model": (
        "artifacts/artificial_humans/switch_joint_exodus/model/"
        "architecture_mlp+rnn+edge__dataset_50ep_doubled.pt"
    ),
    "clone": (
        "artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib"
    ),
}


def load_models(stack, device):
    import joblib

    from aimanager.artificial_humans import GraphNetwork

    models = {
        k: GraphNetwork.load(stack[k], device=device).to(device)
        for k in ("contribution_model", "valid_model", "switch_model")
    }
    models["clone_bundle"] = joblib.load(stack["clone"])
    return models


class MixedFocal:
    """Dispatch the focal seat per batch element: rule or clone.

    The clone is predicted for the whole batch whether or not any row uses
    it, so the RNG a rollout consumes does not depend on how many clone rows
    the design happens to carry -- which is what keeps the clone-against-clone
    control stream-comparable with the rule rows beside it.
    """

    def __init__(self, rule, clone, use_clone):
        self.rule, self.clone = rule, clone
        self.use_clone = use_clone  # (B,) bool

    def predict(self, state, **kw):
        rule_p, _ = self.rule.predict(state, **kw)
        clone_p, _ = self.clone.predict(state, **kw)
        m = self.use_clone.to(rule_p.device).view(-1, *([1] * (rule_p.ndim - 1)))
        return th.where(m, clone_p, rule_p), None


def build_focal(shard, chunk, models, device):
    theta = th.tensor(
        shard[list(PARAM_NAMES)].to_numpy(dtype=float), dtype=th.float
    ).repeat_interleave(chunk, dim=0)
    # a clone row's theta is never read; keep tau positive so the assert holds
    theta[:, 2] = theta[:, 2].clamp(min=1e-9)
    rule = SigmoidRuleBatch(theta).to(device)
    clone = LinearPunisherOpponent(models["clone_bundle"], device=device)
    use_clone = th.tensor(
        (shard["kind"] == "clone").to_numpy(), dtype=th.bool
    ).repeat_interleave(chunk)
    return MixedFocal(rule, clone, use_clone)


def build_rival(kind, models, device):
    if kind == "clone":
        return LinearPunisherOpponent(models["clone_bundle"], device=device)
    if kind == "never":
        return ConstantManager(0)
    raise ValueError(f"unknown rival {kind!r}")


def shape_rows(tables, names):
    """`{seed: (P, 21, 31) counts}` -> a long frame of the non-zero cells."""
    out = []
    for seed, cnt in tables.items():
        pi, ci, ppi = th.nonzero(cnt, as_tuple=True)
        out.append(
            pd.DataFrame(
                {
                    "name": names[pi.numpy()],
                    "seed": np.int32(seed),
                    "contribution": ci.numpy().astype(np.int16),
                    "punishment": ppi.numpy().astype(np.int16),
                    "count": cnt[pi, ci, ppi].numpy().astype(np.int64),
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def run(args):
    device = th.device(args.device)
    stack = dict(DEFAULT_STACK)
    if args.stack:
        with open(args.stack) as f:
            stack.update(yaml.safe_load(f))
    models = load_models(stack, device)

    design = pd.read_csv(args.design)
    seeds = [int(s) for s in args.seeds.split(",")]
    assert args.episodes % args.chunk == 0, "episodes must be a multiple of chunk"
    n_reps = args.episodes // args.chunk

    # One job per contiguous slice of the design. The slice index also seeds
    # the shard index below, so two parts can never produce the same stream
    # for the same design row.
    assert 0 <= args.part < args.n_parts
    edges = np.linspace(0, len(design), args.n_parts + 1).astype(int)
    offset = int(edges[args.part])
    design = design.iloc[offset : int(edges[args.part + 1])].reset_index(drop=True)
    assert len(design), f"part {args.part} of {args.n_parts} is empty"

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"run_args_p{args.part:02d}.json"), "w") as f:
        json.dump({**vars(args), "stack": stack}, f, indent=2)

    shards = [
        design.iloc[i : i + args.shard_size].reset_index(drop=True)
        for i in range(0, len(design), args.shard_size)
    ]
    t0 = time.time()
    for local_si, shard in enumerate(shards):
        si = offset + local_si * args.shard_size
        env = make_env(
            contribution_model=models["contribution_model"],
            valid_model=models["valid_model"],
            switch_model=models["switch_model"],
            batch_size=len(shard) * args.chunk,
            device=device,
        )
        focal = build_focal(shard, args.chunk, models, device)
        rival = build_rival(args.rival, models, device)
        name = np.repeat(shard["name"].to_numpy(), args.chunk)
        param_idx = th.arange(len(shard)).repeat_interleave(args.chunk)
        frames, tables = [], {}
        for seed in seeds:
            for rep in range(n_reps):
                th.manual_seed(seed * 1_000_003 + si * 1009 + rep)
                np.random.seed((seed * 7919 + si * 101 + rep) % (2**31))
                rec = rollout(env, focal, rival)
                summary = summarise(rec)
                frame = pd.DataFrame(
                    {k: v.numpy().astype(np.float32) for k, v in summary.items()}
                )
                frame.insert(0, "name", name)
                frame.insert(1, "seed", np.int32(seed))
                frame.insert(2, "rep", np.int32(rep))
                frames.append(frame)
                cnt = contingency(rec, param_idx, len(shard))
                tables[seed] = cnt if seed not in tables else tables[seed] + cnt
        out = os.path.join(
            args.out, f"episodes_p{args.part:02d}_shard{local_si:03d}.parquet"
        )
        pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
        shape_path = os.path.join(
            args.out, f"shape_p{args.part:02d}_shard{local_si:03d}.parquet"
        )
        shape_rows(tables, shard["name"].to_numpy()).to_parquet(shape_path, index=False)
        done = (local_si + 1) / len(shards)
        el = time.time() - t0
        print(
            f"shard {local_si + 1}/{len(shards)} ({len(shard)} points) "
            f"{el / 60:.1f} min elapsed, {el / done / 60:.1f} min projected",
            flush=True,
        )
        del env, focal, rival
        if device.type == "cuda":
            th.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--shard-size", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--rival", default="clone", choices=("clone", "never"))
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--n-parts", type=int, default=1)
    ap.add_argument("--stack", default=None)
    ap.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
