"""Verify the six contribution arms of the seed-spread measurement load and
differ: PR #188's five seed-ensemble members (copula-stamped on this branch)
and the shipped frontier contributor.

For each artifact: `GraphNetwork.load`, the three copula fields, the number of
parameters and their L2 norm; then the pairwise max |delta| and relative L2
distance over the concatenated parameter vector. Two artifacts that are the
same training would come out at 0.

RUNS ON RAVEN ONLY (the .pt unpickles torch_geometric modules).

    python scripts/data_analysis/seed_spread_verify_members.py
"""

import hashlib
import itertools
import os
import sys

import torch as th

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

STEM = "architecture_node+edge+rnn__dataset_50ep__epochs_575"
ENS = (
    "artifacts/artificial_humans/"
    "group_switching_contribution_50ep_vnode_stimulus_skip_seed_ensemble_copula/model"
)
SHIPPED = (
    "artifacts/artificial_humans/"
    "group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/model/"
    f"{STEM}.pt"
)
ARMS = [(f"seed_{k}", f"{ENS}/{STEM}__seed_{k}.pt") for k in range(1, 6)]
ARMS.append(("shipped", SHIPPED))


def flat_params(model):
    return th.cat([p.detach().reshape(-1) for p in model.parameters()])


def main():
    from aimanager.generic.graph import GraphNetwork

    vecs = {}
    head = f"{'arm':>8} {'rho':>10} {'phi':>5} {'sw':>3} {'n_par':>7} {'|w|':>9}"
    print(head + "  sha256")
    for name, path in ARMS:
        model = GraphNetwork.load(path, device=th.device("cpu"))
        assert model.y_name == "contribution", f"{name}: not a contributor"
        v = flat_params(model)
        vecs[name] = v
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
        print(
            f"{name:>8} {model.copula_rho:10.7f} {model.copula_phi:5.2f} "
            f"{str(model.copula_switch_every):>3} {v.numel():7d} "
            f"{v.norm().item():9.4f}  {sha}"
        )

    names = [n for n, _ in ARMS]
    assert len({vecs[n].numel() for n in names}) == 1, "arms differ in shape"
    print("\npairwise parameter distance (max |delta| / relative L2)")
    for a, b in itertools.combinations(names, 2):
        d = vecs[a] - vecs[b]
        rel = d.norm().item() / max(vecs[a].norm().item(), 1e-12)
        print(f"  {a:>8} vs {b:>8}  {d.abs().max().item():8.4f}  {rel:7.4f}")
        assert d.abs().max().item() > 0, f"{a} and {b} are identical"
    print("\nall six arms load, carry the frozen copula, and differ pairwise")


if __name__ == "__main__":
    main()
