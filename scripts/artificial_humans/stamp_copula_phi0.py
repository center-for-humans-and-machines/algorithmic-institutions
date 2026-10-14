"""Arm C of the closed-loop variance experiment: the stimulus-skip trunk's
copula at its fitted rho but WITHOUT persistence (phi = 0: a fresh shared
latent every round). `make_contribution_copula_artifact.py` refuses phi = 0 by
design, so this is a dict-level copy of the stamped artifact with
`copula_phi` set to 0.0 and every other key carried over untouched (verified
after reload). Runs on Raven (the .pt unpickles torch_geometric modules):

    .venv/bin/python scripts/artificial_humans/stamp_copula_phi0.py
"""

import argparse
import sys
from pathlib import Path

import torch as th

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
NAME = "architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
AH = ROOT / "artifacts/artificial_humans"
BASE = AH / "group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula"
OUT = AH / "group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula_phi0"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=BASE / "model" / NAME)
    ap.add_argument("--out", type=Path, default=OUT / "model" / NAME)
    args = ap.parse_args()
    d = th.load(args.base, map_location="cpu")
    assert d["copula_rho"] > 0 and d["copula_phi"] == 1.0, (
        d["copula_rho"],
        d["copula_phi"],
    )
    d["copula_phi"] = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    th.save(d, args.out)

    from aimanager.generic.graph import GraphNetwork  # noqa: E402  (PyG)

    a = GraphNetwork.load(str(args.base), device="cpu")
    b = GraphNetwork.load(str(args.out), device="cpu")
    assert b.copula_rho == a.copula_rho and b.copula_phi == 0.0
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(th.equal(sa[k], sb[k]) for k in sa)
    print(
        f"wrote {args.out}: rho={b.copula_rho} phi={b.copula_phi} "
        f"switch_every={b.copula_switch_every}; {len(sa)} tensors identical"
    )


if __name__ == "__main__":
    main()
