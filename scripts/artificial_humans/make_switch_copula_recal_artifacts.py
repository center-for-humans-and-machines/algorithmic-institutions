"""Build the recalibrated herding-copula switch artifacts (plan step 4).

Copies the base switch artifact dict and inserts the frozen grid's
(copula_rho, copula_phi, copula_switch_every) fields, one output artifact per
arm of the committed `grid.json`. The pickled modules are byte-shared with
the base; verification reloads every artifact and compares each module's
state_dict bit-identically, and additionally loads the BASE artifact through
GraphNetwork.load as a control -- it must read back (0.0, 0.0, None), proving
the stamped values come from the grid file rather than from __init__
defaults.

The grid is FROZEN: rho and phi are read ONLY from `grid.json`, never
hardcoded here.

RUNS ON RAVEN ONLY (the .pt unpickles torch_geometric modules). Method
details: notes/autoresearch_log/switch-herding-copula-recal.md.

    python scripts/artificial_humans/make_switch_copula_recal_artifacts.py
"""

import hashlib
import json
import sys
from pathlib import Path

import torch as th

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

BASE = ROOT / (
    "artifacts/artificial_humans/switch_pred_opt_50ep_doubled_reanchored"
    "/model/architecture_mlp+rnn+edge__dataset_50ep_doubled.pt"
)
GRID = ROOT / "artifacts/artificial_humans/switch_pred_herding_copula_recal/grid.json"
ARTIFACT_NAME = "architecture_mlp+rnn+edge__dataset_50ep_doubled.pt"
SWITCH_EVERY = 4  # decision cadence of the 23-family protocol (ruling D6)
MODULE_KEYS = ("op1", "op2", "rnn_n", "rnn_g", "bias")
NEW_KEYS = ("copula_rho", "copula_phi", "copula_switch_every")


def state_dicts(d):
    return {k: d[k].state_dict() for k in MODULE_KEYS if d.get(k) is not None}


def assert_identical(a, b, label):
    assert a.keys() == b.keys(), f"{label}: module set differs"
    for mod, sd in a.items():
        assert sd.keys() == b[mod].keys(), f"{label}/{mod}: keys differ"
        for name, t in sd.items():
            assert th.equal(t, b[mod][name]), f"{label}/{mod}/{name} differs"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    from aimanager.generic.graph import GraphNetwork

    grid = json.loads(GRID.read_text())
    base = th.load(BASE, map_location="cpu")
    base_sd = state_dicts(base)
    print(f"base      {BASE.relative_to(ROOT)}")
    print(f"grid      {GRID.relative_to(ROOT)}  ({len(grid)} arms)")

    for k in NEW_KEYS:
        assert k not in base, f"base artifact already carries {k}"

    base_control = GraphNetwork.load(str(BASE), device=th.device("cpu"))
    base_control_triple = (
        base_control.copula_rho,
        base_control.copula_phi,
        base_control.copula_switch_every,
    )
    print(
        f"base ctrl (copula_rho, copula_phi, copula_switch_every) = "
        f"{base_control_triple}"
    )
    assert base_control_triple == (
        0.0,
        0.0,
        None,
    ), f"base control must read (0.0, 0.0, None), got {base_control_triple}"

    for arm in grid:
        k = arm["k"]
        rho = float(arm["rho"])
        phi = float(arm["phi"])
        fields = dict(copula_rho=rho, copula_phi=phi, copula_switch_every=SWITCH_EVERY)
        name = f"switch_pred_herding_copula_recal_{k}"
        d = dict(base)
        d.update(fields)
        out_dir = ROOT / "artifacts/artificial_humans" / name / "model"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / ARTIFACT_NAME
        th.save(d, out)

        again = th.load(out, map_location="cpu")
        assert_identical(base_sd, state_dicts(again), name)
        for key, v in fields.items():
            assert again[key] == v, f"{name}: {key} did not round-trip"
        untouched = {key for key in base if base[key] != again.get(key, object())}
        assert not {
            key for key in untouched if key not in MODULE_KEYS
        }, f"{name}: pre-existing non-module key modified"

        model = GraphNetwork.load(str(out), device=th.device("cpu"))
        got = dict(
            copula_rho=model.copula_rho,
            copula_phi=model.copula_phi,
            copula_switch_every=model.copula_switch_every,
        )
        assert got["copula_rho"] == rho, f"{name}: copula_rho mismatch on load"
        assert got["copula_phi"] == phi, f"{name}: copula_phi mismatch on load"
        assert (
            got["copula_switch_every"] == SWITCH_EVERY
        ), f"{name}: copula_switch_every mismatch on load"
        assert isinstance(
            got["copula_switch_every"], int
        ), f"{name}: copula_switch_every is not an int"

        digest = sha256(out)
        print(f"wrote     {out.relative_to(ROOT)}")
        print(f"  loads   {got}")
        print(f"  sha256  {digest}")

    print("all arms verified bit-identical to the base modules")


if __name__ == "__main__":
    main()
