"""Build the herding-copula switch artifacts (plan step 15).

Copies the base switch artifact dict and inserts the calibrated `copula_*`
fields from the params JSON -- arm A (rho only) and arm B (rho + AR(1) phi).
The pickled modules are byte-shared with the base; verification reloads
every artifact and compares each module's state_dict bit-identically.

RUNS ON RAVEN ONLY (the .pt unpickles torch_geometric modules). Method
details: notes/autoresearch_log/switch-herding-copula.md.

    python scripts/artificial_humans/make_switch_copula_artifact.py
"""

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
PARAMS = ROOT / (
    "artifacts/artificial_humans/switch_pred_herding_copula"
    "/calibration/copula_params.json"
)
ARTIFACT_NAME = "architecture_mlp+rnn+edge__dataset_50ep_doubled.pt"
SWITCH_EVERY = 4  # decision cadence of the 23-family protocol (ruling D6)
MODULE_KEYS = ("op1", "op2", "rnn_n", "rnn_g", "bias")


def arms(params):
    rho = float(params["rho"])
    out = {"switch_pred_herding_copula": dict(copula_rho=rho)}
    if params.get("phi") is not None:
        out["switch_pred_herding_copula_ar1"] = dict(
            copula_rho=rho,
            copula_phi=float(params["phi"]),
            copula_switch_every=SWITCH_EVERY,
        )
    return out


def state_dicts(d):
    return {k: d[k].state_dict() for k in MODULE_KEYS if d.get(k) is not None}


def assert_identical(a, b, label):
    assert a.keys() == b.keys(), f"{label}: module set differs"
    for mod, sd in a.items():
        assert sd.keys() == b[mod].keys(), f"{label}/{mod}: keys differ"
        for name, t in sd.items():
            assert th.equal(t, b[mod][name]), f"{label}/{mod}/{name} differs"


def main():
    params = json.loads(PARAMS.read_text())
    base = th.load(BASE, map_location="cpu")
    base_sd = state_dicts(base)
    print(f"base      {BASE.relative_to(ROOT)}")
    print(f"params    rho={params['rho']} phi={params['phi']}")
    for name, fields in arms(params).items():
        for k in fields:
            assert k not in base, f"base artifact already carries {k}"
        d = dict(base)
        d.update(fields)
        out = ROOT / "artifacts/artificial_humans" / name / "model"
        out.mkdir(parents=True, exist_ok=True)
        out = out / ARTIFACT_NAME
        th.save(d, out)
        again = th.load(out, map_location="cpu")
        assert_identical(base_sd, state_dicts(again), name)
        for k, v in fields.items():
            assert again[k] == v, f"{name}: {k} did not round-trip"
        untouched = {k for k in base if base[k] != again.get(k, object())}
        assert not {
            k for k in untouched if k not in MODULE_KEYS
        }, f"{name}: pre-existing non-module key modified"
        from aimanager.generic.graph import GraphNetwork

        model = GraphNetwork.load(str(out), device=th.device("cpu"))
        got = dict(
            copula_rho=model.copula_rho,
            copula_phi=model.copula_phi,
            copula_switch_every=model.copula_switch_every,
        )
        print(f"wrote     {out.relative_to(ROOT)}")
        print(f"  loads   {got}")
    print("all artifacts verified bit-identical to the base modules")


if __name__ == "__main__":
    main()
