"""Build the herding-copula contribution artifact (plan step 10).

Copies the base GNN contributor's artifact dict and inserts the calibrated
`copula_rho` / `copula_phi` from the params JSON written by
`scripts/artificial_humans/contribution_copula_rho.py --write-params`, plus
`copula_switch_every = 1` (contributions are decided every round, so the AR(1)
latent advances every round). Nothing else moves: the pickled modules are
byte-shared with the base, so verification reloads both artifacts and compares
every tensor bit-identically -- module state dicts and the tensor-valued
entries of `default_values` alike.

The honesty check (`--verify-probs`, ON by default) goes further: it runs the
base and the stamped model teacher-forced (`sample=False`, which bypasses the
copula) over the human train split and asserts the predicted probability
matrices are bit-identical. `--no-verify-probs` skips the data pass and leaves
only the dict-level stamping.

RUNS ON RAVEN ONLY (the .pt unpickles torch_geometric modules, and the
verification loads the training data through the PyG data pipeline). Method
details: notes/autoresearch_log/contribution-herding-copula-v2.md.

    .venv/bin/python \\
        scripts/artificial_humans/make_contribution_copula_artifact.py
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch as th

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "artificial_humans"))

ARTIFACT_NAME = "architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
BASE = ROOT / (
    "artifacts/artificial_humans/group_switching_contribution_50ep/model/"
    + ARTIFACT_NAME
)
PARAMS = ROOT / (
    "artifacts/artificial_humans/group_switching_contribution_50ep_herding_copula_v2"
    "/calibration/copula_params.json"
)
OUT = ROOT / (
    "artifacts/artificial_humans/group_switching_contribution_50ep_herding_copula_v2"
    "/model/" + ARTIFACT_NAME
)
SWITCH_EVERY = 1  # contributions are decided every round (plan step 2)
MODULE_KEYS = ("op1", "op2", "rnn_n", "rnn_g", "bias")
FIELDS = ("copula_rho", "copula_phi", "copula_switch_every")


def rel(path):
    """Repo-relative path when possible, absolute otherwise (logging only)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head():
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - provenance is best effort
        return f"unavailable: {exc}"


def fields_from(params, base_path):
    """The three copula fields, with the calibration's own stop-gates
    re-asserted here so a dropped phi can never reach an artifact.

    `phi` in the params JSON is the diagnostic lag-1 ratio, which may exceed 1
    at the boundary; `phi_final` is the value the plan-revision ruling in
    notes/autoresearch_log/contribution-herding-copula-v2.md settled on
    (1.0 exactly for a static episode latent). When `phi_final` is present
    it is the value stamped; the bare `phi` is only used for params JSONs
    written before the lag-profile pass existed."""
    rho = params["rho"]
    phi_lag1 = params.get("phi")
    assert phi_lag1 is not None, "params JSON carries no phi"
    assert params.get("phi_kept", True), (
        f"calibration dropped phi: {params.get('phi_drop_reason')} -- "
        f"no artifact (plan step 8 stop-gate)"
    )
    if "phi_final" in params:
        phi = params["phi_final"]
        assert phi is not None, "params JSON carries phi_final = null"
        assert params.get("phi_final_reason"), (
            "phi_final carries no phi_final_reason -- the decision rule that "
            "produced it must be recorded (plan step 8b)"
        )
    else:
        phi = phi_lag1
    assert 0.0 < float(rho) < 1.0, f"rho {rho} outside (0, 1)"
    assert 0.0 < float(phi) <= 1.0, f"phi {phi} outside (0, 1]"
    if params.get("copula_switch_every") is not None:
        assert int(params["copula_switch_every"]) == SWITCH_EVERY, (
            f"params say copula_switch_every={params['copula_switch_every']}, "
            f"this script stamps {SWITCH_EVERY}"
        )
    if params.get("source_model") is not None:
        assert params["source_model"] == rel(base_path), (
            f"params were calibrated on {params['source_model']}, "
            f"not on {rel(base_path)}"
        )
    if params.get("source_model_sha256") is not None:
        assert params["source_model_sha256"] == sha256(
            base_path
        ), "params' source_model_sha256 does not match the base artifact"
    return dict(
        copula_rho=float(rho),
        copula_phi=float(phi),
        copula_switch_every=SWITCH_EVERY,
    )


def same(a, b):
    """Deep equality; tensors must match dtype, shape and every bit."""
    if isinstance(a, th.Tensor) or isinstance(b, th.Tensor):
        return (
            isinstance(a, th.Tensor)
            and isinstance(b, th.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and th.equal(a, b)
        )
    if isinstance(a, th.nn.Module) or isinstance(b, th.nn.Module):
        return (
            isinstance(a, th.nn.Module)
            and isinstance(b, th.nn.Module)
            and type(a) is type(b)
            and repr(a) == repr(b)
            and same(dict(a.state_dict()), dict(b.state_dict()))
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(same(x, y) for x, y in zip(a, b))
        )
    return type(a) is type(b) and a == b


def n_tensors(obj):
    """Tensors reachable in `obj`, counted the same way `same` walks it."""
    if isinstance(obj, th.Tensor):
        return 1
    if isinstance(obj, th.nn.Module):
        return n_tensors(dict(obj.state_dict()))
    if isinstance(obj, dict):
        return sum(n_tensors(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(n_tensors(v) for v in obj)
    return 0


def assert_only_fields_added(base, again, fields):
    """Every pre-existing key survives untouched; the only new keys are the
    three copula fields."""
    assert set(base) <= set(again), f"keys dropped: {sorted(set(base) - set(again))}"
    added = set(again) - set(base)
    assert added == set(fields), f"unexpected new keys: {sorted(added - set(fields))}"
    for k in sorted(base):
        assert same(base[k], again[k]), f"pre-existing key {k!r} changed"
    for k, v in fields.items():
        assert again[k] == v, f"{k} did not round-trip: {again[k]!r} != {v!r}"


def load_fields(out_path):
    """The three fields as `GraphNetwork.load` sees them (needs PyG)."""
    from aimanager.generic.graph import GraphNetwork

    model = GraphNetwork.load(str(out_path), device=th.device("cpu"))
    assert model.y_name == "contribution", f"not a contributor: {model.y_name}"
    return model, dict(
        copula_rho=model.copula_rho,
        copula_phi=model.copula_phi,
        copula_switch_every=model.copula_switch_every,
    )


def verify_probs(base_path, out_path):
    """Teacher-forced (`sample=False`) probabilities of the human train split
    must be bit-identical between the base and the stamped model -- the copula
    only ever touches the sampling step. Returns the row count."""
    import numpy as np

    import contribution_copula_rho as cc
    from aimanager.generic.graph import GraphNetwork

    data, _, key_to_idx, _ = cc.load_full()
    idx = cc.select_split(key_to_idx, cc.TRAIN, cc.N_TRAIN_EP)
    out = []
    for path in (base_path, out_path):
        model = GraphNetwork.load(str(path), device="cpu")
        model.eval()
        out.append(cc.teacher_forced_rows(model, data, idx))
    ref, got = out
    for key in ("y", "episode", "agent", "round", "group", "cell"):
        assert np.array_equal(ref[key], got[key]), f"{key} rows differ"
    assert ref["shape"] == got["shape"], "tensor shapes differ"
    assert np.array_equal(ref["P"], got["P"]), "teacher-forced probabilities differ"
    return int(len(ref["y"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", type=Path, default=PARAMS, help="copula_params.json")
    ap.add_argument("--base", type=Path, default=BASE, help="base contributor .pt")
    ap.add_argument("--out", type=Path, default=OUT, help="stamped .pt to write")
    ap.add_argument(
        "--verify-probs",
        dest="verify_probs",
        action="store_true",
        default=True,
        help="(default) also assert the teacher-forced train-split "
        "probabilities are bit-identical to the base model's",
    )
    ap.add_argument(
        "--no-verify-probs",
        dest="verify_probs",
        action="store_false",
        help="dict-level stamping only, no data pass",
    )
    args = ap.parse_args()

    params = json.loads(args.params.read_text())
    fields = fields_from(params, args.base)
    base = th.load(args.base, map_location="cpu")
    for k in fields:
        assert k not in base, f"base artifact already carries {k}"
    print(f"base      {rel(args.base)}")
    print(f"  sha256  {sha256(args.base)}")
    print(f"  keys    {sorted(base)}")
    print(f"params    {rel(args.params)}")
    print(f"  sha256  {sha256(args.params)}")
    print(f"  rho={params['rho']} phi_lag1={params['phi']}")
    print(
        f"  stamping copula_phi={fields['copula_phi']} "
        f"({params.get('phi_final_reason', 'lag-1 phi, no phi_final in params')})"
    )

    stamped = dict(base)
    stamped.update(fields)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    th.save(stamped, args.out)

    again = th.load(args.out, map_location="cpu")
    assert_only_fields_added(base, again, fields)
    n_t = n_tensors({k: base[k] for k in base})
    print(f"wrote     {rel(args.out)}")
    print(f"  sha256  {sha256(args.out)}")
    print(f"  stamped {fields}")
    print(f"  {n_t} tensors compared bit-identically vs the base artifact")
    print(f"  modules {[k for k in MODULE_KEYS if base.get(k) is not None]}")

    _, got = load_fields(args.out)
    assert got == fields, f"GraphNetwork.load returned {got}, expected {fields}"
    print(f"  loads   {got}")

    n_rows = None
    if args.verify_probs:
        n_rows = verify_probs(args.base, args.out)
        print(
            f"  probs   {n_rows} teacher-forced train-split rows "
            f"bit-identical to the base model's"
        )
    else:
        print("  probs   SKIPPED (--no-verify-probs)")

    side = args.out.with_suffix(".copula.json")
    side.write_text(
        json.dumps(
            dict(
                copula_rho=fields["copula_rho"],
                copula_phi=fields["copula_phi"],
                copula_switch_every=fields["copula_switch_every"],
                phi_stamped=fields["copula_phi"],
                phi_final_reason=params.get("phi_final_reason"),
                phi_lag1_diagnostic=params.get("phi"),
                params_json=rel(args.params),
                params_json_sha256=sha256(args.params),
                base_artifact=rel(args.base),
                base_artifact_sha256=sha256(args.base),
                output_artifact=rel(args.out),
                output_artifact_sha256=sha256(args.out),
                estimator=params.get("estimator"),
                rho_ci=params.get("rho_ci"),
                phi_ci=params.get("phi_ci"),
                n_tensors_verified=n_t,
                teacher_forced_rows_verified=n_rows,
                note=(
                    "dict-level copy of the base artifact plus the three "
                    "copula fields; every pre-existing key verified "
                    "bit-identical after reload; teacher-forced "
                    "(sample=False) probabilities unchanged"
                ),
                script=rel(__file__),
                date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                git_head=git_head(),
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"  sidecar {rel(side)}")
    print("artifact verified bit-identical to the base outside the copula fields")


if __name__ == "__main__":
    main()
