"""Step 7 (autoresearch step 7,
notes/autoresearch_log/contribution-inflated-gmlp.md): stamp a fitted group-
copula dose onto a base bundle.

Loads a base bundle (``--base``, default ``contribution_gaussian_mlp_v2_best
.joblib``) and a params sidecar (``--params``, default
``contribution_gaussian_mlp_v2_group_copula.params.json``), and writes
``--out`` (default ``contribution_gaussian_mlp_v2_group_copula.joblib``) =
the base dict plus a fixed manifest of ``copula_*`` keys -- exactly ``rho_p =
rho_total`` / ``rho_t = 0.0`` (the declared persistent-only dose), the
sidecar's provenance for it, and the lag-1 falsifier reading carried for
provenance only (never consumed by the sampler -- see
``src/aimanager/simulation/linear_ah.py``'s ``copula_rho_p`` / ``copula_rho_t``
gate). The three paths default to the PR #170 constants, so a bare run is
that recipe unchanged; passing all three re-runs the same stamp for a
different trunk (e.g. the inflated emission of this branch) -- see
Amendment B, notes/autoresearch_log/contribution-inflated-gmlp.md: never run
with defaults against a non-default sidecar, that would overwrite a
committed bundle with the wrong stamp.

Precedent: ``scripts/baselines/punishment_copula_rho.py::save_bundle`` (its
``NEW_KEYS`` manifest / identical-object-check pattern), reused here for a
different bundle and a different key set.

The primary provenance check is bundle-vs-sidecar: the sidecar's own
``base_bundle_sha256`` must equal the sha256 of the ``--base`` bundle that
was actually loaded, whatever ``--base`` is. The literal
``EXPECTED_BASE_SHA256`` is a secondary check that applies only when
``--base`` is left at its default (the PR #170 trunk) -- it is not a stand-in
for the primary check on any other trunk.

Verifies, and prints:
  1. every pre-existing key is the identical object in the new dict;
  2. the added-key set is exactly the declared manifest, nothing removed;
  3. reloaded from disk, ``predict`` / ``predict_std`` on the train rows are
     bit-identical (``np.array_equal``) to the base bundle's, using the
     step-2/step-6 row builder (``gmlp_group_copula_diagnostic.build_rows``);
     when the estimator also exposes ``predict_proba`` (the inflated
     emission's actual categorical surface -- ``predict``/``predict_std``
     degrade to its Gaussian body and would pass even if the emission were
     broken), that is checked bit-identical too;
  4. ``LinearAHAdapter`` built from the base and from the stamped bundle
     (``n_agents=8, n_contributions=21``) return identical levels on a fixed
     6-round switching sequence with ``sample=False`` (the copula path is
     never taken there, so this only guards against a broken reload);
  5. the reloaded bundle reports ``copula_rho_p`` / ``copula_rho_t`` through
     the adapter, matching the sidecar's own ``rho_total`` / 0.0 -- never a
     literal -- and the adapter accepts the pair (the step-3 gate allows it);
  6. the sha256 of the new .joblib, on its own line (re-checked on Raven).

Local run (CPU torch, no PyG):
    uv run python scripts/baselines/stamp_contribution_group_copula.py
    uv run python scripts/baselines/stamp_contribution_group_copula.py \\
        --base artifacts/baselines/CANDIDATE_best.joblib \\
        --params artifacts/baselines/CANDIDATE_group_copula.params.json \\
        --out artifacts/baselines/CANDIDATE_group_copula.joblib
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import torch as th  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

from aimanager.simulation.linear_ah import LinearAHAdapter  # noqa: E402
from gmlp_group_copula_diagnostic import TRAIN_CFG, build_rows  # noqa: E402
from handcrafted_grid import load_config  # noqa: E402

# Today's constants -- the PR #170 recipe -- doubling as the CLI defaults, so
# a bare run reproduces PR #170 exactly (Amendment B).
BASE_PATH = ROOT / "artifacts/baselines/contribution_gaussian_mlp_v2_best.joblib"
PARAMS_PATH = (
    ROOT / "artifacts/baselines/contribution_gaussian_mlp_v2_group_copula.params.json"
)
OUT = ROOT / "artifacts/baselines/contribution_gaussian_mlp_v2_group_copula.joblib"
# Only checked when --base is left at its default (BASE_PATH above); see the
# module docstring -- the bundle-vs-sidecar sha256 (load_inputs) is primary.
EXPECTED_BASE_SHA256 = (
    "2f0b02e2588dbd8b2c4860ca3918d670095a6eb32851bcec931392c2d6a02e75"
)

# The declared manifest of keys this step adds -- nothing else. `copula_rho_p`
# / `copula_rho_t` are the two the sampler reads; `copula_rho_lag1*` and
# `copula_n_pairs_lag1` are the pre-registered falsifier (Note 28: it inverted
# and is no longer discriminating, but it is still carried as provenance, per
# plan), never stamped as the dose; `copula_base_bundle_sha256` records what
# this bundle was built from.
NEW_KEYS = {
    "copula_rho_p",
    "copula_rho_t",
    "copula_rho_total_ci",
    "copula_structure",
    "copula_estimator",
    "copula_cell_key",
    "copula_data_file",
    "copula_n_pairs",
    "copula_rho_lag1",
    "copula_rho_lag1_ci",
    "copula_n_pairs_lag1",
    "copula_base_bundle_sha256",
}

N_AGENTS = 8
N_CONTRIBUTIONS = 21
# fixed state sequence for verification 4: two groups of four, one switch at
# round 3 (agent 0 <-> agent 4) so the sequence exercises _group bookkeeping,
# not just a static membership.
GROUPS_BY_ROUND = [
    [0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 1, 1],
]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(base_path, params_path):
    import joblib

    base_sha = sha256_of(base_path)
    params = json.loads(params_path.read_text())

    # PRIMARY check: the sidecar was fitted against exactly this bundle on
    # disk, whatever --base is. This is what actually binds -- a sidecar
    # fitted against a different trunk's marginal must never be stamped onto
    # this one.
    assert params["base_bundle_sha256"] == base_sha, (
        "params sidecar's own base_bundle_sha256 disagrees with the --base "
        f"bundle on disk: sidecar says {params['base_bundle_sha256']!r}, "
        f"{base_path.relative_to(ROOT)} hashes to {base_sha!r} -- stop, do "
        "not stamp"
    )

    # Secondary check, default --base only: pins the PR #170 recipe to its
    # known trunk so a bare run is provably that recipe and nothing else.
    if base_path == BASE_PATH:
        assert base_sha == EXPECTED_BASE_SHA256, (
            f"base bundle sha256 mismatch: got {base_sha}, expected "
            f"{EXPECTED_BASE_SHA256} -- stop, do not stamp"
        )

    base = joblib.load(base_path)
    print(f"base bundle sha256 verified: {base_sha}")
    print(f"loaded params sidecar {params_path.relative_to(ROOT)}")
    return base, params


def build_new_bundle(base, params):
    """base dict + the declared copula_* manifest -- rho_p = rho_total (the
    dose, taken from the JSON, never retyped), rho_t = 0.0."""
    new = dict(base)
    new.update(
        copula_rho_p=float(params["rho_total"]),
        copula_rho_t=0.0,
        copula_rho_total_ci=(
            float(params["rho_total_ci"][0]),
            float(params["rho_total_ci"][1]),
        ),
        copula_structure=str(params["structure"]),
        copula_estimator=str(params["estimator"]),
        copula_cell_key=str(params["cell_key"]),
        copula_data_file=str(params["data_file"]),
        copula_n_pairs=int(params["n_pairs_within"]),
        # provenance only -- the pre-registered falsifier (Note 28: inverted,
        # not discriminating, kept for the record), NOT consumed by the
        # sampler.
        copula_rho_lag1=float(params["rho_lag1"]),
        copula_rho_lag1_ci=(
            float(params["rho_lag1_ci"][0]),
            float(params["rho_lag1_ci"][1]),
        ),
        copula_n_pairs_lag1=int(params["n_pairs_lag1"]),
        copula_base_bundle_sha256=str(params["base_bundle_sha256"]),
    )

    # verification 1: every pre-existing key is the identical object.
    n_checked = 0
    for k, v in base.items():
        assert new[k] is v, f"pre-existing bundle key modified: {k}"
        n_checked += 1
    print(f"[1] identical-object check: {n_checked} pre-existing keys, all `is` base")

    # verification 2: exactly the declared manifest added, nothing removed.
    added = set(new) - set(base)
    removed = set(base) - set(new)
    assert added == NEW_KEYS, (added, NEW_KEYS)
    assert not removed, removed
    print(f"[2] key-set check: added={sorted(added)}")
    print("    removed: none")

    return new


def save_and_reload(new, out_path):
    import joblib

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(new, out_path)
    return joblib.load(out_path)


def check_predict_bit_identical(base, reloaded):
    """Verification 3: predict / predict_std on the train rows, base vs
    reloaded stamped bundle, bit-identical (np.array_equal, not allclose).

    On an inflated bundle those two are the mixture's Gaussian BODY only --
    they would stay bit-identical even if the emission actually sampled
    (predict_proba) were broken by the reload. So when the estimator exposes
    predict_proba (the inflated model's real categorical surface, per
    gaussian_regressor.py's docstring), that is checked bit-identical too;
    a bundle without it (the plain Gaussian models) is unaffected."""
    cfg = load_config(TRAIN_CFG)
    rows = build_rows(cfg)
    X = np.column_stack([rows["pool"][k][rows["mask"]] for k in base["features"]])

    Xs_base = base["scaler"].transform(X)
    mu_base = np.asarray(base["estimator"].predict(Xs_base))
    sd_base = np.asarray(base["estimator"].predict_std(Xs_base))

    Xs_new = reloaded["scaler"].transform(X)
    mu_new = np.asarray(reloaded["estimator"].predict(Xs_new))
    sd_new = np.asarray(reloaded["estimator"].predict_std(Xs_new))

    mu_ok = bool(np.array_equal(mu_base, mu_new))
    sd_ok = bool(np.array_equal(sd_base, sd_new))
    assert mu_ok, "predict() is NOT bit-identical after reload"
    assert sd_ok, "predict_std() is NOT bit-identical after reload"
    print(
        f"[3] reload check on {len(X)} train rows: "
        f"predict bit-identical={mu_ok}  predict_std bit-identical={sd_ok}"
    )

    has_proba = hasattr(base["estimator"], "predict_proba")
    if has_proba:
        P_base = np.asarray(base["estimator"].predict_proba(Xs_base))
        P_new = np.asarray(reloaded["estimator"].predict_proba(Xs_new))
        proba_ok = bool(np.array_equal(P_base, P_new))
        assert proba_ok, "predict_proba() is NOT bit-identical after reload"
        print(
            f"[3b] reload check on {len(X)} train rows: "
            f"predict_proba bit-identical={proba_ok} "
            f"(shape={P_base.shape}, the emission actually sampled)"
        )
    else:
        print("[3b] estimator has no predict_proba -- not an inflated model, skipped")


def _adapter_state(t, groups, prev_groups=None, n_agents=N_AGENTS):
    ag = th.tensor(groups, dtype=th.int64).reshape(1, n_agents, 1)
    st = {
        "round_number": th.full((1, n_agents, 1), t, dtype=th.int64),
        "agent_group": ag,
    }
    if t > 0:
        prev = (
            ag
            if prev_groups is None
            else th.tensor(prev_groups, dtype=th.int64).reshape(1, n_agents, 1)
        )
        st["prev_contribution"] = th.full((1, n_agents, 1), 9.0)
        st["prev_punishment"] = th.zeros((1, n_agents, 1))
        st["prev_common_good"] = th.full((1, n_agents, 1), 12.0)
        st["prev_agent_group"] = prev
    return st


def _levels_over_fixed_episode(bundle):
    """Drive `LinearAHAdapter(bundle, sample=False)` over GROUPS_BY_ROUND and
    return the [rounds, agents] deterministic levels."""
    ad = LinearAHAdapter(
        bundle, n_agents=N_AGENTS, n_contributions=N_CONTRIBUTIONS, sample=False
    )
    out = []
    for t, g in enumerate(GROUPS_BY_ROUND):
        st = _adapter_state(t, g, GROUPS_BY_ROUND[t - 1] if t else None)
        pred, _ = ad.predict(st, reset_rnn=(t == 0))
        out.append(pred.reshape(-1).numpy().copy())
    return np.stack(out), ad


def check_adapter_equivalence(base, reloaded, params):
    """Verification 4: base vs stamped adapters give the same levels on the
    deterministic path (sample=False), over the fixed 6-round switching
    sequence GROUPS_BY_ROUND. Verification 5: the stamped bundle's
    copula_rho_p / copula_rho_t come back through a sample=True adapter,
    read against the sidecar's own rho_total -- never a literal."""
    levels_base, _ = _levels_over_fixed_episode(base)
    levels_new, _ = _levels_over_fixed_episode(reloaded)
    same = bool(np.array_equal(levels_base, levels_new))
    assert same, "sample=False adapter levels differ between base and stamped"
    print(
        f"[4] adapter equivalence (sample=False) over {len(GROUPS_BY_ROUND)} "
        f"rounds x {N_AGENTS} agents: identical levels={same}"
    )

    ad_sample = LinearAHAdapter(
        reloaded, n_agents=N_AGENTS, n_contributions=N_CONTRIBUTIONS, sample=True
    )
    rho_p = ad_sample.copula_rho_p
    rho_t = ad_sample.copula_rho_t
    expected_rho_p = float(params["rho_total"])
    assert rho_p == expected_rho_p, (rho_p, expected_rho_p)
    assert rho_t == 0.0, rho_t
    print(
        f"[5] adapter accepted the pair (contribution bundle); reads "
        f"copula_rho_p={rho_p!r} copula_rho_t={rho_t!r} "
        f"(sidecar rho_total={expected_rho_p!r})"
    )


def check_lfs(joblib_path):
    gitattributes = (ROOT / ".gitattributes").read_text()
    tracked = any(
        line.split()[0] == "*.joblib"
        for line in gitattributes.splitlines()
        if line.strip() and not line.startswith("#")
    )
    print(
        f"\n*.joblib LFS-tracked per .gitattributes: {tracked} "
        f"({joblib_path.relative_to(ROOT)} is a plain git-tracked file, "
        "not LFS, if False)"
    )
    return tracked


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base",
        type=Path,
        default=BASE_PATH,
        help="base model bundle the dose is stamped onto (default: %(default)s)",
    )
    ap.add_argument(
        "--params",
        type=Path,
        default=PARAMS_PATH,
        help="params sidecar carrying the fitted dose (default: %(default)s)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT,
        help="stamped bundle written here (default: %(default)s)",
    )
    return ap


def main():
    args = build_parser().parse_args()
    base_path = args.base.resolve()
    params_path = args.params.resolve()
    out_path = args.out.resolve()

    base, params = load_inputs(base_path, params_path)
    new = build_new_bundle(base, params)
    reloaded = save_and_reload(new, out_path)

    check_predict_bit_identical(base, reloaded)
    check_adapter_equivalence(base, reloaded, params)
    check_lfs(out_path)

    new_sha = sha256_of(out_path)
    print(f"\nsaved {out_path.relative_to(ROOT)}")
    print(
        f"  copula_rho_p={new['copula_rho_p']!r}  copula_rho_t={new['copula_rho_t']!r}"
    )
    print(f"  copula_structure={new['copula_structure']!r}")
    print("[6] NEW BUNDLE SHA256:")
    print(new_sha)


if __name__ == "__main__":
    main()
