"""Step 7 (autoresearch step 7,
notes/autoresearch_log/contribution-gmlp-group-copula.md): stamp the step-6
fitted dose into a new model bundle.

Loads the base ``contribution_gaussian_mlp_v2_best.joblib`` bundle (asserting
its sha256 against the value recorded in the step-6 params sidecar) and the
sidecar JSON ``contribution_gaussian_mlp_v2_group_copula.params.json``, and
writes ``contribution_gaussian_mlp_v2_group_copula.joblib`` = the base dict
plus a fixed manifest of ``copula_*`` keys -- exactly ``rho_p =
rho_total`` / ``rho_t = 0.0`` (the declared persistent-only dose), the
sidecar's provenance for it, and the lag-1 falsifier reading carried for
provenance only (never consumed by the sampler -- see
``src/aimanager/simulation/linear_ah.py``'s ``copula_rho_p`` / ``copula_rho_t``
gate).

Precedent: ``scripts/baselines/punishment_copula_rho.py::save_bundle`` (its
``NEW_KEYS`` manifest / identical-object-check pattern), reused here for a
different bundle and a different key set.

Verifies, and prints:
  1. every pre-existing key is the identical object in the new dict;
  2. the added-key set is exactly the declared manifest, nothing removed;
  3. reloaded from disk, ``predict`` / ``predict_std`` on the train rows are
     bit-identical (``np.array_equal``) to the base bundle's, using the
     step-2/step-6 row builder (``gmlp_group_copula_diagnostic.build_rows``);
  4. ``LinearAHAdapter`` built from the base and from the stamped bundle
     (``n_agents=8, n_contributions=21``) return identical levels on a fixed
     state sequence with ``sample=False`` (the copula path is never taken
     there, so this only guards against a broken reload);
  5. the reloaded bundle reports ``copula_rho_p`` / ``copula_rho_t`` through
     the adapter, and the adapter accepts the pair (gaussian_mlp contribution
     bundle -- the step-3 gate allows it);
  6. the sha256 of the new .joblib, on its own line (re-checked on Raven).

Local run (CPU torch, no PyG):
    uv run python scripts/baselines/stamp_contribution_group_copula.py
"""

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

BASE_PATH = ROOT / "artifacts/baselines/contribution_gaussian_mlp_v2_best.joblib"
PARAMS_PATH = (
    ROOT / "artifacts/baselines/contribution_gaussian_mlp_v2_group_copula.params.json"
)
OUT = ROOT / "artifacts/baselines/contribution_gaussian_mlp_v2_group_copula.joblib"
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


def load_inputs():
    import joblib

    base_sha = sha256_of(BASE_PATH)
    assert base_sha == EXPECTED_BASE_SHA256, (
        f"base bundle sha256 mismatch: got {base_sha}, expected "
        f"{EXPECTED_BASE_SHA256} -- stop, do not stamp"
    )
    params = json.loads(PARAMS_PATH.read_text())
    assert params["base_bundle_sha256"] == EXPECTED_BASE_SHA256, (
        "params sidecar's own base_bundle_sha256 disagrees with the base "
        f"bundle on disk: {params['base_bundle_sha256']!r} vs {base_sha!r}"
    )
    base = joblib.load(BASE_PATH)
    print(f"base bundle sha256 verified: {base_sha}")
    print(f"loaded params sidecar {PARAMS_PATH.relative_to(ROOT)}")
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


def save_and_reload(new):
    import joblib

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(new, OUT)
    return joblib.load(OUT)


def check_predict_bit_identical(base, reloaded):
    """Verification 3: predict / predict_std on the train rows, base vs
    reloaded stamped bundle, bit-identical (np.array_equal, not allclose)."""
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


def check_adapter_equivalence(base, reloaded):
    """Verification 4: base vs stamped adapters give the same levels on the
    deterministic path (sample=False). Verification 5: the stamped bundle's
    copula_rho_p / copula_rho_t come back through a sample=True adapter."""
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
    assert rho_p == 0.04378520865574197, rho_p
    assert rho_t == 0.0, rho_t
    print(
        f"[5] adapter accepted the pair (gaussian_mlp contribution bundle); "
        f"reads copula_rho_p={rho_p!r} copula_rho_t={rho_t!r}"
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


def main():
    base, params = load_inputs()
    new = build_new_bundle(base, params)
    reloaded = save_and_reload(new)

    check_predict_bit_identical(base, reloaded)
    check_adapter_equivalence(base, reloaded)
    check_lfs(OUT)

    new_sha = sha256_of(OUT)
    print(f"\nsaved {OUT.relative_to(ROOT)}")
    print(
        f"  copula_rho_p={new['copula_rho_p']!r}  copula_rho_t={new['copula_rho_t']!r}"
    )
    print(f"  copula_structure={new['copula_structure']!r}")
    print("[6] NEW BUNDLE SHA256:")
    print(new_sha)


if __name__ == "__main__":
    main()
