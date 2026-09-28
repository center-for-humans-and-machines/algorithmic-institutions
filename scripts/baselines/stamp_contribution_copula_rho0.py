"""Copula-off copies of the two Gaussian-MLP contributor bundles for the
head-state-spread diagnostic (notes/autoresearch_log/head-state-spread-
diagnostic.md): the like-for-like counterpart of PR #186's arm B (bare
categorical trunk, rho = 0).

`copula_rho_p` and `copula_rho_t` both 0.0 close the group-copula gate in
`LinearAHAdapter.predict`, which restores the independent sampler and its
exact RNG stream (PR #170). Every other key is the identical object; the
reloaded estimator is checked weight-identical and bit-identical in
predict / predict_std (/ predict_proba) on the training rows. Nothing is
recalibrated -- the fitted dose stays on the source bundle untouched.

Local run (CPU torch, no PyG):
    uv run python scripts/baselines/stamp_contribution_copula_rho0.py
"""

import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch as th  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

from aimanager.simulation.linear_ah import LinearAHAdapter  # noqa: E402
from gmlp_group_copula_diagnostic import build_rows  # noqa: E402
from handcrafted_grid import load_config  # noqa: E402
from stamp_contribution_group_copula import (  # noqa: E402
    GROUPS_BY_ROUND,
    _levels_over_fixed_episode,
)

BASELINES = ROOT / "artifacts/baselines"
BUNDLES = [
    "contribution_gaussian_mlp_inflated_group_copula",
    "contribution_gaussian_mlp_v2_group_copula",
]


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp(name):
    src = BASELINES / f"{name}.joblib"
    out = BASELINES / f"{name}_rho0.joblib"
    base = joblib.load(src)
    assert base["copula_rho_p"] > 0 and base["copula_rho_t"] == 0.0, (
        base["copula_rho_p"],
        base["copula_rho_t"],
    )
    new = dict(base)
    new["copula_rho_p"] = 0.0
    new["copula_rho_t"] = 0.0
    new["copula_rho0_source"] = f"{name}.joblib"
    new["copula_rho0_source_sha256"] = sha256_of(src)
    for k, v in base.items():
        if k not in ("copula_rho_p", "copula_rho_t"):
            assert new[k] is v, k
    joblib.dump(new, out)
    re = joblib.load(out)

    # weights: every tensor of the estimator's net identical after reload
    sa, sb = base["estimator"].net.state_dict(), re["estimator"].net.state_dict()
    assert sa.keys() == sb.keys() and all(th.equal(sa[k], sb[k]) for k in sa)
    # conditionals on the training rows bit-identical
    rows = build_rows(load_config(ROOT / base["config"]))
    X = np.column_stack([rows["pool"][k][rows["mask"]] for k in base["features"]])
    Xa, Xb = base["scaler"].transform(X), re["scaler"].transform(X)
    assert np.array_equal(Xa, Xb)
    for fn in ("predict", "predict_std", "predict_proba"):
        if hasattr(base["estimator"], fn):
            a = getattr(base["estimator"], fn)(Xa)
            b = getattr(re["estimator"], fn)(Xb)
            assert np.array_equal(np.asarray(a), np.asarray(b)), fn
    # the adapter: deterministic levels identical, the gate closed
    la, _ = _levels_over_fixed_episode(base)
    lb, ad = _levels_over_fixed_episode(re)
    assert np.array_equal(la, lb)
    ad = LinearAHAdapter(re, n_agents=8, n_contributions=21, sample=True)
    assert ad.copula_rho_p == 0.0 and ad.copula_rho_t == 0.0
    print(
        f"{out.relative_to(ROOT)}: rho_p {base['copula_rho_p']!r} -> 0.0, "
        f"rho_t 0.0; {len(sa)} tensors identical; predict* bit-identical on "
        f"{len(X)} rows; adapter levels identical over {len(GROUPS_BY_ROUND)} "
        f"rounds; sha256 {sha256_of(out)}"
    )


if __name__ == "__main__":
    for name in BUNDLES:
        stamp(name)
