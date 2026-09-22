"""The reference-column guard for the bootstrapped-DQN arm.

Runs locally: no PyG imports.

The failure this guards against is silent. At `0ff44a9` `RuleBasedManager` is a
single fixed formula, `(self, k=1, n_punishments=31, **_)`, so a simulation
config line reading `rule: never` is swallowed by `**_` and ignored. The run
completes clean and the output carries the label you asked for on the default
formula's behaviour. Nothing raises, so the label has to be checked against
the realised behaviour instead.
"""

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "rl_bootstrapped"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import guard  # noqa: E402

BINS = ["{0}", "1-5", "6-10", "11-15", "16-19", "{20}"]


def _reference(**cols):
    return pd.DataFrame(cols, index=pd.Index(BINS, name="contribution_bin"))


def test_a_never_column_that_punishes_is_rejected():
    ref = _reference(never=[2.57, 2.4, 2.2, 2.0, 1.8, 1.5])
    with pytest.raises(SystemExit, match="punishes"):
        guard.validate_reference(ref)


def test_a_never_column_of_exact_zeros_passes():
    guard.validate_reference(_reference(never=[0.0] * 6))


def test_two_rules_agreeing_to_two_decimals_are_rejected():
    ref = _reference(
        never=[0.0] * 6,
        thr9_p10=[4.001, 3.0, 2.0, 1.0, 0.5, 0.25],
        prop10=[4.002, 3.0, 2.0, 1.0, 0.5, 0.25],
    )
    with pytest.raises(SystemExit, match="agree to two decimal"):
        guard.validate_reference(ref)


def test_genuinely_distinct_rules_pass():
    guard.validate_reference(
        _reference(
            never=[0.0] * 6,
            prop10=[20.0, 16.58, 10.26, 6.86, 2.68, 0.0],
            thr9_p10=[10.0, 10.0, 4.22, 0.0, 0.0, 0.0],
        )
    )


def test_the_committed_reference_columns_are_sound():
    """The file this arm actually ships, checked rather than eyeballed."""
    path = os.path.join(
        ROOT,
        "plots/data_analysis/evaluation/rl_manager_bootstrapped_dqn",
        "reference_policy_shape.csv",
    )
    ref = pd.read_csv(path, index_col="contribution_bin")
    guard.validate_reference(ref)
    assert ref["never"].abs().max() == 0.0
    assert ref["prop10"].max() == pytest.approx(20.0)
    assert not ref["prop10"].round(2).equals(ref["thr9_p10"].round(2))
