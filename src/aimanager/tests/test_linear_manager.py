"""MultiManager wiring test for the linear punishment manager (#127).

Runs on Raven (api_manager imports GraphNetwork -> torch_scatter). The
feature-parity of the underlying adapter is covered locally in
tests/baselines/test_baseline_features.py.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = ROOT / "artifacts/baselines/punishment_multinomial_best_with_contr.joblib"


def _round(t, punishments):
    return {
        "contribution": [5, 10, 20, 0, 7, 14, 3, 18],
        "contribution_valid": [True] * 8,
        "punishment": punishments,
        "punishment_valid": [p is not None for p in punishments],
        "agent_group": [0, 0, 0, 0, 1, 1, 1, 1],
        "group": ["lin"] * 4 + ["zero"] * 4,
        "round": t,
    }


def test_multimanager_linear_side():
    from aimanager.manager.api_manager import MultiManager

    mm = MultiManager(
        {
            "lin": {"type": "linear", "model_path": str(BUNDLE)},
            "zero": {"type": "dummy", "constant_punishment": 0},
        }
    )
    rounds = [_round(0, [2, 0, 0, 5, 1, 0, 4, 0]), _round(1, [None] * 8)]
    matched, per_manager = mm.get_punishments(rounds)
    assert len(matched) == 8
    assert all(0 <= p <= 30 for p in matched)
    assert matched[4:] == [0, 0, 0, 0]  # dummy side of the pairing
    assert set(per_manager) == {"lin", "zero"}
