"""Smoke tests for the evaluation-suite visuals (#137)."""

import pandas as pd
import pytest

from aimanager.evaluation_suite import visuals


@pytest.fixture()
def tiny():
    return pd.DataFrame(
        {
            "episode_id": [0, 0],
            "round_number": [0, 1],
            "contribution": [5.0, 10.0],
        }
    )


def test_plot_all_writes_registered_figures(tmp_path, tiny):
    @visuals.plot("TEST_dummy")
    def _dummy(ax, human, sims):
        visuals.lineplot(
            ax,
            human,
            sims,
            lambda df: df.groupby("round_number")["contribution"].mean(),
            "round",
            "mean contribution",
        )

    try:
        paths = visuals.plot_all(tiny, {"run a": tiny, "run b": tiny}, tmp_path)
        assert (tmp_path / "TEST_dummy.jpg").exists()
        assert str(tmp_path / "TEST_dummy.jpg") in paths
    finally:
        visuals.PLOTS[:] = [p for p in visuals.PLOTS if p[0] != "TEST_dummy"]


def test_series_fixed_order_and_no_cycling(tiny):
    sims = {f"run {i}": tiny for i in "abc"}
    labels = [label for label, _, _ in visuals.series(sims)]
    assert labels == ["run a", "run b", "run c"]
    colors = [c for _, _, c in visuals.series(sims)]
    assert colors == visuals.PAIRING_COLORS[:3]

    too_many = {f"run {i}": tiny for i in range(9)}
    with pytest.raises(ValueError, match="facet or fold"):
        visuals.series(too_many)
