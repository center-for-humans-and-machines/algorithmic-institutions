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


def test_plot_all_writes_registered_figures(tmp_path, tiny, monkeypatch):
    def _dummy(ax, human, sims):
        visuals.lineplot(
            ax,
            human,
            sims,
            lambda df: df.groupby("round_number")["contribution"].mean(),
            "round",
            "mean contribution",
        )

    monkeypatch.setattr(visuals, "PLOTS", [("TEST_dummy", _dummy)])
    paths = visuals.plot_all(tiny, {"run a": tiny, "run b": tiny}, tmp_path)
    assert (tmp_path / "TEST_dummy.jpg").exists()
    assert str(tmp_path / "TEST_dummy.jpg") in paths


def test_registered_plots_render_on_real_data(tmp_path):
    from pathlib import Path

    from aimanager.evaluation_suite.convert import (
        HUMAN_DATA_FILE,
        SIM_EXAMPLE_FILE,
        load_human,
        load_sim,
    )

    repo = Path(__file__).resolve().parents[3]
    human = load_human(repo / HUMAN_DATA_FILE)
    sims = load_sim(repo / SIM_EXAMPLE_FILE)
    paths = visuals.plot_all(human, sims, tmp_path)
    names = {Path(p).stem for p in paths}
    assert names >= {
        "CA_hist",
        "CB_line",
        "CC_hist",
        "CD_hist",
        "CE_hist",
        "CE_std_line",
        "CF_line",
        "CG_bar",
        "CG_excess_line",
        "SB_line",
        "SC_hist",
        "SC_line",
        "PA_hist",
        "PB_line",
        "PC_line",
        "PD_bar",
        "PD_excess_line",
        "RCA_bar",
        "RCB_line",
        "RCC_bar",
        "RCD_scatter",
        "RSA_line",
        "RPA_line",
        "RPB_line",
    }
    for p in paths:
        assert Path(p).stat().st_size > 0


def _independent_frame(seed=0, n_episodes=200):
    """Every player drawn independently of their group."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = [
        (ep, f"p{i}", r, i // 4, float(rng.integers(0, 21)), float(rng.integers(0, 31)))
        for ep in range(n_episodes)
        for r in range(24)
        for i in range(8)
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "episode_id",
            "participant_code",
            "round_number",
            "group_id",
            "contribution",
            "punishment",
        ],
    )


def test_independence_floor_follows_group_size():
    frame = _independent_frame(n_episodes=2)
    # every cell holds 4 players, so sqrt(mean(1 / n)) is exactly 1/2
    assert visuals.independence_floor(frame, "contribution") == pytest.approx(0.5)
    # drop a player from one cell: that cell's 1/n rises and so does the floor
    smaller = frame.drop(frame.index[0])
    assert visuals.independence_floor(smaller, "contribution") > 0.5


def test_spread_excess_is_one_under_independence():
    """The claim the dashed 1.0 line makes."""
    frame = _independent_frame()
    for measure in ("contribution", "punishment"):
        excess = visuals.spread_excess(frame, measure, block=6)
        assert len(excess) == 4
        assert excess.min() > 0.9 and excess.max() < 1.1


def _oklab(hex_color):
    """sRGB -> OKLab, for perceptual color distance."""
    rgb = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    lms = (
        0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
        0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
        0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b,
    )
    lp, mp, sp = [c ** (1 / 3) for c in lms]
    return (
        0.2104542553 * lp + 0.7936177850 * mp - 0.0040720468 * sp,
        1.9779984951 * lp - 2.4285922050 * mp + 0.4505937099 * sp,
        0.0259040371 * lp + 0.7827717662 * mp - 0.8086757660 * sp,
    )


def _min_pairwise_de(colors):
    out = float("inf")
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            la, lb = _oklab(a), _oklab(b)
            out = min(out, 100 * sum((x - y) ** 2 for x, y in zip(la, lb)) ** 0.5)
    return out


def test_palette_pairwise_distinct():
    # curated prefix (incl. the black human reference): OKLab distance
    # x100 >= 18 on every pair; the first three hues all sit >= 21
    assert _min_pairwise_de([visuals.HUMAN_COLOR] + visuals.PAIRING_COLORS) >= 18
    # generated extensions stay usable and deterministic
    ten = visuals.pairing_colors(10)
    assert ten[:5] == visuals.PAIRING_COLORS
    assert _min_pairwise_de([visuals.HUMAN_COLOR] + ten) >= 8
    assert visuals.pairing_colors(10) == ten


def test_series_fixed_order(tiny):
    sims = {f"run {i}": tiny for i in "abc"}
    labels = [label for label, _, _ in visuals.series(sims)]
    assert labels == ["run a", "run b", "run c"]
    colors = [c for _, _, c in visuals.series(sims)]
    assert colors == visuals.PAIRING_COLORS[:3]

    many = {f"run {i}": tiny for i in range(9)}
    assert len({c for _, _, c in visuals.series(many)}) == 9  # all distinct
