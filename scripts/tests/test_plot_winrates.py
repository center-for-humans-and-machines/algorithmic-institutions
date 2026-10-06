"""Tests for the five win definitions in scripts/plotting/plot_winrates.py."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "plot_winrates", ROOT / "scripts/plotting/plot_winrates.py"
)
pw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pw)


def _run(pairing, contrib, empty_group=None, n_episodes=4, n_rounds=2):
    """Two agents per group, constant contribution per group, no punishment;
    `empty_group` has no members in round 1."""
    rows = []
    for ep in range(n_episodes):
        for t in range(n_rounds):
            for group, c in enumerate(contrib):
                if group == empty_group and t == 1:
                    continue
                for _ in range(2):
                    rows.append(
                        {
                            "run": f"ah group_switching managed by {pairing}",
                            "episode": ep,
                            "round_number": t,
                            "agent_group": group,
                            "contribution": c,
                            "punishment": 0,
                            "payoff": 0.0,
                            "common_good": 0.0,
                        }
                    )
    return rows


def _df(*runs):
    df = pd.DataFrame([r for run in runs for r in run])
    df[pw.POOL] = 1.6 * df["contribution"] - df["punishment"]
    return df


def test_pool_head_to_head_margin():
    # x gives 10 (pool 32 per round), y gives 5 (pool 16): margin +16
    df = _df(_run("x_vs_y", (10, 5)), _run("y_vs_x", (5, 10)))
    tbl = pw.pool_h2h_table(df, pw.pairing_sides(df))
    row = tbl.set_index("matchup (a vs b)").loc["x vs y"]
    assert row["episodes"] == 8 and row["a_win%"] == 100.0
    assert row["margin a - b [95%]"] == "+16.0 [+16.0, +16.0]"


def test_empty_group_round_counts_as_zero():
    # y gives 10 in round 0 (pool 32) and is empty in round 1 (pool 0)
    df = _df(_run("x_vs_y", (10, 10), empty_group=1))
    tbl = pw.pool_h2h_table(df, pw.pairing_sides(df))
    assert tbl.iloc[0]["b_pool"] == 16.0


def test_anchor_ranks_and_warns():
    df = _df(
        _run("x_vs_ah", (10, 6)),
        _run("ah_vs_zero", (6, 5)),
        _run("y_vs_zero", (3, 5)),  # y never plays ah
    )
    tbl, warnings = pw.anchor_table(df, "ah", "zero")
    assert list(tbl["manager"]) == ["x", "zero (reference)"]
    assert tbl.iloc[0]["vs zero [95%]"] == "+16.0 [+16.0, +16.0]"
    assert warnings == ["not compared, never played `ah`: y"]


def test_anchor_without_reference_warns():
    df = _df(_run("x_vs_ah", (10, 6)))
    tbl, warnings = pw.anchor_table(df, "ah", "zero")
    assert "vs zero [95%]" not in tbl.columns
    assert any("reference `zero` never played `ah`" in w for w in warnings)
