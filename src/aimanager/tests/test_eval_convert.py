"""Structure and semantics tests for the evaluation-suite conversion layer.

Confirms that the human and simulation conversions produce exactly the same
canonical structure (#128) by converting both tracked sources fresh.
"""

from pathlib import Path

import pandas as pd
import pytest

from aimanager.evaluation_suite.convert import (
    CANONICAL_COLUMNS,
    HUMAN_DATA_FILE,
    SIM_EXAMPLE_FILE,
    load_human,
    load_sim,
)

REPO = Path(__file__).resolve().parents[3]
DECISION_ROUNDS = {3, 7, 11, 15, 19}


@pytest.fixture(scope="module")
def human():
    return load_human(REPO / HUMAN_DATA_FILE)


@pytest.fixture(scope="module")
def sims():
    return load_sim(REPO / SIM_EXAMPLE_FILE)


def test_structures_match_exactly(human, sims):
    assert list(human.columns) == CANONICAL_COLUMNS
    for sim in sims.values():
        assert list(sim.columns) == CANONICAL_COLUMNS
        assert sim.dtypes.astype(str).to_dict() == human.dtypes.astype(str).to_dict()


def test_human_dedupe_and_masking(human):
    raw = pd.read_csv(REPO / HUMAN_DATA_FILE)
    assert raw["episode_id"].nunique() == 100  # flip augmentation doubles
    assert human["episode_id"].nunique() == 50
    deduped_raw = raw[
        raw["episode_id"] == raw.groupby("pair_id")["episode_id"].transform("min")
    ]
    assert (
        human["contribution"].isna().sum()
        == (deduped_raw["player_no_input"] != 0).sum()
    )
    assert (
        human["punishment"].isna().sum() == (deduped_raw["manager_no_input"] != 0).sum()
    )


def test_switch_labelling(human, sims):
    for df in [human, *sims.values()]:
        assert set(df.loc[df["switch_mask"], "round_number"]) == DECISION_ROUNDS
        assert (df["does_switch"] <= df["switch_mask"]).all()
        assert (df["switch_valid"] <= df["switch_mask"]).all()

        # membership changes exactly where a decision row said "switch"
        df = df.sort_values(["episode_id", "participant_code", "round_number"])
        by_player = df.groupby(["episode_id", "participant_code"])
        changes = (
            by_player["group_id"].shift(-1).ne(df["group_id"])
            & by_player["group_id"].shift(-1).notna()
        )
        assert changes.equals(df["does_switch"])


def test_human_timeouts_excluded(human):
    assert human["switch_valid"].sum() < human["switch_mask"].sum()


def test_sim_has_no_invalid_rows(sims):
    for sim in sims.values():
        assert sim["contribution"].notna().all()
        assert sim["punishment"].notna().all()
        assert sim["switch_valid"].equals(sim["switch_mask"])


def test_episode_shapes(human, sims):
    for df in [human, *sims.values()]:
        rounds = df.groupby("episode_id")["round_number"].agg(["min", "max", "size"])
        assert (rounds["min"] == 0).all()
        assert (rounds["max"] == 23).all()
        assert (rounds["size"] == 8 * 24).all()
        assert set(df["group_id"]) <= {0, 1}
