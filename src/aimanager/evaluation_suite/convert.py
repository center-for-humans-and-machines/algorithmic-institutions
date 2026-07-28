"""Convert simulation output and human experiment data into one canonical
agent-round format for the evaluation suite (#128).

Canonical frame, one row per (episode, participant, round):

    episode_id        int64    game identifier, unique within the frame
    participant_code  str      participant identifier, unique within the frame
    round_number      int64    0..23
    group_id          int64    membership at this round (0 or 1)
    contribution      float64  0..20; NaN where the player gave no input
    punishment        float64  0..30 received; NaN where the manager gave
                               no input
    does_switch       bool     True at a decision round whose choice was
                               a switch
    switch_mask       bool     True at decision rounds (rounds 3, 7, 11, 15,
                               19 with switch_every=4)
    switch_valid      bool     switch_mask minus timed-out human choices;
                               identical to switch_mask for the simulation

`common_good` and `payoff` are excluded: no metric uses them, and
`common_good` is stored on different scales in the two sources (per-capita
in the simulation, per-group pool in the human CSVs).

Human games appear twice in the CSVs with group labels swapped (the flip
augmentation); `load_human` keeps one copy per game. No-input and timeout
semantics mirror the training pipeline (generic/data.py::parse_agent_rounds).

Running this module as a script dumps small converted examples of both
sources to examples/ (gitignored) for eyeballing.
"""

from pathlib import Path

import pandas as pd
import pandera as pa
from pandera.typing import Series

HUMAN_DATA_FILE = "experiments/2group_8agent_50ep.csv"
SIM_EXAMPLE_FILE = "plots/simulation/22_2g8a_linear_self_ridge_contr/per_round.parquet"

CANONICAL_COLUMNS = [
    "episode_id",
    "participant_code",
    "round_number",
    "group_id",
    "contribution",
    "punishment",
    "does_switch",
    "switch_mask",
    "switch_valid",
]


class CanonicalAgentRound(pa.DataFrameModel):
    episode_id: Series[int]
    participant_code: Series[str]
    round_number: Series[int]
    group_id: Series[int]
    contribution: Series[float] = pa.Field(nullable=True, ge=0, le=20)
    punishment: Series[float] = pa.Field(nullable=True, ge=0, le=30)
    does_switch: Series[bool]
    switch_mask: Series[bool]
    switch_valid: Series[bool]


def _derive_switching(df, switch_every):
    """Mirror of generic/data.py::parse_agent_rounds' switch labelling.

    does_switch sits at the DECISION round s (the round played right before
    each arrival); the membership change realises between s and s+1. The
    episode's last round has no following arrival, so it is never a decision
    row. selection_timeout is logged at the arrival round s+1 and aligned
    back to the decision row.
    """
    df = df.sort_values(["episode_id", "participant_code", "round_number"])
    by_player = df.groupby(["episode_id", "participant_code"])
    next_group = by_player["group_id"].shift(-1)
    is_decision = ((df["round_number"] + 1) % switch_every == 0) & next_group.notna()
    switches = next_group.notna() & next_group.ne(df["group_id"])
    df["does_switch"] = (switches & is_decision).astype(bool)
    df["switch_mask"] = is_decision.astype(bool)
    if "selection_timeout" in df.columns:
        next_timeout = by_player["selection_timeout"].shift(-1)
        df["switch_valid"] = df["switch_mask"] & (
            next_timeout.fillna(0).astype(int) == 0
        )
    else:
        df["switch_valid"] = df["switch_mask"]
    return df


def _finalize(df):
    df = df.astype(
        {
            "episode_id": "int64",
            "round_number": "int64",
            "group_id": "int64",
            "contribution": "float64",
            "punishment": "float64",
        }
    )
    df = (
        df[CANONICAL_COLUMNS]
        .sort_values(["episode_id", "participant_code", "round_number"])
        .reset_index(drop=True)
    )
    CanonicalAgentRound(df)
    return df


def load_human(csv_path, switch_every=4):
    """Human experiment CSV -> canonical frame."""
    df = pd.read_csv(csv_path)
    if "pair_id" in df.columns:
        keep = df.groupby("pair_id")["episode_id"].transform("min")
        df = df[df["episode_id"] == keep].copy()
    df["contribution"] = df["contribution"].where(df["player_no_input"] == 0)
    df["punishment"] = df["punishment"].where(df["manager_no_input"] == 0)
    df = _derive_switching(df, switch_every)
    return _finalize(df)


def load_sim(parquet_path, switch_every=4):
    """Simulation per_round.parquet -> {run name: canonical frame}, one per
    pairing. episode/participant_code are already unique within a run."""
    df = pd.read_parquet(parquet_path)
    out = {}
    for run, run_df in df.groupby("run"):
        run_df = run_df.rename(columns={"episode": "episode_id"})
        run_df = _derive_switching(run_df, switch_every)
        out[run] = _finalize(run_df)
    return out


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[3]
    example_dir = Path(__file__).parent / "examples"
    example_dir.mkdir(exist_ok=True)

    human = load_human(repo / HUMAN_DATA_FILE)
    episodes = sorted(human["episode_id"].unique())[:2]
    human_example = human[human["episode_id"].isin(episodes)]
    human_example.to_csv(example_dir / "human_example.csv", index=False)

    sims = load_sim(repo / SIM_EXAMPLE_FILE)
    run = sorted(sims)[0]
    sim = sims[run]
    episodes = sorted(sim["episode_id"].unique())[:2]
    sim_example = sim[sim["episode_id"].isin(episodes)]
    sim_example.to_csv(example_dir / "sim_example.csv", index=False)

    print(f"human_example.csv: {human_example.shape} from {HUMAN_DATA_FILE}")
    print(f"sim_example.csv: {sim_example.shape} from run '{run}'")
