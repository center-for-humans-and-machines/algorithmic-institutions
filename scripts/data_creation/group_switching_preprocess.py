from pathlib import Path
import ast
import pandas as pd
from argparse import ArgumentParser


def _parse_list(value):
    """Safely parse a stringified Python list from the CSV."""
    if pd.isna(value):
        return []
    # ah_data.csv stores things like "[1, 2, 3, 4]"
    return ast.literal_eval(value)


def main(in_path: str, n_agents: int) -> None:
    """
    Create a player-round CSV from ah_data.csv, keeping only rounds
    with 8 active agents, and shaping it like the pilot
    `*_player_round_slim` data (plus a group_id column).
    """
    repo_root = Path(__file__).resolve().parents[2]

    folder_name = Path(in_path).parent.name
    file_name = Path(in_path).name.replace(".csv", "")
    out_path = repo_root / "experiments" / f"{folder_name}_{file_name}_{n_agents}_agents.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # Parse list-valued columns from their string representations.
    df["contributions_list"] = df["contributions"].apply(_parse_list)
    df["punishments_list"] = df["punishments"].apply(_parse_list)
    df["missing_inputs_list"] = df["missing_inputs"].apply(_parse_list)
    df["participant_codes_list"] = df["participant_codes"].apply(_parse_list)
    df["groups_list"] = df["groups"].apply(_parse_list)

    # ---- Filter to rounds with exactly 8 active agents ----
    has_8_agents = df["contributions_list"].apply(len) == n_agents
    all_active = df["missing_inputs_list"].apply(
        lambda xs: len(xs) == n_agents and all(not x for x in xs)
    )
    df = df[has_8_agents & all_active].copy()

    # We will assign a dense episode_id over (session, group_id, episode).
    episode_id_lookup: dict[tuple[str, int, int], int] = {}

    rows = []

    for _, row in df.iterrows():
        session = row["session"]
        round_raw = int(row["round"])  # 1-based in ah_data.csv

        contributions = row["contributions_list"]
        punishments = row["punishments_list"]
        missing_inputs = row["missing_inputs_list"]
        participant_codes = row["participant_codes_list"]
        groups_list = row["groups_list"]

        # Map each distinct group label in this round to an integer 0,1,...
        # Assumption: groups_list encodes which of the (up to) 2 groups of 4
        # each participant belongs to in this session/round.
        unique_group_labels = sorted(set(groups_list))
        group_label_to_idx = {label: i for i, label in enumerate(unique_group_labels)}

        # Treat each (session, group_id) as a single episode, like a full game.
        # We therefore create a separate episode_id per (session, per-group).
        episode = 1
        for group_label, group_id in group_label_to_idx.items():
            episode_key = (str(session), group_id, episode)
            if episode_key not in episode_id_lookup:
                episode_id_lookup[episode_key] = len(episode_id_lookup)
            episode_id = episode_id_lookup[episode_key]

            # 0-based round_number within episode.
            round_number = round_raw - 1

            # "session #<group_id>" identifier; this will later be used
            # together with episode_id in parse_agent_rounds to create
            # a dense group_idx (the tensor batch dimension).
            global_group_id = f"{session} #{group_id}"

            # Simple, deterministic definition of common_good; adjust if needed.
            common_good = float(sum(contributions))

            # Manager no-input is at the round level; player_no_input is per agent.
            manager_no_input = int(bool(row.get("missing_governor_input", False)))

            for player_id in range(n_agents):
                # Only include players belonging to this group_label
                if group_label_to_idx[groups_list[player_id]] != group_id:
                    continue

                rows.append(
                    {
                        "session": session,
                        "global_group_id": global_group_id,
                        "group_id": group_id,  # sub-group membership 0,1,...
                        "episode": episode,
                        "episode_id": episode_id,
                        "experiment_name": "ah_group_switching",
                        "round_number": round_number,
                        "participant_code": participant_codes[player_id],
                        "player_no_input": int(bool(missing_inputs[player_id])),
                        "manager_no_input": manager_no_input,
                        "player_id": player_id,
                        "contribution": float(contributions[player_id]),
                        "punishment": float(punishments[player_id]),
                        "payoff": 0.0,  # placeholder; not used in the GNN pipeline
                        "common_good": common_good,
                    }
                )

    out_df = pd.DataFrame(rows)

    # Match the column order of dummy_group_switching + pilot_random1_slim.
    cols = [
        "session",
        "global_group_id",
        "group_id",
        "episode",
        "episode_id",
        "experiment_name",
        "round_number",
        "participant_code",
        "player_no_input",
        "manager_no_input",
        "player_id",
        "contribution",
        "punishment",
        "payoff",
        "common_good",
    ]
    out_df = out_df[cols]

    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--in_path", type=str, required=True)
    parser.add_argument(
        "--n_agents",
        help="Number of agents active in each round to filter for",
        type=int, required=True)
    args = parser.parse_args()
    main(args.in_path, args.n_agents)
