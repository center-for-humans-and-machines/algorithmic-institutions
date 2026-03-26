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


def _preprocess_single(in_path: str, n_agents: int):
    """Parse one wide-format CSV into long agent-round rows."""
    df = pd.read_csv(in_path)

    # Parse list-valued columns from their string representations.
    df["contributions_list"] = df["contributions"].apply(_parse_list)
    df["punishments_list"] = df["punishments"].apply(_parse_list)
    df["missing_inputs_list"] = df["missing_inputs"].apply(_parse_list)
    df["participant_codes_list"] = df["participant_codes"].apply(_parse_list)
    df["groups_list"] = df["groups"].apply(_parse_list)

    # Keep only rounds with the expected number of agents.
    has_n_agents = df["contributions_list"].apply(len) == n_agents
    df = df[has_n_agents].copy()

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

        # Map each distinct group label to a stable integer.
        unique_group_labels = sorted(set(groups_list))
        group_label_to_idx = {
            label: i for i, label in enumerate(unique_group_labels)
        }

        # One episode per competition (session + group_idx).
        competition_idx = int(row["group_idx"])
        episode_key = (str(session), competition_idx)
        if episode_key not in episode_id_lookup:
            episode_id_lookup[episode_key] = len(episode_id_lookup)
        episode_id = episode_id_lookup[episode_key]

        round_number = round_raw - 1
        global_group_id = f"{session} #{competition_idx}"
        common_good = float(sum(contributions))
        manager_no_input = int(
            bool(row.get("missing_governor_input", False))
        )

        for player_id in range(n_agents):
            rows.append(
                {
                    "session": session,
                    "global_group_id": global_group_id,
                    "group_id": group_label_to_idx[
                        groups_list[player_id]
                    ],
                    "episode": competition_idx,
                    "episode_id": episode_id,
                    "experiment_name": "ah_group_switching",
                    "round_number": round_number,
                    "participant_code": participant_codes[player_id],
                    "player_no_input": int(
                        bool(missing_inputs[player_id])
                    ),
                    "manager_no_input": manager_no_input,
                    "player_id": player_id,
                    "contribution": float(contributions[player_id]),
                    "punishment": float(punishments[player_id]),
                    "payoff": 0.0,
                    "common_good": common_good,
                }
            )

    return rows


COLS = [
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


def main(in_paths: list[str], n_agents: int, out_path: str = None):
    """Preprocess one or more wide-format CSVs into a single long CSV."""
    repo_root = Path(__file__).resolve().parents[2]

    all_rows = []

    for in_path in in_paths:
        rows = _preprocess_single(in_path, n_agents)
        # Re-key episode_ids to avoid collisions across files
        if all_rows:
            max_id = max(r["episode_id"] for r in all_rows) + 1
            for r in rows:
                r["episode_id"] += max_id
        all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)[COLS]

    if out_path is None:
        if len(in_paths) == 1:
            p = Path(in_paths[0])
            name = f"{p.parent.name}_{p.stem}_{n_agents}_agents.csv"
        else:
            name = f"group_switching_combined_{n_agents}_agents.csv"
        out_path = repo_root / "experiments" / name

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "in_paths", nargs="+", help="One or more wide-format CSVs"
    )
    parser.add_argument(
        "--n_agents",
        help="Number of agents active in each round to filter for",
        type=int, required=True,
    )
    parser.add_argument("--out_path", type=str, default=None)
    args = parser.parse_args()
    main(args.in_paths, args.n_agents, args.out_path)
