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
    """Parse one wide-format CSV into long agent-round rows.

    Emits both label-to-group_id mappings for every episode (governorA→0
    + governorA→1) as distinct episodes sharing a pair_id. This breaks
    the alphabetical governorA→0 bias by data augmentation: any
    behaviour observed under a given governor appears under both
    group_id values, so models can't bind asymmetric behaviour to a
    specific group_id.
    """
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

    # Pre-pass: collect the union of group labels per competition, so the
    # flipped mapping is well-defined even on rounds where every agent
    # ended up in the same group (then a single-row sorted(set(...)) of
    # length 1 would give a no-op flip).
    pair_labels: dict[tuple[str, int], list[str]] = {}
    for _, row in df.iterrows():
        pair_key = (str(row["session"]), int(row["group_idx"]))
        pair_labels.setdefault(pair_key, set()).update(row["groups_list"])
    pair_labels = {k: sorted(v) for k, v in pair_labels.items()}

    # pair_id is dense over the original episodes (one per competition).
    pair_id_lookup: dict[tuple[str, int], int] = {}

    rows = []

    for _, row in df.iterrows():
        session = row["session"]
        round_raw = int(row["round"])  # 1-based in ah_data.csv

        contributions = row["contributions_list"]
        punishments = row["punishments_list"]
        missing_inputs = row["missing_inputs_list"]
        participant_codes = row["participant_codes_list"]
        groups_list = row["groups_list"]

        # One pair per competition (session + group_idx). Both
        # augmentations of a competition share its pair_id.
        competition_idx = int(row["group_idx"])
        pair_key = (str(session), competition_idx)
        if pair_key not in pair_id_lookup:
            pair_id_lookup[pair_key] = len(pair_id_lookup)
        pair_id = pair_id_lookup[pair_key]

        labels = pair_labels[pair_key]
        n_labels = len(labels)
        # mapping[0]: alphabetical (governorA→0); mapping[1]: flipped.
        mappings = [
            {label: i for i, label in enumerate(labels)},
            {label: n_labels - 1 - i for i, label in enumerate(labels)},
        ]

        round_number = round_raw - 1
        # Per-governor missing-input flags. Each governor manages one
        # group label; sorted labels map to the flag columns in order
        # (governorA -> missing_governor_input, governorB ->
        # missing_governor2_input). Verified against the raw data: when
        # a governor's flag is set, his group's punishments are all 0
        # while the other group's punishments are real.
        governor_flag_cols = [
            "missing_governor_input",
            "missing_governor2_input",
        ]
        label_manager_no_input = {
            label: int(bool(row.get(col, False)))
            for label, col in zip(labels, governor_flag_cols)
        }

        for aug_idx, group_label_to_idx in enumerate(mappings):
            aug_suffix = "" if aug_idx == 0 else " (flipped)"
            aug_global_group_id = (
                f"{session} #{competition_idx}{aug_suffix}"
            )
            # Distinct episode_id per augmentation; the global_group_id
            # split alone is enough for downstream tensor-row uniqueness
            # but giving each a unique episode_id keeps groupby keys
            # clean.
            aug_episode_id = pair_id * 2 + aug_idx

            # Per-group common good (depends on the mapping).
            group_contrib = {}
            group_punish = {}
            for pid in range(n_agents):
                gidx = group_label_to_idx[groups_list[pid]]
                is_valid = not bool(missing_inputs[pid])
                if is_valid:
                    group_contrib.setdefault(gidx, 0.0)
                    group_punish.setdefault(gidx, 0.0)
                    group_contrib[gidx] += contributions[pid]
                    group_punish[gidx] += punishments[pid]
            common_good_per_group = {}
            for gidx in set(group_label_to_idx.values()):
                sc = group_contrib.get(gidx, 0.0)
                sp = group_punish.get(gidx, 0.0)
                common_good_per_group[gidx] = sc * 1.6 - sp

            for player_id in range(n_agents):
                gidx = group_label_to_idx[groups_list[player_id]]
                rows.append(
                    {
                        "session": session,
                        "global_group_id": aug_global_group_id,
                        "group_id": gidx,
                        "episode": competition_idx,
                        "episode_id": aug_episode_id,
                        "pair_id": pair_id,
                        "experiment_name": "ah_group_switching",
                        "round_number": round_number,
                        "participant_code": participant_codes[player_id],
                        "player_no_input": int(
                            bool(missing_inputs[player_id])
                        ),
                        "manager_no_input": label_manager_no_input.get(
                            groups_list[player_id], 0
                        ),
                        "player_id": player_id,
                        "contribution": float(contributions[player_id]),
                        "punishment": float(punishments[player_id]),
                        "payoff": 0.0,
                        "common_good": common_good_per_group[gidx],
                    }
                )

    return rows


COLS = [
    "session",
    "global_group_id",
    "group_id",
    "episode",
    "episode_id",
    "pair_id",
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


def main(
    in_paths: list[str],
    n_agents: int,
    out_path: str = None,
):
    """Preprocess one or more wide-format CSVs into a single long CSV."""
    repo_root = Path(__file__).resolve().parents[2]

    all_rows = []

    for in_path in in_paths:
        rows = _preprocess_single(in_path, n_agents)
        # Re-key episode_id and pair_id to avoid collisions across files.
        if all_rows:
            max_eid = max(r["episode_id"] for r in all_rows) + 1
            max_pid = max(r["pair_id"] for r in all_rows) + 1
            for r in rows:
                r["episode_id"] += max_eid
                r["pair_id"] += max_pid
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
