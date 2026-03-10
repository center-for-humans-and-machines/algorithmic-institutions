import random
import warnings
from argparse import ArgumentParser

import pandas as pd


def main(in_path: str, out_path: str, seed: int = 42) -> None:
    """
    Transform an old 4-player pilot CSV into the new 8-player format
    by randomly pairing two 4-player episodes that have the same
    number of rounds.
    """
    df = pd.read_csv(in_path)

    # Identify distinct episodes and count their rounds.
    episodes = (
        df.groupby(["global_group_id", "episode_id"])
        .agg(n_rounds=("round_number", "nunique"))
        .reset_index()
    )

    # Bucket episodes by round count, shuffle, and pair.
    rng = random.Random(seed)
    pairs = []

    for n_rounds, bucket in episodes.groupby("n_rounds"):
        keys = list(zip(bucket["global_group_id"], bucket["episode_id"]))
        rng.shuffle(keys)
        if len(keys) % 2 != 0:
            dropped = keys.pop()
            warnings.warn(
                f"Dropping unpaired episode {dropped} " f"({n_rounds} rounds)"
            )
        for i in range(0, len(keys), 2):
            pairs.append((keys[i], keys[i + 1]))

    # Build the 8-player rows for each pair.
    rows = []
    pair_episode_id = 0

    for ep_a, ep_b in pairs:
        df_a = df[
            (df["global_group_id"] == ep_a[0]) & (df["episode_id"] == ep_a[1])
        ].copy()
        df_b = df[
            (df["global_group_id"] == ep_b[0]) & (df["episode_id"] == ep_b[1])
        ].copy()

        global_gid = f"pair_{pair_episode_id}"

        # Sum contributions per round for combined common_good.
        cg_a = df_a.groupby("round_number")["contribution"].sum().to_dict()
        cg_b = df_b.groupby("round_number")["contribution"].sum().to_dict()

        for round_num in sorted(cg_a.keys()):
            combined_cg = cg_a.get(round_num, 0.0) + cg_b.get(round_num, 0.0)

            # Group A: group_id=0, player_id unchanged (0-3)
            for _, r in df_a[df_a["round_number"] == round_num].iterrows():
                rows.append(
                    _make_row(
                        r,
                        global_gid,
                        group_id=0,
                        player_id=int(r["player_id"]),
                        episode_id=pair_episode_id,
                        common_good=combined_cg,
                    )
                )

            # Group B: group_id=1, player_id remapped to 4-7
            for _, r in df_b[df_b["round_number"] == round_num].iterrows():
                rows.append(
                    _make_row(
                        r,
                        global_gid,
                        group_id=1,
                        player_id=int(r["player_id"]) + 4,
                        episode_id=pair_episode_id,
                        common_good=combined_cg,
                    )
                )

        pair_episode_id += 1

    out_df = pd.DataFrame(rows)
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
    print(f"Wrote {len(out_df)} rows ({pair_episode_id} pairs) to {out_path}")


def _make_row(r, global_group_id, group_id, player_id, episode_id, common_good):
    return {
        "session": r["session"],
        "global_group_id": global_group_id,
        "group_id": group_id,
        "episode": int(r["episode"]),
        "episode_id": episode_id,
        "experiment_name": r["experiment_name"],
        "round_number": int(r["round_number"]),
        "participant_code": r["participant_code"],
        "player_no_input": int(r["player_no_input"]),
        "manager_no_input": (
            int(r["manager_no_input"]) if not pd.isna(r["manager_no_input"]) else 0
        ),
        "player_id": player_id,
        "contribution": float(r["contribution"]),
        "punishment": float(r["punishment"]),
        "payoff": float(r["payoff"]),
        "common_good": common_good,
    }


def combine(transformed_path: str, new_data_path: str, out_path: str) -> None:
    """Concatenate the pseudo-matched old data with new group-switching data."""
    old = pd.read_csv(transformed_path)
    new = pd.read_csv(new_data_path)
    combined = pd.concat([old, new], ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"Combined {len(old)} + {len(new)} = {len(combined)} rows " f"to {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    # Transform subcommand
    transform_p = sub.add_parser("transform")
    transform_p.add_argument("--in_path", type=str, required=True)
    transform_p.add_argument("--out_path", type=str, required=True)
    transform_p.add_argument("--seed", type=int, default=42)

    # Combine subcommand
    combine_p = sub.add_parser("combine")
    combine_p.add_argument("--transformed_path", type=str, required=True)
    combine_p.add_argument("--new_data_path", type=str, required=True)
    combine_p.add_argument("--out_path", type=str, required=True)

    args = parser.parse_args()
    if args.command == "transform":
        main(args.in_path, args.out_path, args.seed)
    elif args.command == "combine":
        combine(args.transformed_path, args.new_data_path, args.out_path)
    else:
        parser.print_help()
