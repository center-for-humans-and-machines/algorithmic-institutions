"""Scrambled test copies of the data a single-idea agent may analyse.

A single-idea agent runs its one analysis script on the data exactly once
(.claude/agents/policy-finder-single-idea.md). So that it can debug the script
first, its export holds `test_data/`: for every data file under ROOTS a copy at
the same relative path, with the same columns and structure and no usable
content. The script takes the data root as its argument: `test_data` while
debugging, `.` for the one real run.

  - tables (`.csv`, `.parquet`): a few whole games (UNITS), the identifier
    columns as they are, every other column permuted over the rows, each on
    its own. A parquet file is read by its first row group only, so a sweep's
    millions of rows are never loaded.
  - `.json` (sweep.json, landscapes.json): the structure and every string,
    integer and boolean as they are; the floats under one key name permuted
    among themselves, and a list of more than two floats shuffled in place.

Other files (plots, configs, notes) get no copy. The copies are not a secret:
the agent may read the real files. They exist so that a debugging run returns
a number that means nothing.

Usage:
    python scripts/policy_finder/scramble_data.py <root> [--out test_data]
"""

import argparse
import json
import os
import random
import re

#: Folders, relative to the export, whose data files get a test copy
ROOTS = ("experiments", "plots/simulation")
OUT = "test_data"
#: Columns kept as they are: what a row is, not what happened in it
KEYS = {
    "session", "episode", "round_number", "participant_code", "experiment_name",
    "run", "player_id", "group_id",
}
KEY_PATTERN = re.compile(r"_id$")
#: Whole games kept: this many runs (a file without `run`: one), and this many
#: episodes of each
MAX_RUNS = 3
MAX_EPISODES = 4
#: A csv is read up to this many rows
CSV_ROWS = 200_000
SEED = 20261126


def is_key(column):
    return column in KEYS or bool(KEY_PATTERN.search(column))


def scramble_table(df, rng):
    """A few whole games of `df`, the value columns permuted over the rows."""
    episode = next((c for c in ("episode_id", "episode") if c in df.columns), None)
    if episode is not None:
        df = df.copy()
        df["_run"] = df["run"] if "run" in df.columns else ""
        keep = []
        runs = list(dict.fromkeys(df["_run"]))
        for run in rng.sample(runs, min(MAX_RUNS, len(runs))):
            rows = df[df["_run"] == run]
            episodes = list(dict.fromkeys(rows[episode]))
            # a parquet row group may cut its last episode short
            whole = episodes[:-1] if len(episodes) > 1 else episodes
            chosen = set(rng.sample(whole, min(MAX_EPISODES, len(whole))))
            keep.append(rows[rows[episode].isin(chosen)])
        import pandas as pd

        df = pd.concat(keep).drop(columns="_run")
    else:
        df = df.sample(n=min(len(df), 500), random_state=rng.randrange(2**31))
    df = df.reset_index(drop=True)
    for column in df.columns:
        if not is_key(column):
            order = list(range(len(df)))
            rng.shuffle(order)
            df[column] = df[column].to_numpy()[order]
    return df


def scramble_json(value, rng):
    """`value` with its floats permuted per key name; the structure stays."""
    slots = {}  # key name -> [(container, index or key)]

    def walk(node, name):
        items = node.items() if isinstance(node, dict) else enumerate(node)
        for key, child in items:
            child_name = key if isinstance(node, dict) else name
            if isinstance(child, (dict, list)):
                if (isinstance(child, list) and len(child) > 2
                        and all(type(x) is float for x in child)):
                    rng.shuffle(child)
                else:
                    walk(child, child_name)
            elif type(child) is float and isinstance(node, dict):
                slots.setdefault(child_name, []).append((node, key))

    if isinstance(value, (dict, list)):
        walk(value, "")
    for places in slots.values():
        values = [node[key] for node, key in places]
        rng.shuffle(values)
        for (node, key), new in zip(places, values):
            node[key] = new
    return value


def copy_file(source, target, rng):
    """Write the test copy of one data file; False if it is not a data file."""
    extension = os.path.splitext(source)[1].lower()
    if extension not in (".csv", ".parquet", ".json"):
        return False
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if extension == ".json":
        with open(source) as f:
            value = json.load(f)
        with open(target, "w") as f:
            json.dump(scramble_json(value, rng), f, indent=2)
            f.write("\n")
        return True
    import pandas as pd

    if extension == ".csv":
        df = pd.read_csv(source, nrows=CSV_ROWS)
        scramble_table(df, rng).to_csv(target, index=False)
    else:
        import pyarrow.parquet as pq

        table = pq.ParquetFile(source)
        first = table.read_row_group(0) if table.num_row_groups else table.read()
        scramble_table(first.to_pandas(), rng).to_parquet(target, index=False)
    return True


def build(root, out=OUT, roots=ROOTS):
    """Write `<root>/<out>/` from the data files under `roots`; return the copies."""
    written = []
    for base in roots:
        for folder, _, files in sorted(os.walk(os.path.join(root, base))):
            for name in sorted(files):
                source = os.path.join(folder, name)
                relative = os.path.relpath(source, root)
                # one seed per file: a copy does not depend on the other files
                rng = random.Random(f"{SEED}:{relative}")
                if copy_file(source, os.path.join(root, out, relative), rng):
                    written.append(relative)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    for path in build(args.root, args.out):
        print(path)


if __name__ == "__main__":
    main()
