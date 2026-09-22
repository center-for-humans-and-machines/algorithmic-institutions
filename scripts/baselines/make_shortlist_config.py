"""Build a shortlist config from a coarser grid's top-N rows (issue #119).

The full feature grid is huge, so it is first swept cheaply -- with ridge/MSE
(ridge.yml), or with the linear gaussian for a nonlinear re-rank. This script
reads that CV-output CSV, keeps the per-block set-variants that appear among the
top-N unique feature-sets, and writes a new config over just those blocks -- a
tight shortlist to re-rank with a richer model (--model, default `gaussian`).

The enumerator always adds an implicit OFF per block, so the shortlist covers
those top-N feature-sets plus their lower-order OFF combinations (a superset,
not exactly N). Set-index -> feature mapping is taken from --source-config and
cross-checked against the CSV's own `features` column (fails fast on a mismatch).

The emitted `setting:` block carries exactly the target model's knobs (see
baseline_models._SPEC), defaulted here and overridden with `--set KEY=V[,V...]`;
a comma-separated value becomes a grid axis.

Usage:
    .venv/bin/python scripts/baselines/make_shortlist_config.py \
        data/baselines/large_set_ridge_cv.csv --top-n 100
    # -> configs/training/baselines/contribution/large_set_ridge_cv_top_100_features.yml

    .venv/bin/python scripts/baselines/make_shortlist_config.py \
        data/baselines/gaussian_top500_feats.csv --top-n 3 \
        --source-config configs/training/baselines/contribution/gaussian.yml \
        --model gaussian_mlp --set hidden=16,32,64 --set lr=0.01,0.05 \
        --set epochs=500 --set weight_decay=0.0,0.0001 \
        --out configs/training/baselines/contribution/gaussian_mlp.yml
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_models import setting_keys  # noqa: E402

DEFAULT_SOURCE = ROOT / "configs/training/baselines/contribution/ridge.yml"
CONFIG_DIR = ROOT / "configs/training/baselines/contribution"

# per-model defaults baked into the generated config (all griddable -- override
# with --set, or edit the emitted yml freely)
SETTING_DEFAULTS = {
    "multinomial": {"C": 1.0},
    "ridge": {"alpha": 1.0},
    "gaussian": {"weight_decay": [0.0, 0.001], "lr": 0.05, "epochs": 3000},
    "gaussian_mlp": {"hidden": 32, "weight_decay": 0.0, "lr": 0.05, "epochs": 500},
}


def _fmt(x):
    """Compact inline YAML for a scalar / list / list-of-lists."""
    if isinstance(x, list):
        return "[" + ", ".join(_fmt(v) for v in x) + "]"
    return str(x)


def used_sets(df, top_n):
    """{block: sorted set-indices} appearing among the top-N unique feature-sets.

    Dedup BEFORE the rank cut: setting sweeps put near-tied duplicates of the
    same feature-set in the top ranks, so filtering `rank <= top_n` first would
    silently shrink the candidate pool (~half of top_n)."""
    uniq = df.sort_values("rank").drop_duplicates("config")
    top = uniq[uniq["config"] != "floor"].head(top_n)
    used = defaultdict(set)
    for cfg_label in top["config"]:
        for part in cfg_label.split("+"):
            blk, s = part.split(":")
            used[blk].add(int(s[1:]))  # 'sN' -> N
    return {b: sorted(v) for b, v in used.items()}, top


def check_mapping(top, blocks):
    """Assert set-index -> features (from source config) matches the CSV features."""
    for _, row in top.iterrows():
        if row["config"] == "floor":
            continue
        mapped = []
        for part in row["config"].split("+"):
            blk, s = part.split(":")
            for f in blocks[blk]["sets"][int(s[1:])]:
                if f not in mapped:
                    mapped.append(f)
        actual = str(row["features"]).split(";")
        if set(mapped) != set(actual):
            raise ValueError(
                f"source-config blocks do not match the CSV for '{row['config']}':"
                f"\n  mapped={mapped}\n  actual={actual}\n"
                "Pass the --source-config that produced this CSV."
            )


def parse_setting(model, overrides):
    """The `setting:` block for `model`: this script's defaults, overridden by
    the `--set KEY=V[,V...]` items. Values are passed through verbatim as YAML
    scalars (the CV driver casts them); a comma makes the key a grid axis,
    e.g. `--set hidden=16,32,64` -> `hidden: [16, 32, 64]`."""
    setting = dict(SETTING_DEFAULTS[model])
    if sorted(setting) != sorted(setting_keys(model)):  # guard against drift
        raise ValueError(
            f"SETTING_DEFAULTS[{model!r}]={sorted(setting)} does not match "
            f"baseline_models' keys {sorted(setting_keys(model))}"
        )
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE[,VALUE...], got {item!r}")
        key, raw = item.split("=", 1)
        if key not in setting:
            raise ValueError(
                f"setting key {key!r} is not valid for model {model!r}; "
                f"allowed: {sorted(setting)}"
            )
        vals = [v.strip() for v in raw.split(",") if v.strip()]
        if not vals:
            raise ValueError(f"--set {item!r} has no value")
        setting[key] = vals[0] if len(vals) == 1 else vals
    return setting


def render(src, used, csv_name, out_stem, setting, model):
    """Render the shortlist config text for `model`."""
    d = src["data"]
    out_csv = f"data/baselines/{out_stem}_cv.csv"
    lines = [
        f"# Auto-generated by make_shortlist_config.py from {csv_name}.",
        f"# {model} shortlist: each block keeps only the set-variants seen in the",
        "# top-N feature-sets of the source grid (+ implicit OFF per block).",
        "",
        "data:",
    ]
    for k in (
        "data_file",
        "experiment_names",
        "target",
        "target_type",
        "categorical_levels",
        "mask",
        "exclude_flipped",
        "switch_every",
    ):
        if k in d:
            lines.append(f"  {k}: {_fmt(d[k])}")
    lines.append(f"  model: {model}")
    lines += [
        "",
        "cv:",
        f"  seed: {src['cv']['seed']}",
        f"  n_folds: {src['cv']['n_folds']}",
        f"  output: {out_csv}",
        "  show_ce: true          # binned 21-way cross-entropy alongside NLL",
        "",
        f"setting:               # {model} knobs; every key griddable (scalar|list)",
    ]
    for k, v in setting.items():
        lines.append(f"  {k}: {_fmt(v)}")
    lines += ["", "blocks:"]
    for blk, spec in src["blocks"].items():
        idxs = used.get(blk, [])
        if not idxs:  # block never used in the top-N -> drop it
            continue
        sets = [spec["sets"][i] for i in idxs]
        lines.append(
            f"  {blk}:                    # top-N sets: {['s%d' % i for i in idxs]}"
        )
        lines.append(f"    components: {_fmt(spec['components'])}")
        lines.append(f"    sets: {_fmt(sets)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "csv", help="ridge CV-output CSV (has `config` + `features` columns)"
    )
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument(
        "--source-config",
        default=str(DEFAULT_SOURCE),
        help="config that produced the CSV (for the block/set definitions)",
    )
    ap.add_argument("--out", default=None, help="output config path")
    ap.add_argument(
        "--model",
        default="gaussian",
        choices=sorted(SETTING_DEFAULTS),
        help="model the shortlist config is written for (default: gaussian)",
    )
    ap.add_argument(
        "--set",
        dest="settings",
        action="append",
        default=[],
        metavar="KEY=V[,V...]",
        help="setting override, repeatable: --set hidden=16,32,64 --set epochs=500",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    src = yaml.safe_load(open(args.source_config))

    used, top = used_sets(df, args.top_n)
    check_mapping(top, src["blocks"])
    setting = parse_setting(args.model, args.settings)

    out = (
        Path(args.out).resolve()  # resolved: a relative --out is cwd-relative
        if args.out
        else (CONFIG_DIR / f"{csv_path.stem}_top_{args.top_n}_features.yml")
    )
    out.write_text(render(src, used, csv_path.name, out.stem, setting, args.model))

    n_blocks = sum(1 for b in src["blocks"] if used.get(b))
    grid = 1
    for b in src["blocks"]:
        if used.get(b):
            grid *= len(used[b]) + 1  # +1 implicit OFF
    print(f"top-{args.top_n}: {len(top)} unique feature-sets over {n_blocks} blocks")
    print(f"per-block sets kept: {used}")
    print(f"-> {grid} enumerated feature-set combos (superset of the {len(top)})")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
