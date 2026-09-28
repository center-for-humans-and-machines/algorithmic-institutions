"""Held-out teacher-forced RCB test: is the punishment response learned, or
memorised?

PR #181 measured the PR #179 group-vnode trunk teacher-forced on the human
trajectories and found an RCB statistic of 0.093 (inside the 0.348 noise
ceiling) against 0.797 in self-play, and read the gap as state drift. But the
shipped artifact is trained on all 50 games, so that 0.093 is an IN-SAMPLE
number. This script asks the discriminating question: does the teacher-forced
response survive on games the model never saw?

Inputs: five fold artifacts trained by
`configs/training/artificial_humans/contribution/
group_switching_contribution_50ep_group_vnode_holdout_folds.yml` (the parent
config plus `holdout_fold: k`, same seed), one per CV fold, plus the shipped
full-data artifact. For every fold model the script rebuilds the fold's
held-out episodes through `get_cross_validations` itself (same seed, same
code path as train.py) and VERIFIES the partition against the fold job's own
recorded held-out log loss before using it.

Quantities, all on the canonical single-copy human frame (one copy per game,
the same 50 episodes `rcb_teacher_forced.py` uses), model column =
teacher-forced E[c_{t+1}] - c_t over the RCB population:

  * the RCB statistic (human-frequency-weighted mean |bin mean - human bin
    mean| over the 4 punishment-rate bins) and its score (/ noise ceiling);
  * the 4 RCB bin means;
  * the within-contribution-band OLS slopes of dc on punishment received.

Reported for (a) each fold's held-out games, (b) each fold's in-sample games,
(c) the pooled held-out predictions over all 50 games (every game predicted
by the model that did not see it), (c') the pooled in-sample predictions
(every game predicted by the four fold models that did see it), and (d) the
shipped full-data artifact in-sample (the 0.093 reference). The observed
human numbers on exactly the same rows are printed next to every model row,
because on 10 games the human slopes are themselves noisy.

Measurement only; imports graph.py, so it needs torch_geometric:
    python scripts/data_analysis/rcb_holdout_teacher_forced.py \
        [--fold-dir DIR] [--full-model PT] [--out CSV]
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch as th  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for sub in (
    "src",
    "scripts/artificial_humans",
    "scripts/baselines",
    "scripts/data_analysis",
):
    sys.path.insert(0, str(ROOT / sub))

import contribution_copula_rho as cc  # noqa: E402
import rcb_teacher_forced as tf  # noqa: E402

from aimanager.generic.data import get_cross_validations  # noqa: E402
from aimanager.generic.graph import GraphNetwork  # noqa: E402

# the human-vs-human RCB noise ceiling (scoring.py: 500 repeats, master seed
# 42) -- the denominator that turns the statistic into a band-scale score
CEILING = 0.3479
DEFAULT_FOLD_DIR = (
    "artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode_holdout"
)
N_FOLDS = 5


def fold_test_indices(n_ep, pair_id, k, seed, n_folds):
    """Episode rows of fold k, through `get_cross_validations` itself.

    train.py seeds `random` with the config seed, builds the tensors (which
    draw nothing from `random`), then `get_cross_validations` shuffles the
    episode index ONCE and splits the pairs round-robin before it looks at
    `holdout_fold` -- so re-seeding right before the call reproduces the
    partition the training job used, and fold k here is also fold k of the
    shipped artifact's own 5-fold CV (same seed). Verified downstream against
    the fold job's recorded held-out log loss, not assumed.
    """
    random.seed(seed)
    fake = {"contribution": th.arange(n_ep)}
    _, _, test = next(
        get_cross_validations(fake, n_folds, 1.0, holdout_fold=k, group_key=pair_id)
    )
    return np.sort(test["contribution"].numpy())


def recorded_test_log_loss(fold_dir, job_id):
    """The last recorded held-out log loss of the fold job (the Recorder's
    metrics parquet; set=test, full mask, no feature perturbation)."""
    m = pd.read_parquet(fold_dir / "metrics" / f"{job_id}.parquet")
    s = m[(m["name"] == "log_loss") & (m["set"] == "test") & (m["mask"] == 0)]
    for c in (
        "shuffle_feature",
        "ablate_feature",
        "leave_one_in_shuffle_feature",
        "leave_one_in_ablate_feature",
    ):
        if c in s.columns:
            s = s[s[c].isna()]
    return float(s.sort_values("epoch")["value"].iloc[-1])


def teacher_forced_nll(model, data, idx):
    """Mean -log P[observed level] over the valid rows of `idx`, i.e. the
    quantity eval_model records as log_loss (sklearn clips at ~1e-15)."""
    rows = cc.teacher_forced_rows(model, data, idx)
    p = rows["P"][np.arange(len(rows["y"])), rows["y"]]
    return float(-np.log(np.clip(p, 1e-15, 1.0)).mean())


def summary(pop, col):
    slopes = tf.band_slopes(pop, col)["slope"].to_numpy()
    tab, stat = tf.bin_means(pop, col)
    out = {"n": int(len(pop)), "rcb_stat": stat, "score": stat / CEILING}
    out.update({f"bin_{b}": v for b, v in zip(tf.RATE_LABELS, tab["mean_dc"])})
    out.update({f"slope_{b}": v for b, v in zip(tf.BAND_LABELS, slopes)})
    return out


def rows_for(df, set_name, fold):
    """Model and observed-human summary rows over the same stimulus rows."""
    pop = tf.rcb_population(df)
    return [
        {"set": set_name, "fold": fold, "source": "model", **summary(pop, "dc_model")},
        {"set": set_name, "fold": fold, "source": "human", **summary(pop, "dc_human")},
    ]


def fmt(v):
    return f"{v: .4f}" if isinstance(v, float) else str(v)


def show(rows):
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, formatters={c: fmt for c in df.columns}))


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold-dir", default=str(ROOT / DEFAULT_FOLD_DIR))
    ap.add_argument("--full-model", default=str(ROOT / tf.DEFAULT_MODEL))
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=cc.SEED)
    ap.add_argument("--out", default=None, help="CSV of every summary row")
    args = ap.parse_args()
    fold_dir = Path(args.fold_dir).resolve()

    data, pair_id, key_to_idx, _ = cc.load_full()
    n_ep = data["contribution"].shape[0]
    tr = cc.select_split(key_to_idx, cc.TRAIN, cc.N_TRAIN_EP)
    te = cc.select_split(key_to_idx, cc.TEST, cc.N_TEST_EP)
    canon = np.array(sorted(set(tr.tolist()) | set(te.tolist())))
    print(f"data      {cc.rel(cc.FULL)}: {n_ep} ep, canonical single copy {len(canon)}")

    # (d) the shipped full-data artifact, in-sample on the 50 games
    full = GraphNetwork.load(args.full_model, device="cpu")
    full.eval()
    assert full.y_name == "contribution"
    print(f"full model {cc.rel(args.full_model)}")
    label = "shipped full-data trunk, in-sample (50 games)"
    df_full, dense_full = tf.stimulus_frame(full, data, canon, label)
    tf.check_alignment(full, data, df_full, dense_full, label)
    rows = rows_for(df_full, "full_in_sample", "all")
    pop_full = tf.rcb_population(df_full)
    assert tf.selfcheck(pop_full, label), "canonical population does not match"

    held_frames, in_frames, covered = [], [], []
    for k in range(args.n_folds):
        pts = sorted(fold_dir.glob(f"model/*fold_{k}.pt"))
        assert len(pts) == 1, f"fold {k}: expected one artifact, found {pts}"
        job_id = pts[0].stem
        model = GraphNetwork.load(str(pts[0]), device="cpu")
        model.eval()
        assert model.y_name == "contribution"

        test_idx = fold_test_indices(n_ep, pair_id, k, args.seed, args.n_folds)
        pairs = set(pair_id[test_idx].tolist())
        assert len(test_idx) == 2 * len(pairs), "a held-out game misses its flip copy"
        assert set(np.nonzero(np.isin(pair_id, list(pairs)))[0]) == set(test_idx)

        nll = teacher_forced_nll(model, data, test_idx)
        rec = recorded_test_log_loss(fold_dir, job_id)
        print(
            f"\nfold {k}: {job_id}\n"
            f"  held-out episodes {len(test_idx)} ({len(pairs)} games, both copies)\n"
            f"  teacher-forced held-out NLL {nll:.6f} vs recorded test log_loss "
            f"{rec:.6f} (diff {nll - rec:+.2e})"
        )
        assert abs(nll - rec) < 2e-3, (
            f"fold {k}: reconstructed partition does not reproduce the job's "
            f"recorded held-out log loss -- wrong seed or fold assignment"
        )

        held = np.array(sorted(set(test_idx.tolist()) & set(canon.tolist())))
        ins = np.array(sorted(set(canon.tolist()) - set(test_idx.tolist())))
        assert len(held) == len(pairs) and len(held) + len(ins) == len(canon)
        covered += held.tolist()
        df_h, _ = tf.stimulus_frame(model, data, held, f"  fold {k} held-out")
        df_i, _ = tf.stimulus_frame(model, data, ins, f"  fold {k} in-sample")
        held_frames.append(df_h)
        in_frames.append(df_i)
        rows += rows_for(df_h, "held_out", k) + rows_for(df_i, "in_sample", k)

    assert (
        sorted(covered) == canon.tolist()
    ), "folds do not cover each game exactly once"

    # (c) pooled held-out: every game predicted by the model that never saw it
    pooled = pd.concat(held_frames, ignore_index=True)
    pop = tf.rcb_population(pooled)
    assert tf.selfcheck(pop, "pooled held-out (50 games)")
    rows += rows_for(pooled, "pooled_held_out", "all")
    # (c') pooled in-sample: every game predicted by the 4 models that saw it
    rows += rows_for(pd.concat(in_frames, ignore_index=True), "pooled_in_sample", "all")

    print("\n================ summary (model rows vs observed human on the same rows)")
    print(f"RCB score = statistic / {CEILING} (human-vs-human noise ceiling)")
    print(f"human reference slopes {tf.HUMAN_SLOPES}, bin means {tf.HUMAN_BIN_MEANS}")
    show(rows)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nwrote {cc.rel(out)}")
    print(f"\nwall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
