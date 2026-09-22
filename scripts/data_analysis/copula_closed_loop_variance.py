"""Where does the contribution copula's variance go, round by round?

Ported from `auto/copula-closed-loop-variance` (PR #188) unchanged except for
the arms it reads: same measurements, same conventions, same teacher-forced
trunk, so `var_cond_mean` here is the same quantity that branch and PR #186
report against the human `Var(E[c | history])` of 27.9. Because a change to
the contribution model's inputs must be judged with the shared-noise
machinery off as well as on, the arms are a 2x2 of (serving fix off / on) x
(copula off / on) around the frontier stack:

  A  parent, copula on   -- 23_..._gnn_switch_timeout (PR #194's frontier)
  B  candidate, copula on -- 23_..._gnn_switch_simtimeout
  C  candidate, copula OFF -- 23_2g8a_sim_timeout_rho0 (bare trunk)
  D  parent, copula OFF   -- 23_2g8a_sim_timeout_rho0_base (bare trunk, the
     parent's two source files restored in an isolated remote copy)

C is the noise-off run the protocol asks for; D is its before. Nothing is
retrained anywhere in this 2x2 -- the four arms run the same artifacts.

Per round t, for each arm and the humans: (i) SD of the group-mean
contribution over (game, group); (ii) SD of individual contributions; (iii)
their ratio (the CG ingredient); (iv) the within-group residual correlation
given the state: c - E[c | that arm's own realised history] from a
teacher-forced pass of the bare trunk (as rcb_teacher_forced.py --sim-parquet
does), then the pairwise Pearson correlation of those residuals between
members of one (game, round, group); (v) the trunk's predictive SD and
entropy at the visited states. Then a decomposition of the group-mean
variance into a between-(episode, group) persistent part and a
round-to-round part. (The logged-latent regression needs a latent-logged
rerun; none exists here, so it is skipped.)

Two stages, because the trunk unpickles torch_geometric modules:

  Raven:  .venv/bin/python scripts/data_analysis/copula_closed_loop_variance.py \\
              --teacher-force              # writes tf_<arm>.parquet
  local:  python scripts/data_analysis/copula_closed_loop_variance.py \\
              --analyse                    # tables + figures

Outputs under plots/data_analysis/evaluation/copula_closed_loop/.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "artificial_humans"))
sys.path.insert(0, str(ROOT / "scripts" / "data_analysis"))

OUT = ROOT / "plots/data_analysis/evaluation/sim_timeout_imputation/copula_closed_loop"
SIM = ROOT / "plots/simulation"
TRUNK = ROOT / (
    "artifacts/artificial_humans/group_switching_contribution_50ep_vnode_stimulus_skip"
    "/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
)
SKIP = "23_2g8a_contr_stimulus_skip_self_gnncopar1_contr_gnn_switch"
ARMS = {
    "A": SKIP + "_timeout",  # parent, copula on
    "B": SKIP + "_simtimeout",  # candidate, copula on
    "C": "23_2g8a_sim_timeout_rho0",  # candidate, copula off
    "D": "23_2g8a_sim_timeout_rho0_base",  # parent, copula off
}
ZLOG = "23_2g8a_copula_cl_variance_a_zlog"  # arm A rerun with the latent logged
LABELS = {"human": "human", "A": "A parent, copula on",
          "B": "B fix, copula on", "C": "C fix, copula OFF",
          "D": "D parent, copula OFF"}  # fmt: skip
BLOCKS = [("1-8", 0, 8), ("9-16", 8, 16), ("17-24", 16, 24)]
CELL = ["episode", "round", "group"]


# --------------------------------------------------------------------------- #
# stage 1 (Raven): teacher-forced conditionals at each arm's visited states
# --------------------------------------------------------------------------- #
def tf_frame(model, data, idx):
    """One row per valid agent-round: observed c, E[c | history], predictive
    SD and entropy of the trunk's teacher-forced marginal."""
    import contribution_copula_rho as cc

    rows = cc.teacher_forced_rows(model, data, idx)
    P = rows["P"]
    lev = np.arange(P.shape[1], dtype=np.float64)
    e = P @ lev
    var = P @ lev**2 - e**2
    ent = -(P * np.log(np.clip(P, 1e-12, None))).sum(1)
    return pd.DataFrame(
        dict(
            episode=rows["episode"],
            agent=rows["agent"],
            round=rows["round"],
            group=rows["group"],
            c=rows["y"].astype(float),
            e=e,
            sd=np.sqrt(np.clip(var, 0, None)),
            entropy=ent,
        )
    )


def teacher_force():
    import contribution_copula_rho as cc
    import torch as th
    from rcb_teacher_forced import check_sim_defaults, load_sim

    from aimanager.generic.graph import GraphNetwork

    OUT.mkdir(parents=True, exist_ok=True)
    model = GraphNetwork.load(str(TRUNK), device="cpu")
    model.eval()
    assert model.copula_rho == 0.0, "teacher-force the BARE trunk"
    data, pair_id, key_to_idx, defaults = cc.load_full()
    tr = cc.select_split(key_to_idx, cc.TRAIN, cc.N_TRAIN_EP)
    te = cc.select_split(key_to_idx, cc.TEST, cc.N_TEST_EP)
    idx = np.array(sorted(set(tr.tolist()) | set(te.tolist())))
    df = tf_frame(model, data, idx)
    df["episode"] = idx[df["episode"]]
    df.to_parquet(OUT / "tf_human.parquet", index=False)
    print(f"human: {len(df)} rows, {len(idx)} single-copy games")

    check_sim_defaults(model, defaults)
    for arm, name in ARMS.items():
        path = SIM / name / "per_round.parquet"
        sim, _ = load_sim(path, defaults)
        n_ep = sim["contribution"].shape[0]
        with th.no_grad():
            df = tf_frame(model, sim, np.arange(n_ep))
        df.to_parquet(OUT / f"tf_{arm}.parquet", index=False)
        print(f"arm {arm}: {len(df)} rows from {name}")


# --------------------------------------------------------------------------- #
# stage 2 (local): the five per-round series, the decomposition, the figures
# --------------------------------------------------------------------------- #
def canonical_frames():
    from aimanager.evaluation_suite.convert import HUMAN_DATA_FILE, load_human, load_sim

    frames = {"human": load_human(ROOT / HUMAN_DATA_FILE)}
    for arm, name in ARMS.items():
        (frames[arm],) = load_sim(SIM / name / "per_round.parquet").values()
    return frames


def spread_by_round(df):
    """(i)-(iii) per round from a canonical frame (NaN contributions dropped,
    empty groups produce no group mean -- the CG conventions)."""
    v = df.dropna(subset=["contribution"])
    gm = v.groupby(["round_number", "episode_id", "group_id"])["contribution"].mean()
    out = pd.DataFrame(
        {
            "sd_group_mean": gm.groupby("round_number").std(),
            "sd_individual": v.groupby("round_number")["contribution"].std(),
        }
    )
    out["ratio"] = out["sd_group_mean"] / out["sd_individual"]
    return out


def cg_ratio(df):
    v = df.dropna(subset=["contribution"])
    gm = v.groupby(["episode_id", "round_number", "group_id"])["contribution"].mean()
    return gm.std() / v["contribution"].std()


def pair_corr(df, col):
    """Pairwise Pearson correlation of `col` between members of one cell
    (all ordered within-cell pairs; the copula estimator's pairing)."""
    g = df.groupby(CELL)[col]
    n = g.transform("size")
    s1 = g.transform("sum")
    x = df[col]
    # sum over ordered pairs i != j of x_i x_j = (sum x)^2 - sum x^2, per cell
    cross = ((s1 * x) - x**2).sum()
    # each row i pairs with n-1 partners: sum_i (n_i - 1) x_i^2 on both sides
    norm = ((n - 1) * x**2).sum()
    return cross / norm if norm > 0 else np.nan


def residual_by_round(tf):
    """(iv) within-group residual correlation and (v) predictive SD / entropy
    per round, from a teacher-forced frame."""
    tf = tf.copy()
    tf["resid"] = tf["c"] - tf["e"]
    tf["zresid"] = tf["resid"] / tf["sd"].clip(lower=1e-6)
    rows = []
    for r, d in tf.groupby("round"):
        rows.append(
            dict(
                round_number=r,
                resid_corr=pair_corr(d, "resid"),
                zresid_corr=pair_corr(d, "zresid"),
                pred_sd_mean=d["sd"].mean(),
                pred_sd_sd=d["sd"].std(),
                entropy_mean=d["entropy"].mean(),
                entropy_sd=d["entropy"].std(),
                resid_sd=d["resid"].std(),
            )
        )
    return pd.DataFrame(rows).set_index("round_number")


def decompose(df):
    """Var of the group mean over (episode, round, group) split into the
    between-(episode, group) part (variance of each group's episode-level
    mean) and the round-to-round part (mean within-(episode, group) variance
    over rounds), with the SDs the CG ratio uses."""
    v = df.dropna(subset=["contribution"])
    gm = v.groupby(["episode_id", "group_id", "round_number"])["contribution"].mean()
    eg = gm.groupby(["episode_id", "group_id"])
    total = gm.var()
    between = eg.mean().var()
    within = eg.var().mean()
    return dict(
        var_total=total,
        var_between_episode_group=between,
        var_round_to_round=within,
        share_between=between / total,
        sd_group_mean=gm.std(),
        sd_individual=v["contribution"].std(),
        ratio=gm.std() / v["contribution"].std(),
        n_cells=len(gm),
    )


def latent_regression():
    """Arm A (latent-logged rerun): regress the group mean on the drawn
    latent at the (episode, round, group) level and at the episode level."""
    path = SIM / ZLOG / "per_round.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    assert df["copula_z"].notna().all(), "latent not logged"
    src = pd.read_parquet(SIM / ARMS["A"] / "per_round.parquet")
    same = (
        df.sort_values(["episode", "participant_code", "round_number"])[
            ["contribution", "punishment", "agent_group"]
        ].to_numpy()
        == src.sort_values(["episode", "participant_code", "round_number"])[
            ["contribution", "punishment", "agent_group"]
        ].to_numpy()
    ).all()
    gm = (
        df.groupby(["episode", "round_number", "agent_group"])
        .agg(
            gmean=("contribution", "mean"),
            z=("copula_z", "mean"),
            n=("copula_z", "size"),
        )
        .reset_index()
    )
    assert (gm["n"] >= 1).all()
    # within a cell every member shares the latent, so the cell mean IS z
    zsd = df.groupby(["episode", "round_number", "agent_group"])["copula_z"].std().max()
    assert zsd < 1e-6, f"latent differs within a cell: {zsd}"
    out = {"rerun_identical_to_A": bool(same)}
    for label, d in (
        ("cell", gm),
        (
            "episode_group",
            gm.groupby(["episode", "agent_group"])[["gmean", "z"]].mean(),
        ),
    ):
        x, y = d["z"].to_numpy(), d["gmean"].to_numpy()
        slope = np.cov(x, y)[0, 1] / x.var(ddof=1)
        r = np.corrcoef(x, y)[0, 1]
        out[f"{label}_slope"] = slope
        out[f"{label}_r2"] = r**2
        out[f"{label}_var_explained"] = r**2 * y.var(ddof=1)
        out[f"{label}_var_gmean"] = y.var(ddof=1)
    # the latent's ONE-SHOT per-round effect: the teacher-forced residual
    # c - E[c | history] regressed on z (the compounding factor is the group-mean
    # slope above over this)
    tf = pd.read_parquet(OUT / "tf_A.parquet")
    # the tensor's episode axis is parse_agent_rounds' dense rank of the STRING
    # key "sim__<episode>", i.e. lexicographic order of the episode number
    order = sorted(df["episode"].unique(), key=lambda e: f"sim__{e}")
    tf["episode"] = np.asarray(order)[tf["episode"]]
    z = df.assign(agent=df["participant_code"].str.split("_").str[0].astype(int))
    tf = tf.merge(
        z[["episode", "agent", "round_number", "copula_z"]].rename(
            columns={"round_number": "round"}
        ),
        on=["episode", "agent", "round"],
    )
    resid = tf["c"] - tf["e"]
    out["resid_on_z_slope"] = np.cov(tf["copula_z"], resid)[0, 1] / tf["copula_z"].var(
        ddof=1
    )
    out["e_on_z_slope"] = np.cov(tf["copula_z"], tf["e"])[0, 1] / tf["copula_z"].var(
        ddof=1
    )
    out["compounding_factor"] = out["cell_slope"] / out["resid_on_z_slope"]
    # phi = 1: the latent is constant over the episode -- check it
    zvar = df.groupby(["episode", "agent_group"])["copula_z"].std().max()
    out["latent_static_within_episode"] = bool(zvar < 1e-6)
    return out


def block_table(per_round, cols):
    rows = []
    for arm, t in per_round.items():
        for name, lo, hi in BLOCKS:
            blk = t[(t.index >= lo) & (t.index < hi)]
            rows.append(dict(arm=arm, rounds=name, **{c: blk[c].mean() for c in cols}))
    return pd.DataFrame(rows)


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [
            floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
            for v in r
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(per_round):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    palette = {"human": "#222222", "A": "#1f77b4", "B": "#d62728", "C": "#ff7f0e",
               "D": "#2ca02c"}  # fmt: skip
    panels = [
        ("sd_group_mean", "(i) SD of group-mean contribution"),
        ("sd_individual", "(ii) SD of individual contributions"),
        ("ratio", "(iii) ratio (i)/(ii), the CG ingredient"),
        ("resid_corr", "(iv) within-group residual corr | state"),
        ("pred_sd_mean", "(v) trunk predictive SD at visited states"),
        ("entropy_mean", "(v') trunk predictive entropy (nats)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (col, title) in zip(axes.ravel(), panels):
        for arm, t in per_round.items():
            ax.plot(
                t.index + 1, t[col], label=LABELS[arm], color=palette[arm],
                lw=2.2 if arm == "human" else 1.6, ls="--" if arm == "human" else "-",
            )  # fmt: skip
        ax.set_title(title)
        ax.set_xlabel("round")
        ax.set_xlim(1, 24)
    axes[0, 2].axhline(
        1 / np.sqrt(4), color="grey", ls=":", lw=1, label="independence floor (n=4)"
    )
    axes[1, 0].axhline(0, color="grey", ls=":", lw=1)
    axes[0, 0].legend(fontsize=8)
    axes[0, 2].legend(fontsize=8)
    fig.suptitle(
        "Closed-loop variance, serving fix x copula (stimulus-skip trunk, "
        "fixed punisher, nothing retrained)"
    )
    fig.tight_layout()
    fig.savefig(OUT / "per_round_lines.jpg", dpi=150)
    plt.close(fig)


def analyse():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = canonical_frames()
    per_round, decomp = {}, {}
    for arm, df in frames.items():
        tf = pd.read_parquet(OUT / f"tf_{arm}.parquet")
        t = spread_by_round(df).join(residual_by_round(tf))
        t["arm"] = arm
        per_round[arm] = t
        decomp[arm] = dict(arm=arm, **decompose(df), cg_ratio=cg_ratio(df))
        tf["resid"] = tf["c"] - tf["e"]
        tf["zresid"] = tf["resid"] / tf["sd"].clip(lower=1e-6)
        decomp[arm]["resid_corr_all_rounds"] = pair_corr(tf, "resid")
        decomp[arm]["zresid_corr_all_rounds"] = pair_corr(tf, "zresid")
        decomp[arm]["resid_corr_rounds_2_24"] = pair_corr(tf[tf["round"] > 0], "resid")
        decomp[arm]["pred_sd_mean"] = tf["sd"].mean()
        decomp[arm]["resid_sd"] = tf["resid"].std()
        # Var(c) = Var(E[c | history]) + Var(residual): states vs noise
        decomp[arm]["var_c"] = tf["c"].var()
        decomp[arm]["var_cond_mean"] = tf["e"].var()
        decomp[arm]["var_resid"] = tf["resid"].var()
    long = pd.concat(per_round.values()).reset_index()
    long.to_csv(OUT / "per_round.csv", index=False)
    cols = ["sd_group_mean", "sd_individual", "ratio", "resid_corr", "zresid_corr",
            "pred_sd_mean", "entropy_mean", "resid_sd"]  # fmt: skip
    blocks = block_table(per_round, cols)
    blocks.to_csv(OUT / "round_blocks.csv", index=False)
    dec = pd.DataFrame(decomp.values())
    dec.to_csv(OUT / "cg_decomposition.csv", index=False)
    lat = latent_regression()
    if lat is not None:
        pd.Series(lat).to_csv(OUT / "latent_regression.csv", header=False)
    plot(per_round)

    with open(OUT / "tables.md", "w") as f:
        f.write("## Round-block means\n\n" + md_table(blocks) + "\n\n")
        f.write(
            "## CG decomposition (all rounds)\n\n" + md_table(dec, "{:.4f}") + "\n\n"
        )
        if lat is not None:
            f.write("## Arm A: group mean on the logged latent\n\n")
            f.write("\n".join(f"- {k}: {v}" for k, v in lat.items()) + "\n")
    print(open(OUT / "tables.md").read())


# --------------------------------------------------------------------------- #
# the 22-row evaluation for the three arms (scores.csv from `aimanager evaluate`)
# --------------------------------------------------------------------------- #
def scores():
    from curpun_rebaseline import METRIC_ORDER, RUN, band, read_scores

    from aimanager.evaluation_suite.convert import HUMAN_DATA_FILE, load_human, load_sim
    from aimanager.evaluation_suite.metrics import GROUPS

    run = "lin_multinomial_copula_self"
    tab = pd.DataFrame(index=METRIC_ORDER)
    bands, rce = {}, {}
    human = GROUPS["R"]._rce_fit(load_human(ROOT / HUMAN_DATA_FILE))
    rce["human"] = human["slope"]
    for arm, name in ARMS.items():
        s = read_scores(str(SIM / name / "evaluation" / "scores.csv"), run)
        assert s is not None, f"no scores for arm {arm}: run evaluate first"
        tab[arm] = s
        bands[arm] = s.map(band)
        rce[arm] = GROUPS["R"]._rce_fit(
            load_sim(SIM / name / "per_round.parquet")[RUN + run]
        )["slope"]
    tab.loc["mean"] = tab.loc[METRIC_ORDER].mean()
    tab.loc["rows <= 1"] = (tab.loc[METRIC_ORDER] <= 1).sum()
    tab.index.name = "row"
    tab.to_csv(OUT / "scores_22.csv")
    rce_df = pd.DataFrame(rce).T
    rce_df.index.name = "arm"
    rce_df.to_csv(OUT / "rce_bands.csv")
    show = tab.copy()
    for arm in ARMS:
        show[arm] = [
            f"{v:.4f} ({bands[arm][r]})" if r in METRIC_ORDER else f"{v:.4f}"
            for r, v in tab[arm].items()
        ]
    md = "## 22-row evaluation, three arms\n\n" + md_table(show.reset_index(), "{:.4f}")
    md += "\n\n## RCE band slopes (0-4 / 5-9 / 10-14 / 15-19)\n\n"
    md += md_table(rce_df.reset_index(), "{:+.3f}") + "\n"
    (OUT / "scores_22.md").write_text(md)
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-force", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--scores", action="store_true")
    a = ap.parse_args()
    if a.teacher_force:
        teacher_force()
    if a.analyse:
        analyse()
    if a.scores:
        scores()
