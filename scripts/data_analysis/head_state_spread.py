"""Does a location-scale emission head resist closed-loop state contraction
better than a categorical one?

PR #186 (`copula_closed_loop_variance.py`, whose decomposition and per-round
functions are imported here unchanged) showed on the categorical stimulus-skip
trunk that the closed loop keeps the per-round noise (Var(residual) 11.4 vs
11.8 on human histories) but loses state spread: Var(E[c | history]) 18.9
without the copula (arm B) against 27.9 on human histories. This script runs
the same decomposition on the two Gaussian-MLP contributor stacks (case c:
`gaussian_mlp_inflated`, a discrete mixture of a binned Gaussian body with
status-quo and corner atoms; case d: `gaussian_mlp_v2`, a rounded
heteroscedastic Gaussian), each with its group copula on (the committed
_curpun sims) and off (rho-zero stamps of the same bundles, weight-identical,
`scripts/baselines/stamp_contribution_copula_rho0.py`). The bundle is
teacher-forced over each arm's own realised trajectories through the
adapter's own feature code (`LinearAHAdapter._pool_from_arrays`, the closed
loop's exact features) and over the 50 human games through the training
loader.

Second probe: the off-manifold gain. Every realised contribution in a human
game is shifted by delta in {-6..6}, clipped to the grid, `prev_contribution`
rebuilt by the same t-1 roll the loader uses, and the model's conditional
expectation recomputed; gain(delta) = mean(E_delta - E_0) / delta over rows
whose OWN previous contribution stays inside the grid (so the agent's own
signal is unclipped; other members' clipping is reported as the realised
shift of the group-mean feature), and over a fixed common set (own previous
contribution in [6, 14]). A head that extrapolates keeps a flat gain; one
that relaxes toward its marginal shows the gain decaying with |delta|.

Stages:
  local:  python scripts/data_analysis/head_state_spread.py --teacher-force
          python scripts/data_analysis/head_state_spread.py --gain
  Raven:  HEAD_DIAG_TREE=<PR #186 checkout> python ... --gnn-gain   (PyG)
  local:  python scripts/data_analysis/head_state_spread.py --analyse

Outputs under plots/data_analysis/evaluation/head_state_spread/.
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
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(ROOT / "scripts" / "data_analysis"))

import copula_closed_loop_variance as ccv  # noqa: E402  (PR #186, verbatim)

OUT = ROOT / "plots/data_analysis/evaluation/head_state_spread"
SIM = ROOT / "plots/simulation"
BASELINES = ROOT / "artifacts/baselines"
K = 21
LEV = np.arange(K, dtype=np.float64)
BUNDLES = {
    "infl": "contribution_gaussian_mlp_inflated_group_copula",
    "v2": "contribution_gaussian_mlp_v2_group_copula",
}
ARMS = {  # arm -> (bundle, sim dir)
    "c_infl": (
        "infl",
        "23_2g8a_infl_self_gaussian_mlp_inflated_group_copula_contr_gnn_joint_exodus"
        "_k_onehot_switch_curpun",
    ),
    "c_infl_rho0": ("infl", "23_2g8a_head_diag_c_infl_rho0"),
    "d_kexo": (
        "v2",
        "23_2g8a_kexo_self_gaussian_mlp_v2_group_copula_contr_gnn_joint_exodus"
        "_k_onehot_switch_curpun",
    ),
    "d_kexo_rho0": ("v2", "23_2g8a_head_diag_d_kexo_rho0"),
}
# The switch-matched CATEGORICAL arms. PR #186's arms A and B ran with
# switch_joint_exodus while every Gaussian arm here runs with
# switch_exodus_k_onehot, which left the switch model as the one cross-lineage
# difference in the headline table. These two close it: the same stimulus-skip
# contributor (copula on / off) under the Gaussian arms' own switch head.
GNN_ARMS = {
    "e_skip_kexo": (  # copula on -- PR #190's committed run, job 30317123
        "23_2g8a_switch_kexo_port_self_gnncopar1_contr_stimulus_skip"
        "_contr_gnn_kexo_switch_curpun"
    ),
    "e_skip_kexo_rho0": "23_2g8a_head_diag_e_skip_kexo_rho0",  # copula off
}

DELTAS = [-6, -4, -2, 2, 4, 6]
COMMON = (6, 14)  # own previous contribution range that stays in-grid for all deltas

# PR #186's numbers (categorical stimulus-skip trunk; its Var(E|hist) on human
# histories is that trunk's own conditional). Quoted, not recomputed.
PR186 = pd.DataFrame(
    [
        ("human (skip trunk)", 39.92, 27.93, 11.79, 28.70, 0.8480, 0.032),
        ("skip A (rho .0395, phi 1)", 35.97, 23.88, 11.40, 23.65, 0.8109, 0.022),
        ("skip B (no copula)", 29.48, 18.88, 11.36, 18.16, 0.7848, 0.001),
        ("skip C (rho .0395, phi 0)", 29.57, 18.86, 11.18, 18.75, 0.7964, 0.031),
    ],
    columns=["arm", "var_c", "var_cond_mean", "var_resid", "var_group_mean",
             "ratio", "resid_corr_all_rounds"],  # fmt: skip
)
PR186["sd_group_mean"] = np.sqrt(PR186["var_group_mean"])
PR186["sd_individual"] = np.sqrt(PR186["var_c"])
PR186["model"] = "skip_categorical"


# --------------------------------------------------------------------------- #
# the bundle's 21-level law at a row
# --------------------------------------------------------------------------- #
def load_bundle(key):
    import joblib

    return joblib.load(BASELINES / f"{BUNDLES[key]}.joblib")


def level_law(b, X):
    """(P [n, 21], mu, sigma): the law the sampler realises -- predict_proba
    for the inflated mixture, the binned Gaussian for the plain bundle
    (contribution_gmlp_copula_rho.score_bundle's rule) -- and the body."""
    from gaussian_mlp_preflight import bin_probs

    Xs = b["scaler"].transform(X)
    est = b["estimator"]
    mu = np.asarray(est.predict(Xs), float).reshape(-1)
    sd = np.asarray(est.predict_std(Xs), float).reshape(-1)
    if b["model"] == "gaussian_mlp_inflated":
        P = np.clip(np.asarray(est.predict_proba(Xs), float), 1e-12, None)
        P /= P.sum(1, keepdims=True)
    else:
        P = bin_probs(mu, sd, K)
    return P, mu, sd


def tf_frame(P, mu, sigma, y, episode, agent, round_, group):
    e = P @ LEV
    var = P @ LEV**2 - e**2
    return pd.DataFrame(
        dict(
            episode=episode,
            agent=agent,
            round=round_,
            group=group,
            c=np.asarray(y, float),
            e=e,
            sd=np.sqrt(np.clip(var, 0, None)),
            entropy=-(P * np.log(P)).sum(1),
            mu=mu,
            sigma=sigma,
        )
    )


# --------------------------------------------------------------------------- #
# human games through the training loader; sims through the adapter's features
# --------------------------------------------------------------------------- #
def human_data(b):
    """create_torch_data on the 50 games (train + test split files, the same
    games PR #186 teacher-forced) with the bundle's own defaults."""
    from aimanager.generic.data import create_torch_data
    from handcrafted_grid import load_config, load_episodes

    cfg = load_config(ROOT / b["config"])
    d = cfg["data"]
    frames = []
    splits = ("2group_8agent_50ep_bline_train.csv",
              "2group_8agent_50ep_bline_test.csv")  # fmt: skip
    for f in splits:
        d["data_file"] = f"experiments/baseline/{f}"
        frames.append(load_episodes(cfg, ROOT))
    df = pd.concat(frames, ignore_index=True)
    data, dv, _ = create_torch_data(
        df, default_values=b["default_values"], switch_every=b["switch_every"]
    )
    assert data["contribution"].shape[0] == 50, data["contribution"].shape
    return data


def pool_rows(b, pool, mask, y, group):
    g, a, t = np.nonzero(mask)
    X = np.column_stack([pool[f][mask] for f in b["features"]])
    P, mu, sd = level_law(b, X)
    return tf_frame(P, mu, sd, y[mask], g, a, t, group[mask])


def human_frame(b):
    from handcrafted_grid import build_feature_pool

    data = human_data(b)
    pool = build_feature_pool(data, b["switch_every"])
    mask = data["contribution_valid"].numpy().astype(bool)
    return pool_rows(
        b, pool, mask, data["contribution"].numpy(), data["agent_group"].numpy()
    )


def sim_arrays(path):
    df = pd.read_parquet(path)
    assert df["run"].nunique() == 1
    df["agent"] = df["participant_code"].str.split("_").str[0].astype(int)
    df = df.sort_values(["episode", "agent", "round_number"])
    A, T = df["agent"].nunique(), df["round_number"].nunique()
    assert len(df) == df["episode"].nunique() * A * T, "incomplete grid"
    for ep, d in df.groupby("episode", sort=True):
        arr = {
            k: d[k].to_numpy().reshape(A, T)
            for k in ("contribution", "punishment", "common_good", "agent_group")
        }
        yield int(ep), arr


def sim_frame(b, path):
    """The bundle teacher-forced over a simulation's realised rounds through
    `_pool_from_arrays` -- the exact feature code the closed loop ran, fed the
    parquet's own per-capita common good and post-arrival membership."""
    from aimanager.simulation.linear_ah import LinearAHAdapter

    ad = LinearAHAdapter(b, n_agents=8, n_contributions=K)
    frames = []
    for ep, arr in sim_arrays(path):
        c, p, cg, ag = (
            arr["contribution"].astype(float),
            arr["punishment"].astype(float),
            arr["common_good"].astype(float),
            arr["agent_group"].astype(int),
        )
        pool = ad._pool_from_arrays(c, p, cg, ag, c.shape[1])
        mask = np.ones(c.shape, bool)[None]
        f = pool_rows(b, pool, mask, c[None], ag[None])
        f["episode"] = ep
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def teacher_force():
    OUT.mkdir(parents=True, exist_ok=True)
    for key in BUNDLES:
        b = load_bundle(key)
        df = human_frame(b)
        df.to_parquet(OUT / f"tf_human_{key}.parquet", index=False)
        print(f"human under {key}: {len(df)} rows, var(c) {df['c'].var():.3f}")
    for arm, (key, name) in ARMS.items():
        path = SIM / name / "per_round.parquet"
        if not path.exists():
            print(f"arm {arm}: {path} missing, skipped")
            continue
        df = sim_frame(load_bundle(key), path)
        df.to_parquet(OUT / f"tf_{arm}.parquet", index=False)
        print(f"arm {arm}: {len(df)} rows from {name}")


# --------------------------------------------------------------------------- #
# off-manifold gain: shift the realised history, re-evaluate E[c | history]
# --------------------------------------------------------------------------- #
def shifted(data, delta, default):
    """A copy of the tensors with every VALID realised contribution moved by
    delta (clipped to the grid) and prev_contribution rebuilt by the loader's
    own t-1 roll (aimanager.generic.data.shift)."""
    from aimanager.generic.data import shift

    d = dict(data)
    c = data["contribution"].clone()
    valid = data["contribution_valid"]
    c[valid] = (c[valid] + delta).clamp(0, K - 1)
    d["contribution"] = c
    d["prev_contribution"] = shift(c, default)
    return d


def gain_table(evaluate, data, default, model, extra=None):
    """`evaluate(d) -> (e, aux)` on the masked rows of a data dict; the gain
    of e (and of aux, e.g. the location mu) per unit delta."""
    from aimanager.generic.data import shift

    assert (shift(data["contribution"], default) == data["prev_contribution"]).all()
    mask = data["contribution_valid"].numpy().astype(bool)
    rnd = np.nonzero(mask)[2]
    prev0 = data["prev_contribution"].numpy()[mask].astype(float)
    e0, aux0 = evaluate(data)
    rows = []
    for delta in DELTAS:
        e, aux = evaluate(shifted(data, delta, default))
        sets = {
            "own_in_grid": (rnd >= 1) & (prev0 + delta >= 0) & (prev0 + delta <= K - 1),
            "common_6_14": (rnd >= 1) & (prev0 >= COMMON[0]) & (prev0 <= COMMON[1]),
        }
        for name, sel in sets.items():
            r = dict(
                model=model, delta=delta, set=name, n=int(sel.sum()),
                gain_e=float((e[sel] - e0[sel]).mean() / delta),
                e0_mean=float(e0[sel].mean()),
            )  # fmt: skip
            for k, v in (aux or {}).items():
                r[f"gain_{k}"] = float((v[sel] - aux0[k][sel]).mean() / delta)
            if extra is not None:
                r.update(extra(data, delta, sel))
            rows.append(r)
    return pd.DataFrame(rows)


def gain_gaussian():
    from handcrafted_grid import build_feature_pool

    OUT.mkdir(parents=True, exist_ok=True)
    tabs = []
    for key in BUNDLES:
        b = load_bundle(key)
        data = human_data(b)
        mask = data["contribution_valid"].numpy().astype(bool)
        pool0 = build_feature_pool(data, b["switch_every"])

        def evaluate(d):
            pool = build_feature_pool(d, b["switch_every"])
            X = np.column_stack([pool[f][mask] for f in b["features"]])
            P, mu, sd = level_law(b, X)
            return P @ LEV, {"mu": mu}

        def realised_shift(d, delta, sel):
            """How far the group-mean feature actually moved (others clipped)."""
            pool = build_feature_pool(
                shifted(d, delta, b["default_values"]["contribution"]),
                b["switch_every"],
            )
            f = "prev_contribution_mean_group"
            return {
                "group_feature_shift": float(
                    (pool[f][mask] - pool0[f][mask])[sel].mean() / delta
                )
            }

        tabs.append(
            gain_table(
                evaluate, data, b["default_values"]["contribution"], key, realised_shift
            )
        )
        print(tabs[-1].to_string())
    pd.concat(tabs).to_csv(OUT / "gain_gaussian.csv", index=False)


def gnn_teacher_force():
    """PR #186's teacher-forcing, run over the two switch-matched categorical
    arms. The BARE stimulus-skip trunk (copula_rho == 0) supplies
    E[c | history] at each arm's own realised states, exactly as
    `copula_closed_loop_variance.teacher_force` does for arms A/B/C -- the
    same function (`ccv.tf_frame` -> `cc.teacher_forced_rows`), so the new rows
    are produced by the code that produced the quoted ones.

    Also re-runs the human pass, whose Var(E[c|hist]) must reproduce PR #186's
    27.93: that is the check that this path is measuring the same thing.
    """
    tree = Path(os.environ["HEAD_DIAG_TREE"]).expanduser().resolve()
    sys.path.insert(0, str(tree / "src"))
    sys.path.insert(0, str(tree / "scripts" / "artificial_humans"))
    sys.path.insert(0, str(tree / "scripts" / "data_analysis"))
    import contribution_copula_rho as cc
    import torch as th
    from rcb_teacher_forced import check_sim_defaults, load_sim

    from aimanager.generic.graph import GraphNetwork

    OUT.mkdir(parents=True, exist_ok=True)
    trunk = tree / (
        "artifacts/artificial_humans"
        "/group_switching_contribution_50ep_vnode_stimulus_skip"
        "/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
    )
    model = GraphNetwork.load(str(trunk), device="cpu")
    model.eval()
    assert model.copula_rho == 0.0, "teacher-force the BARE trunk"

    data, _, key_to_idx, defaults = cc.load_full()
    tr = cc.select_split(key_to_idx, cc.TRAIN, cc.N_TRAIN_EP)
    te = cc.select_split(key_to_idx, cc.TEST, cc.N_TEST_EP)
    idx = np.array(sorted(set(tr.tolist()) | set(te.tolist())))
    df = ccv.tf_frame(model, data, idx)
    df["episode"] = idx[df["episode"]]
    df.to_parquet(OUT / "tf_human_skip.parquet", index=False)
    v = (df["c"] - df["e"]).var()
    print(f"human under the bare skip trunk: {len(df)} rows, "
          f"var(c) {df['c'].var():.3f}, var(E) {df['e'].var():.3f} "
          f"(PR #186: 27.93), var(resid) {v:.3f} (PR #186: 11.79)")  # fmt: skip

    check_sim_defaults(model, defaults)
    for arm, name in GNN_ARMS.items():
        path = SIM / name / "per_round.parquet"
        if not path.exists():
            print(f"arm {arm}: {path} missing, skipped")
            continue
        sim, _ = load_sim(path, defaults)
        n_ep = sim["contribution"].shape[0]
        with th.no_grad():
            df = ccv.tf_frame(model, sim, np.arange(n_ep))
        df.to_parquet(OUT / f"tf_{arm}.parquet", index=False)
        print(f"arm {arm}: {len(df)} rows, var(E) {df['e'].var():.3f}")


def gain_gnn():
    """The categorical stimulus-skip trunk (PR #186's arm B model), run under
    that PR's code tree (HEAD_DIAG_TREE) on Raven: its loaders give the same
    50 games, its GraphNetwork the teacher-forced conditional."""
    tree = Path(os.environ["HEAD_DIAG_TREE"]).expanduser().resolve()
    sys.path.insert(0, str(tree / "src"))
    sys.path.insert(0, str(tree / "scripts" / "artificial_humans"))
    import contribution_copula_rho as cc
    import torch as th
    from aimanager.generic.graph import GraphNetwork

    trunk = tree / (
        "artifacts/artificial_humans"
        "/group_switching_contribution_50ep_vnode_stimulus_skip"
        "/model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
    )
    model = GraphNetwork.load(str(trunk), device="cpu")
    model.eval()
    assert model.copula_rho == 0.0
    data, _, key_to_idx, defaults = cc.load_full()
    tr = cc.select_split(key_to_idx, cc.TRAIN, cc.N_TRAIN_EP)
    te = cc.select_split(key_to_idx, cc.TEST, cc.N_TEST_EP)
    idx = th.as_tensor(np.array(sorted(set(tr.tolist()) | set(te.tolist()))))
    sub = {k: v[idx] for k, v in data.items()}
    mask = sub["contribution_valid"].numpy().astype(bool)
    n_ep, n_ag, _ = sub["contribution"].shape
    edge_index = model.create_fully_connected(n_ag, n_batch=n_ep)

    def evaluate(d):
        with th.no_grad():
            _, proba = model.predict_independent(
                d, sample=False, reset_rnn=True, edge_index=edge_index
            )
        P = proba.double().numpy()[mask]
        return P @ LEV, {}

    OUT.mkdir(parents=True, exist_ok=True)
    tab = gain_table(evaluate, sub, defaults["contribution"], "skip_categorical")
    print(tab.to_string())
    tab.to_csv(OUT / "gain_skip.csv", index=False)


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def decomposition(df, tf):
    d = dict(**ccv.decompose(df), cg_ratio=ccv.cg_ratio(df))
    tf = tf.copy()
    tf["resid"] = tf["c"] - tf["e"]
    d["resid_corr_all_rounds"] = ccv.pair_corr(tf, "resid")
    d["pred_sd_mean"] = tf["sd"].mean()
    d["var_c"] = tf["c"].var()
    d["var_cond_mean"] = tf["e"].var()
    d["var_resid"] = tf["resid"].var()
    d["var_mu"] = tf["mu"].var() if "mu" in tf else np.nan
    d["n_rows"] = len(tf)
    return d


def bootstrap_retention(n_boot=2000, seed=0):
    """Episode-cluster bootstrap of Var(E[c | history]) and of the retention
    against the SAME model's human-history value. Episodes are the independent
    unit (agents within a game share a trajectory), so both the sim arm (100
    episodes) and the human reference (50 games) are resampled whole, and
    independently -- the two are different data, not paired."""
    rng = np.random.default_rng(seed)
    out = []
    hum = {}
    for key in BUNDLES:
        tf = pd.read_parquet(OUT / f"tf_human_{key}.parquet")
        hum[key] = [g["e"].to_numpy() for _, g in tf.groupby("episode")]
    arms = {a: k for a, (k, _) in ARMS.items()}
    hs = OUT / "tf_human_skip.parquet"
    if hs.exists():
        tf = pd.read_parquet(hs)
        hum["skip_categorical"] = [g["e"].to_numpy() for _, g in tf.groupby("episode")]
        arms.update({a: "skip_categorical" for a in GNN_ARMS})
    for arm, key in arms.items():
        path = OUT / f"tf_{arm}.parquet"
        if not path.exists():
            continue
        tf = pd.read_parquet(path)
        sim = [g["e"].to_numpy() for _, g in tf.groupby("episode")]
        draws = []
        for _ in range(n_boot):
            a = np.concatenate([sim[i] for i in rng.integers(0, len(sim), len(sim))])
            b = np.concatenate(
                [hum[key][i] for i in rng.integers(0, len(hum[key]), len(hum[key]))]
            )
            draws.append((a.var(), b.var()))
        d = np.array(draws)
        ret = d[:, 0] / d[:, 1]
        out.append(
            dict(
                arm=arm, model=key,
                var_cond_mean=tf["e"].var(),
                var_lo=np.percentile(d[:, 0], 2.5),
                var_hi=np.percentile(d[:, 0], 97.5),
                retention=float(np.mean(ret)),
                ret_lo=np.percentile(ret, 2.5),
                ret_hi=np.percentile(ret, 97.5),
                n_episodes=len(sim),
            )
        )  # fmt: skip
    return pd.DataFrame(out)


def analyse():
    from aimanager.evaluation_suite.convert import HUMAN_DATA_FILE, load_human, load_sim

    OUT.mkdir(parents=True, exist_ok=True)
    human = load_human(ROOT / HUMAN_DATA_FILE)
    per_round, dec = {}, []
    for key in BUNDLES:
        tf = pd.read_parquet(OUT / f"tf_human_{key}.parquet")
        arm = f"human ({key})"
        dec.append(dict(arm=arm, model=key, **decomposition(human, tf)))
        per_round[arm] = ccv.spread_by_round(human).join(ccv.residual_by_round(tf))
    hs = OUT / "tf_human_skip.parquet"
    if hs.exists():
        tf = pd.read_parquet(hs)
        dec.append(
            dict(arm="human (skip recomputed)", model="skip_categorical",
                 **decomposition(human, tf))
        )  # fmt: skip
    arms = {a: (k, n) for a, (k, n) in ARMS.items()}
    arms.update({a: ("skip_categorical", n) for a, n in GNN_ARMS.items()})
    for arm, (key, name) in arms.items():
        p = OUT / f"tf_{arm}.parquet"
        if not p.exists():
            continue
        tf = pd.read_parquet(p)
        (df,) = load_sim(SIM / name / "per_round.parquet").values()
        dec.append(dict(arm=arm, model=key, **decomposition(df, tf)))
        per_round[arm] = ccv.spread_by_round(df).join(ccv.residual_by_round(tf))
    dec = pd.DataFrame(dec)
    dec.to_csv(OUT / "state_spread_decomposition.csv", index=False)
    cols = ["sd_group_mean", "sd_individual", "ratio", "resid_corr", "pred_sd_mean",
            "entropy_mean", "resid_sd"]  # fmt: skip
    blocks = ccv.block_table(per_round, cols)
    blocks.to_csv(OUT / "round_blocks.csv", index=False)
    pd.concat(
        per_round.values(), keys=per_round.keys(), names=["arm"]
    ).reset_index().to_csv(OUT / "per_round.csv", index=False)

    # headline: state spread, quoted PR #186 rows first
    head_cols = ["arm", "model", "var_c", "var_cond_mean", "var_resid",
                 "sd_group_mean", "sd_individual", "ratio",
                 "resid_corr_all_rounds"]  # fmt: skip
    head = pd.concat([PR186[head_cols], dec[head_cols]], ignore_index=True)
    ref = {"skip_categorical": float(PR186.loc[0, "var_cond_mean"])}  # 27.93
    for key in BUNDLES:
        row = dec.loc[dec["arm"] == f"human ({key})", "var_cond_mean"]
        ref[key] = float(row.iloc[0])
    head["human_var_cond_mean"] = head["model"].map(ref)
    head["retention"] = head["var_cond_mean"] / head["human_var_cond_mean"]
    head.to_csv(OUT / "headline.csv", index=False)

    boot = bootstrap_retention()
    boot.to_csv(OUT / "retention_bootstrap.csv", index=False)

    gains = [pd.read_csv(OUT / "gain_gaussian.csv")]
    if (OUT / "gain_skip.csv").exists():
        gains.append(pd.read_csv(OUT / "gain_skip.csv"))
    gain = pd.concat(gains, ignore_index=True)
    shift_ref = gain.dropna(subset=["group_feature_shift"])
    chk = shift_ref.groupby(["delta", "set"])["group_feature_shift"].nunique()
    assert (chk == 1).all(), "clipping differs between bundles"
    key = shift_ref.groupby(["delta", "set"])["group_feature_shift"].first()
    gain["group_feature_shift"] = gain.set_index(["delta", "set"]).index.map(key)
    gain["gain_e_norm"] = gain["gain_e"] / gain["group_feature_shift"]
    gain.to_csv(OUT / "gain_curves.csv", index=False)
    piv = {
        s: gain[gain["set"] == s].pivot(index="delta", columns="model", values="gain_e")
        for s in gain["set"].unique()
    }
    plot(per_round, piv)

    with open(OUT / "tables.md", "w") as f:
        f.write("## State spread (all rounds)\n\n" + ccv.md_table(head) + "\n\n")
        f.write("## Retention of Var(E[c|hist]), episode-cluster bootstrap\n\n")
        f.write(ccv.md_table(boot) + "\n\n")
        f.write("## Round-block means\n\n" + ccv.md_table(blocks) + "\n\n")
        f.write("## Full decomposition\n\n" + ccv.md_table(dec, "{:.4f}") + "\n\n")
        for s, t in piv.items():
            f.write(f"## Off-manifold gain of E[c | history], rows: {s}\n\n")
            f.write(ccv.md_table(t.reset_index()) + "\n\n")
        for s_ in gain["set"].unique():
            t = gain[gain["set"] == s_].pivot(
                index="delta", columns="model", values="gain_e_norm"
            )
            f.write("## Gain normalised by the realised group-feature "
                    f"shift, rows: {s_}\n\n")  # fmt: skip
            f.write(ccv.md_table(t.reset_index()) + "\n\n")
        mu = gain[gain["set"] == "common_6_14"].pivot(
            index="delta", columns="model", values="gain_mu"
        )
        f.write("## Gain of the Gaussian body's location mu (common set)\n\n")
        f.write(ccv.md_table(mu.dropna(axis=1, how="all").reset_index()) + "\n\n")
        n = gain[gain["set"] == "own_in_grid"].pivot(
            index="delta", columns="model", values="n"
        )
        f.write(
            "## Rows per delta (own_in_grid)\n\n"
            + ccv.md_table(n.reset_index(), "{:.0f}")
            + "\n"
        )
    print(open(OUT / "tables.md").read())


def plot(per_round, piv):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    colors = {
        "human (infl)": "#222222", "human (v2)": "#666666",
        "c_infl": "#1f77b4", "c_infl_rho0": "#aec7e8",
        "d_kexo": "#2ca02c", "d_kexo_rho0": "#98df8a",
        "e_skip_kexo": "#d62728", "e_skip_kexo_rho0": "#ff9896",
    }  # fmt: skip
    panels = [
        ("sd_group_mean", "(i) SD of group-mean contribution"),
        ("sd_individual", "(ii) SD of individual contributions"),
        ("ratio", "(iii) ratio (i)/(ii)"),
        ("pred_sd_mean", "(v) predictive SD at visited states"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, (col, title) in zip(axes.ravel()[:4], panels):
        for arm, t in per_round.items():
            if col != "pred_sd_mean" and arm == "human (v2)":
                continue  # identical to human (infl): the data, not the model
            ax.plot(
                t.index + 1, t[col], label=arm, color=colors[arm],
                lw=2.2 if arm.startswith("human") else 1.6,
                ls="--" if arm.startswith("human") else "-",
            )  # fmt: skip
        ax.set_title(title)
        ax.set_xlabel("round")
        ax.set_xlim(1, 24)
    axes[0, 0].legend(fontsize=7)
    for ax, (s, t) in zip(axes.ravel()[4:], piv.items()):
        for m in t.columns:
            ax.plot(t.index, t[m], marker="o", label=m)
        ax.axhline(0, color="grey", ls=":", lw=1)
        ax.set_title(f"off-manifold gain dE/d(delta), rows: {s}")
        ax.set_xlabel("history shift delta (contribution points)")
        ax.legend(fontsize=7)
    fig.suptitle("Emission head and closed-loop state spread "
                 "(Gaussian-MLP stacks, fixed punisher)")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "head_state_spread.jpg", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher-force", action="store_true")
    ap.add_argument("--gain", action="store_true")
    ap.add_argument("--gnn-tf", action="store_true")
    ap.add_argument("--gnn-gain", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.teacher_force:
        teacher_force()
    if a.gain:
        gain_gaussian()
    if a.gnn_tf:
        gnn_teacher_force()
    if a.gnn_gain:
        gain_gnn()
    if a.analyse:
        analyse()
