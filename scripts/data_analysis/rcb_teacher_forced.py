"""Step 0 of the RCB experiment: the trunk's TEACHER-FORCED punishment response.

Measures, on the human data and with the model never seeing its own draws, the
parent trunk's conditional response to punishment -- the quantity the RCB row
scores -- so the experiment can decide between its two pre-declared changes
(notes/autoresearch_log/contribution-punishment-response.md, "What the change
will be"):

  (A) within-contribution-band OLS slopes of the contribution change on the
      punishment received, over the RCB population (punishment > 0,
      contribution < 20, next-round contribution valid), in the bands
      0-4 / 5-9 / 10-14 / 15-19;
  (B) the RCB bin means over the punishment-rate bins
      (0,0.25] / (0.25,0.5] / (0.5,1] / >1, rate = punishment / (20 -
      contribution), plus the human-bin-frequency-weighted mean absolute
      discrepancy against the human bin means;
  (C) (A) and (B) recomputed on the OBSERVED human next-round contribution over
      exactly the same rows -- a self-check that the population matches the
      canonical evaluation frame's.

The model's predicted change for a stimulus row at round t is

    E[c_{t+1}] - c_t,   E[c_{t+1}] = sum_k k * P[t+1, k],

where P[t+1] is the teacher-forced predicted marginal AT ROUND t+1 -- the
model's prediction of c_{t+1} given prev_contribution = c_t and
prev_punishment = p_t. The stimulus (c_t, p_t) therefore pairs with the NEXT
row's marginal, not its own; the alignment is asserted (see `check_alignment`).

Model: the BARE trunk
artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode/
model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt
and not the copula-stamped copy. The copula affects sampling only; this is a
teacher-forced conditional measurement, so the two artifacts are weight-
identical for this purpose.

SIM MODE (step 6, the pre-simulation mechanism gate). With `--sim-parquet
PATH` the trajectories come from a finished simulation's `per_round.parquet`
instead of the human CSV, and everything else is unchanged: the same
`create_torch_data` / `parse_agent_rounds` lag construction, the same human
default values (so round 0's `prev_` priors are the ones the simulation's own
environment used), the same teacher-forced forward pass, the same RCB
population and the same human-frequency weights.

The point of that mode: a teacher-forced pass over an ALREADY REALISED
trajectory is exactly that trajectory's conditional expectation. So

  * the PARENT trunk over the PARENT's own sim states must reproduce the
    parent's flat closed-loop slopes -- if it does not, the "the states, not
    the conditional" diagnosis (log note 8) is wrong;
  * the CANDIDATE trunk over those same states predicts whether the
    immediate-stimulus skip restores the response OFF the human manifold.

Both modes use the BARE trunks, never the copula-stamped copies: the copula
changes only how the marginals are turned into correlated draws, so for a
teacher-forced conditional measurement the two artifacts are weight-identical.

Measurement only: this script trains nothing and writes no artifact.

Imports graph.py, so this runs on Raven only:
    .venv/bin/python scripts/data_analysis/rcb_teacher_forced.py \
        [--model PT] [--sim-parquet plots/simulation/<run>/per_round.parquet]
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch as th  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "artificial_humans"))
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))

# the copula calibration's loaders, imported unmodified: that module's top
# level is imports, constants and defs only, so importing it runs nothing.
# It also installs the torch_geometric.nn.meta alias the legacy pickles need.
import contribution_copula_rho as cc  # noqa: E402

from aimanager.generic.data import create_torch_data  # noqa: E402
from aimanager.generic.graph import GraphNetwork  # noqa: E402

DEFAULT_MODEL = (
    "artifacts/artificial_humans/group_switching_contribution_50ep_group_vnode/"
    "model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
)

MASK = cc.MASK  # "contribution_valid"
PMASK = "punishment_valid"

# contribution bands of the hypothesis table
BAND_EDGES = [-0.5, 4.5, 9.5, 14.5, 19.5]
BAND_LABELS = ["0-4", "5-9", "10-14", "15-19"]

# the RCB rate bins, copied from the (frozen, read-only) metric definition
RATE_EDGES = [0.0, 0.25, 0.5, 1.0, float("inf")]
RATE_LABELS = ["(0,0.25]", "(0.25,0.5]", "(0.5,1]", ">1"]

# the human reference numbers quoted in the declaration, for the (C) self-check
HUMAN_SLOPES = {"0-4": 0.1397, "5-9": 0.1038, "10-14": -0.0767, "15-19": -0.1615}
HUMAN_BIN_MEANS = np.array([0.891761, 1.340774, 1.678647, 2.014440])
HUMAN_BIN_COUNTS = np.array([1238.0, 672.0, 473.0, 277.0])

# the parent's OBSERVED closed-loop numbers on its own simulation, quoted in
# the declaration -- the sim-mode self-check target (the observed column must
# reproduce them, since the population is rebuilt here from the parquet)
SIM_REF_SLOPES = {"0-4": 0.0619, "5-9": 0.0115, "10-14": -0.0082, "15-19": -0.0366}
SIM_REF_STAT = 0.7973084747030883
SIM_REF_DIR = "23_2g8a_contr_group_vnode_self_gnncopar1_contr_gnn_switch"

# the simulation records one row per (episode, agent, round) with no timeouts;
# these are the columns mem_to_df writes
SIM_COLUMNS = [
    "episode",
    "participant_code",
    "round_number",
    "punishment",
    "common_good",
    "contribution",
    "agent_group",
    "group_id",
    "run",
]


def f(x):
    """Unrounded float for the log."""
    return repr(float(x))


# --------------------------------------------------------------------------- #
# sim mode: a simulation's realised trajectories through the SAME loaders
# --------------------------------------------------------------------------- #
def sim_raw_frame(path):
    """A simulation `per_round.parquet` reshaped into the raw agent-round
    frame `parse_agent_rounds` consumes, so the tensors, the lag shift, the
    defaults and the validity masks are all built by the human path's code.

    Mapping, column by column:
      * `participant_code` is written by `simulate.mem_to_df` as
        "<agent index>_<episode>", so the agent index is the prefix; it is
        the tensor's `player_idx` axis and is unique within an episode;
      * `episode` is the tensor's batch axis; `global_group_id` is constant,
        so the (global_group_id, episode_id) key that `parse_agent_rounds`
        dense-ranks gives one row per simulated episode;
      * `group_id` == `agent_group` (mem_to_df writes one from the other) is
        the membership the environment used for THAT round: `recorder.add`
        runs after `punish()`, and the switch is applied at the START of the
        round, before `update_contribution` -- so the recorded group is the
        one the contribution model conditioned on;
      * there are no timeouts in a simulation -- the environment always feeds
        a punishment and always records a contribution -- so `player_no_input`
        and `manager_no_input` are 0 everywhere. That is ASSERTED below
        (rows present, nothing missing) rather than assumed, and it is what
        the frozen evaluation suite does too (`convert.load_sim` marks every
        simulated row valid);
      * `common_good` is recorded PER CAPITA by the environment
        ((1.6*sum_c - sum_p) / n_valid), whereas `parse_agent_rounds` expects
        the per-group pool and divides by the valid count itself. It is
        multiplied back up here so the round trip is exact. (No contributor
        `x_encoding` reads it; this only keeps the tensor honest.)
    """
    df = pd.read_parquet(path)
    missing = [c for c in SIM_COLUMNS if c not in df.columns]
    assert not missing, f"{path}: missing columns {missing}"
    runs = sorted(df["run"].unique().tolist())
    if len(runs) > 1:
        print(f"  {len(runs)} runs in the parquet, keeping {runs[0]!r}")
    df = df[df["run"] == runs[0]].copy()
    assert (df["agent_group"] == df["group_id"]).all(), "agent_group != group_id"

    agent = df["participant_code"].astype(str).str.split("_").str[0].astype(int)
    raw = pd.DataFrame(
        {
            "episode_id": df["episode"].astype(int).to_numpy(),
            "round_number": df["round_number"].astype(int).to_numpy(),
            "player_id": agent.to_numpy(),
            "global_group_id": "sim",
            "group_id": df["group_id"].astype(int).to_numpy(),
            "player_no_input": 0,
            "manager_no_input": 0,
            "contribution": df["contribution"].astype(float).to_numpy(),
            "punishment": df["punishment"].astype(float).to_numpy(),
            "common_good": df["common_good"].astype(float).to_numpy(),
        }
    )

    # the simulation has no timeouts: assert the grid is complete and nothing
    # is missing, which is what "every row is valid" means here
    keys = ["episode_id", "player_id", "round_number"]
    assert not raw.duplicated(keys).any(), "duplicate (episode, agent, round)"
    n_ep = raw["episode_id"].nunique()
    n_ag = raw["player_id"].nunique()
    n_rd = raw["round_number"].nunique()
    n_cells = n_ep * n_ag * n_rd
    assert len(raw) == n_cells, f"incomplete grid: {len(raw)} rows vs {n_cells}"
    for col in ("contribution", "punishment", "common_good", "group_id"):
        assert raw[col].notna().all(), f"{col} has missing values"
    assert set(raw["player_id"]) == set(range(n_ag)), "agent index not 0..n-1"
    assert set(raw["round_number"]) == set(range(n_rd)), "round index not 0..T-1"
    print(
        f"  run={runs[0]!r} episodes={n_ep} agents={n_ag} rounds={n_rd} "
        f"rows={len(raw)} (all valid: no timeouts)"
    )

    n_valid = raw.groupby(["episode_id", "round_number", "group_id"])[
        "player_id"
    ].transform("size")
    raw["common_good"] = raw["common_good"] * n_valid
    return raw, runs[0]


def load_sim(path, defaults):
    """`create_torch_data` on a simulation's realised rounds, with the HUMAN
    default values -- the same numbers the simulation's own environment used
    to fill round 0's `prev_` slots (`ArtificialHumanEnv.reset_state` fills
    them from the artificial human's stored `default_values`). Passing them in
    also stops the medians being recomputed from the simulation itself."""
    raw, run = sim_raw_frame(path)
    data, dv, _ = create_torch_data(raw, default_values=defaults)
    assert dv is defaults or dv == defaults, "defaults were recomputed"
    return data, run


# A simulation's parquet records contribution, punishment, agent_group and
# common_good and nothing else, so two derived training columns cannot be
# rebuilt from it faithfully: `contribution_valid` (the environment's validity
# model is not recorded -- every simulated row is taken as valid, as the frozen
# evaluation suite also does) and `own_grp_prev_mean_contr` (which depends on
# that validity and on a median recomputed per file). A model that reads either
# would be fed something the closed loop did not feed it.
SIM_UNFAITHFUL = {
    "contribution_valid",
    "prev_contribution_valid",
    "recorded",
    "prev_recorded",
    "own_grp_prev_mean_contr",
    "prev_own_grp_prev_mean_contr",
}


def check_sim_defaults(model, defaults):
    """Round 0's priors must be the ones the closed loop used, and every
    feature the model reads must be one the parquet rebuilds faithfully."""
    md = getattr(model, "default_values", None) or {}
    for k in ("contribution", "punishment"):
        if k in md:
            got, ref = float(md[k]), float(defaults[k])
            assert got == ref, f"{k} default {got} != human {ref}"
            print(f"  prev_{k} round-0 default = {f(ref)}  (model == human) OK")
        else:
            print(f"  model carries no {k} default; using human {f(defaults[k])}")

    used = []
    for attr in ("x_encoding", "u_encoding", "edge_encoding"):
        enc = getattr(model, attr, None) or []
        names = [e["name"] for e in enc]
        print(f"  {attr} = {names}")
        used += names
    bad = sorted(set(used) & SIM_UNFAITHFUL)
    assert not bad, f"model reads features the parquet cannot rebuild: {bad}"
    print("  no encoded feature depends on unrecorded sim state           OK")


# --------------------------------------------------------------------------- #
# stimulus rows: observed (c_t, p_t, c_{t+1}) and the predicted E[c_{t+1}]
# --------------------------------------------------------------------------- #
def dense_predictions(rows):
    """Scatter the flat teacher-forced rows back into dense [G, A, T] arrays:
    E[c] = sum_k k * P[.., k], and the full marginal P, NaN / 0 where the row
    was masked out."""
    n_ep, n_agents, n_rounds = rows["shape"]
    n_lev = rows["P"].shape[1]
    levels = np.arange(n_lev, dtype=np.float64)
    e = np.full((n_ep, n_agents, n_rounds), np.nan, dtype=np.float64)
    p = np.zeros((n_ep, n_agents, n_rounds, n_lev), dtype=np.float64)
    e[rows["episode"], rows["agent"], rows["round"]] = rows["P"] @ levels
    p[rows["episode"], rows["agent"], rows["round"]] = rows["P"]
    return e, p


def stimulus_frame(model, data, idx, label):
    """One row per valid stimulus round t of the selected episodes.

    Observed arrays come from the SAME tensors `teacher_forced_rows` indexes,
    so observed and predicted are indexed identically. Returns the frame and
    the dense arrays the alignment check needs.
    """
    rows = cc.teacher_forced_rows(model, data, idx)
    sub = {k: v[th.as_tensor(idx)] for k, v in data.items()}
    n_ep, n_agents, n_rounds = rows["shape"]

    e_pred, p_pred = dense_predictions(rows)  # E[c_t], P[c_t] given hist < t
    lev = np.arange(p_pred.shape[-1], dtype=np.float64)
    # Var[c_t | history < t] of the same teacher-forced marginal; used only by
    # the sim mode's sampling-noise table, so no printed output changes here
    v_pred = (p_pred * lev**2).sum(-1) - e_pred**2
    contr = sub["contribution"].numpy().astype(np.float64)
    punish = sub["punishment"].numpy().astype(np.float64)
    c_ok = sub[MASK].numpy().astype(bool)
    p_ok = sub[PMASK].numpy().astype(bool)
    grp = sub["agent_group"].numpy().astype(np.int64)

    # the stimulus round t runs to T-2; t+1 is the response round
    t = slice(0, n_rounds - 1)
    t1 = slice(1, n_rounds)
    valid = c_ok[:, :, t] & c_ok[:, :, t1]  # dc needs both contributions
    g, a, r = np.nonzero(valid)

    df = pd.DataFrame(
        {
            "episode": g,
            "agent": a,
            "round": r,
            "group": grp[:, :, t][valid],
            "c_t": contr[:, :, t][valid],
            "p_t": punish[:, :, t][valid],
            "p_valid": p_ok[:, :, t][valid],
            "c_next": contr[:, :, t1][valid],
            "e_next": e_pred[:, :, t1][valid],  # aligned: predicts c_{t+1}
            "e_self": e_pred[:, :, t][valid],  # off-by-one: predicts c_t
            "v_next": v_pred[:, :, t1][valid],  # Var[c_{t+1} | history]
        }
    )
    assert df["e_next"].notna().all() and df["e_self"].notna().all()
    df["dc_human"] = df["c_next"] - df["c_t"]
    df["dc_model"] = df["e_next"] - df["c_t"]
    df["dc_misaligned"] = df["e_self"] - df["c_t"]
    print(
        f"{label}: episodes={n_ep} agents={n_agents} rounds={n_rounds} "
        f"teacher-forced rows={len(rows['y'])} stimulus rows={len(df)}"
    )
    dense = dict(e=e_pred, p=p_pred, c=contr, c_ok=c_ok)
    return df, dense


def check_alignment(model, data, df, dense, label):
    """Pin down that `e_next` really is the model's prediction of c_{t+1}.

    Four handles, structural first:

      1. the tensor shift identity: prev_contribution[.., t] is
         contribution[.., t - 1] (and the same for punishment), so index t of
         the model's input really is the lagged pair of round t;
      2. the model conditions on NOTHING from the current round -- every
         contribution / punishment feature in x_encoding is a `prev_` one --
         so P[.., t] can only be a prediction of contribution[.., t];
      3. the lag profile: e_pred[.., t] is a deterministic function of its
         conditioning input, so its correlation with contribution[.., t + k]
         must PEAK at k = -1 (the input) and be lower at k = 0 (the target it
         predicts only imperfectly) and k = +1. An off-by-one indexing would
         move the peak;
      4. the requested magnitude check: over the RCB population the OLS slope
         of E[c_{t+1}] on c_t is strongly positive -- own-contribution
         stickiness, human dc-on-c slope ~ -0.20, i.e. ~ +0.80 -- while the
         misaligned E[c_t] on c_t is not the same quantity.

    The mean predictive NLL at each lag is printed as context (it is NOT an
    alignment test: the marginal is concentrated on the lagged contribution,
    so lag -1 can score well by stickiness alone).
    """
    print(f"\n--- alignment check ({label}) ---")

    # (1) structural: the shift the encoder relies on
    for name in ("contribution", "punishment"):
        cur = data[name]
        prev = data[f"prev_{name}"]
        assert th.equal(prev[:, :, 1:], cur[:, :, :-1]), f"prev_{name} misshifted"
    print("  prev_contribution / prev_punishment are the t-1 tensors        OK")

    # (2) structural: no current-round conditioning
    names = [e["name"] for e in model.x_encoding]
    illegal = [n for n in names if n in ("contribution", "punishment")]
    assert not illegal, f"model conditions on the current round: {illegal}"
    print(f"  x_encoding carries no current-round feature ({names})  OK")

    # (3) the lag profile of the predicted mean
    e, c, ok = dense["e"], dense["c"], dense["c_ok"]
    n_rounds = c.shape[2]
    prof = {}
    core = slice(2, n_rounds - 2)
    for k in (-2, -1, 0, 1, 2):
        sl = slice(2 + k, n_rounds - 2 + k)
        m = ok[:, :, core] & ok[:, :, sl] & np.isfinite(e[:, :, core])
        prof[k] = float(np.corrcoef(e[:, :, core][m], c[:, :, sl][m])[0, 1])
    print("  corr(E[c_t], c_{t+k}) by lag k:")
    for k in sorted(prof):
        tags = {-1: "  <- conditioning input", 0: "  <- target"}
        tag = tags.get(k, "")
        print(f"    k={k:+d}  {f(prof[k])}{tag}")
    peak = max(prof, key=prof.get)
    assert peak == -1, f"correlation peaks at lag {peak}, not the input lag -1"
    assert prof[-1] > prof[0] > prof[1], "lag profile is not monotone away from -1"

    # NLL context only
    p = dense["p"]
    lvl = c.astype(np.int64).clip(0, p.shape[3] - 1)
    print("  mean -log P[.., t][c_{t+k}] by lag k (context, not a test):")
    for k in (-1, 0, 1):
        sl = slice(2 + k, n_rounds - 2 + k)
        m = ok[:, :, core] & ok[:, :, sl]
        pr = np.take_along_axis(p[:, :, core], lvl[:, :, sl][..., None], axis=3)[..., 0]
        print(f"    k={k:+d}  {f(-np.log(np.clip(pr[m], 1e-12, None)).mean())}")

    # (4) the slope
    pop = rcb_population(df)
    slope_pred = ols_slope(pop["c_t"].to_numpy(), pop["e_next"].to_numpy())
    slope_mis = ols_slope(pop["c_t"].to_numpy(), pop["e_self"].to_numpy())
    slope_obs = ols_slope(pop["c_t"].to_numpy(), pop["c_next"].to_numpy())
    corr = float(np.corrcoef(pop["c_t"], pop["e_next"])[0, 1])
    print(f"  OLS slope E[c_t+1] ~ c_t (RCB pop)   {f(slope_pred)}  <- aligned")
    print(f"  OLS slope E[c_t]   ~ c_t (RCB pop)   {f(slope_mis)}  <- off by one")
    print(f"  OLS slope c_t+1    ~ c_t (RCB pop)   {f(slope_obs)}  <- human")
    print(f"  corr(c_t, E[c_t+1])                  {f(corr)}")
    assert slope_pred > 0.5, f"E[c_t+1] not sticky in c_t: {slope_pred}"
    assert corr > 0.5, f"E[c_t+1] not correlated with c_t: {corr}"


# --------------------------------------------------------------------------- #
# the two tables
# --------------------------------------------------------------------------- #
def rcb_population(df):
    """The RCB population, built exactly as the frozen metric builds it:
    punished (punishment > 0, and not a manager timeout -> the canonical
    frame's punishment is NaN there), non-full contributors (< 20), with a
    valid next-round contribution (already imposed when the frame was built).
    """
    pop = df[df["p_valid"] & (df["p_t"] > 0) & (df["c_t"] < 20)].copy()
    pop["band"] = pd.cut(pop["c_t"], BAND_EDGES, labels=BAND_LABELS)
    rate = pop["p_t"] / (20.0 - pop["c_t"])
    pop["rate_bin"] = pd.cut(rate, RATE_EDGES, labels=RATE_LABELS)
    return pop


def ols_slope(x, y):
    """Slope of the least-squares line y ~ a + b x."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom == 0.0:
        return float("nan")
    return float((xc * (y - y.mean())).sum() / denom)


def band_slopes(pop, col):
    """(A): OLS slope of `col` on the punishment received, within band."""
    out = []
    for band in BAND_LABELS:
        sel = pop[pop["band"] == band]
        out.append(
            {
                "band": band,
                "n": int(len(sel)),
                "slope": ols_slope(sel["p_t"], sel[col]),
                "mean_dc": float(sel[col].mean()) if len(sel) else float("nan"),
                "mean_p": float(sel["p_t"].mean()) if len(sel) else float("nan"),
            }
        )
    return pd.DataFrame(out)


def bin_means(pop, col):
    """(B): the RCB statistic -- mean `col` per punishment-rate bin, plus the
    human-count-weighted mean absolute discrepancy against the human means."""
    g = pop.groupby("rate_bin", observed=False)[col]
    means = g.mean().reindex(RATE_LABELS).to_numpy(dtype=np.float64)
    counts = g.size().reindex(RATE_LABELS).to_numpy(dtype=np.float64)
    disc = np.abs(means - HUMAN_BIN_MEANS)
    weighted = float((HUMAN_BIN_COUNTS * disc).sum() / HUMAN_BIN_COUNTS.sum())
    tab = pd.DataFrame(
        {
            "rate_bin": RATE_LABELS,
            "n": counts,
            "mean_dc": means,
            "human_mean": HUMAN_BIN_MEANS,
            "abs_disc": disc,
            "human_weight": HUMAN_BIN_COUNTS / HUMAN_BIN_COUNTS.sum(),
        }
    )
    return tab, weighted


def show(tab):
    print(tab.to_string(index=False, float_format=lambda v: f"{v: .10f}"))


def report(pop, label, observed="HUMAN"):
    print(f"\n================ {label} (RCB population n={len(pop)}) ============")
    for col, name in (
        ("dc_model", "(A/B) MODEL, teacher-forced E[c_t+1] - c_t"),
        ("dc_human", f"(C) {observed}, observed c_t+1 - c_t"),
        ("dc_misaligned", "[diagnostic] MISALIGNED E[c_t] - c_t"),
    ):
        print(f"\n--- {name} ---")
        slopes = band_slopes(pop, col)
        slopes["human_slope"] = [HUMAN_SLOPES[b] for b in slopes["band"]]
        slopes["ratio_to_human"] = slopes["slope"] / slopes["human_slope"]
        print("(A) within-contribution-band slope of dc on punishment:")
        show(slopes)
        tab, weighted = bin_means(pop, col)
        print("(B) RCB bin means:")
        show(tab)
        print(f"    human-weighted mean abs discrepancy = {f(weighted)}")


def selfcheck(pop, label):
    """(C) must reproduce the declaration's human numbers to ~2 decimals."""
    ok = True
    slopes = band_slopes(pop, "dc_human").set_index("band")["slope"]
    for band, ref in HUMAN_SLOPES.items():
        got = float(slopes[band])
        if abs(got - ref) > 5e-3:
            print(f"!! SELF-CHECK FAIL {label} slope {band}: {f(got)} vs {ref}")
            ok = False
    tab, weighted = bin_means(pop, "dc_human")
    if weighted > 5e-3:
        print(f"!! SELF-CHECK FAIL {label} bin means: weighted gap {f(weighted)}")
        ok = False
    n_ref = int(HUMAN_BIN_COUNTS.sum())
    if len(pop) != n_ref:
        print(f"!! SELF-CHECK NOTE {label}: n={len(pop)} vs declaration {n_ref}")
    print(f"\nSELF-CHECK {label}: {'PASS' if ok else 'FAIL'}")
    return ok


def selfcheck_sim(pop, label):
    """Sim mode's (C): the OBSERVED column must reproduce the parent's own
    closed-loop numbers, which is the proof that this population is the
    evaluation suite's RCB population rebuilt from the same parquet."""
    ok = True
    slopes = band_slopes(pop, "dc_human").set_index("band")["slope"]
    for band, ref in SIM_REF_SLOPES.items():
        got = float(slopes[band])
        print(f"  observed slope {band:>5}: {f(got)}  vs recorded sim {ref}")
        if abs(got - ref) > 5e-3:
            print(f"!! SELF-CHECK FAIL {label} slope {band}: {f(got)} vs {ref}")
            ok = False
    _, weighted = bin_means(pop, "dc_human")
    print(f"  observed weighted discrepancy: {f(weighted)} vs {f(SIM_REF_STAT)}")
    if abs(weighted - SIM_REF_STAT) > 5e-3:
        print(f"!! SELF-CHECK FAIL {label} statistic: {f(weighted)}")
        ok = False
    print(f"\nSELF-CHECK {label}: {'PASS' if ok else 'FAIL'}")
    return ok


def sampling_noise(pop):
    """How far the OBSERVED closed-loop numbers can sit from their own
    conditional expectation by draw luck alone.

    The teacher-forced column IS E[dc | state]; the observed column is one
    realisation of it. Their difference is the sampling residual
    c_{t+1} - E[c_{t+1} | history], whose variance the model states directly
    (`v_next`). Along rounds those residuals are a martingale difference
    sequence -- each is mean zero given everything before it -- so they are
    uncorrelated, and the diagonal sum below is exact for that component.
    Within one (episode, round, group) cell the copula correlates them, which
    this ignores: the numbers are therefore a LOWER BOUND on the true spread,
    and |z| is an upper bound.
    """
    print("\n--- sampling noise: observed vs its own conditional expectation ---")
    rows = []
    for band in BAND_LABELS:
        sel = pop[pop["band"] == band]
        pv = sel["p_t"].to_numpy(dtype=np.float64)
        w = pv - pv.mean()
        ss = float((w * w).sum())
        var = float(((w / ss) ** 2 * sel["v_next"].to_numpy()).sum())
        obs = ols_slope(pv, sel["dc_human"].to_numpy())
        mod = ols_slope(pv, sel["dc_model"].to_numpy())
        se = float(np.sqrt(var))
        rows.append(
            {
                "band": band,
                "n": int(len(sel)),
                "slope_model": mod,
                "slope_obs": obs,
                "diff": obs - mod,
                "se_indep": se,
                "z": (obs - mod) / se if se > 0 else float("nan"),
            }
        )
    print("(A) within-band slopes:")
    show(pd.DataFrame(rows))

    rows = []
    for lab in RATE_LABELS:
        sel = pop[pop["rate_bin"] == lab]
        n = len(sel)
        se = float(np.sqrt(sel["v_next"].sum())) / n if n else float("nan")
        obs = float(sel["dc_human"].mean())
        mod = float(sel["dc_model"].mean())
        rows.append(
            {
                "rate_bin": lab,
                "n": n,
                "mean_model": mod,
                "mean_obs": obs,
                "diff": obs - mod,
                "se_indep": se,
                "z": (obs - mod) / se if se > 0 else float("nan"),
            }
        )
    print("(B) RCB bin means:")
    show(pd.DataFrame(rows))


def run_sim_mode(model, sim_path, defaults):
    """Teacher-force `model` over a simulation's realised trajectories."""
    print(f"\nsim data  {cc.rel(sim_path)}")
    check_sim_defaults(model, defaults)
    data, run = load_sim(sim_path, defaults)
    n_ep = data["contribution"].shape[0]
    label = f"sim {Path(sim_path).parent.name}"
    idx = np.arange(n_ep, dtype=np.int64)
    df, dense = stimulus_frame(model, data, idx, label)
    check_alignment(model, data, df, dense, label)
    pop = rcb_population(df)
    report(pop, label, observed="SIM")
    sampling_noise(pop)
    print("\n--- (C) self-check: observed sim vs the recorded closed loop ---")
    return selfcheck_sim(pop, label)


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(ROOT / DEFAULT_MODEL))
    ap.add_argument(
        "--sim-parquet",
        default=None,
        help=(
            "teacher-force the trunk over a simulation's realised rounds "
            "instead of the human CSV (step 6's mechanism gate)"
        ),
    )
    args = ap.parse_args()
    in_path = Path(args.model).resolve()

    model = GraphNetwork.load(str(in_path), device="cpu")
    model.eval()
    assert model.y_name == "contribution", f"not a contributor: {model.y_name}"
    print(f"model     {cc.rel(in_path)}")
    print(f"  y_name={model.y_name} y_levels={model.y_levels}")
    print(f"  x_encoding={[e['name'] for e in model.x_encoding]}")
    print(f"  copula_rho={model.copula_rho} copula_phi={model.copula_phi}")

    data, pair_id, key_to_idx, defaults = cc.load_full()
    if args.sim_parquet is not None:
        ok = run_sim_mode(model, Path(args.sim_parquet).resolve(), defaults)
        print(f"\nwall {time.time() - t0:.1f}s")
        if not ok:
            print("!! (C) does not reproduce the recorded sim -- INVALID")
            sys.exit(2)
        return
    tr_idx = cc.select_split(key_to_idx, cc.TRAIN, cc.N_TRAIN_EP)
    te_idx = cc.select_split(key_to_idx, cc.TEST, cc.N_TEST_EP)
    assert not set(tr_idx.tolist()) & set(te_idx.tolist()), "train/test overlap"
    assert not set(pair_id[tr_idx].tolist()) & set(
        pair_id[te_idx].tolist()
    ), "a flip copy of a train game sits in the holdout"
    both_idx = np.array(sorted(set(tr_idx.tolist()) | set(te_idx.tolist())))
    print(f"data      {cc.rel(cc.FULL)} (episodes={len(key_to_idx)})")
    print(f"  train {len(tr_idx)} ep, test {len(te_idx)} ep, union {len(both_idx)} ep")
    print(f"  contribution default={f(defaults['contribution'])}")

    frames, denses = {}, {}
    for label, idx in (
        ("train split (40 ep)", tr_idx),
        ("held-out split (10 ep)", te_idx),
        ("train+test, single copy (50 ep)", both_idx),
    ):
        frames[label], denses[label] = stimulus_frame(model, data, idx, label)

    union = "train+test, single copy (50 ep)"
    check_alignment(model, data, frames[union], denses[union], union)

    ok = True
    for label, df in frames.items():
        pop = rcb_population(df)
        report(pop, label)
        if label == union:
            ok = selfcheck(pop, label)

    print(f"\nwall {time.time() - t0:.1f}s")
    if not ok:
        print("!! (C) does not reproduce the declaration -- comparison INVALID")
        sys.exit(2)


if __name__ == "__main__":
    main()
