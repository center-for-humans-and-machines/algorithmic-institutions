"""Does the artificial contributor's response to punishment depend on what it
contributed?  An INTERVENTIONAL probe of the frontier contributor.

Why not read it off the data.  RCE -- the OLS slope of the next-round
contribution change on the punishment received, within contribution band -- is
an observational quantity.  In every trajectory the punishment an agent gets is
chosen by a manager that already looked at the contribution, so a within-band
slope mixes the causal dose response with whatever the manager conditioned on.
The question this script answers is the causal one, because it is the one an RL
manager's gradient actually sees: hold a real state fixed, FORCE a contribution
and FORCE a punishment, and read the model's next-round contribution.

The design (open loop, teacher forced, one round):

  * contexts are real human states: an (episode, focal agent, round t*) triple
    from the canonical 50 single-copy episodes, with the focal agent's own
    round-t* cell recorded (not a timeout) so the state we perturb is one the
    game really produced;
  * the intervention sets the focal agent's round-t* contribution to c and
    punishment to p -- both the round-t* slot and the ``prev_*[t*+1]`` slot the
    model reads, exactly as ``simulation/intervention_probe`` does, with
    ``common_good[t*]`` recomputed from the overridden values and propagated;
  * the outcome is the model's own conditional mean,
    ``E[c_{t*+1}] = sum_k k * P[t*+1, k]``, and the response is
    ``delta(c, p) = E[c_{t*+1}] - c``;
  * every context sees every grid cell, so the surface is a PAIRED comparison:
    context heterogeneity is common to all cells and cancels in the slopes.

SHARED NOISE IS OFF, AND CANNOT MATTER HERE.  ``predict_independent(sample=
False)`` returns the predicted marginal and draws nothing, so no RNG is
consumed and the herding copula is never entered.  The copula is an
inverse-CDF sampler (``generic/copula.py``): it correlates draws ACROSS agents
of a cell while leaving each row's marginal exactly as predicted, so the
quantity measured here is invariant to rho and phi by construction.  With
``--verify-copula`` that invariance is measured rather than asserted: a subgrid
is redrawn with the copula on and the sampled cell means are compared against
the analytic ones.

THE IMPUTED 9.  A player who gave no input is stored with the median
contribution (9) and ``contribution_valid = False``.  Those cells are excluded
from the human reference population and from the context set; they are never
averaged in.  The model's outcome is a predicted distribution over valid
contributions (the trunk trained with ``mask_name: contribution_valid``), so no
imputed value enters the response either.

Outputs (``--output-dir``):
  * ``intervention_surface.csv``  -- one row per (c, p) grid cell: the mean
    forced response, its paired standard error, and the human cell count;
  * ``band_slopes.csv``           -- the RCE analogue per contribution band,
    human-weighted and uniform, beside the human slopes and a context
    bootstrap CI;
  * ``level_slopes.csv``          -- the same slope at one-point resolution,
    one row per forced contribution level INCLUDING the ceiling c = 20 that
    RCE drops, beside the human slope at that level;
  * ``context_surface.npz``       -- the per-context responses, for the
    bootstrap and for any re-weighting (not committed; regenerable);
  * ``summary.json``              -- the headline numbers.

Imports graph.py, so this runs on Raven only:
    .venv/bin/python scripts/data_analysis/contributor_punishment_intervention.py
"""

import argparse
import json
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

# the copula calibration's loaders, imported unmodified (its top level is
# imports, constants and defs only).  It also installs the
# torch_geometric.nn.meta alias the legacy pickles need.
import contribution_copula_rho as cc  # noqa: E402

from aimanager.generic.graph import GraphNetwork  # noqa: E402

DEFAULT_MODEL = (
    "artifacts/artificial_humans/"
    "group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/"
    "model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
)

MASK = "contribution_valid"
PMASK = "punishment_valid"

BAND_EDGES = [-0.5, 4.5, 9.5, 14.5, 19.5]
BAND_LABELS = ["0-4", "5-9", "10-14", "15-19"]

# the human RCE slopes and the human-vs-human noise ceiling of the RCE score,
# from notes/evaluation_metric_defs.md and notes/autoresearch.md
HUMAN_SLOPES = {"0-4": 0.1397, "5-9": 0.1038, "10-14": -0.0767, "15-19": -0.1615}
RCE_NOISE_CEILING = 0.0860

# per-band seed-to-seed sd of the closed-loop RCE slopes, six contributor
# training seeds, simulation draw held fixed
# (notes/autoresearch_log/seed-spread-noise-floor.md)
SEED_SD = {"0-4": 0.0182, "5-9": 0.0223, "10-14": 0.0263, "15-19": 0.0564}

# features a model must NOT read, because the override cannot be propagated
# into them faithfully from a single forced cell
UNPROPAGATED = {"own_grp_prev_mean_contr", "prev_own_grp_prev_mean_contr"}


def f(x):
    """Unrounded float for the log."""
    return repr(float(x))


def ols_slope(x, y, w=None):
    """Slope of the weighted least-squares line y ~ a + b x."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=np.float64)
    sw = w.sum()
    if sw <= 0:
        return float("nan")
    xc = x - (w * x).sum() / sw
    denom = float((w * xc * xc).sum())
    if denom == 0.0:
        return float("nan")
    return float((w * xc * (y - (w * y).sum() / sw)).sum() / denom)


def band_of(c):
    return pd.cut(np.asarray(c, dtype=float), BAND_EDGES, labels=BAND_LABELS)


# --------------------------------------------------------------------------- #
# the human reference: the RCE population and its (band, c, p) weights
# --------------------------------------------------------------------------- #
def human_frame(data, idx):
    """One row per (episode, agent, stimulus round t) of the selected episodes
    with a recorded contribution at t and t+1 and a recorded punishment at t --
    the canonical RCE population's parent frame, rebuilt from the same tensors
    the probe perturbs so the two are indexed identically."""
    sub = {k: v[th.as_tensor(idx)] for k, v in data.items()}
    n_rounds = sub["contribution"].shape[2]
    contr = sub["contribution"].numpy().astype(np.float64)
    punish = sub["punishment"].numpy().astype(np.float64)
    c_ok = sub[MASK].numpy().astype(bool)
    p_ok = sub[PMASK].numpy().astype(bool)
    t, t1 = slice(0, n_rounds - 1), slice(1, n_rounds)
    valid = c_ok[:, :, t] & c_ok[:, :, t1] & p_ok[:, :, t]
    e, a, r = np.nonzero(valid)
    df = pd.DataFrame(
        {
            "episode": e,
            "agent": a,
            "round": r,
            "c_t": contr[:, :, t][valid],
            "p_t": punish[:, :, t][valid],
            "c_next": contr[:, :, t1][valid],
        }
    )
    df["dc"] = df["c_next"] - df["c_t"]
    return df


def rce_population(df):
    """Punished (p > 0), non-full (c < 20), valid dc -- the frozen definition
    of `metrics._rce_population`, minus the rate bin it does not use."""
    pop = df[(df["p_t"] > 0) & (df["c_t"] < 20)].copy()
    pop["band"] = band_of(pop["c_t"])
    return pop


def human_selfcheck(pop):
    """The rebuilt population must reproduce the recorded human slopes."""
    print(f"\n--- human reference (RCE population n={len(pop)}) ---")
    ok = True
    for band in BAND_LABELS:
        sel = pop[pop["band"] == band]
        got = ols_slope(sel["p_t"], sel["dc"])
        ref = HUMAN_SLOPES[band]
        flag = "" if abs(got - ref) <= 5e-3 else "   !! MISMATCH"
        ok = ok and not flag
        print(f"  {band:>5}  n={len(sel):5d}  slope={f(got)}  recorded={ref}{flag}")
    spread = ols_slope(
        pop[pop["band"] == "0-4"]["p_t"], pop[pop["band"] == "0-4"]["dc"]
    ) - ols_slope(pop[pop["band"] == "15-19"]["p_t"], pop[pop["band"] == "15-19"]["dc"])
    print(f"  targeting gradient (0-4 minus 15-19) = {f(spread)}")
    print(f"SELF-CHECK human slopes: {'PASS' if ok else 'FAIL'}")
    return ok, float(spread)


# --------------------------------------------------------------------------- #
# the intervention
# --------------------------------------------------------------------------- #
def check_model(model):
    """Every feature the model reads must be one the forced cell propagates
    into, and no current-round contribution or punishment may be read (the
    probe's whole alignment rests on the lag)."""
    used = []
    for attr in ("x_encoding", "u_encoding", "edge_encoding", "b_encoding"):
        enc = getattr(model, attr, None) or []
        names = [e.get("name", e.get("etype")) for e in enc]
        print(f"  {attr} = {names}")
        used += [n for n in names if n]
    illegal = [n for n in used if n in ("contribution", "punishment")]
    assert not illegal, f"model reads the current round: {illegal}"
    bad = sorted(set(used) & UNPROPAGATED)
    assert not bad, f"model reads a feature the override cannot propagate: {bad}"
    print("  no current-round feature, no unpropagated feature              OK")


def select_contexts(data, idx, rounds, rng, n_max):
    """(episode, agent, t*) triples whose focal round-t* cell is recorded."""
    sel = th.as_tensor(idx)
    sub_c = data[MASK][sel].numpy().astype(bool)
    sub_p = data[PMASK][sel].numpy().astype(bool)
    nat_c = data["contribution"][sel].numpy()
    nat_p = data["punishment"][sel].numpy()
    rows = []
    for t in rounds:
        e, a = np.nonzero(sub_c[:, :, t] & sub_p[:, :, t])
        rows.append(
            pd.DataFrame(
                {
                    "row": idx[e],
                    "agent": a,
                    "t_star": t,
                    "c_nat": nat_c[e, a, t],
                    "p_nat": nat_p[e, a, t],
                }
            )
        )
    ctx = pd.concat(rows, ignore_index=True)
    if n_max is not None and len(ctx) > n_max:
        take = rng.choice(len(ctx), size=n_max, replace=False)
        ctx = ctx.iloc[np.sort(take)].reset_index(drop=True)
    return ctx


def _edge_cache(model, n_nodes):
    cache = {}

    def get(n_batch):
        if n_batch not in cache:
            cache[n_batch] = model.create_fully_connected(n_nodes, n_batch=n_batch)
        return cache[n_batch]

    return get


def _override(sub, rows, agents, t_star, c, p):
    """Force the focal agent's round-t* cell, in both the slot the model reads
    (``prev_*[t*+1]``) and the slot the pool is recomputed from."""
    for name, val in (("contribution", c), ("punishment", p)):
        sub[name][rows, agents, t_star] = val
        sub[f"{name}_valid"][rows, agents, t_star] = True
        sub[f"prev_{name}"][rows, agents, t_star + 1] = val
        sub[f"prev_{name}_valid"][rows, agents, t_star + 1] = True
    sub["contribution_max"] = sub["contribution"] == 20
    _recompute_common_good(sub, t_star)


def forced_response(model, data, ctx, c_grid, p_grid, chunk, get_edges):
    """delta(c, p) per context: E[c_{t*+1} | do(c_t = c, p_t = p)] - c.

    Returns a float32 array (n_contexts, n_cells) in the row-major order of
    ``[(c, p) for c in c_grid for p in p_grid]``, and that cell list.  Every
    (context, cell) pair is its own batch element, so one forward covers many
    cells and the grid costs no more kernel launches than it has to.
    """
    device = model.device
    cells = [(c, p) for c in c_grid for p in p_grid]
    out = np.full((len(ctx), len(cells)), np.nan, dtype=np.float32)
    levels = th.arange(model.y_levels, dtype=th.float64, device=device)
    cell_c = th.tensor([c for c, _ in cells], dtype=th.int64, device=device)
    cell_p = th.tensor([p for _, p in cells], dtype=th.int64, device=device)
    t0, done, total = time.time(), 0, len(ctx) * len(cells)
    for t_star, grp in ctx.groupby("t_star"):
        t_star = int(t_star)
        pos = grp.index.to_numpy()
        eps = th.as_tensor(grp["row"].to_numpy())
        ags = th.as_tensor(grp["agent"].to_numpy(), device=device)
        T = t_star + 2
        base = {k: v[eps][:, :, :T].to(device) for k, v in data.items()}
        n_ctx = len(grp)
        # the (context, cell) cross product, flattened
        pair_ctx = th.arange(n_ctx, device=device).repeat_interleave(len(cells))
        pair_cell = th.arange(len(cells), device=device).repeat(n_ctx)
        for s in range(0, len(pair_ctx), chunk):
            sl = slice(s, min(s + chunk, len(pair_ctx)))
            pc, pk = pair_ctx[sl], pair_cell[sl]
            sub = {k: v[pc].clone() for k, v in base.items()}
            n_b = sub["contribution"].shape[0]
            rows = th.arange(n_b, device=device)
            _override(sub, rows, ags[pc], t_star, cell_c[pk], cell_p[pk])
            with th.no_grad():
                _, proba = model.predict_independent(
                    sub, sample=False, reset_rnn=True, edge_index=get_edges(n_b)
                )
            pr = proba[rows, ags[pc], t_star + 1].double()
            dc = (pr * levels).sum(-1) - cell_c[pk].double()
            out[pos[pc.cpu().numpy()], pk.cpu().numpy()] = (
                dc.cpu().numpy().astype(np.float32)
            )
            done += n_b
            if done % (50 * chunk) < chunk:
                el = time.time() - t0
                print(
                    f"  [{done}/{total}] {el:.0f}s elapsed, "
                    f"{el / max(done, 1) * (total - done):.0f}s left",
                    flush=True,
                )
    assert np.isfinite(out).all(), "some (context, cell) was never filled"
    return out, cells


def _recompute_common_good(sub, t_star):
    """The pool at t* with the overridden values, propagated to
    ``prev_common_good[t*+1]``.  ``parse_agent_rounds`` stores the PER-CAPITA
    pool (divided by the valid count), so the same convention is used here."""
    ag = sub["agent_group"][:, :, t_star]
    c_t = sub["contribution"][:, :, t_star].float()
    p_t = sub["punishment"][:, :, t_star].float()
    ok = sub["contribution_valid"][:, :, t_star].float()
    new = th.zeros_like(sub["common_good"][:, :, t_star])
    for g in ag.unique().tolist():
        in_g = ag == g
        n_valid = (ok * in_g).sum(-1, keepdim=True).clamp(min=1)
        pool = ((1.6 * c_t - p_t) * in_g).sum(-1, keepdim=True)
        new = th.where(in_g, (pool / n_valid).expand_as(new), new)
    sub["common_good"][:, :, t_star] = new
    sub["prev_common_good"][:, :, t_star + 1] = new


# --------------------------------------------------------------------------- #
# the RCE analogue
# --------------------------------------------------------------------------- #
def cell_weights(pop, cells):
    """Human counts per (c, p) cell, in the order of ``cells``."""
    counts = pop.groupby([pop["c_t"].astype(int), pop["p_t"].astype(int)]).size()
    return np.array([float(counts.get((c, p), 0.0)) for c, p in cells])


def band_slopes(surface, cells, weights):
    """The RCE analogue: per band, the weighted OLS slope of the forced
    response on the forced punishment.  Weighting by the human cell counts
    makes this the human slope's exact interventional twin -- same cells, same
    composition, only the response replaced."""
    cs = np.array([c for c, _ in cells], dtype=float)
    ps = np.array([p for _, p in cells], dtype=float)
    bands = np.asarray(band_of(cs))
    out = {}
    for band in BAND_LABELS:
        m = (bands == band) & (ps > 0) & (weights > 0)
        out[band] = ols_slope(ps[m], surface[m], weights[m])
    return out


def _fe_slope(c, p, y, w):
    """Weighted OLS slope of y on p with one intercept PER CONTRIBUTION LEVEL.

    RCE pools the levels inside a band and fits one line, so its slope also
    absorbs the between-level gradient: dc falls by about three points across
    the 15-19 band, and if the punishment humans gave differs across those
    five levels the pooled slope picks that up as if it were a dose response.
    Demeaning p and y within level removes it and leaves the dose response
    proper.  Reported alongside the frozen pooled statistic, never instead of
    it -- the pooled one is what the suite scores.
    """
    c, p, y, w = (np.asarray(v, dtype=np.float64) for v in (c, p, y, w))
    num = den = 0.0
    for lvl in np.unique(c):
        m = (c == lvl) & (w > 0)
        if m.sum() < 2 or w[m].sum() <= 0:
            continue
        ww, pp, yy = w[m], p[m], y[m]
        pc = pp - (ww * pp).sum() / ww.sum()
        yc = yy - (ww * yy).sum() / ww.sum()
        num += float((ww * pc * yc).sum())
        den += float((ww * pc * pc).sum())
    return num / den if den > 0 else float("nan")


def fe_band_slopes(surface, cells, weights):
    """The model's within-level dose response per band, human cell weights."""
    cs = np.array([c for c, _ in cells], dtype=float)
    ps = np.array([p for _, p in cells], dtype=float)
    bands = np.asarray(band_of(cs))
    out = {}
    for band in BAND_LABELS:
        m = (bands == band) & (ps > 0)
        out[band] = _fe_slope(cs[m], ps[m], surface[m], weights[m])
    return out


def human_fe_band_slopes(pop):
    """The same within-level dose response on the human rows."""
    out = {}
    for band in BAND_LABELS:
        sel = pop[pop["band"] == band]
        out[band] = _fe_slope(sel["c_t"], sel["p_t"], sel["dc"], np.ones(len(sel)))
    return out


def uniform_band_slopes(surface, cells, p_max):
    """The same slope over the grid itself, every cell counted once, up to
    ``p_max``: what the model does over the dose range regardless of how often
    humans were dosed there."""
    cs = np.array([c for c, _ in cells], dtype=float)
    ps = np.array([p for _, p in cells], dtype=float)
    bands = np.asarray(band_of(cs))
    out = {}
    for band in BAND_LABELS:
        m = (bands == band) & (ps > 0) & (ps <= p_max)
        out[band] = ols_slope(ps[m], surface[m])
    return out


def natural_cell_check(model, data, idx, ctx, ctx_surface, cells, c_max, p_max):
    """Forcing a context's OWN recorded cell must be a no-op.

    The probe's whole claim rests on the override machinery writing the forced
    pair into exactly the slots the model reads and nothing else.  That is
    checkable without argument: set (c, p) to what the round really was, and
    the forced response has to equal the plain teacher-forced prediction for
    that row -- a different forward, over the full 24 rounds instead of the
    probe's truncated prefix, through a different code path.
    """
    # cc.teacher_forced_rows returns numpy, so it needs the model on the cpu;
    # moving it there and back also proves the check runs the weights, not a
    # cached device tensor
    dev = model.device
    model.to("cpu")
    model.device = "cpu"
    try:
        tf = cc.teacher_forced_rows(model, data, idx)
    finally:
        model.to(dev)
        model.device = dev
    n_ep, n_agents, n_rounds = tf["shape"]
    levels = np.arange(tf["P"].shape[1], dtype=np.float64)
    e = np.full((n_ep, n_agents, n_rounds), np.nan)
    e[tf["episode"], tf["agent"], tf["round"]] = tf["P"] @ levels
    pos = {int(v): i for i, v in enumerate(idx)}
    cell_of = {(c, p): i for i, (c, p) in enumerate(cells)}
    got, want = [], []
    for i, row in ctx.iterrows():
        c, p = int(row["c_nat"]), int(row["p_nat"])
        if c > c_max or p > p_max:
            continue
        ref = e[pos[int(row["row"])], int(row["agent"]), int(row["t_star"]) + 1]
        if not np.isfinite(ref):
            continue  # the focal's own round t*+1 was a timeout: no target
        got.append(float(ctx_surface[i, cell_of[(c, p)]]))
        want.append(float(ref) - c)
    got, want = np.array(got), np.array(want)
    d = np.abs(got - want)
    err, mean_err = (float(d.max()), float(d.mean())) if len(d) else (np.nan, np.nan)
    print(
        f"\n--- no-op check: forcing the recorded cell ({len(got)} contexts) ---\n"
        f"  max  |forced - teacher forced| = {f(err)}\n"
        f"  mean |forced - teacher forced| = {f(mean_err)}"
    )
    # The two sides are the same weights through different arithmetic: the
    # probe runs a t*+2-round prefix on the gpu, the reference runs all 24
    # rounds on the cpu, both in float32. The tolerance is set for that, not
    # for a behavioural difference -- 1e-2 of a contribution point is 5e-4 of
    # the 0-20 scale and two orders below the smallest effect read off the
    # surface.
    assert len(got) > 100, "too few contexts survived the no-op check"
    assert err < 1e-2, f"forcing the recorded cell is not a no-op: {err}"
    print("  the override writes only what the model reads             OK")
    return err, int(len(got))


def level_slopes(surface, cells, weights, human, p_max):
    """One row per forced contribution level, including the ceiling c = 20 that
    RCE drops (its rate is undefined there, not its slope).  This is the
    targeting gradient at one-point resolution: what one more point of
    punishment buys the manager at each contribution level."""
    cs = np.array([c for c, _ in cells], dtype=float)
    ps = np.array([p for _, p in cells], dtype=float)
    lookup = {(int(c), int(p)): surface[i] for i, (c, p) in enumerate(cells)}
    rows = []
    for c in sorted({int(v) for v in cs}):
        m = (cs == c) & (ps > 0) & (ps <= p_max)
        mw = m & (weights > 0)
        hsel = human[human["c_t"] == c]
        hpun = hsel[hsel["p_t"] > 0]
        rows.append(
            {
                "contribution": c,
                "model_dc_at_p0": lookup[(c, 0)],
                "model_step_p0_to_p1": lookup[(c, 1)] - lookup[(c, 0)],
                "model_slope_uniform": ols_slope(ps[m], surface[m]),
                "model_slope_human_weighted": (
                    ols_slope(ps[mw], surface[mw], weights[mw])
                    if mw.sum() >= 2
                    else float("nan")
                ),
                "human_n_punished": int(len(hpun)),
                "human_slope": (
                    ols_slope(hpun["p_t"], hpun["dc"])
                    if len(hpun) >= 10
                    else float("nan")
                ),
                "human_dc_unpunished": (
                    float(hsel[hsel["p_t"] == 0]["dc"].mean())
                    if (hsel["p_t"] == 0).sum()
                    else float("nan")
                ),
                "human_dc_punished": (
                    float(hpun["dc"].mean()) if len(hpun) else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def ceiling_contrast(surface, cells, human):
    """The ceiling case RCE drops and the inverted managers aim at: c = 20.

    Model: the forced response at p = 0 against the human punishment
    distribution over full contributors, so the contrast is the same
    punished-minus-unpunished statistic RCC scores.  Human: RCC itself."""
    lookup = {(int(c), int(p)): surface[i] for i, (c, p) in enumerate(cells)}
    full = human[human["c_t"] == 20]
    pun = full[full["p_t"] > 0]
    w = pun.groupby(pun["p_t"].astype(int)).size()
    if w.sum() == 0:
        return {}
    model_pun = float(sum(w[p] * lookup[(20, int(p))] for p in w.index) / w.sum())
    return {
        "human_n_full": int(len(full)),
        "human_n_full_punished": int(len(pun)),
        "human_rcc_contrast": float(
            pun["dc"].mean() - full[full["p_t"] == 0]["dc"].mean()
        ),
        "model_rcc_contrast": model_pun - lookup[(20, 0)],
        "model_slope_at_20_human_weighted": ols_slope(
            pun["p_t"].to_numpy(),
            np.array([lookup[(20, int(p))] for p in pun["p_t"]]),
        ),
        "human_slope_at_20": ols_slope(pun["p_t"], pun["dc"]),
        "mean_punishment_on_full_contributors": float(pun["p_t"].mean()),
    }


def near_support_slopes(ctx_surface, ctx, cells, weights):
    """The same band slopes, but each band averaged only over the contexts
    whose OWN round-t* contribution fell in that band -- the near-support
    variant, which never asks the model what a habitual free-rider would do
    after suddenly giving 20."""
    out = {}
    nat = np.asarray(band_of(ctx["c_nat"].to_numpy()))
    for band in BAND_LABELS:
        m = nat == band
        if m.sum() < 20:
            out[band] = float("nan")
            continue
        s = ctx_surface[m].mean(0).astype(np.float64)
        out[band] = band_slopes(s, cells, weights)[band]
    return out


def bootstrap(ctx_surface, cells, weights, n_boot, seed):
    """Resample CONTEXTS (the only random ingredient: the model is
    deterministic and the grid is fixed) and recompute the band slopes."""
    rng = np.random.default_rng(seed)
    n = ctx_surface.shape[0]
    draws = {b: [] for b in BAND_LABELS}
    grads = []
    for _ in range(n_boot):
        take = rng.integers(0, n, size=n)
        s = ctx_surface[take].mean(0).astype(np.float64)
        sl = band_slopes(s, cells, weights)
        for b in BAND_LABELS:
            draws[b].append(sl[b])
        grads.append(sl["0-4"] - sl["15-19"])
    ci = {
        b: (
            float(np.percentile(draws[b], 2.5)),
            float(np.percentile(draws[b], 97.5)),
            float(np.std(draws[b], ddof=1)),
        )
        for b in BAND_LABELS
    }
    ci["gradient"] = (
        float(np.percentile(grads, 2.5)),
        float(np.percentile(grads, 97.5)),
        float(np.std(grads, ddof=1)),
    )
    return ci


# --------------------------------------------------------------------------- #
# the copula invariance check
# --------------------------------------------------------------------------- #
def verify_copula(model, data, ctx, c_grid, p_grid, get_edges, n_draws, seed):
    """Redraw a subgrid with the sampler the simulation uses (copula ON, the
    model's own rho and phi) and compare the sampled cell means against the
    analytic ones over the same contexts.  The copula is an inverse-CDF
    sampler, so the two must agree to Monte-Carlo error; this measures that
    rather than asserting it."""
    print(
        f"\n--- copula invariance check (rho={f(model.copula_rho)}, "
        f"phi={f(model.copula_phi)}, {n_draws} draws per cell) ---"
    )
    device = model.device
    cells = [(c, p) for c in c_grid for p in p_grid]
    acc = {cell: [] for cell in cells}
    for t_star, grp in ctx.groupby("t_star"):
        t_star = int(t_star)
        eps = th.as_tensor(grp["row"].to_numpy())
        ags = th.as_tensor(grp["agent"].to_numpy(), device=device)
        base = {k: v[eps][:, :, : t_star + 2].to(device) for k, v in data.items()}
        n_b = len(grp)
        rows = th.arange(n_b, device=device)
        for ci, (c, p) in enumerate(cells):
            sub = {k: v.clone() for k, v in base.items()}
            _override(
                sub,
                rows,
                ags,
                t_star,
                th.full((n_b,), c, dtype=th.int64, device=device),
                th.full((n_b,), p, dtype=th.int64, device=device),
            )
            for d in range(n_draws):
                th.manual_seed(seed + 1009 * ci + d)
                with th.no_grad():
                    pred, _ = model.predict_independent(
                        sub, sample=True, reset_rnn=True, edge_index=get_edges(n_b)
                    )
                acc[(c, p)].append(pred[rows, ags, t_star + 1].double().cpu().numpy())
    out = []
    for c, p in cells:
        draws = np.concatenate(acc[(c, p)])
        out.append(
            {
                "c": c,
                "p": p,
                "n_draws": len(draws),
                "sampled_mean_dc": float(draws.mean()) - c,
                "sampled_se": float(draws.std(ddof=1) / np.sqrt(len(draws))),
            }
        )
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(ROOT / DEFAULT_MODEL))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument(
        "--rounds",
        default="5,9,13,17,21",
        help="intervention rounds t* (comma separated); t*+1 must exist",
    )
    ap.add_argument("--n-contexts", type=int, default=2000)
    ap.add_argument("--c-max", type=int, default=20)
    ap.add_argument("--p-max", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify-copula", action="store_true")
    ap.add_argument("--copula-draws", type=int, default=40)
    args = ap.parse_args()

    out_dir = Path(
        args.output_dir or ROOT / "plots/data_analysis/contributor_punishment_targeting"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = GraphNetwork.load(str(Path(args.model).resolve()), device=str(device))
    model.eval()
    assert model.y_name == "contribution", f"not a contributor: {model.y_name}"
    print(f"model     {cc.rel(args.model)}")
    print(f"  device={device} y_levels={model.y_levels}")
    print(f"  copula_rho={f(model.copula_rho)} copula_phi={f(model.copula_phi)}")
    print(f"  group_vnode={model.group_vnode_module is not None}")
    print(f"  stimulus_skip={getattr(model, 'stimulus_skip', False)}")
    check_model(model)

    data, _, key_to_idx, _ = cc.load_full()
    tr = cc.select_split(key_to_idx, cc.TRAIN, cc.N_TRAIN_EP)
    te = cc.select_split(key_to_idx, cc.TEST, cc.N_TEST_EP)
    idx = np.array(sorted(set(tr.tolist()) | set(te.tolist())), dtype=np.int64)
    print(f"data      {cc.rel(cc.FULL)}  single-copy episodes={len(idx)}")

    # structural handle: prev_* really is the t-1 tensor
    for name in ("contribution", "punishment"):
        assert th.equal(
            data[f"prev_{name}"][:, :, 1:], data[name][:, :, :-1]
        ), f"prev_{name} misshifted"
    print("  prev_contribution / prev_punishment are the t-1 tensors        OK")

    human = human_frame(data, idx)
    pop = rce_population(human)
    ok, human_gradient = human_selfcheck(pop)

    rounds = [int(r) for r in args.rounds.split(",")]
    n_rounds = data["contribution"].shape[2]
    assert max(rounds) + 1 < n_rounds, "t*+1 must exist"
    rng = np.random.default_rng(args.seed)
    ctx = select_contexts(data, idx, rounds, rng, args.n_contexts)
    print(
        f"\ncontexts  {len(ctx)} (episode, agent, t*) triples over t*={rounds}, "
        f"all with a recorded round-t* cell"
    )

    c_grid = list(range(0, args.c_max + 1))
    p_grid = list(range(0, args.p_max + 1))
    print(
        f"grid      {len(c_grid)} x {len(p_grid)} = {len(c_grid) * len(p_grid)} cells"
    )
    get_edges = _edge_cache(model, data["contribution"].shape[1])

    ctx_surface, cells = forced_response(
        model, data, ctx, c_grid, p_grid, args.chunk, get_edges
    )
    surface = ctx_surface.mean(0).astype(np.float64)
    se = ctx_surface.std(0, ddof=1).astype(np.float64) / np.sqrt(len(ctx))
    weights = cell_weights(pop, cells)

    np.savez_compressed(
        out_dir / "context_surface.npz",
        surface=ctx_surface,
        c=np.array([c for c, _ in cells], dtype=np.int16),
        p=np.array([p for _, p in cells], dtype=np.int16),
        ctx_row=ctx["row"].to_numpy(),
        ctx_agent=ctx["agent"].to_numpy(),
        ctx_t_star=ctx["t_star"].to_numpy(),
        ctx_c_nat=ctx["c_nat"].to_numpy(),
        ctx_p_nat=ctx["p_nat"].to_numpy(),
    )

    surf = pd.DataFrame(
        {
            "contribution": [c for c, _ in cells],
            "punishment": [p for _, p in cells],
            "band": np.asarray(band_of([c for c, _ in cells])),
            "mean_dc": surface,
            "se_dc": se,
            "human_n": weights,
        }
    )
    surf.to_csv(out_dir / "intervention_surface.csv", index=False)
    print(f"\nwrote {out_dir / 'intervention_surface.csv'}")

    noop_err, noop_n = natural_cell_check(
        model, data, idx, ctx, ctx_surface, cells, args.c_max, args.p_max
    )

    lev = level_slopes(surface, cells, weights, human, args.p_max)
    lev.to_csv(out_dir / "level_slopes.csv", index=False)
    print(f"wrote {out_dir / 'level_slopes.csv'}")
    print("\n--- the targeting gradient at one-point resolution ---")
    print(lev.to_string(index=False, float_format=lambda v: f"{v: .5f}"))

    ceiling = ceiling_contrast(surface, cells, human)
    print("\n--- the ceiling (c = 20), the case RCE drops ---")
    for k, v in ceiling.items():
        print(f"  {k:<38} {f(v)}")

    hw = band_slopes(surface, cells, weights)
    uw = uniform_band_slopes(surface, cells, args.p_max)
    ns = near_support_slopes(ctx_surface, ctx, cells, weights)
    fe = fe_band_slopes(surface, cells, weights)
    hfe = human_fe_band_slopes(pop)
    ci = bootstrap(ctx_surface, cells, weights, args.n_boot, args.seed)

    rows = []
    for band in BAND_LABELS:
        lo, hi, sd = ci[band]
        rows.append(
            {
                "band": band,
                "human_slope": HUMAN_SLOPES[band],
                "model_forced_slope": hw[band],
                "boot_lo": lo,
                "boot_hi": hi,
                "boot_sd": sd,
                "seed_sd": SEED_SD[band],
                "uniform_grid_slope": uw[band],
                "near_support_slope": ns[band],
                "model_within_level_slope": fe[band],
                "human_within_level_slope": hfe[band],
                "abs_gap_to_human": abs(hw[band] - HUMAN_SLOPES[band]),
                "gap_in_noise_ceilings": abs(hw[band] - HUMAN_SLOPES[band])
                / RCE_NOISE_CEILING,
                "share_of_human": hw[band] / HUMAN_SLOPES[band],
                "human_n": float(
                    weights[np.asarray(band_of([c for c, _ in cells])) == band].sum()
                ),
            }
        )
    tab = pd.DataFrame(rows)
    tab.to_csv(out_dir / "band_slopes.csv", index=False)
    print(f"wrote {out_dir / 'band_slopes.csv'}")
    print("\n--- the RCE analogue, forced (human cell weights) ---")
    print(tab.to_string(index=False, float_format=lambda v: f"{v: .6f}"))

    gradient = hw["0-4"] - hw["15-19"]
    g_lo, g_hi, g_sd = ci["gradient"]
    seed_gradient_sd = float(np.hypot(SEED_SD["0-4"], SEED_SD["15-19"]))
    mean_slope = float(np.mean([hw[b] for b in BAND_LABELS]))
    summary = {
        "model": cc.rel(args.model),
        "copula_rho": float(model.copula_rho),
        "copula_phi": float(model.copula_phi),
        "shared_noise": "off (sample=False: the marginal is read, nothing drawn)",
        "n_contexts": int(len(ctx)),
        "rounds": rounds,
        "grid": [len(c_grid), len(p_grid)],
        "human_selfcheck": bool(ok),
        "noop_check_max_abs_error": noop_err,
        "noop_check_n": noop_n,
        "human_slopes": HUMAN_SLOPES,
        "model_forced_slopes": hw,
        "model_uniform_slopes": uw,
        "model_near_support_slopes": ns,
        "model_within_level_slopes": fe,
        "human_within_level_slopes": hfe,
        "within_level_gradient_model": float(fe["0-4"] - fe["15-19"]),
        "within_level_gradient_human": float(hfe["0-4"] - hfe["15-19"]),
        "ceiling": ceiling,
        "human_targeting_gradient": human_gradient,
        "model_targeting_gradient": float(gradient),
        "gradient_boot_ci": [g_lo, g_hi],
        "gradient_boot_sd": g_sd,
        "gradient_seed_sd": seed_gradient_sd,
        "share_of_human_gradient": float(gradient / human_gradient),
        "model_mean_slope": mean_slope,
        "rce_noise_ceiling": RCE_NOISE_CEILING,
        "weighted_abs_gap_to_human": float(
            sum(
                weights[np.asarray(band_of([c for c, _ in cells])) == b].sum()
                * abs(hw[b] - HUMAN_SLOPES[b])
                for b in BAND_LABELS
            )
            / weights[
                np.isin(np.asarray(band_of([c for c, _ in cells])), BAND_LABELS)
            ].sum()
        ),
    }

    print("\n=================== the decisive numbers ===================")
    print(f"  human targeting gradient (0-4 minus 15-19)   {f(human_gradient)}")
    print(f"  model forced gradient                        {f(gradient)}")
    print(f"    bootstrap 95% CI over contexts             [{f(g_lo)}, {f(g_hi)}]")
    print(f"    contributor seed sd of the same contrast   {f(seed_gradient_sd)}")
    print(
        f"  share of the human gradient that survives    "
        f"{f(gradient / human_gradient)}"
    )
    print(f"  model mean slope across bands (the LEVEL)    {f(mean_slope)}")
    print(
        f"  weighted |slope gap| in noise ceilings       "
        f"{f(summary['weighted_abs_gap_to_human'] / RCE_NOISE_CEILING)}"
    )
    print(
        f"  within-level gradient, model vs human        "
        f"{f(summary['within_level_gradient_model'])} vs "
        f"{f(summary['within_level_gradient_human'])}"
    )

    if args.verify_copula:
        step = max(1, len(ctx) // 200)
        sub_ctx = ctx.iloc[::step].reset_index(drop=True)
        vg_c, vg_p = [2, 12, 17], [0, 10, 20]
        vdf = verify_copula(
            model, data, sub_ctx, vg_c, vg_p, get_edges, args.copula_draws, args.seed
        )
        # the analytic surface is over all contexts; recompute it on the
        # subgrid's own contexts so the comparison is like for like
        sub_surface, sub_cells = forced_response(
            model, data, sub_ctx, vg_c, vg_p, args.chunk, get_edges
        )
        sub_mean = sub_surface.mean(0)
        vdf["analytic_same_contexts"] = [
            float(sub_mean[sub_cells.index((c, p))]) for c, p in zip(vdf["c"], vdf["p"])
        ]
        vdf["z"] = (vdf["sampled_mean_dc"] - vdf["analytic_same_contexts"]) / vdf[
            "sampled_se"
        ]
        vdf.to_csv(out_dir / "copula_invariance.csv", index=False)
        print(vdf.to_string(index=False, float_format=lambda v: f"{v: .6f}"))
        print(f"  max |z| = {f(vdf['z'].abs().max())}")
        summary["copula_max_abs_z"] = float(vdf["z"].abs().max())

    summary["wall_seconds"] = time.time() - t0
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir / 'summary.json'}")
    print(f"wall {time.time() - t0:.1f}s")
    if not ok:
        print("!! the human self-check failed -- the comparison is INVALID")
        sys.exit(2)


if __name__ == "__main__":
    main()
