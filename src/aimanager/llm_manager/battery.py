"""The standard battery, read off a recorded paired rollout.

Pure torch / numpy / pandas: no torch_geometric, no model loading, so every
number here is exercised by local tests against synthetic records. The
harness that produces the record is in `harness.py`.

Three things in here are deliberate and are not to be quietly simplified.

**The pool and the total contribution are never averaged, combined, or
reduced to one score.** They disagree in this game: punishment leaves the
pool at full price and contributions enter it at 1.6, so a correctly-aimed
rule buys contribution clearly and is close to break-even on the pool
(#207: `thr9_p10` minus never-punishing is +8.12 [4.01, 12.22] contribution
and +1.07 [-5.79, 7.93] pool). A manager can be ahead on one and level on
the other, and a single figure hides exactly the trade the reader needs.

**Targeting is three statistics together, never the rank alone.** A flat
policy scores well on rank -- a profile falling 5.00 to 4.99 ranks like one
falling 4.76 to 0.27 -- and the rank hides force, so the human managers and
the clone are identical on rank while differing by 1.088 on bin means. Rank
is reported beside `magnitude` (the spread of the bin means relative to
their own mean) and `noise_gate` (that spread over its own standard error).
The tie structure is reported with them, because a rank correlation over a
mostly-zero punishment column is attenuated by ties, and how attenuated
depends on how quiet the manager is -- which is a property the statistic is
supposed to be invariant to. `rho_floor` is the most negative rho the
rule's own two marginals permit, and `rho_rel = rho / rho_floor` is the
share of that attainable targeting it actually reached.

**The leaver diagnostic is an ordering, never a sign test.** `c_gap` tracks
policy shape at about r = -0.95 across managers, so the ranking is
informative; but measured across four inverted managers only one crossed
zero, so the zero point does not classify. Its noise floor is
`C_GAP_NOISE_FLOOR`, about the size of the differences a fine contrast
would ask it to resolve. `leaver_ordering` therefore emits a rank and a
"is this neighbour gap bigger than the noise floor" flag and no sign call.
"""

import numpy as np
import pandas as pd
import torch as th
from scipy.stats import norm

from aimanager.manager.paired_rollout import RPA_LABELS, summarise

#: The group's undivided common pool, `1.6 * sum(c) - sum(p)`. The
#: maintainer has settled this as the objective: a manager that increases
#: collaboration and maintains the pool. Every table in this module leads
#: with it and the power calculation is sized on it first.
PRIMARY_OBJECTIVE = "pool"

#: The five quantities reported for every arm, per episode, on each seat,
#: in the order tables carry them. `pool` and `contribution` are seat TOTALS
#: per round -- the competing setting prices a manager on what its whole
#: group produced, so a policy that raises contributions per head while
#: shedding the heads has not gained anything.
#:
#: `contribution` sits second and is a diagnostic, not a co-equal headline,
#: and it is in the battery for one specific reason: it separates a manager
#: that raised collaboration from one that merely refrained from spending.
#: Two managers can reach the same pool by opposite routes -- raising
#: contributions and paying for them, or doing nothing -- and the objective
#: cannot tell them apart on its own. A pool-equal comparison whose
#: contribution columns differ is a finding; `contrasts` exists so it is
#: visible rather than collapsed.
HEADLINE = (
    "pool",
    "contribution",
    "pool_per_member",
    "mean_punishment",
    "members",
)

#: The leaver diagnostic's noise floor, measured in the paired setting
#: (PR #219). Differences smaller than this are not read.
C_GAP_NOISE_FLOOR = 0.577

#: Telemetry a manager may expose through `telemetry()`. Absent keys are
#: reported as NaN rather than 0, so "not measured" never reads as "none".
TELEMETRY_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "n_calls",
    "n_decisions_requested",
    "n_parse_failures",
    "manager_wall_clock_s",
)


# --------------------------------------------------------------------- #
# per-episode frame
# --------------------------------------------------------------------- #
def _share_roundmean(rec, seat_group):
    """Mean over rounds of the seat's pool divided by its valid headcount.

    This is `rule_vs_clone_paired_report.share_corr`, the "per member"
    column of PR #217, kept so this harness can be checked against that
    arm's published table. It is NOT `pool / members`: the divisor is the
    number of players who gave an input that round (the game's own divisor,
    `payoff = 20 - c - p + pool/n_valid`), the ratio is formed per
    group-round rather than once at the end, and a round where the seat held
    nobody who played contributes a 0 rather than being dropped.
    """
    c, p, v, g = (rec[k] for k in ("contribution", "punishment", "valid", "group"))
    seat = g == seat_group
    c_eff = th.where(v, c, th.zeros_like(c)).to(th.float)
    p_eff = th.where(v, p, th.zeros_like(p)).to(th.float)
    n_valid = (seat & v).sum(dim=1).to(th.float)
    pool = (1.6 * (c_eff * seat).sum(dim=1)) - (p_eff * seat).sum(dim=1)
    share = th.where(n_valid > 0, pool / n_valid.clamp(min=1), th.zeros_like(pool))
    return share.mean(dim=-1)


def episode_frame(rec, focal_group=0, rival_group=1, switch_every=4):
    """One row per episode: everything the battery is formed from.

    Sums and counts come through from `paired_rollout.summarise` undivided,
    so a caller can pool episodes without re-weighting; the ratios added
    here are the per-episode versions, which is what the noise floor and the
    minimum detectable difference are computed over.
    """
    cols = {
        k: v.numpy().astype(np.float64)
        for k, v in summarise(
            rec,
            focal_group=focal_group,
            rival_group=rival_group,
            switch_every=switch_every,
        ).items()
    }
    df = pd.DataFrame(cols)
    for seat, grp in (("focal", focal_group), ("rival", rival_group)):
        df[f"{seat}_share_roundmean"] = (
            _share_roundmean(rec, grp).numpy().astype(np.float64)
        )
        df[f"{seat}_pool_per_member"] = _safe_div(
            df[f"{seat}_pool"], df[f"{seat}_members"]
        )
        df[f"{seat}_mean_punishment"] = _safe_div(
            df[f"{seat}_p_num"], df[f"{seat}_p_den"]
        )
        df[f"{seat}_mean_contribution"] = _safe_div(
            df[f"{seat}_c_num"], df[f"{seat}_c_den"]
        )
        # the same spend over the divisor PR #217 used: valid member-rounds
        # rather than member-rounds. They differ by the timeout rate, about
        # 5%, and quoting one against the other is a 5% phantom effect.
        df[f"{seat}_mean_punishment_valid"] = _safe_div(
            df[f"{seat}_p_num"], df[f"{seat}_c_den"]
        )
    df.insert(0, "episode", np.arange(len(df), dtype=np.int64))
    return df


def _safe_div(num, den):
    return np.where(np.asarray(den) > 0, num / np.where(den > 0, den, np.nan), np.nan)


# --------------------------------------------------------------------- #
# policy shape
# --------------------------------------------------------------------- #
def _bin_means_and_se(num, den, min_bin_n=20):
    """Pooled bin means over episodes, and the ratio estimator's error.

    A bin mean is a ratio of two sums over episodes, so its error is
    `sum_e (p_e - m * n_e)^2 / (sum_e n_e)^2`. Episodes are the independent
    unit; the agent-rounds inside one are not, which is why the error is
    taken across episodes and not across cells.
    """
    e = len(num)
    tot_n, tot_p = den.sum(0), num.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(
            tot_n >= min_bin_n, tot_p / np.where(tot_n > 0, tot_n, np.nan), np.nan
        )
        resid = num - np.nan_to_num(m)[None, :] * den
        var = (
            (resid**2).sum(0)
            * (e / max(e - 1, 1))
            / np.where(tot_n > 0, tot_n, np.nan) ** 2
        )
    return m, np.sqrt(var)


def policy_shape(episodes, min_bin_n=20):
    """Mean punishment per contribution bin, on the evaluation suite's bins.

    The bins are `paired_rollout.RPA_EDGES`, and the cells a player timed out
    on were dropped at source in `_policy_shape` rather than binned: their
    recorded contribution is an imputed 9 the manager never saw, so binning
    them would file the rule's timeout behaviour under `6-10`.
    """
    num = episodes[[f"rpa_p_{lab}" for lab in RPA_LABELS]].to_numpy(float)
    den = episodes[[f"rpa_n_{lab}" for lab in RPA_LABELS]].to_numpy(float)
    m, se = _bin_means_and_se(num, den, min_bin_n)
    return pd.DataFrame(
        {
            "bin": list(RPA_LABELS),
            "mean_punishment": m,
            "se": se,
            "n_decisions": den.sum(0),
        }
    )


# --------------------------------------------------------------------- #
# targeting: three statistics, and the tie structure behind the rank
# --------------------------------------------------------------------- #
def _midranks(marginal):
    cum = np.cumsum(marginal)
    return cum - marginal + (marginal + 1) / 2.0


def _moments(counts):
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    rc, rp = _midranks(counts.sum(1)), _midranks(counts.sum(0))
    mc = (counts.sum(1) * rc).sum() / n
    mp = (counts.sum(0) * rp).sum() / n
    vc = (counts.sum(1) * (rc - mc) ** 2).sum() / n
    vp = (counts.sum(0) * (rp - mp) ** 2).sum() / n
    return n, rc, rp, mc, mp, vc, vp


def spearman_from_counts(counts):
    """Tie-corrected Spearman rho from a contribution x punishment table.

    Invariant to any monotone rescaling of the punishment, so a rule that
    punishes the same people three times as hard scores the same -- which a
    difference of bin means does not. Negative means low contributors are
    punished (correct targeting); a rule that never punishes has no variance
    and scores nan rather than 0.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.sum() == 0:
        return np.nan
    n, rc, rp, mc, mp, vc, vp = _moments(counts)
    if vc <= 0 or vp <= 0:
        return np.nan
    cov = (counts * (rc - mc)[:, None] * (rp - mp)[None, :]).sum() / n
    return float(cov / np.sqrt(vc * vp))


def rho_floor_from_counts(counts):
    """The most negative rho these two marginals permit.

    Ties put a ceiling on |rho| that has nothing to do with aim. A manager
    that punishes 4% of its decisions has 96% of its punishment column tied
    at zero, and no arrangement of the joint can then reach -1; a manager
    that punishes half of them can get much closer. So the same aim scores
    differently depending on how quiet the manager is, and quiet is exactly
    what a language model asked to be careful may be.

    The bound is exact and cheap. Midranks depend only on the marginals, so
    rho is a linear functional of the joint with those margins fixed, and
    `r_c * r_p` is supermodular; the Frechet lower bound (the countermonotone
    coupling, built here by a north-west corner fill that pairs the lowest
    contributions with the highest punishments) attains the minimum.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.sum() == 0:
        return np.nan
    n, rc, rp, mc, mp, vc, vp = _moments(counts)
    if vc <= 0 or vp <= 0:
        return np.nan
    a, b = counts.sum(1).copy(), counts.sum(0).copy()
    i, j, cov = 0, len(b) - 1, 0.0
    while i < len(a) and j >= 0:
        if a[i] <= 0:
            i += 1
            continue
        if b[j] <= 0:
            j -= 1
            continue
        m = min(a[i], b[j])
        cov += m * (rc[i] - mc) * (rp[j] - mp)
        a[i] -= m
        b[j] -= m
    return float((cov / n) / np.sqrt(vc * vp))


def tie_structure(counts):
    """How much of the rank information the two marginals have already spent.

    `tie_frac_p` is the share of unordered decision pairs tied on punishment,
    `tie_frac_c` the same on contribution, and `zero_share` the single tie
    block that dominates a quiet manager: the decisions that punished
    nothing. They are reported beside the rank so a weak rho can be told
    apart from an attenuated one.
    """
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    if n < 2:
        return {"tie_frac_c": np.nan, "tie_frac_p": np.nan, "zero_share": np.nan}
    pairs = n * (n - 1) / 2.0

    def tied(marg):
        return float((marg * (marg - 1) / 2.0).sum() / pairs)

    return {
        "tie_frac_c": tied(counts.sum(1)),
        "tie_frac_p": tied(counts.sum(0)),
        "zero_share": float(counts.sum(0)[0] / n),
    }


def targeting_triple_from_arrays(num, den, min_bin_n=20):
    """`(episodes, bins)` punishment sums and counts -> magnitude and gate."""
    m, se = _bin_means_and_se(num, den, min_bin_n)
    ok = ~np.isnan(m)
    if ok.sum() < 2:
        return {
            "n_bins": int(ok.sum()),
            "bin_mean_range": np.nan,
            "magnitude": np.nan,
            "noise_gate": np.nan,
        }
    hi, lo = int(np.nanargmax(m)), int(np.nanargmin(m))
    rng = float(m[hi] - m[lo])
    se_rng = float(np.sqrt(se[hi] ** 2 + se[lo] ** 2))
    mean_m = float(np.nanmean(m))
    if se_rng > 0:
        gate = rng / se_rng
    elif rng > 0:
        # a deterministic rule's extreme bins are the same number in every
        # episode, so they carry no sampling error at all: the gate is
        # unbounded rather than undefined
        gate = np.inf
    else:
        gate = np.nan
    return {
        "n_bins": int(ok.sum()),
        "bin_mean_range": rng,
        "magnitude": rng / mean_m if mean_m else np.nan,
        "noise_gate": gate,
    }


def targeting(counts, episodes, min_bin_n=20):
    """Aim, force and the tie structure -- reported together, always."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum()
    n_pos = counts[:, 1:].sum()
    levels = np.arange(counts.shape[1])[None, :]
    total_p = (counts * levels).sum()
    rho = spearman_from_counts(counts)
    floor = rho_floor_from_counts(counts)
    num = episodes[[f"rpa_p_{lab}" for lab in RPA_LABELS]].to_numpy(float)
    den = episodes[[f"rpa_n_{lab}" for lab in RPA_LABELS]].to_numpy(float)
    return {
        "rho": rho,
        "rho_floor": floor,
        "rho_rel": (float(rho / floor) if floor is not None and floor < 0 else np.nan),
        **tie_structure(counts),
        **targeting_triple_from_arrays(num, den, min_bin_n),
        "n_decisions": int(n),
        "punish_rate": float(n_pos / n) if n else np.nan,
        "mean_p_valid": float(total_p / n) if n else np.nan,
        "mean_p_given_positive": float(total_p / n_pos) if n_pos else np.nan,
    }


# --------------------------------------------------------------------- #
# the battery row
# --------------------------------------------------------------------- #
def _mean_se(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    sd = float(x.std(ddof=1)) if len(x) > 1 else np.nan
    return float(x.mean()), sd, (sd / np.sqrt(len(x)) if len(x) > 1 else np.nan)


#: The two headline quantities that are ratios, and what they are a ratio
#: OF. Their reported LEVEL is the pooled ratio, sum of numerators over sum
#: of denominators, which is the convention every paired arm in this
#: project uses (`rule_sigmoid.aggregate.RATIOS`, `_ratio`). Averaging the
#: per-episode ratios instead is a different number -- measured here at 2.06
#: against 1.85 for the clone's spend, an 11% gap and 35 standard errors at
#: 6,144 episodes -- because episodes differ in how many member-rounds they
#: contain and the small ones carry the larger ratios. The per-episode
#: version is kept as `_episodemean` and is what the spread, the noise floor
#: and the minimum detectable difference are taken over, since those need an
#: independent value per episode.
POOLED_RATIOS = {
    "pool_per_member": ("pool", "members"),
    "mean_punishment": ("p_num", "p_den"),
}


def battery_row(name, episodes, counts, telemetry=None, wall_clock_s=np.nan):
    """One arm's whole battery as a flat row.

    `pool` and `contribution` both appear and are never combined; there is
    deliberately no summary objective column for a caller to sort on.
    """
    row = {"arm": name, "n_episodes": int(len(episodes))}
    for seat in ("focal", "rival"):
        for q in HEADLINE:
            mean, sd, se = _mean_se(episodes[f"{seat}_{q}"])
            row[f"{seat}_{q}"] = mean
            row[f"{seat}_{q}_sd"] = sd
            row[f"{seat}_{q}_se"] = se
        for q, (num, den) in POOLED_RATIOS.items():
            row[f"{seat}_{q}_episodemean"] = row[f"{seat}_{q}"]
            row[f"{seat}_{q}"] = float(
                _safe_div(
                    episodes[f"{seat}_{num}"].sum(), episodes[f"{seat}_{den}"].sum()
                )
            )
    row["focal_share_roundmean"] = float(episodes["focal_share_roundmean"].mean())
    row["rival_share_roundmean"] = float(episodes["rival_share_roundmean"].mean())
    row["focal_mean_contribution"] = float(
        _safe_div(episodes["focal_c_num"].sum(), episodes["focal_c_den"].sum())
    )
    row["focal_mean_punishment_valid"] = float(
        _safe_div(episodes["focal_p_num"].sum(), episodes["focal_c_den"].sum())
    )
    row.update(targeting(counts, episodes))
    shape = policy_shape(episodes)
    for lab, m in zip(shape["bin"], shape["mean_punishment"]):
        row[f"rpa_{lab}"] = m
    for lab, nd in zip(shape["bin"], shape["n_decisions"]):
        row[f"rpa_n_{lab}"] = nd
    row.update(_leaver_row(episodes))
    row["wall_clock_s"] = wall_clock_s
    row.update(_telemetry_row(telemetry, row["n_episodes"]))
    return row


def _leaver_row(episodes):
    """Leavers' mean contribution minus stayers', pooled over episodes.

    Reported for the ORDERING it induces over managers. Do not add a sign
    test here: see the module docstring and `leaver_ordering`.
    """
    tot = {k: float(episodes[k].sum()) for k in ("lv_n", "lv_c", "st_n", "st_c")}
    c_lv = tot["lv_c"] / tot["lv_n"] if tot["lv_n"] else np.nan
    c_st = tot["st_c"] / tot["st_n"] if tot["st_n"] else np.nan
    denom = tot["lv_n"] + tot["st_n"]
    return {
        "c_leavers": c_lv,
        "c_stayers": c_st,
        "c_gap": c_lv - c_st,
        "leave_rate": tot["lv_n"] / denom if denom else np.nan,
    }


def _telemetry_row(telemetry, n_episodes):
    t = dict(telemetry or {})
    out = {k: float(t[k]) if k in t else np.nan for k in TELEMETRY_KEYS}
    asked = out["n_decisions_requested"]
    out["parse_failure_rate"] = (
        out["n_parse_failures"] / asked if asked and asked > 0 else np.nan
    )
    tok = out["prompt_tokens"] + out["completion_tokens"]
    out["total_tokens"] = tok
    out["tokens_per_episode"] = tok / n_episodes if n_episodes else np.nan
    return out


def _unpaired_ci(a, b, n_boot=4000, seed=1):
    """Interval for mean(a) - mean(b), resampling each arm's episodes.

    Two arms are two different worlds -- a manager that punishes differently
    makes the players act differently and consumes a different number of
    draws -- so the difference is unpaired, which is the conservative
    reading and the construction #217 and #219 both used.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)[~np.isnan(np.asarray(a, float))]
    b = np.asarray(b, float)[~np.isnan(np.asarray(b, float))]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    da = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(1)
    d = da - db
    return (
        float(a.mean() - b.mean()),
        float(np.percentile(d, 2.5)),
        float(np.percentile(d, 97.5)),
    )


def contrasts(episodes, reference, seat="focal", quantities=HEADLINE, seed=1):
    """Every arm minus a reference arm, pool first, on all five quantities.

    This table is what makes the route visible. The objective discriminates
    between managers but does not reward punishment for its own sake: a
    correctly-targeted threshold rule is indistinguishable from never
    punishing on the pool (#207: +1.07 [-5.79, 7.93]; #219's held-out
    seeds: +0.51) while the capped fitted rule beats that threshold rule by
    +5.00. So a manager sitting at never-punishing on the pool has not
    necessarily failed, and one above the capped rule has done something
    new -- and neither reading can be made without the contribution column
    beside the pool.
    """
    out = []
    ref = episodes[episodes["arm"] == reference]
    assert len(ref), f"no reference arm {reference!r} in this run"
    # only arms that faced the same rival are comparable: a symmetric
    # control differs from its own arm in the OTHER seat, and putting that
    # difference in a table of manager contrasts would read as the manager
    ref_rival = set(ref["rival"]) if "rival" in ref else None
    for arm, g in episodes.groupby("arm", sort=False):
        if arm == reference:
            continue
        if ref_rival is not None and set(g["rival"]) != ref_rival:
            continue
        for q in quantities:
            d, lo, hi = _unpaired_ci(g[f"{seat}_{q}"], ref[f"{seat}_{q}"], seed=seed)
            out.append(
                {
                    "arm": arm,
                    "reference": reference,
                    "quantity": q,
                    "delta": d,
                    "lo": lo,
                    "hi": hi,
                    "crosses_zero": bool(lo <= 0 <= hi) if lo == lo else None,
                }
            )
    return pd.DataFrame(out)


def leaver_ordering(battery, noise_floor=C_GAP_NOISE_FLOOR):
    """Rank the arms by `c_gap`, and say which neighbours are separated.

    An ORDERING, not a classification. The zero point does not separate
    correctly- from incorrectly-targeted managers, so nothing here reads the
    sign; `gap_to_next` against the noise floor is the only claim made.
    """
    out = battery[["arm", "c_gap"]].sort_values("c_gap").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["gap_to_next"] = out["c_gap"].shift(-1) - out["c_gap"]
    sep = (out["gap_to_next"] > noise_floor).astype("boolean")
    sep.iloc[-1] = pd.NA
    out["separated_from_next"] = sep
    return out


# --------------------------------------------------------------------- #
# noise floor and the minimum detectable difference
# --------------------------------------------------------------------- #
def _boot_paired(d, n_boot=4000, seed=1):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return np.nan, np.nan, np.nan
    draws = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return (
        float(d.mean()),
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def noise_floor(episodes, arm="symmetric"):
    """What the harness reports when there is nothing to report.

    Both seats hold the same manager, so every quantity's focal-minus-rival
    difference has an expectation of zero and anything it shows is the seat,
    not the policy. Three numbers per quantity:

      * `seat_bias` with a paired bootstrap interval -- the two seats share
        an episode, so the difference is paired and the paired interval is
        the honest one. PR #217 measured a real one here (`never_vs_never`
        at -7.69 [-14.06, -1.31] on the pool), so this is not a formality.
      * `sd_episode` -- the spread of the focal seat's own value across
        episodes. This is what sets the error of an arm measured on its own
        and so of any comparison between two separate rollouts.
      * `sd_diff` -- the spread of the focal-minus-rival difference. Smaller
        than `sqrt(2) * sd_episode` to the extent that the two seats share
        an episode's luck, which is what makes the within-run contrast the
        cheaper of the two designs.
    """
    rows = []
    for q in HEADLINE:
        f = episodes[f"focal_{q}"].to_numpy(float)
        r = episodes[f"rival_{q}"].to_numpy(float)
        bias, lo, hi = _boot_paired(f - r)
        d = (f - r)[~np.isnan(f - r)]
        rows.append(
            {
                "arm": arm,
                "quantity": q,
                "n_episodes": int(len(episodes)),
                "focal_mean": float(np.nanmean(f)),
                "rival_mean": float(np.nanmean(r)),
                "seat_bias": bias,
                "seat_bias_lo": lo,
                "seat_bias_hi": hi,
                "sd_episode": float(np.nanstd(f, ddof=1)),
                "sd_diff": float(d.std(ddof=1)) if len(d) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


#: Budgets the power table is published at. The low end is what a language
#: model was feared to be stuck at; the high end is where the existing
#: baselines live (#217 at 300, #219's validation at 6,144), and on four
#: A100s a rollout is 24 sequential batched calls however wide the batch is,
#: so the wall clock stops scaling with the episode count once the server
#: saturates. The table therefore has to span the whole range rather than
#: settle the feasibility question at the bottom of it.
EPISODE_BUDGETS = (50, 200, 500, 1000, 3000)

#: Differences this project has already measured, so "an effect worth
#: caring about" is anchored on the world rather than on a convention.
#: Pool first, because that is the objective a result will be judged on;
#: they span the range from a difference nobody should expect to resolve to
#: one that is unmissable.
REFERENCE_EFFECTS = (
    {
        "effect": "thr9_p10 over never",
        "quantity": "pool",
        "size": 0.51,
        "source": "PR #219, 60.19 - 59.68 on the held-out seeds",
    },
    {
        "effect": "capped_sigmoid over thr9_p10",
        "quantity": "pool",
        "size": 5.00,
        "source": "PR #219, held-out seeds",
    },
    {
        "effect": "inverted rule against never",
        "quantity": "pool",
        "size": 32.75,
        "source": "PR #217, inv_thr11_p10",
    },
    {
        "effect": "thr9_p10 over never",
        "quantity": "contribution",
        "size": 7.34,
        "source": "PR #219, 44.64 - 37.30 on the held-out seeds",
    },
)


def mdd_table(floor, episode_counts=EPISODE_BUDGETS, alpha=0.05, power=0.8):
    """Smallest difference each design could detect, at each budget.

    `MDD = (z_{1-alpha/2} + z_{power}) * se`, the standard two-sided
    normal-approximation calculation. Two designs, because this harness
    supports two comparisons and they do not cost the same:

      * `unpaired` -- arm A against arm B, two separate rollouts of `n`
        episodes each. `se = sqrt(2) * sd_episode / sqrt(n)`. This is the
        design for "does the language model beat `thr9_p10`".
      * `within_run` -- the focal seat against the rival seat of the SAME
        rollout, `n` episodes in total. `se = sd_diff / sqrt(n)`. Cheaper,
        because the two seats share the episode's luck, but it answers a
        different question: it compares the manager with whatever sits in
        the other seat, and it carries the seat bias measured above.

    The numbers are the floor, not the budget: a real run also pays for
    multiple comparisons, and an effect at exactly the MDD is detected half
    the time by construction.
    """
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    rows = []
    for _, r in floor.iterrows():
        for n in episode_counts:
            rows.append(
                {
                    "quantity": r["quantity"],
                    "episodes": int(n),
                    "sd_episode": r["sd_episode"],
                    "sd_diff": r["sd_diff"],
                    "mdd_unpaired": z * np.sqrt(2) * r["sd_episode"] / np.sqrt(n),
                    "mdd_within_run": z * r["sd_diff"] / np.sqrt(n),
                }
            )
    return pd.DataFrame(rows)


def detectability(floor, effects=REFERENCE_EFFECTS, alpha=0.05, power=0.8):
    """How many episodes each already-measured effect would need.

    The inverse of `mdd_table`, and the more useful direction now that the
    budget is not the binding constraint: `n = (z * se_unit / delta)^2`,
    where `se_unit` is the standard error at one episode
    (`sqrt(2) * sd_episode` unpaired, `sd_diff` within-run). It says what an
    episode count buys rather than whether the run is feasible.

    A quantity whose spread is zero in the control -- a constant manager's
    own spend, say -- needs no episodes at all and reports 0.
    """
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    f = floor.set_index("quantity")
    rows = []
    for e in effects:
        if e["quantity"] not in f.index:
            continue
        r = f.loc[e["quantity"]]
        row = {**e, "sd_episode": r["sd_episode"], "sd_diff": r["sd_diff"]}
        for tag, se_unit in (
            ("unpaired", np.sqrt(2) * r["sd_episode"]),
            ("within_run", r["sd_diff"]),
        ):
            row[f"n_episodes_{tag}"] = (
                int(np.ceil((z * se_unit / e["size"]) ** 2))
                if se_unit == se_unit
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)
