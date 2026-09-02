"""Preflight for the contribution group copula -- A DIAGNOSTIC, NEVER A GATE.

Four arms, all driven through the ONE two-component code path
(``LinearAHAdapter._sample_levels_gaussian_copula``):

  1. ``(rho_p, rho_t) = (0, 0)``            -- the reference every delta is
                                               measured against.
  2. the candidate ``(rho_total, 0)``       -- read from the stamped bundle.
  3. the falsifier reading                  -- ``(rho_lag1, rho_total -
                                               rho_lag1)`` from the params JSON.
  4. transient only ``(0, rho_total)``      -- for completeness.

Two independent proxies, both local and CPU:

  A. Reduced-form Monte Carlo of the parent's measured AR(1) group dynamics
     (linear mean, no punisher feedback), whose rho = 0 arm is expected near
     ratio 0.6551 (log Note 8).
  B. Closed-loop rollout of the REAL gaussian_mlp_v2 contribution bundle
     against the REAL PR #160 severity-copula punisher, fixed 4/4 groups, no
     switching, mirroring the env's round order, whose rho = 0 arm is expected
     near ratio 0.72 (log Note 9).

Both proxies are OPEN-LOOP APPROXIMATIONS of the real simulation and are read
as DELTA predictors only: proxy A has a linear mean and no punisher, proxy B
freezes group membership and disables switching, and neither carries the
switch-slot interaction the real simulation has. Their absolute ratio levels
are off in opposite directions (A below, B above), which is exactly why every
prediction here applies the proxy's delta to the parent's real baseline ratio.

The experiment proceeds to simulation at the fitted dose whatever this script
prints. Nothing here adjusts, tunes or recommends a dose.

Usage (from the repo root):

    uv run python scripts/baselines/gmlp_group_copula_preflight.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "baselines"))

from aimanager.simulation.linear_ah import LinearAHAdapter  # noqa: E402

CONTR_BUNDLE = "artifacts/baselines/contribution_gaussian_mlp_v2_group_copula.joblib"
PARAMS_JSON = (
    "artifacts/baselines/contribution_gaussian_mlp_v2_group_copula.params.json"
)
PUNISH_BUNDLE = "artifacts/baselines/punishment_multinomial_severity_copula.joblib"

# --- the parent's exact numbers, the frame every prediction is stated in ----
HUMAN_RATIO = 0.8480163543652899
PARENT_RATIO = 0.6915274426731914  # PR #167's candidate, the real sim
CG_DENOM = 0.026448679600274437
PARENT_CA_STD_DIFF = -2.621590294686563  # sim SD(participant means) - human's

# --- proxy A: the parent's measured reduced-form dynamics (log Note 8) -----
MC_A = 0.7316
MC_B = 0.2015
MC_SIGMA = 3.410
MC_K = 9.0 * (1.0 - MC_A - MC_B)
MC_EPISODES = 4000
MC_ROUNDS = 24
MC_BURN_IN = 4
MC_N_GROUPS = 2
MC_GROUP_SIZE = 4  # 8 agents per episode in two groups, as in the sim
MC_SEED = 7

# --- proxy B: the sim's own protocol, minus switching -----------------------
SIM_AGENT_GROUPS = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
SIM_N_AGENTS = 8
SIM_N_CONTRIBUTIONS = 21
SIM_EPISODES = 100
SIM_ROUNDS = 24
SIM_SEED = 42

DOSE_RESPONSE = [0.0, 0.02, 0.0438, 0.06, 0.08]


def band(score):
    """The scoring schema's band for a normalised score."""
    if score <= 1.0:
        return "<= 1"
    if score <= 2.0:
        return "1-2"
    if score <= 5.0:
        return "2-5"
    return "> 5"


def implied_cg(delta_ratio):
    """The parent's REAL baseline ratio moved by a proxy's delta, scored."""
    ratio = PARENT_RATIO + delta_ratio
    return abs(HUMAN_RATIO - ratio) / CG_DENOM, ratio


def spread_stats(contrib, episode, agent, group):
    """SD(group mean) / SD(individual) over (episode, round, group) cells plus
    SD(participant means) -- the eval suite's CG and CA-diagnostic quantities,
    with pandas' ddof = 1 in both places.

    ``contrib`` is [n_rows]; ``episode`` / ``agent`` / ``group`` are the same
    length, ``group`` already keyed per (episode, round, group) cell."""
    contrib = np.asarray(contrib, dtype=float)
    cell_means = _means_by(contrib, group)
    part_means = _means_by(contrib, np.stack([episode, agent], axis=1))
    sd_gmean = float(cell_means.std(ddof=1))
    sd_ind = float(contrib.std(ddof=1))
    return {
        "sd_gmean": sd_gmean,
        "sd_ind": sd_ind,
        "ratio": sd_gmean / sd_ind,
        "sd_participant": float(part_means.std(ddof=1)),
    }


def _means_by(values, keys):
    """Group means of ``values`` by row-wise integer key(s)."""
    keys = np.asarray(keys)
    if keys.ndim == 1:
        keys = keys[:, None]
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    return sums / counts


# --------------------------------------------------------------------------- #
# proxy A: reduced-form Monte Carlo
# --------------------------------------------------------------------------- #
def proxy_a(rho_p, rho_t):
    """c_it = clip(rint(a c_i,t-1 + b m_g,t-1 + k + sigma z_it), 0, 20) with
    z_it = sqrt(rho_p) u_g + sqrt(rho_t) v_gt + sqrt(1 - rho_p - rho_t) e_it,
    u_g drawn ONCE PER EPISODE (per group), v_gt every round. Linear mean, no
    punisher feedback: an open-loop delta predictor."""
    rg = np.random.default_rng(MC_SEED)
    e_, g_, n_ = MC_EPISODES, MC_N_GROUPS, MC_GROUP_SIZE
    c = np.full((e_, g_, n_), 9.0)
    u = rg.standard_normal((e_, g_))  # once per (episode, group)
    w_p, w_t = np.sqrt(rho_p), np.sqrt(rho_t)
    w_e = np.sqrt(1.0 - rho_p - rho_t)
    kept = []
    for t in range(MC_ROUNDS):
        v = rg.standard_normal((e_, g_))  # fresh every round
        eps = rg.standard_normal((e_, g_, n_))
        z = w_p * u[..., None] + w_t * v[..., None] + w_e * eps
        m = c.mean(2, keepdims=True)
        c = np.clip(np.rint(MC_A * c + MC_B * m + MC_K + MC_SIGMA * z), 0, 20)
        if t >= MC_BURN_IN:
            kept.append(c.copy())
    arr = np.stack(kept, axis=1)  # [episode, round, group, member]
    n_rounds = arr.shape[1]
    ep = np.arange(e_)[:, None, None, None] * np.ones_like(arr, dtype=np.int64)
    rd = np.arange(n_rounds)[None, :, None, None] * np.ones_like(arr, dtype=np.int64)
    gp = np.arange(g_)[None, None, :, None] * np.ones_like(arr, dtype=np.int64)
    mb = np.arange(n_)[None, None, None, :] * np.ones_like(arr, dtype=np.int64)
    cell = np.stack([ep.ravel(), rd.ravel(), gp.ravel()], axis=1)
    agent = (gp * n_ + mb).ravel()  # stable participant id within an episode
    return spread_stats(arr.ravel(), ep.ravel(), agent, cell)


# --------------------------------------------------------------------------- #
# proxy B: closed-loop rollout of the real models
# --------------------------------------------------------------------------- #
class ForcedCopulaAdapter(LinearAHAdapter):
    """Route EVERY arm through ``_sample_levels_gaussian_copula``, the (0, 0)
    reference included.

    ``predict``'s gate falls back to the independent ``_sample_levels`` when
    both weights are zero, and the two paths consume a different number of
    torch draws (1n vs 3n), so a rho = 0 arm taken through the legacy path
    would sit on a different RNG stream than the other arms and its delta
    would carry that difference. Forcing the two-component path keeps common
    random numbers across arms; at (0, 0) the sampler is algebraically the
    independent one (weights 0, 0, 1)."""

    def _sample_levels(self, Xs, n_levels):
        t = max(self._group)  # the round ``predict`` just recorded
        return self._sample_levels_gaussian_copula(Xs, n_levels, self._group[t])


def _state(t, prev, groups, device="cpu"):
    """The env state keys the adapter reads, in the env's [1, A, 1] shape."""

    def col(x, dtype):
        return th.tensor(np.asarray(x).reshape(1, -1, 1), dtype=dtype, device=device)

    return {
        "round_number": col(np.full(len(groups), t), th.int64),
        "prev_contribution": col(prev["contribution"], th.int64),
        "prev_punishment": col(prev["punishment"], th.int64),
        "prev_common_good": col(prev["common_good"], th.float),
        "prev_agent_group": col(groups, th.int64),
        "agent_group": col(groups, th.int64),
    }


def proxy_b(bundle, punisher_bundle, rho_p, rho_t):
    """Roll the real contribution bundle out against the real punisher, fixed
    4/4 groups, no switching, mirroring the env's round order: contribution ->
    punishment -> per-group common good (1.6 sum c - sum p) / n_valid ->
    prev_* shift. Returns the spread stats plus the observed persistent
    latents, for the per-episode-reset check (log Note 26)."""
    contr = dict(bundle)
    contr["copula_rho_p"] = float(rho_p)
    contr["copula_rho_t"] = float(rho_t)
    ah = ForcedCopulaAdapter(
        contr, n_agents=SIM_N_AGENTS, n_contributions=SIM_N_CONTRIBUTIONS
    )
    pm = LinearAHAdapter(punisher_bundle, n_agents=SIM_N_AGENTS)

    groups = SIM_AGENT_GROUPS
    defaults = ah.default_values
    th.manual_seed(SIM_SEED)
    rows_c, rows_ep, rows_rd, rows_ag = [], [], [], []
    latents = []
    for episode in range(SIM_EPISODES):
        prev = {
            "contribution": np.full(SIM_N_AGENTS, defaults["contribution"]),
            "punishment": np.full(SIM_N_AGENTS, defaults["punishment"]),
            "common_good": np.full(SIM_N_AGENTS, defaults["common_good"]),
        }
        rounds = []
        for t in range(SIM_ROUNDS):
            # contribution: reset_rnn at t == 0 clears the episode history AND
            # the persistent latents (`_reset_history`), so every episode draws
            # its own u_g -- without this the whole run is one giant episode
            # with a frozen offset per group (log Note 26).
            pred, _ = ah.predict(_state(t, prev, groups), reset_rnn=(t == 0))
            c = pred.reshape(-1).numpy().astype(np.int64)
            rd = {
                "contribution": c.tolist(),
                "contribution_valid": [True] * SIM_N_AGENTS,
                "punishment": [None] * SIM_N_AGENTS,
                "punishment_valid": [False] * SIM_N_AGENTS,
                "agent_group": groups.tolist(),
                "round": t,
            }
            # punishment: the manager sees the episode so far plus this round's
            # contributions, exactly as simulate.run_simulation calls it.
            p = pm.get_punishments(rounds + [rd]).numpy().astype(np.int64)
            rd = {
                **rd,
                "punishment": p.tolist(),
                "punishment_valid": [True] * SIM_N_AGENTS,
            }
            rounds.append(rd)
            cg = np.empty(SIM_N_AGENTS, dtype=float)
            for g in np.unique(groups):
                sel = groups == g
                n_valid = int(sel.sum())  # every agent is valid in this proxy
                cg[sel] = (1.6 * c[sel].sum() - p[sel].sum()) / n_valid
            rows_c.append(c)
            rows_ep.append(np.full(SIM_N_AGENTS, episode))
            rows_rd.append(np.full(SIM_N_AGENTS, t))
            rows_ag.append(groups.copy())
            prev = {"contribution": c, "punishment": p, "common_good": cg}
        latents.append(dict(ah._copula_z))  # captured before the next reset

    contrib = np.concatenate(rows_c)
    ep = np.concatenate(rows_ep)
    rdn = np.concatenate(rows_rd)
    agp = np.concatenate(rows_ag)
    agent = np.concatenate([np.arange(SIM_N_AGENTS)] * (SIM_EPISODES * SIM_ROUNDS))
    cell = np.stack([ep, rdn, agp], axis=1)
    out = spread_stats(contrib, ep, agent, cell)
    out["latents"] = latents
    out["mean_contribution"] = float(contrib.mean())
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def table(header, rows):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = "  ".join(str(h).ljust(w) for h, w in zip(header, widths))
    print("  " + line)
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print("  " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def arm_rows(arms, stats):
    """One table row per arm: levels, the delta vs the rho = 0 arm, and the
    prediction that delta implies at the parent's real baseline."""
    ref = stats[arms[0][0]]
    rows = []
    for label, rho_p, rho_t in arms:
        s = stats[label]
        d = s["ratio"] - ref["ratio"]
        cg, ratio = implied_cg(d)
        d_ca = s["sd_participant"] - ref["sd_participant"]
        rows.append(
            [
                label,
                f"{rho_p:.6g}",
                f"{rho_t:.6g}",
                f"{s['sd_gmean']:.4f}",
                f"{s['sd_ind']:.4f}",
                f"{s['ratio']:.4f}",
                f"{d:+.4f}",
                f"{ratio:.4f}",
                f"{cg:.3f}",
                band(cg),
                f"{s['sd_participant']:.4f}",
                f"{d_ca:+.4f}",
                f"{PARENT_CA_STD_DIFF + d_ca:+.4f}",
            ]
        )
    return rows


ARM_HEADER = [
    "arm",
    "rho_p",
    "rho_t",
    "SD(gmean)",
    "SD(ind)",
    "ratio",
    "d(ratio)",
    "pred ratio",
    "impl CG",
    "band",
    "SD(pmean)",
    "d(pmean)",
    "impl CA sd_diff",
]


def main():
    t_start = time.time()
    import joblib

    bundle = joblib.load(_ROOT / CONTR_BUNDLE)
    punisher = joblib.load(_ROOT / PUNISH_BUNDLE)
    params = json.loads((_ROOT / PARAMS_JSON).read_text())

    rho_total = float(bundle["copula_rho_p"])  # the stamped dose, read not typed
    rho_t_stamped = float(bundle["copula_rho_t"])
    rho_lag1 = float(params["rho_lag1"])
    two_comp = params["two_component_reading"]

    print("=" * 78)
    print("contribution group-copula preflight -- DIAGNOSTIC, NOT A GATE")
    print("=" * 78)
    print("The experiment proceeds to simulation at the fitted dose whatever")
    print("this prints; nothing here adjusts or recommends a dose.")
    print()
    print(f"  stamped dose        rho_p = {rho_total!r}, rho_t = {rho_t_stamped!r}")
    print(f"  structure           {bundle['copula_structure']}")
    print(f"  falsifier reading   rho_lag1 = {rho_lag1!r}")
    print(
        "  two-component       "
        f"rho_p = {two_comp['rho_p']!r}, rho_t = {two_comp['rho_t']!r}"
    )
    print(f"  parent baseline     ratio = {PARENT_RATIO!r} (the real sim)")
    print(f"  human               ratio = {HUMAN_RATIO!r}, denom {CG_DENOM!r}")
    print()

    arms = [
        ("(0, 0) reference", 0.0, 0.0),
        ("candidate (rho_total, 0)", rho_total, 0.0),
        ("falsifier (lag1, rest)", two_comp["rho_p"], two_comp["rho_t"]),
        ("transient (0, rho_total)", 0.0, rho_total),
    ]
    assert abs(rho_lag1 - two_comp["rho_p"]) < 1e-12
    assert abs(rho_lag1 + two_comp["rho_t"] - rho_total) < 1e-12

    # ---------------- proxy A ---------------- #
    print("-" * 78)
    print("PROXY A -- reduced-form Monte Carlo (linear mean, NO punisher)")
    print(
        f"  a = {MC_A}, b = {MC_B}, sigma = {MC_SIGMA}, k = {MC_K:.6g}, "
        f"{MC_N_GROUPS} groups x {MC_GROUP_SIZE} agents"
    )
    print(
        f"  {MC_EPISODES} episodes x {MC_ROUNDS} rounds, first {MC_BURN_IN} "
        f"rounds discarded, seed {MC_SEED}"
    )
    print("-" * 78)
    t0 = time.time()
    stats_a = {label: proxy_a(rp, rt) for label, rp, rt in arms}
    secs_a = time.time() - t0
    table(ARM_HEADER, arm_rows(arms, stats_a))
    ref_a = stats_a[arms[0][0]]["ratio"]
    print(
        f"\n  calibration: rho = 0 arm ratio {ref_a:.4f} against the log's "
        f"Note-8 expectation 0.6551 (diff {ref_a - 0.6551:+.4f})"
    )
    print(f"  proxy A took {secs_a:.1f} s")

    # ---------------- proxy B ---------------- #
    print()
    print("-" * 78)
    print("PROXY B -- closed-loop rollout of the REAL models")
    print(f"  contribution: {CONTR_BUNDLE}")
    print(f"  punisher:     {PUNISH_BUNDLE}")
    print(
        f"  fixed groups {SIM_AGENT_GROUPS.tolist()}, no switching, "
        f"{SIM_EPISODES} episodes x {SIM_ROUNDS} rounds, seed {SIM_SEED}"
    )
    print("  round order: contribution -> punishment -> common good -> prev_ shift")
    print("-" * 78)
    t0 = time.time()
    stats_b = {}
    for label, rp, rt in arms:
        stats_b[label] = proxy_b(bundle, punisher, rp, rt)
    secs_b = time.time() - t0
    table(ARM_HEADER, arm_rows(arms, stats_b))
    print(f"  proxy B took {secs_b:.1f} s ({len(arms)} rollouts)")
    ref_b = stats_b[arms[0][0]]["ratio"]
    print(
        f"\n  calibration: rho = 0 arm ratio {ref_b:.4f} against the log's "
        f"Note-9 expectation ~0.72 (diff {ref_b - 0.7216:+.4f} vs its 0.7216)"
    )
    for label, _, _ in arms:
        print(
            f"  mean contribution, {label}: "
            f"{stats_b[label]['mean_contribution']:.3f}"
        )

    # ---------------- per-episode latent reset ---------------- #
    print()
    print("-" * 78)
    print("PER-EPISODE LATENT RESET CHECK (log Note 26)")
    print("-" * 78)
    expected = 2 * SIM_EPISODES
    for label, _, _ in arms:
        lat = stats_b[label]["latents"]
        assert all(sorted(d) == [0, 1] for d in lat), "a group lost its latent"
        vals = [v for d in lat for v in d.values()]
        n_distinct = len(set(vals))
        print(
            f"  {label:<26} {len(lat)} episodes x 2 groups -> {len(vals)} "
            f"latents, {n_distinct} distinct (expected {expected})"
        )
        assert n_distinct == expected, "the persistent latent was not re-drawn"
    print(
        "  CONFIRMED: u_g is re-drawn at every episode boundary; the run is\n"
        "  not one giant episode with a frozen offset per group."
    )

    # ---------------- dose-response ---------------- #
    print()
    print("-" * 78)
    print("DOSE-RESPONSE, proxy B (persistent only, rho_t = 0)")
    print("-" * 78)
    rows = []
    for rp in DOSE_RESPONSE:
        hit = [lab for lab, p, t in arms if abs(p - rp) < 1e-12 and t == 0.0]
        s = stats_b[hit[0]] if hit else proxy_b(bundle, punisher, rp, 0.0)
        d = s["ratio"] - ref_b
        cg, ratio = implied_cg(d)
        rows.append(
            [
                f"{rp:.4f}",
                f"{s['sd_gmean']:.4f}",
                f"{s['sd_ind']:.4f}",
                f"{s['ratio']:.4f}",
                f"{d:+.4f}",
                f"{ratio:.4f}",
                f"{cg:.3f}",
                band(cg),
            ]
        )
    table(
        [
            "rho_p",
            "SD(gmean)",
            "SD(ind)",
            "ratio",
            "d(ratio)",
            "pred ratio",
            "CG",
            "band",
        ],
        rows,
    )
    slope = (float(rows[3][4]) - float(rows[1][4])) / (
        DOSE_RESPONSE[3] - DOSE_RESPONSE[1]
    )
    print(f"\n  local slope d(ratio)/d(rho_p) around the dose: {slope:.3f} per unit")

    # ---------------- the prediction ---------------- #
    d_a = stats_a[arms[1][0]]["ratio"] - stats_a[arms[0][0]]["ratio"]
    d_b = stats_b[arms[1][0]]["ratio"] - stats_b[arms[0][0]]["ratio"]
    cg_a, ratio_a = implied_cg(d_a)
    cg_b, ratio_b = implied_cg(d_b)
    lo, hi = sorted([cg_a, cg_b])
    print()
    print("=" * 78)
    print("PRE-REGISTERED PREDICTION for the real simulation (candidate arm)")
    print("=" * 78)
    print(f"  proxy A: d(ratio) {d_a:+.4f} -> ratio {ratio_a:.4f} -> CG {cg_a:.2f}")
    print(f"           band {band(cg_a)}")
    print(f"  proxy B: d(ratio) {d_b:+.4f} -> ratio {ratio_b:.4f} -> CG {cg_b:.2f}")
    print(f"           band {band(cg_b)}")
    print(f"  range spanned by the two proxies: CG {lo:.2f} to {hi:.2f}")
    print(
        f"  bands: {band(lo)} to {band(hi)}; the parent's CG is 5.91060457046713 "
        "(> 5)"
    )
    print()
    print(
        f"  THE SINGLE PRE-REGISTERED PREDICTION: CG {cg_b:.2f}, band "
        f"{band(cg_b)} -- proxy B's,"
    )
    print("  because it rolls out the REAL contribution and punisher models in a")
    print("  closed loop while proxy A only mimics them with a linear AR(1) mean.")
    print(
        f"  Range the two proxies span: CG {lo:.2f} ({band(lo)}) to "
        f"{hi:.2f} ({band(hi)})."
    )
    print()
    print("  Both proxies are OPEN-LOOP APPROXIMATIONS: proxy A has a linear")
    print("  mean and no punisher feedback, proxy B fixes group membership and")
    print("  disables switching (and treats every contribution as valid). They")
    print("  therefore predict the DELTA, not the level -- their rho = 0 arms")
    print("  sit below (A) and above (B) the real sim's 0.6915 -- and NEITHER")
    print("  includes the switch-slot interaction the real simulation has.")
    print()
    print(f"total runtime {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
