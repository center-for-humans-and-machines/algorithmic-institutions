"""Unit tests for the contribution copula sampler
(``aimanager.generic.copula.sample_levels_copula``): inverse-CDF convention,
marginal preservation, within-cell (not cross-cell) correlation, determinism,
composition-stable RNG consumption, the first-member latent rule, and parity
with the punisher adapter's ``_sample_levels_copula`` (PR #146), which this
module mirrors. Invariants and rationale:
notes/autoresearch_log/contribution-cg-copula.md and
notes/autoresearch_log/punisher-severity-copula.md (appendix).

The module under test is PyG-free by construction, so this runs locally:
    uv run python -m pytest tests/copula -v
"""

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch as th  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # tests/copula -> repo root
# this checkout's src must win over any installed/editable aimanager, so the
# sampler under test is the one in THIS worktree
sys.path.insert(0, str(ROOT / "src"))

from aimanager.generic import copula  # noqa: E402
from aimanager.generic.copula import sample_levels_copula  # noqa: E402
from aimanager.simulation.linear_ah import LinearAHAdapter  # noqa: E402

N_LEVELS = 21  # contribution levels 0..20
N_AGENTS = 8
GROUPS = [0, 0, 0, 0, 1, 1, 1, 1]
N_DRAW = 20000  # sampling repeats for the marginal / correlation tests
FEATS = ["prev_contribution", "prev_punishment", "round_number", "is_first"]
DEFAULTS = {
    "punishment": 0.0,
    "contribution": 9.0,
    "common_good": 12.333333333333334,
    "agent_group": 0.0,
    "contribution_valid": 0.0,
    "punishment_valid": 0.0,
    "recorded": 0.0,
}
# dyadic rows: exactly representable, so cumsum has no rounding slack and the
# <= convention can be pinned AT the bin edge rather than merely near it
DYADIC = [0.5, 0.25, 0.125, 0.0625, 0.0625]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def contribution_proba(peak, n_levels=N_LEVELS, scale=2.5, seed=0):
    """A contribution-like row: spike at the previous level with a smooth tail,
    plus a little asymmetry so no test can pass on a symmetric accident."""
    k = np.arange(n_levels, dtype=float)
    w = np.exp(-np.abs(k - peak) / scale) + 0.02 * np.exp(-((k - 0.0) ** 2) / 8.0)
    w = w * (1.0 + 0.15 * np.sin(k + seed))
    return w / w.sum()


def proba_block(peaks):
    """(len(peaks), N_LEVELS) float32 proba, one contribution-like row each."""
    rows = np.stack([contribution_proba(p, seed=i) for i, p in enumerate(peaks)])
    return th.tensor(rows, dtype=th.float32)


def repeat_rows(proba, n_draw):
    """(n_draw, A, K) view of one A-row proba block, one repeat per row."""
    return proba.unsqueeze(0).expand(n_draw, *proba.shape).contiguous()


def repeat_cells(groups, n_draw):
    """(n_draw, A) cells: the group pattern, made distinct per repeat so the
    repeats are independent cells rather than one giant cell."""
    g = th.tensor(groups, dtype=th.int64)
    return g.unsqueeze(0) + th.arange(n_draw).unsqueeze(1) * (int(g.max()) + 1)


def reference_levels(proba, cells, rho, member="first"):
    """Hand-computed reference: 2N draws (zs then eps), latent from the FIRST
    member of each cell in flattened row order (dict-based, as the punisher
    adapter does it), inverted through each row's own CDF. Reimplemented here
    so the comparison does not depend on the code under test. ``member="last"``
    builds the rule the sampler must NOT be following."""
    flat = proba.reshape(-1, proba.shape[-1]).double().numpy()
    n = len(flat)
    zs = th.randn(n, dtype=th.float64).numpy()
    eps = th.randn(n, dtype=th.float64).numpy()
    ids = [int(c) for c in cells.reshape(-1).tolist()]
    slots = {}
    for i, cid in enumerate(ids):
        if member == "last" or cid not in slots:
            slots[cid] = i
    pick = np.array([slots[cid] for cid in ids], dtype=np.int64)
    x = math.sqrt(rho) * zs[pick] + math.sqrt(1.0 - rho) * eps
    u = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    cum = np.cumsum(flat, axis=1)
    lvl = np.array([int(np.searchsorted(c, ui, "left")) for c, ui in zip(cum, u)])
    return th.tensor(np.clip(lvl, 0, flat.shape[1] - 1)).reshape(cells.shape)


def pair_corr(draws, pairs):
    """Mean Pearson correlation of sampled levels over the given column pairs."""
    return float(
        np.mean([np.corrcoef(draws[:, i], draws[:, j])[0, 1] for i, j in pairs])
    )


WITHIN = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7)]
CROSS = [(0, 4), (0, 5), (1, 6), (2, 7), (3, 4), (3, 7)]


# --------------------------------------------------------------------------- #
# (a) inverse-CDF correctness on a hand-built proba
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "u, expected",
    [
        (1e-12, 0),  # far below F(0)
        (0.25, 0),
        (0.5, 0),  # exactly F(0) -> still level 0 (F(a-1) < u <= F(a))
        (0.5 + 1e-9, 1),  # just above F(0) -> next level
        (0.75 - 1e-9, 1),
        (0.75 + 1e-9, 2),  # just above F(1)
        (0.875 + 1e-9, 3),
        (1.0 - 1e-12, 4),  # top level, no overflow past K - 1
    ],
)
def test_inverse_cdf_convention(monkeypatch, u, expected):
    """rho = 0.25 so sqrt(rho) = 0.5 exactly: the latent carries ndtri(u) / 0.5
    and comes back as ndtri(u) with no rounding, while the idiosyncratic draw
    is zeroed -- u reaches the inversion exactly as prescribed."""
    from scipy.special import ndtri

    rho = 0.25
    proba = th.tensor([DYADIC], dtype=th.float64)  # cumsum .5/.75/.875/.9375/1
    latent = np.array([ndtri(u) / math.sqrt(rho)])
    seen = []

    def fake_randn(n, dtype=None, device=None):
        seen.append(n)
        vals = latent if len(seen) == 1 else np.zeros(n)
        return th.tensor(vals, dtype=th.float64)

    monkeypatch.setattr(copula.th, "randn", fake_randn)
    lvl = sample_levels_copula(proba, th.zeros(1, dtype=th.int64), rho)
    assert seen == [1, 1], "expected exactly 2 randn calls of size N"
    assert lvl.tolist() == [expected]
    assert lvl.dtype == th.int64


def test_shape_and_dtype_follow_cells():
    """The copula branch must return what ``IntEncoder.decode`` returns at the
    same call site: int64 of shape ``proba.shape[:-1]``."""
    proba = th.rand(3, 4, N_LEVELS).softmax(-1)
    cells = th.zeros(3, 4, dtype=th.int64)
    th.manual_seed(0)
    lvl = sample_levels_copula(proba, cells, 0.4)
    assert lvl.shape == (3, 4) and lvl.dtype == th.int64
    assert int(lvl.min()) >= 0 and int(lvl.max()) <= N_LEVELS - 1


# --------------------------------------------------------------------------- #
# (b) marginals preserved at every rho
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rho", [0.0, 0.35, 0.9])
def test_marginals_match_proba(rho):
    """Per-row level frequencies must match the given proba within 4 binomial
    SEs (SE = sqrt(p(1-p)/N)); levels with fewer than 20 expected counts are
    skipped, the normal approximation not being usable there. rho changes the
    dependence only -- never the marginal."""
    proba = proba_block([3, 7, 10, 12, 15, 18, 20, 0])
    th.manual_seed(11)
    draws = sample_levels_copula(
        repeat_rows(proba, N_DRAW), repeat_cells(GROUPS, N_DRAW), rho
    ).numpy()
    P = proba.double().numpy()
    worst = 0.0
    for agent in range(N_AGENTS):
        freq = np.bincount(draws[:, agent], minlength=N_LEVELS) / N_DRAW
        for lvl in range(N_LEVELS):
            p = P[agent, lvl]
            if p * N_DRAW < 20:
                continue
            se = math.sqrt(p * (1.0 - p) / N_DRAW)
            worst = max(worst, abs(freq[lvl] - p) / se)
    assert worst < 4.0, f"rho={rho}: worst deviation {worst:.2f} binomial SEs"


def test_mean_contribution_unchanged():
    """Blunter but load-bearing: the per-agent mean level is what the group
    statistics are built from, and the copula must not move it."""
    proba = proba_block([3, 7, 10, 12, 15, 18, 20, 0])
    P = proba.double().numpy()
    k = np.arange(N_LEVELS)
    target = (P * k).sum(1)
    sd = np.sqrt((P * k**2).sum(1) - target**2)
    for rho in (0.0, 0.5):
        th.manual_seed(13)
        draws = sample_levels_copula(
            repeat_rows(proba, N_DRAW), repeat_cells(GROUPS, N_DRAW), rho
        ).numpy()
        dev = np.abs(draws.mean(0) - target) / (sd / math.sqrt(N_DRAW))
        assert dev.max() < 4.0, f"rho={rho}: worst mean shift {dev.max():.2f} SEs"


# --------------------------------------------------------------------------- #
# (c) within-cell correlation, cross-cell independence
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def same_row_draws():
    """All eight agents share one proba row, so any level correlation can only
    come from the shared latent, never from shared features."""
    proba = proba_block([9] * N_AGENTS)
    out = {}
    for name, rho in (("rho05", 0.5), ("rho0", 0.0)):
        th.manual_seed(12)
        out[name] = sample_levels_copula(
            repeat_rows(proba, N_DRAW), repeat_cells(GROUPS, N_DRAW), rho
        ).numpy()
    return out


def test_within_cell_correlation_induced(same_row_draws):
    """Discretising the latent costs dependence, so the level correlation lands
    below rho itself; asserted as "many SEs above zero" (the SE of a zero
    correlation is 1/sqrt(N))."""
    within = pair_corr(same_row_draws["rho05"], WITHIN)
    assert within > 6.0 / math.sqrt(N_DRAW), f"within-cell corr {within:.4f}"
    assert within > 0.2, f"within-cell corr {within:.4f} implausibly weak at rho=0.5"


def test_cross_cell_correlation_absent(same_row_draws):
    cross = pair_corr(same_row_draws["rho05"], CROSS)
    assert abs(cross) < 4.0 / math.sqrt(N_DRAW), f"cross-cell corr {cross:.4f}"


def test_rho_zero_leaves_rows_independent(same_row_draws):
    for pairs, label in ((WITHIN, "within"), (CROSS, "cross")):
        corr = pair_corr(same_row_draws["rho0"], pairs)
        assert abs(corr) < 4.0 / math.sqrt(N_DRAW), f"{label} corr {corr:.4f} at rho=0"


def test_one_cell_shares_one_latent():
    """Every row of a cell reads one latent: with identical proba rows and rho
    at the independence-free end, the sampled levels coincide."""
    proba = proba_block([9] * 4)
    th.manual_seed(3)
    lvl = sample_levels_copula(proba, th.zeros(4, dtype=th.int64), 0.999999)
    assert len(set(lvl.tolist())) == 1


# --------------------------------------------------------------------------- #
# (d) determinism and composition-stable RNG consumption
# --------------------------------------------------------------------------- #
def test_determinism_under_seed():
    proba = proba_block([3, 7, 10, 12, 15, 18, 20, 0])
    cells = th.tensor(GROUPS)
    th.manual_seed(42)
    first = [sample_levels_copula(proba, cells, 0.35) for _ in range(20)]
    th.manual_seed(42)
    again = [sample_levels_copula(proba, cells, 0.35) for _ in range(20)]
    for a, b in zip(first, again):
        assert th.equal(a, b)
    # ...and the sampler is genuinely drawing, not returning a fixed vector
    assert not all(th.equal(first[0], x) for x in first[1:])


@pytest.mark.parametrize("rho", [0.0, 0.35])
def test_cell_composition_does_not_shift_the_stream(rho):
    """Exactly 2N draws per call whatever the cell composition (rho = 0
    included), so partitions differ only through the latent assignment, never
    through a desynchronised RNG."""
    proba = proba_block([3, 7, 10, 12, 15, 18, 20, 0])
    partitions = [
        GROUPS,
        list(range(N_AGENTS)),  # all singletons
        [0] * N_AGENTS,  # one cell
        [5, 2, 5, 2, 5, 2, 5, 2],  # interleaved, unsorted ids
    ]
    tails = []
    for cells in partitions:
        th.manual_seed(7)
        sample_levels_copula(proba, th.tensor(cells), rho)
        tails.append(th.randn(1).item())
    assert len(set(tails)) == 1, f"RNG consumption varies with composition: {tails}"
    th.manual_seed(7)
    th.randn(2 * N_AGENTS, dtype=th.float64)
    assert th.randn(1).item() == tails[0], "expected exactly 2N float64 draws"


# --------------------------------------------------------------------------- #
# (e) the first-member latent rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cells",
    [
        [0, 0, 1, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 1, 0, 1, 0],  # same partition, first members swapped
        [7, 3, 7, 3, 3, 7, 3, 7],  # unsorted ids: th.unique sorts, order must not
        [4, 4, 4, 4, 4, 4, 4, 4],
    ],
)
def test_first_member_latent_rule(cells):
    """Each cell's latent is the `zs` slot of its FIRST member in flattened row
    order -- pinned against a dict-based reimplementation, so relabelling or
    reordering the ids cannot silently change which draw is shared."""
    proba = proba_block([3, 7, 10, 12, 15, 18, 20, 0])
    cells = th.tensor(cells)
    th.manual_seed(5)
    want = reference_levels(proba, cells, 0.35)
    th.manual_seed(5)
    got = sample_levels_copula(proba, cells, 0.35)
    assert th.equal(want, got), f"{want.tolist()} != {got.tolist()}"


def test_first_member_rule_matters():
    """Guard against a vacuous pin: reading each cell's LAST member's slot
    instead of its first gives different levels, so the rule above has teeth."""
    proba = proba_block([3, 7, 10, 12, 15, 18, 20, 0])
    cells = th.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    th.manual_seed(5)
    got = sample_levels_copula(proba, cells, 0.9)
    th.manual_seed(5)
    last = reference_levels(proba, cells, 0.9, member="last")
    assert not th.equal(got, last), "sampler is not using the first member"


def test_relabelling_cells_leaves_the_draw_unchanged():
    """Cell ids are labels, not an ordering: relabelling a partition moves no
    row, so the first member of each cell -- and the result -- is unchanged."""
    proba = proba_block([3, 7, 10, 12, 15, 18, 20, 0])
    th.manual_seed(5)
    a = sample_levels_copula(proba, th.tensor([0, 0, 1, 1, 0, 1, 0, 1]), 0.9)
    th.manual_seed(5)
    b = sample_levels_copula(proba, th.tensor([1, 1, 0, 0, 1, 0, 1, 0]), 0.9)
    assert th.equal(a, b)


def test_multi_dim_cells_flatten_row_major():
    """(rows, rounds) input: the flattened order the GNN decode sees is row
    then round, and the (row, round) result must equal the flat-call result."""
    proba = proba_block([3, 7, 10, 12]).reshape(2, 2, N_LEVELS)
    cells = th.tensor([[0, 1], [0, 1]])
    th.manual_seed(17)
    nested = sample_levels_copula(proba, cells, 0.4)
    th.manual_seed(17)
    flat = sample_levels_copula(
        proba.reshape(-1, N_LEVELS), cells.reshape(-1), 0.4
    )
    assert th.equal(nested.reshape(-1), flat)


# --------------------------------------------------------------------------- #
# (f) parity with the punisher adapter (PR #146)
# --------------------------------------------------------------------------- #
def _fixed_proba_adapter(rows, rho):
    """LinearAHAdapter over a stub estimator whose predict_proba returns `rows`
    verbatim. Dyadic rows sum to exactly 1, so _class_probs' floor-and-
    renormalise is the identity and both samplers invert the same CDF."""

    class FixedProba:
        classes_ = np.arange(len(rows[0]))

        def predict_proba(self, Xs):
            return np.asarray(rows, dtype=float).copy()

    return LinearAHAdapter(
        dict(
            model="multinomial",
            estimator=FixedProba(),
            scaler=None,
            features=list(FEATS),
            target="punishment",
            n_levels=len(rows[0]),
            default_values=dict(DEFAULTS),
            copula_rho=rho,
        )
    )


@pytest.mark.parametrize("groups", [[0, 0, 1, 1], [7, 3, 7, 3], [2, 2, 2, 2]])
@pytest.mark.parametrize("rho", [0.0, 0.35, 0.9])
def test_parity_with_linear_adapter(groups, rho):
    """Same probabilities, same cell ids, same seed -> the same levels as
    ``LinearAHAdapter._sample_levels_copula``. This is what lets the two slots
    be calibrated by one estimator."""
    rows = [
        DYADIC,
        DYADIC[::-1],
        [0.25, 0.25, 0.25, 0.125, 0.125],
        [0.125, 0.125, 0.5, 0.125, 0.125],
    ]
    ad = _fixed_proba_adapter(rows, rho)
    th.manual_seed(23)
    want = ad._sample_levels_copula(np.zeros((4, len(FEATS))), len(rows[0]), groups)
    th.manual_seed(23)
    got = sample_levels_copula(
        th.tensor(rows, dtype=th.float64), th.tensor(groups), rho
    )
    assert got.tolist() == want.tolist(), f"{got.tolist()} != {want.tolist()}"


def test_parity_leaves_the_stream_where_the_adapter_does():
    """Same 2N consumption as the adapter, so a mixed stack (linear punisher +
    copula contributor) keeps one reproducible stream."""
    rows = [DYADIC, DYADIC[::-1], DYADIC, DYADIC[::-1]]
    ad = _fixed_proba_adapter(rows, 0.35)
    th.manual_seed(29)
    ad._sample_levels_copula(np.zeros((4, len(FEATS))), len(rows[0]), [0, 0, 1, 1])
    after_adapter = th.randn(1).item()
    th.manual_seed(29)
    sample_levels_copula(
        th.tensor(rows, dtype=th.float64), th.tensor([0, 0, 1, 1]), 0.35
    )
    assert th.randn(1).item() == after_adapter


# --------------------------------------------------------------------------- #
# (g) input guards
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rho", [1.0, 1.5, -0.1, -1e-9])
def test_rho_outside_unit_interval_raises(rho):
    proba = proba_block([9, 9])
    with pytest.raises(AssertionError, match=r"\[0, 1\)"):
        sample_levels_copula(proba, th.zeros(2, dtype=th.int64), rho)


@pytest.mark.parametrize(
    "shape", [(3,), (2, 2), (4, 1), ()]  # too short, wrong rank, wrong length
)
def test_cells_shape_mismatch_raises(shape):
    proba = proba_block([9, 9])  # (2, K) -> cells must be (2,)
    with pytest.raises(AssertionError, match="does not match"):
        sample_levels_copula(proba, th.zeros(shape, dtype=th.int64), 0.35)
