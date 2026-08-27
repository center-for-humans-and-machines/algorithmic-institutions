"""Calibrate ``copula_rho`` for the AR-GNN punisher's severity copula.

The AR punisher (``artifacts/artificial_humans/punishment_ar_gnn_50ep_doubled``,
2750 epochs) already conditions each agent's punishment on the groupmate
punishments revealed before it. What it cannot see is the manager's round-level
severity mood -- the shared latent the multinomial punisher's copula supplied
outright. This script estimates that residual within-(episode, round, group)
correlation ``rho`` **against the AR model's own conditional CDFs**, then writes
a copy of the shipped checkpoint carrying ``copula_rho`` and nothing else.

Estimator (planning ruling R3, ``notes/autoresearch_log/punisher-ar-copula.md``):

* Conditionals are harvested by replaying the *training-time* reveal scheme.
  For each of ``R`` seeded reveal replicates a permutation of the 8 agents is
  drawn per episode; for ``k = 0 .. 7`` the agents at reveal position ``< k``
  carry their true punishments and the rest are masked to the default, exactly
  as ``aimanager.artificial_humans.train.apply_mask_pattern`` builds them, and
  one forward pass over the whole (n_episodes, 8, n_rounds) batch yields the
  conditional row of the agent at position ``k``. Every valid agent-round
  therefore gets one conditional per replicate, conditioned on the TRUE
  punishments of the agents revealed before it.
* Each observation ``y`` contributes the latent interval
  ``l = Phi^-1(F(y - 1))``, ``u = Phi^-1(F(y))`` (``F(-1) = 0``, clipped off
  0/1), where ``F`` is that conditional CDF.
* A cell is one (episode, round, agent_group) group with >= 2 valid agents.
  Its exact shared-latent log-likelihood is the 1-D integral

      log \\int phi(z) prod_i [ Phi((u_i - sqrt(rho) z) / sqrt(1 - rho))
                              - Phi((l_i - sqrt(rho) z) / sqrt(1 - rho)) ] dz

  evaluated with 64-node Gauss-Hermite quadrature
  (``\\int phi(z) g(z) dz = (1 / sqrt(pi)) sum_j w_j g(sqrt(2) x_j)``). A
  pairwise likelihood -- the estimator the linear severity copula used -- would
  be biased here because the AR conditionals are not exchangeable within a cell.
* The objective is the mean over reveal replicates of the summed cell
  log-likelihood; it is maximised over a coarse grid (0 .. 0.9 step 0.05)
  followed by bounded Brent, clipped to [0, 0.95].
* Uncertainty: cluster bootstrap over the 50 ``pair_id``s (both flip copies of a
  drawn pair enter together), 200 resamples, percentile 95% CI.
* ``--roundtrip`` is an ACCEPTANCE GATE, not a diagnostic: data are generated
  through the model itself under the shipped copula sampler semantics and the
  estimator must recover ``rho_true`` to within 0.03, otherwise the script
  exits non-zero.
* Every other split (per replicate, single copy vs doubled, randomized-PIT
  moment estimator, round thirds, cell sizes, CV folds) and ``--preflight`` are
  printed for context only and NEVER select ``rho``.

Runs inside an sbatch job on Raven (PyG is needed for the forward passes);
``python -m py_compile`` is the only local check. Every PyG / aimanager import
is deferred into a function so the pure-math helpers stay importable locally.

Usage (the documented cluster path; see
``configs/training/artificial_humans/punishment/ar_copula_rho.yml``):

    python scripts/artificial_humans/punishment_ar_copula_rho.py \\
        <job_file> --roundtrip --preflight
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

import numpy as np  # noqa: E402
import torch as th  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.special import logsumexp, ndtr, ndtri  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SEED = 38381  # the AR punisher's training seed (ar_gnn_50ep_doubled.yml)
N_REVEAL = 8  # reveal replicates averaged over in the likelihood
N_BOOT = 200  # cluster bootstrap resamples over the 50 pair_ids
N_QUAD = 64  # Gauss-Hermite nodes for the shared-latent integral
U_EPS = 1e-12  # CDF clip, keeps Phi^-1 finite
P_TINY = 1e-300  # log floor for an interval probability
RHO_GRID = np.arange(0.0, 0.9001, 0.05)
RHO_MAX = 0.95  # quadrature stays accurate well inside rho = 1
ROUNDTRIP_RHOS = (0.1, 0.3, 0.5)
N_ROUNDTRIP = 2  # synthetic datasets per rho_true
ROUNDTRIP_TOL = 0.03  # acceptance-gate tolerance on max |rho_hat - rho_true|
BOOT_HALFWIDTH = 0.2  # bootstrap grid half-width around the full-sample rho

# Seed offsets: disjoint streams, all derived from SEED so the run is fully
# reproducible. Reveal replicate r of the estimator uses SEED + r.
GEN_ORDER_SEED = SEED + 90000  # roundtrip generating reveal orders
GEN_DRAW_SEED = SEED + 91000  # roundtrip copula draws (torch)
PREFLIGHT_SEED = SEED + 92000  # pre-flight generative sweeps
PIT_SEED = SEED + 93000  # randomized-PIT diagnostic

GH_X, GH_W = np.polynomial.hermite.hermgauss(N_QUAD)
LOG_GH_W = np.log(GH_W) - 0.5 * np.log(np.pi)  # weights of the phi-integral
SQRT2 = float(np.sqrt(2.0))

DEFAULTS = {
    "device": "cpu",
    "seed": SEED,
    "basedir": ".",
    "data_file": "experiments/2group_8agent_50ep.csv",
    "experiment_names": ["ah_group_switching"],
    "switch_every": 4,
    "mask_name": "punishment_valid",
    "y_name": "punishment",
    "n_cross_val": 5,
    "n_reveal_replicates": N_REVEAL,
    "n_bootstrap": N_BOOT,
    "source_model": (
        "artifacts/artificial_humans/punishment_ar_gnn_50ep_doubled/model/"
        "architecture_node+edge+rnn+ar__dataset_50ep_doubled__epochs_2750.pt"
    ),
    "source_md5": "4774e934f08a96da01da875851ad7a2c",
    "output_dir": "artifacts/artificial_humans/punishment_ar_gnn_copula_50ep_doubled",
    "job_id": "architecture_node+edge+rnn+ar+copula__dataset_50ep_doubled__epochs_2750",
    "edge_encoding": [{"name": "ar_punishment", "n_levels": 31}],
}


def f(x):
    """Unrounded float for the log."""
    return repr(float(x))


class Tee:
    """Mirror stdout into the calibration log."""

    def __init__(self, path):
        self.stream = sys.__stdout__
        self.file = open(path, "w")

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return out.stdout.decode().strip()
    except Exception:  # pragma: no cover - provenance is best effort
        return None


def load_params(job_file):
    """Job params with standalone-runnable defaults.

    Accepts both the ``job.yml`` written by ``artificial_humans/run.py``
    (``params_only: true`` -> params at the top level) and the experiment
    config itself (params nested under ``params``), so the script can be
    dry-run without submitting anything.
    """
    params = dict(DEFAULTS)
    if job_file is None:
        return params
    import yaml

    with open(job_file, "r") as fh:
        loaded = yaml.safe_load(fh) or {}
    if isinstance(loaded.get("params"), dict):
        loaded = loaded["params"]
    unknown = sorted(set(loaded) - set(DEFAULTS))
    params.update({k: v for k, v in loaded.items() if k in DEFAULTS})
    params["_unknown_job_keys"] = unknown
    return params


# --------------------------------------------------------------------------- #
# model and data -- both rebuilt exactly as training built them
# --------------------------------------------------------------------------- #
def load_model(path, expect_md5, device, edge_encoding):
    """Load the shipped AR checkpoint and assert its identity."""
    # pytorch geometric moved the meta module after the checkpoint was written
    import torch_geometric.nn.models.meta as meta_module

    sys.modules["torch_geometric.nn.meta"] = meta_module
    from aimanager.generic.graph import GraphNetwork

    got = md5sum(path)
    assert got == expect_md5, f"source .pt md5 {got} != expected {expect_md5}"
    model = GraphNetwork.load(path, device=device)
    model = model.to(device)
    model.eval()
    assert model.autoregressive is True, "source checkpoint is not autoregressive"
    assert model.y_name == "punishment", f"y_name is {model.y_name!r}"
    assert model.copula_rho is None, f"source already has rho {model.copula_rho!r}"
    assert (
        model.edge_encoding == edge_encoding
    ), f"edge_encoding {model.edge_encoding!r} != {edge_encoding!r}"
    assert model.edge_encoder.size == 2, "ar_punishment must contribute 2 channels"
    return model


def build_data(params):
    """The full flip-doubled training tensors, rebuilt with train.py's path.

    Mirrors ``aimanager.artificial_humans.train.main`` lines "seed -> read_csv ->
    experiment filter -> create_torch_data(switch_every)". The shipped
    checkpoint is the ``i is None`` full-data fit of ``get_cross_validations``
    (all 100 doubled episodes), so no fold subsetting is applied; that yield
    only permutes the episode axis, which the estimator is invariant to.
    """
    import pandas as pd
    from aimanager.generic.data import create_torch_data

    seed = int(params["seed"])
    th.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    df = pd.read_csv(os.path.join(params["basedir"], params["data_file"]))
    df = df[df["experiment_name"].isin(params["experiment_names"])]
    data, default_values, pair_id = create_torch_data(
        df, switch_every=params["switch_every"]
    )
    return data, default_values, np.asarray(pair_id)


def cv_fold_episodes(n_episodes, n_splits, pair_id, seed):
    """Training's CV fold membership, recovered by calling the real splitter.

    ``get_cross_validations`` shuffles with the ``random`` module only, and
    ``create_torch_data`` never touches it, so re-seeding here reproduces the
    exact folds the training run used. A probe tensor holding the episode index
    turns the yielded data dicts back into index lists.
    """
    from aimanager.generic.data import get_cross_validations

    random.seed(seed)
    probe = {"contribution": th.arange(n_episodes).reshape(n_episodes, 1, 1)}
    folds = []
    for i, _, test in get_cross_validations(probe, n_splits, 1.0, group_key=pair_id):
        if i is None or test is None:
            continue
        folds.append((i, test["contribution"].flatten().numpy()))
    return folds


# --------------------------------------------------------------------------- #
# reveal scheme
# --------------------------------------------------------------------------- #
def reveal_orders(n_episodes, n_agents, seed):
    """One reveal permutation of the agents per episode.

    ``predict_autoreg`` draws a single ``np.random.permutation`` of the nodes
    per call; harvesting per episode is this script's calibration choice (it
    multiplies the conditional coverage by n_episodes at no extra forward-pass
    cost) and uses the same legacy RandomState algorithm.
    """
    rng = np.random.RandomState(seed)
    return np.stack([rng.permutation(n_agents) for _ in range(n_episodes)])


def reveal_positions(orders):
    """``pos[e, a]`` = reveal position of agent ``a`` in episode ``e``."""
    n_episodes, n_agents = orders.shape
    pos = np.empty_like(orders)
    ranks = np.tile(np.arange(n_agents), (n_episodes, 1))
    np.put_along_axis(pos, orders, ranks, axis=1)
    return pos


def step_probs(model, data, pattern, default_values, edge_index, params):
    """Conditional class probabilities for one reveal step.

    ``pattern`` is True for the agents still to be predicted, the polarity
    ``apply_mask_pattern`` and ``predict_autoreg`` both use: their punishments
    are masked to the default and their ``autoreg_mask`` is True, so the
    ``ar_punishment`` gate only lets the already-revealed groupmates through.
    """
    from aimanager.artificial_humans.train import apply_mask_pattern

    n_episodes, n_agents, n_rounds = data[params["y_name"]].shape
    masked = apply_mask_pattern(
        data,
        pattern,
        params["y_name"],
        params["mask_name"],
        default_values,
    )
    with th.no_grad():
        encoded = model.encode(
            masked, y_encode=False, edge_index=edge_index, device=model.device
        )
        logit = model(encoded)
        proba = th.nn.functional.softmax(logit, dim=-1)
    return proba.reshape(n_episodes, n_agents, n_rounds, model.y_levels).double()


def harvest_one(model, data, default_values, edge_index, orders, obs, params):
    """(F(y - 1), F(y)) for every valid observation under one reveal order."""
    y = data[params["y_name"]]
    n_episodes, n_agents, n_rounds = y.shape
    pos = reveal_positions(orders)
    ep_all = th.arange(n_episodes)
    cdf_lo = th.zeros((n_episodes, n_agents, n_rounds), dtype=th.float64)
    cdf_hi = th.zeros((n_episodes, n_agents, n_rounds), dtype=th.float64)
    for k in range(n_agents):
        pattern = th.from_numpy(pos >= k)
        proba = step_probs(model, data, pattern, default_values, edge_index, params)
        target = th.from_numpy(orders[:, k])
        rows = proba[ep_all, target]  # (n_episodes, n_rounds, n_levels)
        cum = rows.cumsum(-1)
        idx = y[ep_all, target].unsqueeze(-1)  # (n_episodes, n_rounds, 1)
        hi = cum.gather(-1, idx).squeeze(-1)
        p_y = rows.gather(-1, idx).squeeze(-1)
        cdf_hi[ep_all, target] = hi
        cdf_lo[ep_all, target] = hi - p_y
    ep, ag, rd = obs["episode"], obs["agent"], obs["round"]
    lo = cdf_lo.numpy()[ep, ag, rd]
    hi = cdf_hi.numpy()[ep, ag, rd]
    # every agent is the target of exactly one reveal step, so every valid
    # observation must carry a strictly positive conditional mass
    assert np.all(hi > lo), "an observation was left without a conditional row"
    return lo, hi


def harvest(model, data, default_values, edge_index, obs, params, n_rep, seed_base):
    """Stacked (R, n_obs) conditional CDF bounds over the reveal replicates."""
    y = data[params["y_name"]]
    n_episodes, n_agents, _ = y.shape
    lo, hi = [], []
    for r in range(n_rep):
        orders = reveal_orders(n_episodes, n_agents, seed_base + r)
        a, b = harvest_one(model, data, default_values, edge_index, orders, obs, params)
        lo.append(a)
        hi.append(b)
    return np.stack(lo), np.stack(hi)


def latent_bounds(cdf_lo, cdf_hi):
    """Latent interval endpoints, clipped off 0 and 1."""
    return (
        ndtri(np.clip(cdf_lo, U_EPS, 1.0 - U_EPS)),
        ndtri(np.clip(cdf_hi, U_EPS, 1.0 - U_EPS)),
    )


# --------------------------------------------------------------------------- #
# the estimator: exact cell-level shared-latent MLE
# --------------------------------------------------------------------------- #
class CellLikelihood:
    """Per-cell log-likelihood of the exchangeable Gaussian copula.

    Observations must already be sorted by cell so ``np.add.reduceat`` sees
    contiguous blocks. ``__call__(rho)`` returns an ``(R, n_cells)`` array over
    the cells with >= 2 valid agents; the objective is a weighted sum of its
    columns averaged over the replicate rows, which makes every bootstrap
    resample and diagnostic split a re-weighting of the same array.
    """

    def __init__(self, lo, hi, starts, keep):
        self.lo = np.ascontiguousarray(lo)
        self.hi = np.ascontiguousarray(hi)
        self.starts = starts
        self.keep = keep
        self.n_rep = lo.shape[0]
        self.n_cells = int(keep.sum())
        self.evals = 0

    def __call__(self, rho, rows=None):
        rho = float(np.clip(rho, 0.0, RHO_MAX))
        self.evals += 1
        a = float(np.sqrt(rho))
        b = float(np.sqrt(1.0 - rho))
        shift = (a * SQRT2) * GH_X[None, :]  # (1, Q)
        reps = range(self.n_rep) if rows is None else list(rows)
        out = np.empty((len(reps), self.n_cells))
        for j, r in enumerate(reps):
            upper = ndtr((self.hi[r][:, None] - shift) / b)
            lower = ndtr((self.lo[r][:, None] - shift) / b)
            logp = np.log(np.maximum(upper - lower, P_TINY))
            block = np.add.reduceat(logp, self.starts, axis=0)[self.keep]
            out[j] = logsumexp(block + LOG_GH_W[None, :], axis=1)
        return out


def make_objective(cell_lik, weights, rows=None):
    """Total log-likelihood: sum over weighted cells, mean over replicates."""

    def ll(rho):
        return float((cell_lik(rho, rows=rows) @ weights).mean())

    return ll


def maximise(ll, grid=RHO_GRID, cached=None):
    """Coarse grid then bounded Brent inside the bracketing interval."""
    if cached is None:
        vals = np.array([ll(r) for r in grid])
    else:
        vals = np.asarray(cached)
    b = int(np.argmax(vals))
    lo = float(grid[max(b - 1, 0)])
    hi = float(min(grid[min(b + 1, len(grid) - 1)], RHO_MAX))
    if hi <= lo:
        return float(np.clip(grid[b], 0.0, RHO_MAX)), float(vals[b])
    res = minimize_scalar(
        lambda r: -ll(r), bounds=(lo, hi), method="bounded", options=dict(xatol=1e-6)
    )
    rho = float(np.clip(res.x, 0.0, RHO_MAX))
    return rho, float(-res.fun)


def quadrature_error(seed=SEED, n=100):
    """Max relative error of the 64-node quadrature, per rho (provenance).

    Checks the 2-agent case, the only cell size with an independent reference:
    the shared-latent integral there is a bivariate-normal rectangle, which
    scipy's ``multivariate_normal.cdf`` computes to ~1e-12. Printed at startup
    so the log records how much the quadrature can be trusted at the rho the
    MLE actually lands on -- Gauss-Hermite loses accuracy as rho approaches 1.
    """
    try:
        from scipy.stats import multivariate_normal as mvn
    except ImportError:  # pragma: no cover - reference is optional
        return None
    rng = np.random.default_rng(seed)
    lo = rng.uniform(-2.5, 0.5, (1, 2 * n))
    hi = lo + rng.uniform(0.2, 2.5, (1, 2 * n))
    starts = np.arange(0, 2 * n, 2)
    keep = np.ones(n, bool)
    lik = CellLikelihood(lo, hi, starts, keep)
    kw = dict(mean=[0.0, 0.0], abseps=1e-12, releps=1e-12, maxpts=10**7)
    out = {}
    for rho in (0.1, 0.3, 0.5, 0.7, 0.9, RHO_MAX):
        mine = np.exp(lik(rho)[0])
        cov = [[1.0, rho], [rho, 1.0]]
        ref = np.array(
            [
                mvn.cdf([hi[0, 2 * c], hi[0, 2 * c + 1]], cov=cov, **kw)
                - mvn.cdf([lo[0, 2 * c], hi[0, 2 * c + 1]], cov=cov, **kw)
                - mvn.cdf([hi[0, 2 * c], lo[0, 2 * c + 1]], cov=cov, **kw)
                + mvn.cdf([lo[0, 2 * c], lo[0, 2 * c + 1]], cov=cov, **kw)
                for c in range(n)
            ]
        )
        out[round(float(rho), 4)] = float(np.abs(mine / ref - 1.0).max())
    return out


def cell_blocks(cell):
    """Sort observations by cell; return (order, starts, sizes)."""
    order = np.argsort(cell, kind="stable")
    k = cell[order]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    sizes = np.diff(np.r_[starts, len(k)])
    return order, starts, sizes


# --------------------------------------------------------------------------- #
# cluster bootstrap over pair_ids
# --------------------------------------------------------------------------- #
def bootstrap(cell_lik, pair_of_cell, n_pairs, n_boot, seed, rho_hat):
    """Percentile bootstrap: resample the 50 pairs, re-maximise per resample.

    Both flip copies of a drawn pair enter together because a pair's cells are
    pooled by ``pair_of_cell``. The coarse grid is narrowed around the
    full-sample estimate and its cell-likelihood arrays are computed once and
    reused by every resample; only the Brent refinement costs new evaluations.
    """
    grid = np.clip(
        np.arange(rho_hat - BOOT_HALFWIDTH, rho_hat + BOOT_HALFWIDTH + 1e-9, 0.05),
        0.0,
        RHO_MAX,
    )
    grid = np.unique(np.round(grid, 6))
    cache = {float(r): cell_lik(r) for r in grid}
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        draw = rng.integers(0, n_pairs, size=n_pairs)
        weights = np.bincount(draw, minlength=n_pairs).astype(float)[pair_of_cell]
        cached = [float((cache[float(r)] @ weights).mean()) for r in grid]
        ll = make_objective(cell_lik, weights)
        out[i], _ = maximise(ll, grid=grid, cached=cached)
    return out


# --------------------------------------------------------------------------- #
# generative sweep: the shipped copula sampler on the real covariates
# --------------------------------------------------------------------------- #
def generate(model, data, default_values, edge_index, orders, rho, draw_seed, params):
    """Model-generated punishments under the copula sampler semantics.

    Exactly the process ``GraphNetwork.predict_autoreg`` implements -- one
    shared latent ``z`` per (episode, group, round) held fixed across the AR
    steps, per-agent noise ``eps``, ``u = Phi(sqrt(rho) z + sqrt(1-rho) eps)``
    inverted through the agent's own conditional CDF with the ``min{a : F(a)
    >= u}`` convention -- except that the node features (``prev_punishment``
    included) are held at their real values, so a synthetic dataset differs
    from the real one only in the punishments being calibrated. Values are
    revealed into the conditioning tensor as they are drawn, so later agents
    condition on the synthetic groupmates, not the human ones.
    """
    y_name = params["y_name"]
    y = data[y_name]
    n_episodes, n_agents, n_rounds = y.shape
    n_levels = model.y_levels
    n_groups = int(data["agent_group"].max()) + 1
    gen = th.Generator()
    gen.manual_seed(int(draw_seed))
    z = th.randn((n_episodes, n_groups, n_rounds), generator=gen, dtype=th.float64)
    a = float(np.sqrt(rho))
    b = float(np.sqrt(1.0 - rho))

    y_syn = th.full_like(y, int(default_values[y_name]))
    pos = reveal_positions(orders)
    ep_all = th.arange(n_episodes)
    synthetic = dict(data)
    for k in range(n_agents):
        synthetic[y_name] = y_syn
        pattern = th.from_numpy(pos >= k)
        proba = step_probs(
            model, synthetic, pattern, default_values, edge_index, params
        )
        target = th.from_numpy(orders[:, k])
        rows = proba[ep_all, target]  # (n_episodes, n_rounds, n_levels)
        group = data["agent_group"][ep_all, target].long()
        eps = th.randn((n_episodes, n_rounds), generator=gen, dtype=th.float64)
        z_g = z.gather(1, group.unsqueeze(1)).squeeze(1)
        u = th.special.ndtr(a * z_g + b * eps)
        cum = rows.cumsum(-1).contiguous()
        lvl = th.searchsorted(cum, u.unsqueeze(-1).contiguous()).squeeze(-1)
        y_syn[ep_all, target] = lvl.clamp(0, n_levels - 1).to(y_syn.dtype)
    return y_syn


def estimate(model, data, default_values, edge_index, obs, params, n_rep, seed_base):
    """Full estimator on one dataset: harvest, then maximise."""
    cdf_lo, cdf_hi = harvest(
        model, data, default_values, edge_index, obs, params, n_rep, seed_base
    )
    lo, hi = latent_bounds(cdf_lo, cdf_hi)
    cell_lik = CellLikelihood(lo, hi, obs["starts"], obs["keep"])
    ones = np.ones(cell_lik.n_cells)
    rho, ll_hat = maximise(make_objective(cell_lik, ones))
    return rho, ll_hat, cell_lik, (cdf_lo, cdf_hi)


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def pit_moment_rho(cdf_lo, cdf_hi, starts, sizes, seed):
    """Randomized-PIT exchangeable moment estimator (ATTENUATED).

    Kept only as a cross-check of the MLE's sign and rough size: randomising
    inside each observation's CDF interval adds independent noise, which
    shrinks the estimate towards zero.
    """
    rng = np.random.default_rng(seed)
    keep = sizes >= 2
    n_pairs = (sizes[keep] * (sizes[keep] - 1) // 2).sum()
    out = []
    for r in range(cdf_lo.shape[0]):
        v = rng.random(cdf_lo.shape[1])
        u = cdf_lo[r] + v * (cdf_hi[r] - cdf_lo[r])
        z = ndtri(np.clip(u, U_EPS, 1.0 - U_EPS))
        z = z - z.mean()
        var = z.var(ddof=1)
        s = np.add.reduceat(z, starts)[keep]
        q = np.add.reduceat(z**2, starts)[keep]
        out.append(((s**2 - q) / 2.0).sum() / (n_pairs * var))
    return np.asarray(out)


def spread_stats(values, cell_index, n_cells):
    """(std of per-cell mean punishment, that std / std of the individuals).

    The PD-flavoured group-spread statistic in its simple form: one mean per
    (episode, round, agent_group) cell, its spread across cells, and the ratio
    the evaluation suite's PD row is built from.
    """
    counts = np.bincount(cell_index, minlength=n_cells)
    sums = np.bincount(cell_index, weights=values.astype(float), minlength=n_cells)
    means = sums[counts > 0] / counts[counts > 0]
    sd_cell = float(np.std(means, ddof=1))
    sd_ind = float(np.std(values.astype(float), ddof=1))
    return sd_cell, sd_cell / sd_ind


# --------------------------------------------------------------------------- #
# checkpoint
# --------------------------------------------------------------------------- #
def write_checkpoint(source, target, rho, device, edge_encoding):
    """Source pickle dict + ``copula_rho``, nothing else touched."""
    from aimanager.generic.graph import GraphNetwork

    src = th.load(source, map_location=device)
    assert "copula_rho" not in src, "source checkpoint already carries copula_rho"
    out = dict(src)
    out["copula_rho"] = float(rho)
    target.parent.mkdir(parents=True, exist_ok=True)
    th.save(out, target)

    back = th.load(target, map_location=device)
    assert set(back) - set(src) == {"copula_rho"}, "unexpected new checkpoint keys"
    assert set(src) - set(back) == set(), "checkpoint keys were dropped"
    assert back["copula_rho"] == float(rho)

    ref = GraphNetwork.load(source, device=device)
    got = GraphNetwork.load(str(target), device=device)
    ref_sd, got_sd = ref.state_dict(), got.state_dict()
    assert set(ref_sd) == set(got_sd), "parameter names differ after reload"
    for k in ref_sd:
        assert ref_sd[k].dtype == got_sd[k].dtype, f"dtype changed for {k}"
        assert th.equal(ref_sd[k], got_sd[k]), f"parameter {k} is not bit-identical"
    assert got.autoregressive is True
    assert got.copula_rho == float(rho)
    assert got.edge_encoding == edge_encoding
    assert got.y_levels == ref.y_levels and got.y_name == ref.y_name
    assert got.default_values == ref.default_values
    return len(ref_sd)


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description="Calibrate the AR punisher's rho.")
    ap.add_argument("job_file", nargs="?", default=None, help="job.yml with params")
    ap.add_argument(
        "--roundtrip",
        action="store_true",
        help="ACCEPTANCE GATE: recover a known rho from model-generated data",
    )
    ap.add_argument(
        "--preflight",
        action="store_true",
        help="print the group-spread statistics (go/no-go context only)",
    )
    args = ap.parse_args()

    params = load_params(args.job_file)
    device = th.device(params["device"])
    assert params["device"] == "cpu", "calibration is pinned to cpu for determinism"
    out_dir = Path(params["output_dir"])
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tee = Tee(metrics_dir / "copula_rho_calibration.log")
    sys.stdout = tee
    status = 0
    try:
        status = run(args, params, device, out_dir, metrics_dir, t0)
    finally:
        sys.stdout = tee.stream
        tee.close()
    sys.exit(status)


def run(args, params, device, out_dir, metrics_dir, t0):
    th.manual_seed(int(params["seed"]))
    n_rep = int(params["n_reveal_replicates"])
    n_boot = int(params["n_bootstrap"])
    y_name = params["y_name"]

    print("=== AR punisher severity-copula calibration ===")
    print(f"job_file            {args.job_file}")
    if params.get("_unknown_job_keys"):
        print(
            "  job keys unused by this script (run.py bookkeeping): "
            f"{params['_unknown_job_keys']}"
        )
    print(f"device              {params['device']}   torch {th.__version__}")
    print(f"seed                {params['seed']}   reveal replicates R={n_rep}")
    print(f"git commit          {git_commit()}")

    source = params["source_model"]
    model = load_model(source, params["source_md5"], device, params["edge_encoding"])
    print(f"source checkpoint   {source}")
    print(f"  md5               {params['source_md5']} (asserted)")
    print(
        f"  autoregressive={model.autoregressive} copula_rho={model.copula_rho} "
        f"y_levels={model.y_levels} edge_encoding={model.edge_encoding}"
    )

    data, default_values, pair_id = build_data(params)
    assert float(default_values[y_name]) == float(
        model.default_values[y_name]
    ), "rebuilt default punishment differs from the checkpoint's"
    y = data[y_name]
    n_episodes, n_agents, n_rounds = y.shape
    valid = data[params["mask_name"]].numpy().astype(bool)
    ep, ag, rd = np.nonzero(valid)
    group = data["agent_group"].numpy()[ep, ag, rd]
    n_groups = int(data["agent_group"].max()) + 1
    cell = (ep * n_rounds + rd) * n_groups + group
    order, starts, sizes = cell_blocks(cell)
    keep = sizes >= 2

    obs = {
        "episode": ep[order],
        "agent": ag[order],
        "round": rd[order],
        "group": group[order],
        "cell": cell[order],
        "starts": starts,
        "sizes": sizes,
        "keep": keep,
    }
    cell_episode = obs["episode"][starts]
    cell_round = obs["round"][starts]
    pairs = np.unique(pair_id)
    pair_rank = {int(p): i for i, p in enumerate(pairs)}
    pair_of_cell = np.array(
        [pair_rank[int(pair_id[e])] for e in cell_episode[keep]], dtype=np.int64
    )

    print(f"data                {params['data_file']}")
    print(
        f"  episodes={n_episodes} agents={n_agents} rounds={n_rounds} "
        f"pair_ids={len(pairs)} groups={n_groups}"
    )
    print(
        f"  valid rows={len(ep)} of {valid.size}   cells={len(sizes)} "
        f"cells>=2={int(keep.sum())}"
    )
    print(f"  cell size histogram={np.bincount(sizes).tolist()}")
    print(f"  default punishment={f(default_values[y_name])}")

    quad_err = quadrature_error()
    print(
        f"quadrature          {N_QUAD}-node Gauss-Hermite, max relative error "
        f"vs scipy mvn on 2-agent cells"
    )
    print(f"  per rho           {quad_err}")

    edge_index = model.create_fully_connected(n_agents, n_batch=n_episodes)

    # ------------------------- the estimator ------------------------- #
    t_h = time.time()
    rho_hat, ll_hat, cell_lik, cdf = estimate(
        model, data, default_values, edge_index, obs, params, n_rep, int(params["seed"])
    )
    ones = np.ones(cell_lik.n_cells)
    ll_zero = make_objective(cell_lik, ones)(0.0)
    print("\n=== rho: exact cell-level shared-latent MLE (full doubled data) ===")
    print(f"rho_hat                      {f(rho_hat)}")
    print(f"  log-likelihood(rho_hat)    {f(ll_hat)}")
    print(f"  log-likelihood(0)          {f(ll_zero)}")
    print(f"  2*(ll(rho_hat) - ll(0))    {f(2.0 * (ll_hat - ll_zero))}")
    print(
        f"  cells={cell_lik.n_cells}  quadrature nodes={N_QUAD}  "
        f"likelihood evals={cell_lik.evals}   [{time.time() - t_h:.1f}s]"
    )

    t_b = time.time()
    boot = bootstrap(
        cell_lik, pair_of_cell, len(pairs), n_boot, int(params["seed"]), rho_hat
    )
    se = float(boot.std(ddof=1))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    print(f"cluster bootstrap ({n_boot} resamples over {len(pairs)} pair_ids)")
    print(f"  SE                         {f(se)}")
    print(f"  95% percentile CI          [{f(ci[0])}, {f(ci[1])}]")
    print(
        f"  min/max                    {f(boot.min())} / {f(boot.max())}"
        f"   [{time.time() - t_b:.1f}s]"
    )

    # ------------------------- acceptance gate ------------------------- #
    roundtrip_rows = []
    gate_pass = None
    if args.roundtrip:
        t_r = time.time()
        print(
            f"\n=== ACCEPTANCE GATE: round-trip recovery "
            f"({N_ROUNDTRIP} synthetic dataset(s) per rho, tolerance "
            f"{ROUNDTRIP_TOL}) ==="
        )
        print(
            "the oracle column re-estimates with the GENERATING reveal order as "
            "the single replicate; it isolates reveal-order mismatch and is "
            "context only, never the gate"
        )
        print(
            "rho_true   rho_hat                 bias                   "
            "rho_hat (oracle reveal order)"
        )
        for i, rho_true in enumerate(ROUNDTRIP_RHOS):
            for d in range(N_ROUNDTRIP):
                tag = 1000 * i + d
                gen_orders = reveal_orders(n_episodes, n_agents, GEN_ORDER_SEED + tag)
                y_syn = generate(
                    model,
                    data,
                    default_values,
                    edge_index,
                    gen_orders,
                    rho_true,
                    GEN_DRAW_SEED + tag,
                    params,
                )
                syn = {**data, y_name: y_syn}
                rho_syn, _, _, _ = estimate(
                    model,
                    data=syn,
                    default_values=default_values,
                    edge_index=edge_index,
                    obs=obs,
                    params=params,
                    n_rep=n_rep,
                    seed_base=int(params["seed"]),
                )
                # context only: the estimator handed the generating order as
                # its single replicate, isolating reveal-order mismatch
                lo_o, hi_o = harvest_one(
                    model, syn, default_values, edge_index, gen_orders, obs, params
                )
                lo_o, hi_o = latent_bounds(lo_o[None], hi_o[None])
                oracle_lik = CellLikelihood(lo_o, hi_o, starts, keep)
                rho_oracle, _ = maximise(
                    make_objective(oracle_lik, np.ones(oracle_lik.n_cells))
                )
                roundtrip_rows.append(
                    dict(
                        rho_true=float(rho_true),
                        dataset=d,
                        rho_hat=float(rho_syn),
                        bias=float(rho_syn - rho_true),
                        rho_hat_oracle_order=float(rho_oracle),
                    )
                )
                print(
                    f"{rho_true:<10.2f} {f(rho_syn):<23} "
                    f"{f(rho_syn - rho_true):<22} {f(rho_oracle)}"
                )
        bias = max(abs(r["bias"]) for r in roundtrip_rows)
        gate_pass = bool(bias <= ROUNDTRIP_TOL)
        print(
            f"max |bias| = {f(bias)}  ->  "
            f"{'PASS' if gate_pass else 'FAIL'} (tolerance {ROUNDTRIP_TOL})"
            f"   [{time.time() - t_r:.1f}s]"
        )
        if not gate_pass:
            print("\n!!! ROUND-TRIP ACCEPTANCE GATE FAILED -- rho is NOT usable !!!")
            print("!!! no checkpoint written; see the plan's step 9-10 rulings  !!!")
            return 1

    # ------------------------- diagnostics ------------------------- #
    print("\n=== diagnostics only (never a selection criterion) ===")
    per_rep = [
        maximise(make_objective(cell_lik, ones, rows=[r]))[0] for r in range(n_rep)
    ]
    print(
        f"per-replicate rho            min {f(min(per_rep))} max "
        f"{f(max(per_rep))} sd {f(np.std(per_rep, ddof=1))}"
    )
    print(f"  values                     {[round(v, 6) for v in per_rep]}")

    first_copy = np.zeros(n_episodes, dtype=bool)
    seen = set()
    for e in range(n_episodes):
        p = int(pair_id[e])
        if p not in seen:
            seen.add(p)
            first_copy[e] = True
    w_single = first_copy[cell_episode[keep]].astype(float)
    rho_single, _ = maximise(make_objective(cell_lik, w_single))
    print(
        f"single copy per pair         {f(rho_single)}   "
        f"(doubled {f(rho_hat)}; cells={int(w_single.sum())})"
    )

    pit = pit_moment_rho(cdf[0], cdf[1], starts, sizes, PIT_SEED)
    print(
        f"randomized-PIT moment        {f(pit.mean())}  ATTENUATED "
        f"(min {f(pit.min())} max {f(pit.max())})"
    )

    kept_round = cell_round[keep]
    for third in np.array_split(np.arange(n_rounds), 3):
        w = np.isin(kept_round, third).astype(float)
        rho_t, _ = maximise(make_objective(cell_lik, w))
        print(
            f"rounds {third[0]:>2}-{third[-1]:<2}                 {f(rho_t)}   "
            f"cells={int(w.sum())}"
        )

    kept_size = sizes[keep]
    for s in sorted(set(kept_size.tolist())):
        w = (kept_size == s).astype(float)
        rho_s, _ = maximise(make_objective(cell_lik, w))
        print(f"cell size {s}                  {f(rho_s)}   cells={int(w.sum())}")

    try:
        folds = cv_fold_episodes(
            n_episodes, int(params["n_cross_val"]), pair_id, int(params["seed"])
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        folds = []
        print(f"per-CV-fold rho              skipped ({exc})")
    for i, test_idx in folds:
        member = np.zeros(n_episodes, dtype=bool)
        member[test_idx] = True
        w = member[cell_episode[keep]].astype(float)
        rho_f, _ = maximise(make_objective(cell_lik, w))
        print(
            f"CV fold {i} (test episodes)    {f(rho_f)}   "
            f"episodes={len(test_idx)} cells={int(w.sum())}"
        )

    # ------------------------- pre-flight ------------------------- #
    preflight = {}
    if args.preflight:
        t_p = time.time()
        cell_index = np.searchsorted(np.unique(obs["cell"]), obs["cell"])
        n_cells_all = len(np.unique(obs["cell"]))
        y_human = y.numpy()[obs["episode"], obs["agent"], obs["round"]]
        rows = [("human", y_human)]
        for label, rho in (("independent", 0.0), ("copula rho_hat", rho_hat)):
            orders = reveal_orders(n_episodes, n_agents, PREFLIGHT_SEED)
            y_sw = generate(
                model,
                data,
                default_values,
                edge_index,
                orders,
                rho,
                PREFLIGHT_SEED + 1,
                params,
            ).numpy()
            rows.append(
                (
                    f"AR {label} (free sweep)",
                    y_sw[obs["episode"], obs["agent"], obs["round"]],
                )
            )
        for label, rho in (("independent", None), ("copula rho_hat", rho_hat)):
            saved = model.copula_rho
            model.copula_rho = rho
            np.random.seed(PREFLIGHT_SEED)
            th.manual_seed(PREFLIGHT_SEED)
            with th.no_grad():
                y_pa, _ = model.predict_autoreg(
                    dict(data), sample=True, edge_index=edge_index
                )
            model.copula_rho = saved
            rows.append(
                (
                    f"AR {label} (predict_autoreg)",
                    y_pa.numpy()[obs["episode"], obs["agent"], obs["round"]],
                )
            )
        print("\n=== pre-flight (go/no-go context; rho is NEVER tuned to it) ===")
        print(
            "source                                std(cell mean)"
            "          spread ratio"
        )
        for label, values in rows:
            sd_cell, ratio = spread_stats(values, cell_index, n_cells_all)
            preflight[label] = dict(sd_cell_mean=sd_cell, spread_ratio=ratio)
            print(f"{label:<37} {f(sd_cell):<23} {f(ratio)}")
        print(f"  [{time.time() - t_p:.1f}s]")

    # ------------------------- outputs ------------------------- #
    target = out_dir / "model" / f"{params['job_id']}.pt"
    n_params = write_checkpoint(
        params["source_model"], target, rho_hat, device, params["edge_encoding"]
    )
    print(f"\nsaved {target}")
    print(f"  copula_rho={f(rho_hat)}  md5={md5sum(str(target))}")
    print(
        f"  reload self-check: {n_params} parameter tensors bit-identical to "
        "the source, autoregressive True, new keys == {'copula_rho'}"
    )

    payload = dict(
        rho=float(rho_hat),
        rho_se=se,
        rho_ci_95=[ci[0], ci[1]],
        log_likelihood=float(ll_hat),
        log_likelihood_at_zero=float(ll_zero),
        n_cells=int(cell_lik.n_cells),
        n_cells_total=int(len(sizes)),
        n_valid_rows=int(len(ep)),
        n_pairs_effective=int(len(pairs)),
        n_episodes=int(n_episodes),
        reveal_replicates=int(n_rep),
        n_bootstrap=int(n_boot),
        quadrature_nodes=int(N_QUAD),
        quadrature_rel_error=quad_err,
        rho_max=float(RHO_MAX),
        seeds=dict(
            base=int(params["seed"]),
            reveal=[int(params["seed"]) + r for r in range(n_rep)],
            bootstrap=int(params["seed"]),
            roundtrip_order=int(GEN_ORDER_SEED),
            roundtrip_draw=int(GEN_DRAW_SEED),
            preflight=int(PREFLIGHT_SEED),
            pit=int(PIT_SEED),
        ),
        estimator=(
            "exact cell-level shared-latent Gaussian-copula MLE on the AR "
            "model's own conditional CDFs; conditionals harvested by replaying "
            "the training reveal scheme (apply_mask_pattern) over "
            f"{n_rep} seeded per-episode reveal permutations; cells are "
            "(episode, round, agent_group) with >= 2 valid agents; "
            f"{N_QUAD}-node Gauss-Hermite; coarse grid 0-0.9 step 0.05 then "
            "bounded Brent, clipped to [0, 0.95]"
        ),
        cell_key="episode_round_agent_group",
        data_file=str(params["data_file"]),
        source_model=str(params["source_model"]),
        source_md5=str(params["source_md5"]),
        target_model=str(target),
        target_md5=md5sum(str(target)),
        roundtrip=dict(
            tolerance=ROUNDTRIP_TOL,
            passed=gate_pass,
            rows=roundtrip_rows,
        ),
        preflight=preflight,
        diagnostics=dict(
            per_replicate_rho=[float(v) for v in per_rep],
            single_copy_rho=float(rho_single),
            pit_moment_rho_attenuated=float(pit.mean()),
        ),
        git_commit=git_commit(),
        runtime_seconds=float(time.time() - t0),
    )
    json_path = metrics_dir / "copula_rho.json"
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"saved {json_path}")
    print(f"total runtime {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    main()
