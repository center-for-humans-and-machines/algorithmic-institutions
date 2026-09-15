"""Heteroscedastic Gaussian linear regressor for the continuous contribution
baseline (issue #119 / #121).

Replaces the Ridge point-estimator for the continuous target: instead of a single
conditional mean, this predicts a full conditional distribution

    y | x  ~  N( mu(x), sigma(x)^2 ),   mu(x) = x . w_mu + b_mu,
                                        log sigma(x) = x . w_s + b_s

both heads being a single linear layer (nn.Linear(in_features, 2)), fit by
maximum likelihood with nn.GaussianNLLLoss. The log-sigma parameterisation keeps
sigma > 0. In the simulation the AH samples contribution ~ N(mu(x), sigma(x)),
which restores the human-like contribution spread that a deterministic
mean-predictor collapses (see #121 Q4 / diversity analysis).

sklearn-ish surface so it drops into inspect_best_model with minimal churn:
`.fit`, `.predict` (-> mu), `.predict_std` (-> sigma), `.nll`, `.coef_`,
`.intercept_` (all on the MEAN head). Pickles via joblib (torch tensors pickle).

`GaussianMLPRegressor` is the nonlinear variant: the same heteroscedastic
Gaussian heads fed by a 2-layer net (Linear(d, hidden) -> tanh -> Linear(h, 2))
instead of a single affine map, so both mu(x) and log sigma(x) become
state-dependent. It shares the whole surface above except `.coef_` /
`.intercept_`, which have no meaning without a linear mean map.

`InflatedGaussianMLPRegressor` keeps that trunk but replaces the emission: a
discrete mixture of the binned Gaussian body with probability atoms at the
status quo (`prev_contribution`) and at the two endowment corners, fitted by
the 21-way cross-entropy. It is a proper distribution over the levels, so its
surface is the categorical one (`predict_proba`, `classes_`, `nll` = the
cross-entropy) and `predict` / `predict_std` degrade to body diagnostics.
"""

import numpy as np
import torch as th
from torch import nn


def binned_probs(mu, sigma, k_levels):
    """[N, k_levels] probabilities of N(mu, sigma) discretised onto the integer
    levels 0..k_levels-1: unit-width bins, the left tail folded into level 0 and
    the right tail into the top level, floored at 1e-12 and renormalised.

    This is THE discretisation convention of this module: `binned_logloss`
    scores it, `binned_log_probs` is its differentiable torch twin (log space),
    and the inflated emission head uses that twin as its Gaussian body. Keep the
    three in step. sigma may be scalar or per-row (heteroscedastic)."""
    from scipy.stats import norm

    mu = np.asarray(mu, float).reshape(-1)
    sigma = np.broadcast_to(np.asarray(sigma, float).reshape(-1), mu.shape)
    ks = np.arange(k_levels)
    P = norm.cdf((ks + 0.5 - mu[:, None]) / sigma[:, None]) - norm.cdf(
        (ks - 0.5 - mu[:, None]) / sigma[:, None]
    )
    P[:, 0] = norm.cdf((0.5 - mu) / sigma)
    P[:, -1] = 1.0 - norm.cdf((k_levels - 1.5 - mu) / sigma)
    P = np.clip(P, 1e-12, None)
    P /= P.sum(1, keepdims=True)
    return P


def binned_log_probs(mu, log_sigma, k_levels):
    """[N, k_levels] LOG probabilities of the `binned_probs` convention, in
    torch and differentiable w.r.t. (mu, log_sigma). Computed in float64 and in
    log space: the tail-side CDF difference is taken on whichever side of the
    bin is small, so the far bins keep their precision instead of cancelling.
    The CDF is evaluated on the k_levels + 1 bin EDGES, which adjacent bins
    share -- two passes over [N, k+1] per call rather than four over [N, k]."""
    mu = mu.reshape(-1, 1).double()
    sigma = th.exp(log_sigma.reshape(-1, 1).double())
    edges = th.arange(k_levels + 1, dtype=th.float64).reshape(1, -1) - 0.5 - mu
    edges = edges / sigma
    up, down = th.special.ndtr(edges), th.special.ndtr(-edges)
    lo, hi = edges[:, :-1], edges[:, 1:]
    P = th.where(hi + lo > 0.0, down[:, :-1] - down[:, 1:], up[:, 1:] - up[:, :-1])
    # tails: everything below 0.5 folds into level 0, everything above the top
    # bin's lower edge into the top level.
    P = th.cat([up[:, 1:2], P[:, 1:-1], down[:, -2:-1]], dim=1)
    log_P = th.log(P.clamp_min(1e-12))
    return log_P - th.logsumexp(log_P, dim=1, keepdim=True)


def binned_logloss(mu, y, sigma, k_levels):
    """Discrete log-loss of N(mu, sigma) binned onto integer levels 0..k_levels-1
    (left tail folds into level 0, right tail into the top level). This is a
    proper 21-way cross-entropy, directly comparable to the categorical /
    GNN contribution log-loss. sigma may be scalar or per-row (heteroscedastic)."""
    P = binned_probs(mu, sigma, k_levels)
    yi = np.clip(np.rint(np.asarray(y).reshape(-1)), 0, k_levels - 1).astype(int)
    return float(-np.mean(np.log(P[np.arange(len(yi)), yi])))


class _Head(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 2)

    def forward(self, x):
        out = self.linear(x)
        return out[:, 0], out[:, 1]  # mu, log_sigma


class _MLPHead(nn.Module):
    def __init__(self, in_features, hidden):
        super().__init__()
        self.hidden = nn.Linear(in_features, hidden)
        self.out = nn.Linear(hidden, 2)

    def forward(self, x):
        out = self.out(th.tanh(self.hidden(x)))
        return out[:, 0], out[:, 1]  # mu, log_sigma


class GaussianRegressor:
    """MLE heteroscedastic Gaussian linear model (torch, CPU)."""

    def __init__(self, weight_decay=0.0, epochs=3000, lr=0.05, seed=0):
        self.weight_decay = float(weight_decay)  # Adam L2 (all params)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.seed = int(seed)
        self.net = None

    def _make_net(self, in_features):
        """The net mapping features -> (mu, log_sigma). Subclass hook."""
        return _Head(in_features)

    def _out_layer(self):
        """The final nn.Linear, whose biases carry the warm start."""
        return self.net.linear

    def fit(self, X, y):
        X = th.as_tensor(np.asarray(X), dtype=th.float32)
        y = th.as_tensor(np.asarray(y), dtype=th.float32).reshape(-1)
        th.manual_seed(self.seed)
        self.net = self._make_net(X.shape[1])
        # warm start: mu bias -> mean(y), log-sigma bias -> log std(y). Keeps the
        # NLL well-conditioned from step 0 (var neither explodes nor collapses).
        with th.no_grad():
            out = self._out_layer()
            out.bias[0] = y.mean()
            out.bias[1] = th.log(y.std().clamp(min=1e-3))
        opt = th.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.GaussianNLLLoss(full=True)
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            mu, log_sigma = self.net(X)
            var = th.exp(2.0 * log_sigma)
            loss = loss_fn(mu, y, var)
            loss.backward()
            opt.step()
        return self

    def _forward(self, X):
        self.net.eval()
        with th.no_grad():
            mu, log_sigma = self.net(th.as_tensor(np.asarray(X), dtype=th.float32))
        return mu.numpy(), np.exp(log_sigma.numpy())

    def predict(self, X):
        return self._forward(X)[0]

    def predict_std(self, X):
        return self._forward(X)[1]

    def nll(self, X, y):
        """Mean Gaussian negative log-likelihood (nats) on (X, y)."""
        mu, sigma = self._forward(X)
        y = np.asarray(y).reshape(-1)
        var = sigma**2
        return float(np.mean(0.5 * (np.log(2 * np.pi * var) + (y - mu) ** 2 / var)))

    @property
    def coef_(self):
        """Standardised coefficients of the MEAN head (mu)."""
        return self.net.linear.weight.detach().numpy()[0]

    @property
    def intercept_(self):
        return float(self.net.linear.bias.detach().numpy()[0])


class GaussianMLPRegressor(GaussianRegressor):
    """MLE heteroscedastic Gaussian model with a 2-layer (tanh) net.

    Same objective, optimiser and surface as `GaussianRegressor`; only the map
    from features to (mu, log_sigma) is nonlinear, so both the conditional mean
    and the conditional spread can vary with the state.
    """

    def __init__(self, hidden=32, weight_decay=0.0, epochs=500, lr=0.05, seed=0):
        super().__init__(weight_decay=weight_decay, epochs=epochs, lr=lr, seed=seed)
        self.hidden = int(hidden)

    def _make_net(self, in_features):
        net = _MLPHead(in_features, self.hidden)
        # zero output weights so that, with the warm-started output biases, the
        # net emits the marginal N(mean(y), std(y)) at step 0 -- the same
        # starting point as the linear model. The hidden layer keeps torch's
        # default init, which breaks the symmetry once gradients flow.
        with th.no_grad():
            net.out.weight.zero_()
        return net

    def _out_layer(self):
        return self.net.out

    @property
    def coef_(self):
        raise NotImplementedError(
            "GaussianMLPRegressor has no single affine mean map; coef_ is only "
            "defined for the linear GaussianRegressor."
        )

    @property
    def intercept_(self):
        raise NotImplementedError(
            "GaussianMLPRegressor has no single affine mean map; intercept_ is "
            "only defined for the linear GaussianRegressor."
        )


ATOM_NAMES = ("prev", "0", "20")
ATOM_LOGIT_INIT = -3.0  # warm start: each atom starts at ~4% of the mass
_NEG = -1e30  # a finite "log 0": logaddexp(_NEG, x) == x for any finite x


class _InflatedMLPHead(nn.Module):
    """`_MLPHead`'s trunk with a wider output: (mu, log_sigma, atom logits)."""

    def __init__(self, in_features, hidden, n_atoms):
        super().__init__()
        self.hidden = nn.Linear(in_features, hidden)
        self.out = nn.Linear(hidden, 2 + n_atoms)

    def forward(self, x):
        out = self.out(th.tanh(self.hidden(x)))
        return out[:, 0], out[:, 1], out[:, 2:]


class InflatedGaussianMLPRegressor(GaussianMLPRegressor):
    """Status-quo- and corner-inflated binned Gaussian emission.

    Same trunk as `GaussianMLPRegressor` (Linear(d, hidden) -> tanh), a wider
    output layer, and a discrete distribution over the 21 contribution levels

        P(c | x) = pi_body(x) BinnedN(c; mu(x), sigma(x))
                 + sum_a pi_a(x) 1[c = level_a(x)]

    with the mixture weights a softmax over [0, atom logits] (the body's logit
    is pinned at 0) and the atoms drawn from {"prev", "0", "20"}: the previous
    contribution (the status quo), and the game's two endowment corners. The
    behaviour it encodes is exact repetition -- humans repeat last round's
    contribution in 44% of rounds, which a Gaussian body structurally cannot
    represent, since it has no point mass.

    Two atoms may land on the same level (prev == 0 or prev == 20); their
    masses add. Fitted by the mean 21-way cross-entropy -- the same quantity
    this lineage already reports as `test_logloss_binned` -- with the
    incumbent's optimiser, warm start and (lr, wd, epochs, seed) knobs.

    `atoms=()` degenerates to a binned Gaussian fitted by that same objective.

    Surface: `fit(Z, y, prev=...)` (the RAW prev_contribution column is
    required: it recovers the standardiser's affine map for `prev_index`, so
    the atom can be placed at the right level from `Z` alone afterwards),
    `log_probs` / `predict_proba` / `classes_` / `mixture_probs`, `nll` (the
    cross-entropy, the model's primary metric) and `predict` / `predict_std`
    (the BODY's mu / sigma -- diagnostics only, not the mixture's).
    """

    def __init__(
        self,
        hidden=32,
        weight_decay=0.0,
        epochs=500,
        lr=0.05,
        seed=0,
        atoms=ATOM_NAMES,
        prev_index=0,
        k_levels=21,
    ):
        super().__init__(
            hidden=hidden, weight_decay=weight_decay, epochs=epochs, lr=lr, seed=seed
        )
        self.atoms = self._check_atoms(atoms)
        self.prev_index = int(prev_index)
        self.k_levels = int(k_levels)
        self.prev_mean_ = None
        self.prev_scale_ = None

    @staticmethod
    def _check_atoms(atoms):
        atoms = tuple(str(a) for a in atoms)
        bad = [a for a in atoms if a not in ATOM_NAMES]
        if bad:
            raise ValueError(f"unknown atom(s) {bad}; allowed: {list(ATOM_NAMES)}")
        if len(set(atoms)) != len(atoms):
            raise ValueError(f"duplicate atoms in {atoms}")
        return atoms

    # ------------------------------------------------------------------ #
    # the standardiser's affine map for the prev column
    # ------------------------------------------------------------------ #
    def _recover_affine(self, z, raw):
        """(mean, scale) with raw == z * scale + mean, recovered exactly from a
        standardised column and its raw original. Placing the atom at the wrong
        level is silent until the simulation, so a column that is not an affine
        image of `raw` is a hard error here."""
        z = np.asarray(z, float).reshape(-1)
        raw = np.asarray(raw, float).reshape(-1)
        if len(z) != len(raw):
            raise ValueError(f"prev has {len(raw)} rows for {len(z)} feature rows")
        var = float(z.var())
        if var <= 0.0:
            raise ValueError(
                f"feature column {self.prev_index} is constant; the standardiser's "
                "affine map for prev_contribution cannot be recovered"
            )
        scale = float(((z - z.mean()) * (raw - raw.mean())).mean() / var)
        mean = float(raw.mean() - scale * z.mean())
        resid = float(np.abs(z * scale + mean - raw).max())
        if resid > 1e-4 * max(1.0, abs(scale)):
            raise ValueError(
                f"feature column {self.prev_index} is not an affine image of the "
                f"raw prev column (max residual {resid:.3e}); prev_index is wrong"
            )
        return mean, scale

    def _prev_levels(self, Z):
        """The status-quo atom's level per row, from the standardised features."""
        if self.prev_scale_ is None:
            raise ValueError("not fitted: the prev column's affine map is unknown")
        raw = np.asarray(Z, float)[:, self.prev_index] * self.prev_scale_
        raw = raw + self.prev_mean_
        return np.clip(np.rint(raw), 0, self.k_levels - 1).astype(np.int64)

    # ------------------------------------------------------------------ #
    # the mixture
    # ------------------------------------------------------------------ #
    def _make_net(self, in_features):
        net = _InflatedMLPHead(in_features, self.hidden, len(self.atoms))
        with th.no_grad():
            net.out.weight.zero_()  # see GaussianMLPRegressor._make_net
        return net

    def _atom_column(self, name, prev_levels, n):
        if name == "prev":
            return prev_levels.reshape(-1, 1)
        level = 0 if name == "0" else self.k_levels - 1
        return th.full((n, 1), level, dtype=th.int64)

    def _log_probs_t(self, Xt, prev_levels):
        """[N, k_levels] log-probabilities as a differentiable tensor."""
        mu, log_sigma, atom_logits = self.net(Xt)
        log_body = binned_log_probs(mu, log_sigma, self.k_levels)
        if not self.atoms:
            return log_body
        n = Xt.shape[0]
        logits = th.cat([th.zeros(n, 1, dtype=th.float64), atom_logits.double()], 1)
        log_w = th.log_softmax(logits, dim=1)
        # atoms accumulate by logaddexp, so two atoms on the same level (prev
        # at a corner) ADD their masses instead of overwriting each other.
        log_atoms = th.full((n, self.k_levels), _NEG, dtype=th.float64)
        for j, name in enumerate(self.atoms):
            col = self._atom_column(name, prev_levels, n)
            here = th.logaddexp(log_atoms.gather(1, col), log_w[:, j + 1 : j + 2])
            log_atoms = log_atoms.scatter(1, col, here)
        return th.logaddexp(log_body + log_w[:, :1], log_atoms)

    def fit(self, X, y, prev=None):
        """Minimise the mean 21-way cross-entropy. `prev` is the RAW
        prev_contribution column of the same rows as `X` -- without it the
        status-quo atom has no level, so the estimator refuses to fit."""
        if prev is None:
            raise ValueError(
                "InflatedGaussianMLPRegressor.fit needs the raw prev_contribution "
                "column: fit(Z, y, prev=X_raw[:, prev_index])"
            )
        Z = np.asarray(X, float)
        self.prev_mean_, self.prev_scale_ = self._recover_affine(
            Z[:, self.prev_index], prev
        )
        Xt = th.as_tensor(Z, dtype=th.float32)
        yc = np.asarray(y, float).reshape(-1)
        yt = th.as_tensor(yc, dtype=th.float32)
        levels = th.as_tensor(
            np.clip(np.rint(yc), 0, self.k_levels - 1).astype(np.int64)
        ).reshape(-1, 1)
        prev_levels = th.as_tensor(self._prev_levels(Z))
        th.manual_seed(self.seed)
        self.net = self._make_net(Xt.shape[1])
        with th.no_grad():
            out = self._out_layer()
            out.bias[0] = yt.mean()
            out.bias[1] = th.log(yt.std().clamp(min=1e-3))
            out.bias[2:] = ATOM_LOGIT_INIT
        opt = th.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            log_p = self._log_probs_t(Xt, prev_levels)
            loss = -log_p.gather(1, levels).mean()
            loss.backward()
            opt.step()
        return self

    def _forward(self, X):
        """The BODY's (mu, sigma) -- the mixture's own mean is not this."""
        self.net.eval()
        with th.no_grad():
            mu, log_sigma, _ = self.net(th.as_tensor(np.asarray(X), dtype=th.float32))
        return mu.numpy(), np.exp(log_sigma.numpy())

    def log_probs(self, X):
        """[N, k_levels] log-probabilities over the contribution levels."""
        Z = np.asarray(X, float)
        self.net.eval()
        with th.no_grad():
            log_p = self._log_probs_t(
                th.as_tensor(Z, dtype=th.float32),
                th.as_tensor(self._prev_levels(Z)),
            )
        return log_p.numpy()

    def predict_proba(self, X):
        """[N, k_levels] probabilities; rows sum to one."""
        return np.exp(self.log_probs(X))

    def mixture_probs(self, X):
        """[N, 1 + len(atoms)] mixture weights, column 0 the body and the rest
        in `atoms` order (so pi_rep is column 1 + atoms.index('prev'))."""
        self.net.eval()
        with th.no_grad():
            _, _, atom_logits = self.net(
                th.as_tensor(np.asarray(X, float), dtype=th.float32)
            )
        n = atom_logits.shape[0]
        logits = th.cat([th.zeros(n, 1, dtype=th.float64), atom_logits.double()], 1)
        return th.softmax(logits, dim=1).numpy()

    def nll(self, X, y):
        """Mean 21-way cross-entropy (nats) -- the model's primary metric."""
        log_p = self.log_probs(X)
        yi = np.clip(np.rint(np.asarray(y, float).reshape(-1)), 0, self.k_levels - 1)
        yi = yi.astype(int)
        return float(-np.mean(log_p[np.arange(len(yi)), yi]))

    @property
    def classes_(self):
        return np.arange(self.k_levels)
