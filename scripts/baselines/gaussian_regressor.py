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
"""

import numpy as np
import torch as th
from torch import nn


def binned_logloss(mu, y, sigma, k_levels):
    """Discrete log-loss of N(mu, sigma) binned onto integer levels 0..k_levels-1
    (left tail folds into level 0, right tail into the top level). This is a
    proper 21-way cross-entropy, directly comparable to the categorical /
    GNN contribution log-loss. sigma may be scalar or per-row (heteroscedastic)."""
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
