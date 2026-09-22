"""Single-hidden-layer MLP point-regressor for the continuous contribution
baseline.

Same target as the Ridge / GaussianRegressor mean head, but with a nonlinear
conditional mean:

    y | x  ~  point estimate  mu(x) = tanh(x . W1 + b1) . w2 + b2

fit by plain least squares (nn.MSELoss), full-batch Adam, CPU torch. This
isolates the effect of nonlinearity in the mean alone: no variance head, no
sampling, no categorical output -- the deterministic counterpart of the linear
baseline, so a score difference is attributable to curvature rather than to a
change in the noise model.

sklearn-ish surface mirroring GaussianRegressor: `.fit`, `.predict` (-> mu).
`.coef_` / `.intercept_` are deliberately unavailable (the model is nonlinear,
so there is no single coefficient vector to report). Pickles via joblib (torch
tensors pickle).
"""

import numpy as np
import torch as th
from torch import nn


class _MLPHead(nn.Module):
    def __init__(self, in_features, hidden):
        super().__init__()
        self.hidden = nn.Linear(in_features, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.out(th.tanh(self.hidden(x)))[:, 0]  # mu


class MLPRegressor:
    """Least-squares single-hidden-layer tanh MLP (torch, CPU)."""

    def __init__(self, hidden=16, weight_decay=0.0, epochs=3000, lr=0.05, seed=0):
        self.hidden = int(hidden)
        self.weight_decay = float(weight_decay)  # Adam L2 (all params)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.seed = int(seed)
        self.net = None

    def fit(self, X, y):
        X = th.as_tensor(np.asarray(X), dtype=th.float32)
        y = th.as_tensor(np.asarray(y), dtype=th.float32).reshape(-1)
        th.manual_seed(self.seed)
        self.net = _MLPHead(X.shape[1], self.hidden)
        # warm start: output bias -> mean(y), so step 0 predicts the marginal
        # mean and the optimiser only has to learn the residual structure.
        with th.no_grad():
            self.net.out.bias[0] = y.mean()
        opt = th.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self.net(X), y)
            loss.backward()
            opt.step()
        return self

    def predict(self, X):
        self.net.eval()
        with th.no_grad():
            mu = self.net(th.as_tensor(np.asarray(X), dtype=th.float32))
        return mu.numpy()

    @property
    def coef_(self):
        raise NotImplementedError("MLPRegressor is nonlinear: no single coef_ vector.")

    @property
    def intercept_(self):
        raise NotImplementedError("MLPRegressor is nonlinear: no scalar intercept_.")
