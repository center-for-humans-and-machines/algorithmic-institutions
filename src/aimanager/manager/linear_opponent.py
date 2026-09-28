"""Serve a saved linear punishment baseline as the fixed opponent manager in
RL training (``aimanager.rl_manager.run_batch``).

The simulation path drives a linear punisher through
``LinearAHAdapter.get_punishments(rounds)``: one episode, a Python round-dict
history, ``build_feature_pool`` rebuilt over the whole ``[1, A, T]`` history
every round. RL training runs ``batch_size`` episodes in parallel inside
``ArtificialHumanEnv`` and calls its opponent once per round, so that path is
both the wrong shape and far too slow to sit inside a 4000-step training loop.

Every feature the punishment bundles select is a *local* function of the
agent's own current and previous round -- no group aggregates, no tenure
windows, no since-switch means. So this adapter recomputes exactly those
columns straight from the env state, batched over episodes.
``SUPPORTED_FEATURES`` is asserted against the bundle at load, so a bundle that
ever selects a non-local feature fails loudly instead of quietly reading a
wrong number. Parity with the simulation path (same bundle, same features, same
deterministic levels) is pinned by
``src/aimanager/tests/test_linear_opponent.py``.

The env hands this adapter ``environment.served_state()``, i.e. a timed-out
player's contribution reads the recorded 0 rather than the imputed default --
the same value ``LinearAHAdapter._pool_from_rounds`` puts back before building
the punisher's features.
"""

import numpy as np
import torch as th

# Kept in sync with scripts/baselines/handcrafted_grid.py by
# test_linear_opponent.test_constants_match_training.
ENDOWMENT = 20.0


def _f(x):
    """Env state entry -> float64 numpy (B, A)."""
    return x.detach().to("cpu").numpy().reshape(x.shape[0], x.shape[1]).astype(float)


class LinearPunisherOpponent:
    """A ``.joblib`` punishment bundle behind the env's opponent interface.

    ``predict(state, ...) -> (punishment[B, A, 1] int64, None)``, matching the
    GNN punisher's call site in ``run_batch``.
    """

    # The local feature columns this adapter can reproduce exactly. Mirrors the
    # definitions in scripts/baselines/handcrafted_grid.build_feature_pool.
    SUPPORTED_FEATURES = frozenset(
        {
            "contribution",
            "contribution_max",
            "contribution_zero",
            "contribution_valid",
            "prev_contribution",
            "prev_punishment",
            "round_number",
            "is_first",
        }
    )

    autoregressive = False

    def __init__(self, bundle, *, n_groups=2, device=None, sample=True):
        assert bundle["target"] == "punishment", (
            f"opponent bundle must have target 'punishment', "
            f"got {bundle['target']!r}"
        )
        assert bundle["model"] == "multinomial", (
            "only the multinomial punisher is served here (the gaussian/ridge "
            f"samplers are not batched), got {bundle['model']!r}"
        )
        self.estimator = bundle["estimator"]
        self.scaler = bundle["scaler"]
        self.features = list(bundle["features"])
        unsupported = sorted(set(self.features) - self.SUPPORTED_FEATURES)
        assert not unsupported, (
            "bundle selects features this batched adapter cannot reproduce "
            f"from the env state: {unsupported}"
        )
        self.n_levels = int(bundle["n_levels"])
        self.copula_rho = float(bundle.get("copula_rho", 0.0) or 0.0)
        assert 0.0 <= self.copula_rho < 1.0
        self.default_values = dict(bundle["default_values"])
        self.temperature = float(bundle.get("temperature", 1.0))
        self.n_groups = int(n_groups)
        self.device = device
        self.sample = bool(sample)

    def to(self, device):
        self.device = device
        return self

    # ------------------------------------------------------------------ #
    def _columns(self, state):
        c = _f(state["contribution"])
        cols = {
            "contribution": c,
            "contribution_max": (c == ENDOWMENT).astype(float),
            "contribution_zero": (c == 0).astype(float),
            "prev_contribution": _f(state["prev_contribution"]),
            "prev_punishment": _f(state["prev_punishment"]),
        }
        if "contribution_valid" in self.features:
            cols["contribution_valid"] = _f(state["contribution_valid"])
        r = _f(state["round_number"])
        cols["round_number"] = r
        cols["is_first"] = (r == 0).astype(float)
        return cols

    def _class_probs(self, Xs):
        """[n, n_levels] class probabilities; mirrors LinearAHAdapter."""
        proba = self.estimator.predict_proba(Xs)
        P = np.full((len(Xs), self.n_levels), 1e-12)
        P[:, self.estimator.classes_] = proba
        if self.temperature != 1.0:
            P = P ** (1.0 / self.temperature)
        P /= P.sum(1, keepdims=True)
        return P

    def _sample(self, P, groups):
        """P is [B, A, L]; groups is an int64 [B, A] tensor of memberships."""
        B, A, L = P.shape
        Pt = th.from_numpy(P)
        if not self.sample:
            return Pt.argmax(-1)
        if self.copula_rho == 0.0:
            return th.multinomial(Pt.reshape(B * A, L), 1).reshape(B, A)
        # one shared severity latent per (episode, group), drawn as the
        # simulation path draws it: a full [B, A] field, read at each group's
        # first member.
        zs = th.randn(B, A, dtype=th.float64)
        eps = th.randn(B, A, dtype=th.float64)
        same = groups.unsqueeze(2) == groups.unsqueeze(1)  # (B, A, A)
        first = same.to(th.uint8).argmax(dim=2)  # (B, A)
        a = float(np.sqrt(self.copula_rho))
        b = float(np.sqrt(1.0 - self.copula_rho))
        u = th.special.ndtr(a * zs.gather(1, first) + b * eps)  # (B, A)
        cum = Pt.cumsum(-1).contiguous()
        lvl = th.searchsorted(cum, u.unsqueeze(-1).contiguous())
        return lvl.squeeze(-1).clamp(0, L - 1)

    def predict(self, state, *, reset_rnn=False, edge_index=None, **_):
        cols = self._columns(state)
        B, A = cols["contribution"].shape
        X = np.column_stack([cols[f].reshape(-1) for f in self.features])
        P = self._class_probs(self.scaler.transform(X)).reshape(B, A, self.n_levels)
        groups = state["agent_group"].detach().to("cpu").reshape(B, A).to(th.int64)
        lvl = self._sample(P, groups)
        dev = state["contribution"].device
        return lvl.to(dev).to(th.int64).reshape(B, A, 1), None


def load_opponent(path, *, n_groups=2, device=None, sample=True):
    """``.joblib`` -> batched linear punisher; anything else -> GNN punisher."""
    if str(path).endswith(".joblib"):
        import joblib

        return LinearPunisherOpponent(
            joblib.load(path), n_groups=n_groups, device=device, sample=sample
        )
    from aimanager.artificial_humans import GraphNetwork

    return GraphNetwork.load(path, device=device)
