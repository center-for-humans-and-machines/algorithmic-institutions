"""Run the hand-crafted linear baselines (issue #119, saved as joblib bundles:
ridge / gaussian / multinomial) as drop-in artificial humans in the simulation
pipeline (#121).

The env drives artificial humans through a single interface:

    model.predict(state, reset_rnn=..., edge_index=...) -> (prediction, extra)

and reads ``model.default_values``. The GNN consumes the whole graph + an RNN
state; a linear model instead needs a flat, standardised feature vector per agent
-- exactly the ``build_feature_pool`` features it was trained on. Those features
are functions of PREVIOUS-round quantities, the membership timeline and the
switch history, so this adapter accumulates a per-episode history (the env state
only ever holds the current round + its ``prev_`` sibling) and rebuilds the same
``[G, A, T]`` tensors ``create_torch_data`` would, then calls the *same*
``build_feature_pool`` used in training -- parity by construction.

Timing / leakage (mirrors handcrafted_grid, re-anchored per #123):
  * Contribution model: called before round t is played; its prev-family
    features need only realised values through t-1 plus the current membership.
    Current-valued features are illegal for it (asserted against
    ``CURRENT_VALUED`` at load).
  * Switch model: the env calls it at the END of round s (outcomes complete,
    membership still pre-decision), so the current family at row s is fully
    realised -- exactly the training-time anchoring of ``does_switch[s]``.
  * Arrival markers (window resets, tenure counters) are derived inside
    ``build_feature_pool`` from the membership timeline; no ``does_switch``
    input is needed.
"""

import sys
from pathlib import Path

import numpy as np
import torch as th

# build_feature_pool (scripts/baselines) is the single source of truth for the
# feature engineering (spec: notes/baseline_feature_defs.md; parity test:
# tests/baselines). Import it so sim features can never drift from training.
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "scripts" / "baselines"))
from handcrafted_grid import CURRENT_VALUED, build_feature_pool  # noqa: E402


def _shift(a, default):
    """t-1 shift on the round axis (axis 2); round 0 -> default. Mirrors
    aimanager.generic.data.shift so reconstructed prev_ tensors match training."""
    out = np.roll(a, 1, axis=2)
    out[:, :, 0] = default
    return out


class LinearAHAdapter:
    """Wrap a saved linear-baseline bundle behind the GraphNetwork predict API."""

    # env checks GraphNetwork.autoregressive on the contribution model; linear
    # models are never autoregressive.
    autoregressive = False

    # history buffers we accumulate over an episode (round -> per-agent array)
    _MEASURES = ("contribution", "punishment", "common_good")

    def __init__(
        self, bundle, *, n_agents, n_contributions, device=None, switch_sample=True
    ):
        self.model_type = bundle["model"]  # 'ridge' | 'gaussian' | 'multinomial'
        self.estimator = bundle["estimator"]
        self.scaler = bundle["scaler"]
        self.features = list(bundle["features"])
        self.target = bundle["target"]
        self.is_switch = self.target == "does_switch"
        self.n_levels = int(bundle.get("n_levels", 0))
        # contribution sampling: ridge N(mu, sigma), gaussian N(mu, sigma(x)),
        # multinomial = predict_proba tempered by `temperature` (T=1 as-is).
        self.sigma = float(bundle.get("sigma") or 0.0)
        self.temperature = float(bundle.get("temperature", 1.0))
        self.default_values = dict(bundle["default_values"])
        self.switch_every = bundle.get("switch_every")
        self.n_agents = int(n_agents)
        self.n_contributions = int(n_contributions)
        self.device = device
        # match the GNN switch predictor (sample=True); set False for a
        # deterministic proba>0.5 threshold.
        self.switch_sample = switch_sample
        if not self.is_switch:
            illegal = sorted(set(self.features) & CURRENT_VALUED)
            assert not illegal, (
                "contribution bundle contains current-valued features "
                f"(illegal, they read the target's round): {illegal}"
            )
        self._reset_history()

    def to(self, device):
        self.device = device
        return self

    # ------------------------------------------------------------------ #
    # per-episode history
    # ------------------------------------------------------------------ #
    def _reset_history(self):
        self._measure = {m: {} for m in self._MEASURES}  # round -> (A,) float
        self._group = {}  # round -> (A,) int membership

    def _record(self, state, t):
        """Fold the current env state into the episode history.

        Contribution model (called before round t is played): prev_* holds the
        realised values of round t-1, membership is post-arrival. Switch model
        (called at the END of round t): the current keys additionally hold
        round t's realised outcomes, membership is still pre-decision.
        Re-recording the same round is idempotent."""
        A = self.n_agents

        def col(key):
            return state[key].detach().to("cpu").numpy().reshape(A)

        if t > 0:
            for m in self._MEASURES:
                self._measure[m][t - 1] = col(f"prev_{m}").astype(float)
            self._group[t - 1] = col("prev_agent_group").astype(int)
        self._group[t] = col("agent_group").astype(int)
        if self.is_switch:
            for m in self._MEASURES:
                self._measure[m][t] = col(m).astype(float)

    def _build_pool(self, t):
        """Reconstruct the [1, A, t+1] create_torch_data tensors from history and
        return build_feature_pool's feature dict."""
        A, T = self.n_agents, t + 1
        dv = self.default_values

        def stack(store, default, dtype):
            arr = np.full((A, T), default, dtype=dtype)
            for r, v in store.items():
                if r < T:
                    arr[:, r] = v
            return arr

        c = stack(self._measure["contribution"], dv["contribution"], float)
        p = stack(self._measure["punishment"], dv["punishment"], float)
        cg = stack(self._measure["common_good"], dv["common_good"], float)
        ag = stack(self._group, 0, int)
        rec = np.ones((A, T), dtype=float)  # every agent is present in the sim
        rounds = np.tile(np.arange(T, dtype=float), (A, 1))

        # -> contiguous [1, A, T] torch tensor (build_feature_pool .numpy()s it)
        def b(x):
            return th.from_numpy(np.ascontiguousarray(x[None]))

        # Current tensors carry defaults at the not-yet-played round for the
        # contribution model; only the (asserted prev-family) features it was
        # trained on are read, so those cells are never consumed.
        d = {
            "contribution": b(c),
            "punishment": b(p),
            "common_good": b(cg),
            "prev_contribution": b(_shift(c[None], dv["contribution"])[0]),
            "prev_punishment": b(_shift(p[None], dv["punishment"])[0]),
            "prev_common_good": b(_shift(cg[None], dv["common_good"])[0]),
            "prev_agent_group": b(_shift(ag[None], 0)[0].astype(int)),
            "prev_recorded": b(_shift(rec[None], 0.0)[0].astype(bool)),
            "agent_group": b(ag.astype(int)),
            "recorded": b(rec.astype(bool)),
            "round_number": b(rounds),
        }
        return build_feature_pool(d, self.switch_every)

    # ------------------------------------------------------------------ #
    # env-facing predict
    # ------------------------------------------------------------------ #
    def predict(self, state, *, reset_rnn=False, edge_index=None, **_):
        t = int(state["round_number"].reshape(-1)[0].item())
        if reset_rnn or t == 0:
            self._reset_history()
        self._record(state, t)

        pool = self._build_pool(t)
        X = np.column_stack([pool[f][0, :, t] for f in self.features])
        Xs = self.scaler.transform(X)
        dev = state["round_number"].device

        if self.is_switch:  # multinomial logistic switch bundle
            proba = self.estimator.predict_proba(Xs)
            classes = list(self.estimator.classes_)
            p1 = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
            if self.switch_sample:
                sw = np.random.random(len(X)) < p1
            else:
                sw = p1 >= 0.5
            pred = th.tensor(sw, dtype=th.bool, device=dev).reshape(1, -1, 1)
            return pred, None

        if self.model_type == "multinomial":  # sample the 21-way predict_proba
            proba = self.estimator.predict_proba(Xs)
            P = np.full((len(Xs), self.n_contributions), 1e-12)
            P[:, self.estimator.classes_] = proba
            if self.temperature != 1.0:
                P = P ** (1.0 / self.temperature)
            P /= P.sum(1, keepdims=True)
            lvl = np.array(
                [np.random.choice(self.n_contributions, p=P[i]) for i in range(len(P))],
                dtype=np.int64,
            )
            return th.tensor(lvl, dtype=th.int64, device=dev).reshape(1, -1, 1), None

        # contribution: sample from the conditional Gaussian, then discretise.
        mu = self.estimator.predict(Xs)
        if self.model_type == "gaussian":  # heteroscedastic sigma(x) from the head
            sd = self.estimator.predict_std(Xs)
            yhat = mu + np.random.normal(0.0, 1.0, size=mu.shape) * sd
        elif self.sigma > 0:  # ridge: homoscedastic scalar sigma
            yhat = mu + np.random.normal(0.0, self.sigma, size=mu.shape)
        else:  # no sigma stored -> deterministic point prediction
            yhat = mu
        lvl = np.clip(np.rint(yhat), 0, self.n_contributions - 1).astype(np.int64)
        pred = th.tensor(lvl, dtype=th.int64, device=dev).reshape(1, -1, 1)
        return pred, None


def load_ah_model(
    path, *, device=None, n_agents=None, n_contributions=None, switch_sample=True
):
    """Load an artificial-human model by extension: ``.joblib`` -> linear-baseline
    adapter, anything else -> GraphNetwork (GNN). Lets a sim config mix the two."""
    if str(path).endswith(".joblib"):
        import joblib

        bundle = joblib.load(path)
        return LinearAHAdapter(
            bundle,
            n_agents=n_agents,
            n_contributions=n_contributions,
            device=device,
            switch_sample=switch_sample,
        )
    from aimanager.artificial_humans import GraphNetwork

    return GraphNetwork.load(path, device=device)
