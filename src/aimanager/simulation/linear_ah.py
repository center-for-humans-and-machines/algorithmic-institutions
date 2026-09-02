"""Run the hand-crafted linear baselines (issues #119/#127, saved as joblib
bundles: ridge / gaussian[_mlp] / multinomial) as drop-in artificial humans AND as
punishment managers in the simulation pipeline (#121).

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

The punishment manager (#127) reuses the SAME adapter via a second entry point,
``get_punishments(rounds)``: it rebuilds the tensors from the raw round-dict
history instead of env states, recomputing per-capita common good with the env
formula since round dicts carry none.

Timing / leakage (mirrors handcrafted_grid, re-anchored per #123):
  * Contribution model: called before round t is played -> prev family only
    (current-valued features asserted illegal at load).
  * Switch model: called at the END of round s -> current family realised,
    matching the training anchoring of ``does_switch[s]``.
  * Punishment model (#127): called AFTER round t's contributions, BEFORE its
    punishments -> prev-anchored (same assert).
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
    """Wrap a saved linear-baseline bundle behind the GraphNetwork predict API
    (contribution / switch targets, env-driven) or behind the manager API
    (punishment target, rounds-driven via ``get_punishments``)."""

    # env checks GraphNetwork.autoregressive on the contribution model; linear
    # models are never autoregressive.
    autoregressive = False

    # history buffers we accumulate over an episode (round -> per-agent array)
    _MEASURES = ("contribution", "punishment", "common_good")

    def __init__(
        self,
        bundle,
        *,
        n_agents=None,
        n_contributions=None,
        device=None,
        switch_sample=True,
        sample=True,
    ):
        self.model_type = bundle["model"]  # 'ridge' | 'gaussian[_mlp]' | 'multinomial'
        self.estimator = bundle["estimator"]
        self.scaler = bundle["scaler"]
        self.features = list(bundle["features"])
        self.target = bundle["target"]
        self.is_switch = self.target == "does_switch"
        self.is_punishment = self.target == "punishment"
        self.n_levels = int(bundle.get("n_levels", 0))
        # level sampling: ridge N(mu, sigma), gaussian/_mlp N(mu, sigma(x)),
        # multinomial = predict_proba tempered by `temperature` (T=1 as-is).
        self.sigma = float(bundle.get("sigma") or 0.0)
        self.temperature = float(bundle.get("temperature", 1.0))
        # Severity copula: `copula_rho` on a punishment bundle correlates a
        # group's punishments via one shared latent per round (calibrated by
        # scripts/baselines/punishment_copula_rho.py). Absent or 0.0 keeps the
        # independent path -- and its exact RNG consumption -- unchanged.
        self.copula_rho = float(bundle.get("copula_rho", 0.0) or 0.0)
        assert (
            0.0 <= self.copula_rho < 1.0
        ), f"copula_rho must lie in [0, 1), got {self.copula_rho}"
        assert self.copula_rho == 0.0 or (
            self.is_punishment and self.model_type == "multinomial"
        ), (
            "copula_rho is implemented for the multinomial punishment sampler "
            f"only, got target={self.target!r} model={self.model_type!r}"
        )
        # Group copula, Gaussian contribution sampler: `copula_rho_p` /
        # `copula_rho_t` weight a shared latent that correlates a group's
        # contribution draws -- persistent per (episode, group) and transient
        # per (round, group) respectively, entering the standard normal as
        # z = sqrt(rho_p) * u_g + sqrt(rho_t) * v_g + sqrt(1 - rho_p - rho_t) * e_i
        # (calibrated by scripts/baselines/contribution_gmlp_copula_rho.py).
        # Absent or both 0.0 keeps the independent path -- and its exact RNG
        # consumption -- unchanged; this step only opens the gate, it draws
        # nothing.
        self.copula_rho_p = float(bundle.get("copula_rho_p", 0.0) or 0.0)
        self.copula_rho_t = float(bundle.get("copula_rho_t", 0.0) or 0.0)
        assert (
            0.0 <= self.copula_rho_p
        ), f"copula_rho_p must be >= 0, got {self.copula_rho_p}"
        assert (
            0.0 <= self.copula_rho_t
        ), f"copula_rho_t must be >= 0, got {self.copula_rho_t}"
        assert self.copula_rho_p + self.copula_rho_t < 1.0, (
            "copula_rho_p + copula_rho_t must be < 1 (residual idiosyncratic "
            f"variance would vanish), got {self.copula_rho_p} + {self.copula_rho_t}"
        )
        assert self.copula_rho_p + self.copula_rho_t == 0.0 or (
            self.target == "contribution"
            and self.model_type in ("gaussian", "gaussian_mlp")
        ), (
            "copula_rho_p / copula_rho_t are implemented for the Gaussian "
            f"contribution sampler only, got target={self.target!r} "
            f"model={self.model_type!r}"
        )
        self.default_values = dict(bundle["default_values"])
        self.switch_every = bundle.get("switch_every")
        # env-driven use (predict) needs both; the rounds-driven manager path
        # derives the agent count from the rounds themselves.
        self.n_agents = None if n_agents is None else int(n_agents)
        self.n_contributions = None if n_contributions is None else int(n_contributions)
        self.device = device
        self.sample = bool(sample)  # False -> deterministic levels (mu / argmax)
        # match the GNN switch predictor (sample=True); set False for a
        # deterministic proba>0.5 threshold.
        self.switch_sample = switch_sample
        if not self.is_switch:
            illegal = sorted(set(self.features) & CURRENT_VALUED)
            assert not illegal, (
                f"{self.target} bundle contains current-valued features "
                f"(illegal, they read the target's round): {illegal}"
            )
        self._reset_history()

    def to(self, device):
        self.device = device
        return self

    # ------------------------------------------------------------------ #
    # per-episode history (env-driven targets)
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

    # ------------------------------------------------------------------ #
    # feature-pool reconstruction (shared)
    # ------------------------------------------------------------------ #
    def _pool_from_arrays(self, c, p, cg, ag, T):
        """[A, T] measure/membership arrays -> the [1, A, T] create_torch_data
        tensors -> build_feature_pool's feature dict."""
        A = c.shape[0]
        dv = self.default_values
        rec = np.ones((A, T), dtype=float)  # every agent is present in the sim
        rounds = np.tile(np.arange(T, dtype=float), (A, 1))

        # -> contiguous [1, A, T] torch tensor (build_feature_pool .numpy()s it)
        def b(x):
            return th.from_numpy(np.ascontiguousarray(x[None]))

        # Not-yet-realised cells stay at defaults; the (asserted prev-family)
        # features never consume them.
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

    def _build_pool(self, t):
        """Env-driven path: reconstruct the [1, A, t+1] tensors from the
        accumulated episode history."""
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
        return self._pool_from_arrays(c, p, cg, ag, T)

    def _pool_from_rounds(self, rounds):
        """Rounds-driven path (punishment manager): rebuild the tensors from
        the round-dict history, recomputing per-capita common good with the
        env formula."""
        A, T = len(rounds[0]["contribution"]), len(rounds)
        dv = self.default_values

        def series(key, default):
            out = np.full((A, T), float(default))
            for t, r in enumerate(rounds):
                vals = r.get(key) or [None] * A
                for a, v in enumerate(vals):
                    if v is not None:
                        out[a, t] = float(v)
            return out

        c = series("contribution", dv["contribution"])
        p = series("punishment", dv["punishment"])
        ag = series("agent_group", 0).astype(int)
        cv = np.array(
            [[bool(v) for v in r["contribution_valid"]] for r in rounds]
        ).T  # [A, T]

        cg = np.full((A, T), float(dv["common_good"]))
        cz = np.where(cv, c, 0.0)
        pz = np.where(cv, p, 0.0)
        for t in range(T - 1):
            for g in (0, 1):
                sel = ag[:, t] == g
                nv = int(cv[sel, t].sum())
                if nv:
                    cg[sel, t] = (1.6 * cz[sel, t].sum() - pz[sel, t].sum()) / nv
        return self._pool_from_arrays(c, p, cg, ag, T)

    # ------------------------------------------------------------------ #
    # level sampling (shared by contribution and punishment)
    # ------------------------------------------------------------------ #
    def _class_probs(self, Xs, n_levels):
        """[n, n_levels] class probabilities of a multinomial bundle; shared
        by the independent and copula samplers so marginals cannot drift."""
        proba = self.estimator.predict_proba(Xs)
        P = np.full((len(Xs), n_levels), 1e-12)
        P[:, self.estimator.classes_] = proba
        if self.temperature != 1.0:
            P = P ** (1.0 / self.temperature)
        P /= P.sum(1, keepdims=True)
        return P

    def _sample_levels(self, Xs, n_levels):
        """Discrete levels [n]; sample=False -> deterministic. Randomness is
        drawn from the torch RNG so th.manual_seed governs linear and GNN
        sampling alike."""
        if self.model_type == "multinomial":
            P = self._class_probs(Xs, n_levels)
            if self.sample:
                lvl = th.multinomial(th.from_numpy(P), 1).reshape(-1)
                return lvl.numpy().astype(np.int64)
            return P.argmax(1).astype(np.int64)

        mu = self.estimator.predict(Xs)
        if self.model_type in ("gaussian", "gaussian_mlp") and self.sample:
            sd = self.estimator.predict_std(Xs)  # heteroscedastic sigma(x) head
            yhat = mu + th.randn(len(mu)).numpy() * sd
        elif self.sample and self.sigma > 0:  # ridge: homoscedastic scalar sigma
            yhat = mu + th.randn(len(mu)).numpy() * self.sigma
        else:  # no sigma stored / sample=False -> deterministic point prediction
            yhat = mu
        return np.clip(np.rint(yhat), 0, n_levels - 1).astype(np.int64)

    def _sample_levels_copula(self, Xs, n_levels, groups):
        """Discrete levels [A], one shared severity latent per group id:
        u_i = Phi(sqrt(rho) z_g(i) + sqrt(1-rho) eps_i), inverted through the
        agent's own CDF. Same marginals as ``_sample_levels``; always exactly
        2A torch draws per call. Conventions and rationale:
        notes/autoresearch_log/punisher-severity-copula.md (appendix)."""
        P = self._class_probs(Xs, n_levels)
        if not self.sample:
            return P.argmax(1).astype(np.int64)

        n = len(Xs)
        zs = th.randn(n, dtype=th.float64)  # fixed 2A draws, composition-stable
        eps = th.randn(n, dtype=th.float64)
        g = np.asarray(groups).reshape(-1)
        assert len(g) == n, f"groups has {len(g)} entries for {n} agents"
        first, pick = {}, np.empty(n, dtype=np.int64)
        for i, gid in enumerate(g):
            pick[i] = first.setdefault(int(gid), i)

        a = float(np.sqrt(self.copula_rho))
        b = float(np.sqrt(1.0 - self.copula_rho))
        u = th.special.ndtr(a * zs[th.from_numpy(pick)] + b * eps)
        cum = th.from_numpy(np.cumsum(P, axis=1))
        lvl = th.searchsorted(cum.contiguous(), u.reshape(-1, 1).contiguous())
        return lvl.reshape(-1).clamp(0, n_levels - 1).numpy().astype(np.int64)

    # ------------------------------------------------------------------ #
    # env-facing predict (contribution / switch)
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
                sw = th.rand(len(X)).numpy() < p1  # torch RNG: seeded draw
            else:
                sw = p1 >= 0.5
            pred = th.tensor(sw, dtype=th.bool, device=dev).reshape(1, -1, 1)
            return pred, None

        lvl = self._sample_levels(Xs, self.n_contributions)
        return th.tensor(lvl, dtype=th.int64, device=dev).reshape(1, -1, 1), None

    # ------------------------------------------------------------------ #
    # manager-facing get_punishments (punishment, #127)
    # ------------------------------------------------------------------ #
    def get_punishments(self, rounds):
        """Punishments for the last round in `rounds`, as an int64 [A] tensor."""
        assert (
            self.is_punishment
        ), f"get_punishments needs a punishment bundle, got {self.target!r}"
        T = len(rounds)
        pool = self._pool_from_rounds(rounds)
        X = np.column_stack([pool[f][0, :, T - 1] for f in self.features])
        Xs = self.scaler.transform(X)
        if self.sample and self.copula_rho > 0.0:
            # membership from the same round dict the features come from
            gid = rounds[-1].get("agent_group")
            groups = np.zeros(len(X), np.int64) if gid is None else np.asarray(gid)
            lvl = self._sample_levels_copula(Xs, self.n_levels, groups)
        else:
            lvl = self._sample_levels(Xs, self.n_levels)
        return th.tensor(lvl, dtype=th.int64)


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
