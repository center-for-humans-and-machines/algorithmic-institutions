"""The parametrised sigmoid punishment rule.

    f(c)    = 1 / (1 + exp((c - c0) / tau))
    m_ep(t) = ((T - t) / T) ** gamma_ep            T = n_rounds
    m_sw(s) = ((s + 1) / S) ** gamma_sw            S = switch_every
    p       = round(P_max * f(c) * m_ep(t) * m_sw(s)), clipped to [0, 30]

Five parameters: ``p_max``, ``c0``, ``tau``, ``gamma_ep``, ``gamma_sw``.

**Why a logistic.** It nests the incumbent continuously. As ``tau -> 0`` the
logistic becomes a hard step, so ``p_max = 10, c0 = 9.5, tau -> 0,
gamma_ep = gamma_sw = 0`` is exactly ``thr9_p10`` -- punish 10 whenever the
player contributed 9 or less -- the best manager measured anywhere in this
project (notes/autoresearch_log/rule-based-manager-sweep.md). A family that
could not reproduce the incumbent would not be a generalisation of it;
``tests/test_sigmoid_rule.py::test_reproduces_thr9_p10`` asserts it cell by
cell against ``RuleBasedManager(rule="threshold", ...)``.

**Why the multipliers run on the REMAINING horizon.** Punishment is an
investment: it is charged now and collected later, through the contributions
it raises. What is left to collect is what remains of the episode
(``m_ep``) and what remains of this player's tenure in the group before the
next reshuffle (``m_sw``). ``gamma = 0`` switches a multiplier off, so the
family also nests every memoryless rule in the earlier arms.

``s`` is the number of FURTHER rounds this round included that the manager
holds the player before the next reshuffle: the env decides a switch at the
end of round ``s`` with ``(s + 1) % S == 0`` and applies it at ``s + 1``
(``ArtificialHumanEnv.step``), so ``s = (S - 1) - (t mod S)``. At ``t = 3, 7,
11, ...`` the player may be gone next round and ``m_sw`` is at its smallest;
right after a reshuffle it is 1. The expression is periodic and takes no
notice of the episode ending before the next reshuffle would arrive -- that
horizon is ``m_ep``'s job.

The module deliberately holds no torch_geometric import, so the rule (and its
tests) run anywhere; only the callers that drive artificial humans need PyG.
"""

import torch as th

N_ROUNDS = 24
SWITCH_EVERY = 4
N_PUNISHMENTS = 31

#: Column order of a parameter vector, everywhere in this arm.
PARAM_NAMES = ("p_max", "c0", "tau", "gamma_ep", "gamma_sw")


def sigmoid_punishment(
    contribution,
    round_number,
    *,
    p_max,
    c0,
    tau,
    gamma_ep,
    gamma_sw,
    n_rounds=N_ROUNDS,
    switch_every=SWITCH_EVERY,
    n_punishments=N_PUNISHMENTS,
):
    """Punishment for every cell of ``contribution``.

    ``contribution`` and ``round_number`` are broadcastable tensors of any
    shape; each parameter is a scalar or a tensor broadcastable against them,
    which is what lets one rollout carry a different parameter vector per
    batch element. Returns a float tensor of whole numbers in
    ``[0, n_punishments - 1]``; the caller casts.
    """
    c = contribution.to(th.float)
    t = round_number.to(th.float)

    # sigmoid(-x) == 1 / (1 + exp(x)) but saturates instead of overflowing,
    # which is what makes the tau -> 0 limit usable rather than a nan.
    f = th.sigmoid(-(c - c0) / tau)

    # clamp the bases at 0: a negative base with a fractional exponent is nan,
    # and t > n_rounds (which no rollout produces) would otherwise go there.
    ep_base = ((n_rounds - t) / n_rounds).clamp(min=0.0)
    s = (switch_every - 1) - th.remainder(t, switch_every)
    sw_base = ((s + 1.0) / switch_every).clamp(min=0.0)

    raw = p_max * f * ep_base**gamma_ep * sw_base**gamma_sw
    return raw.round().clamp(0.0, float(n_punishments - 1))


class SigmoidRuleBatch:
    """The rule as a manager, one parameter vector per batch element.

    ``predict(state) -> (punishment, None)`` matches the opponent-manager
    interface ``aimanager.rl_manager.run_batch`` calls (and
    ``LinearPunisherOpponent.predict``), so the rule can hold either seat of
    a paired rollout without the loop knowing which kind of manager it has.

    ``theta`` is ``(B, 5)`` in ``PARAM_NAMES`` order, one row per episode in
    the batch. Rows are repeated by the caller: a shard of 128 design points
    run at 16 episodes each is a ``(2048, 5)`` theta whose rows repeat in
    blocks of 16.
    """

    autoregressive = False

    def __init__(
        self,
        theta,
        *,
        n_rounds=N_ROUNDS,
        switch_every=SWITCH_EVERY,
        n_punishments=N_PUNISHMENTS,
    ):
        theta = th.as_tensor(theta, dtype=th.float)
        assert theta.ndim == 2 and theta.shape[1] == len(PARAM_NAMES), theta.shape
        assert (theta[:, 2] > 0).all(), "tau must be strictly positive"
        self.theta = theta
        self.n_rounds = int(n_rounds)
        self.switch_every = int(switch_every)
        self.n_punishments = int(n_punishments)
        self.default_values = {
            "contribution": 0,
            "punishment": 0,
            "contribution_valid": False,
            "punishment_valid": False,
            "in_group": False,
        }

    def to(self, device):
        self.theta = self.theta.to(device)
        return self

    def predict(self, state, *, reset_rnn=False, edge_index=None, **_):
        c = state["contribution"]
        theta = self.theta.to(c.device)
        assert theta.shape[0] == c.shape[0], (theta.shape, c.shape)
        # (B, 1, 1) so every parameter broadcasts across agents and the
        # trailing singleton the env carries.
        p_max, c0, tau, gamma_ep, gamma_sw = (
            theta[:, i].view(-1, *([1] * (c.ndim - 1))) for i in range(len(PARAM_NAMES))
        )
        p = sigmoid_punishment(
            c,
            state["round_number"],
            p_max=p_max,
            c0=c0,
            tau=tau,
            gamma_ep=gamma_ep,
            gamma_sw=gamma_sw,
            n_rounds=self.n_rounds,
            switch_every=self.switch_every,
            n_punishments=self.n_punishments,
        )
        return p.to(th.int64), None


class ConstantManager:
    """A flat punishment for every cell, batched. ``never`` is ``amount=0``.

    Exists so the controls sit on the same interface as the rule and the
    clone, and so a paired rollout can put any of the three in either seat.
    """

    autoregressive = False

    def __init__(self, amount=0, **_):
        self.amount = int(amount)
        self.default_values = {
            "contribution": 0,
            "punishment": 0,
            "contribution_valid": False,
            "punishment_valid": False,
            "in_group": False,
        }

    def to(self, device):
        return self

    def predict(self, state, **_):
        return th.full_like(state["contribution"], self.amount), None
