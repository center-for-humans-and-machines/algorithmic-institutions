"""Behaviour-policy exploration for the RL manager.

The manager acts on 31 ordinal punishment levels, 0 to 30. The original
behaviour policy is epsilon-greedy with eps fixed at 0.1 and the exploratory
action drawn uniformly over all 31 levels. That injects an expected
``eps * mean(uniform) == 0.1 * 15 == 1.5`` punishment points per member per
round on top of whatever the greedy policy asked for -- the size of the entire
learned signal -- so the policy that is evaluated and the policy that fills the
replay buffer are different policies.

Two switchable changes live here, both off by default so the original
behaviour is recoverable from config alone:

``eps_final`` / ``eps_anneal_steps``
    Linear decay of eps from its starting value to a floor, reached exactly at
    ``eps_anneal_steps`` and held there afterwards. Linear-to-a-floor is the
    DQN standard (Mnih et al. 2015) and is the only common schedule that
    reaches its floor at a stated step rather than asymptotically, which is
    what turns "the behaviour policy has converged onto the evaluated one"
    into a checkable claim.

``sigma``
    Replaces the uniform draw with a discretised Gaussian centred on the
    current greedy action, so exploration respects the ordinal structure of
    the action space and probes near the policy instead of across it.

    Boundaries are handled by truncation and renormalisation, not clipping:
    the sampled action is distributed as the discretised Gaussian
    *conditioned* on landing in ``[0, n_actions - 1]``,

        P(a | a0) = exp(-(a - a0)^2 / 2 sigma^2) / sum_b exp(-(b - a0)^2 / 2 sigma^2)

    over ``b`` in ``[0, n_actions - 1]``. Clipping would instead push the whole
    out-of-range tail onto the endpoint: at ``a0 = 0`` and ``sigma = 2`` that
    is 0.600 of the mass on the single action 0, a spike the ordinal story
    does not justify. Truncation puts 0.333 there, and that 0.333 is not a
    pile-up -- it is the interior value 0.199 times the mass the lost half of
    the kernel used to carry. Every pair of in-range actions keeps the
    Gaussian ratio it would have had; only the normaliser changes.
"""

import torch as th


class Exploration:
    """Epsilon-greedy with an optional epsilon schedule and an optional local
    proposal distribution.

    With ``eps_final`` and ``sigma`` both unset this is exactly the original
    constant-eps, uniform-proposal behaviour.
    """

    def __init__(
        self,
        *,
        eps,
        n_actions,
        device,
        eps_final=None,
        eps_anneal_steps=None,
        sigma=None,
    ):
        if (eps_final is None) != (eps_anneal_steps is None):
            raise ValueError(
                "eps_final and eps_anneal_steps must be set together "
                f"(got eps_final={eps_final}, eps_anneal_steps={eps_anneal_steps})"
            )
        if eps_anneal_steps is not None and eps_anneal_steps <= 0:
            raise ValueError("eps_anneal_steps must be positive")
        if sigma is not None and sigma <= 0:
            raise ValueError("sigma must be positive")

        self.eps = eps
        self.eps_final = eps_final
        self.eps_anneal_steps = eps_anneal_steps
        self.sigma = sigma
        self.n_actions = n_actions
        self.device = device

        # Row a0 of `proposal_probs` is the full proposal distribution for a
        # greedy action of a0: the discretised Gaussian restricted to the
        # action set and renormalised. `softmax` over the in-range logits *is*
        # that renormalisation, so the boundary case needs no special branch.
        # float64, not float32: at sigma = 2 the far tail of a 31-level row is
        # around exp(-112), which underflows float32 to exactly zero and would
        # silently make the claimed ratios wrong at the ends of the scale. The
        # table is 31x31, so the precision is free.
        self.proposal_probs = None
        if sigma is not None:
            a = th.arange(n_actions, device=device, dtype=th.float64)
            d = a.unsqueeze(0) - a.unsqueeze(1)
            self.proposal_probs = (-(d**2) / (2 * sigma**2)).softmax(-1)

    def epsilon(self, update_step=None):
        """Epsilon in force at `update_step`; constant if no schedule is set."""
        if self.eps_final is None or update_step is None:
            return self.eps
        frac = min(max(update_step / self.eps_anneal_steps, 0.0), 1.0)
        return self.eps + frac * (self.eps_final - self.eps)

    def propose(self, greedy_action):
        """The exploratory action, drawn for every cell regardless of whether
        it will be used. Uniform over the action set unless `sigma` is set."""
        if self.proposal_probs is None:
            return th.randint(
                0, self.n_actions, size=greedy_action.shape, device=self.device
            )
        probs = self.proposal_probs[greedy_action.reshape(-1)]
        return th.multinomial(probs, 1).reshape(greedy_action.shape)

    def __call__(self, greedy_action, update_step=None):
        eps = self.epsilon(update_step)
        if eps <= 0:
            return greedy_action
        proposal = self.propose(greedy_action)
        select = th.rand(greedy_action.shape, device=self.device) < eps
        return th.where(select, proposal, greedy_action)

    def describe(self, update_step=None):
        """One-line summary for run logs and guard reports."""
        kind = "uniform" if self.sigma is None else f"local(sigma={self.sigma})"
        sched = (
            "constant"
            if self.eps_final is None
            else f"{self.eps}->{self.eps_final} over {self.eps_anneal_steps}"
        )
        return (
            f"eps={self.epsilon(update_step):.4f} "
            f"(schedule {sched}), proposal {kind}"
        )
