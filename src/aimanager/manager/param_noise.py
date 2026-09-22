"""Parameter-space exploration noise for the RL manager.

Plappert et al. (2018), *Parameter Space Noise for Exploration*: additive
Gaussian noise on the weights of the network that **acts**, resampled once per
episode and held fixed for the whole episode, with the noise scale adapted
online to hold a target divergence in action space.

Why this and not action noise. The manager chooses one of 31 ordinal
punishment levels. Epsilon-greedy at 0.1 replaces a tenth of those choices
with a uniform draw, which injects ~1.5 punishment points per member per round
and -- worse for this task -- applies that punishment *independently of the
contribution it is aimed at*. Weight noise perturbs the function rather than
the output: a perturbed policy still maps contribution to punishment in some
coherent way, and a whole episode is collected under one such mapping.

Three design decisions, all deliberate:

1. **Only the acting network is perturbed.** The perturbation lives on a
   private deep copy (`self.perturbed`). `ArtificalManager.policy_model`,
   `ArtificalManager.target_model` and the opponent are never written to.
   `refresh()` copies the live policy parameters in and adds noise on top, so
   the copy tracks training without the training ever seeing the noise.

2. **Relative (per-tensor) scale.** The paper uses layer normalisation so that
   one global sigma means the same thing in every layer. Adding layer norm
   here would change the network even at zero noise and break the
   reproduction guarantee, so instead each tensor's noise is scaled by that
   tensor's own RMS. This addresses the same failure the paper's layer norm
   addresses -- a sigma that is huge in one layer and negligible in another --
   without touching the architecture. Switchable via ``relative``.

3. **The divergence measure is ordinal, and its target tracks epsilon-greedy.**
   See `MEASURES` and `eps_greedy_displacement` below. Measured on this
   project's own stack, the unweighted measure Plappert uses gives a policy
   shifted one level (0.00445) and a policy with its contribution contingency
   *inverted* (0.00469) the same number to within 5%, while the ordinal
   measure separates them 1.00 against 5.77. For an experiment whose result
   is that two of three seeds inverted their contingency, that settles it.

Zero scale is an exact identity: with ``scale == 0`` no random number is drawn
at all and the perturbed copy holds exactly the policy parameters, so the
acting network is the policy network and the agent is deterministic-greedy.
With no ``param_noise`` block in the config the class is never constructed and
the epsilon-greedy code path is untouched.
"""

import copy

import torch as th

# The three action-space divergence measures, all computed on the same pair of
# Q tensors so a run can report what the other two would have said.
#
#   l2   Plappert's own DQN measure: the RMS difference between the two
#        softmax policies, averaged over actions and states. UNWEIGHTED over
#        the action set -- it cannot tell a policy that shifted every
#        punishment by one level from one that inverted its contingency on
#        contribution, because both simply move probability from one index to
#        another. At 31 ordinal levels it also saturates: once the argmax has
#        moved at all, moving it further barely changes the number.
#   w1   1-Wasserstein between the two softmax policies on the integer
#        punishment line, i.e. the sum over levels of the absolute CDF
#        difference. Ordinal, in punishment points, but it is a distance
#        between *distributions*, and these policies are near-deterministic.
#   mad  Mean |a - a~| between the two greedy actions, in punishment points.
#        Ordinal, and in exactly the units the whole comparison is stated in:
#        epsilon-greedy's 1.5 punishment points per member per round is the
#        same quantity measured on the same axis.
MEASURES = ("mad", "l2", "w1")


def divergence_mad(q_ref, q_pert, temperature=1.0):
    """Mean absolute displacement of the greedy action, in punishment levels."""
    a = q_ref.argmax(-1).to(th.float)
    b = q_pert.argmax(-1).to(th.float)
    return (a - b).abs().mean().item()


def divergence_l2(q_ref, q_pert, temperature=1.0):
    """Plappert's DQN distance: RMS difference of the softmax policies."""
    p = th.softmax(q_ref / temperature, dim=-1)
    q = th.softmax(q_pert / temperature, dim=-1)
    return (p - q).pow(2).mean(-1).mean().sqrt().item()


def divergence_w1(q_ref, q_pert, temperature=1.0):
    """1-Wasserstein between the softmax policies on the ordinal action line."""
    p = th.softmax(q_ref / temperature, dim=-1).cumsum(-1)
    q = th.softmax(q_pert / temperature, dim=-1).cumsum(-1)
    return (p - q).abs().sum(-1).mean().item()


DIVERGENCES = {
    "mad": divergence_mad,
    "l2": divergence_l2,
    "w1": divergence_w1,
}

# The sentinel `target` that tracks epsilon-greedy instead of fixing a number.
EPS_MATCHED = "eps_matched"


def eps_greedy_displacement(q_ref, eps, n_actions):
    """The mean |a_behaviour - a_greedy| that epsilon-greedy WOULD produce on
    exactly these states, in punishment levels. Exact, not sampled.

    For a uniform draw u over {0, ..., N-1} replacing the greedy action a with
    probability eps, the expected displacement is
        eps * E_u|u - a| = eps * (a(a+1)/2 + (N-1-a)(N-a)/2) / N.

    This is what makes the arm a one-variable change. The displacement
    epsilon-greedy injects is NOT a constant: it depends on where the greedy
    action sits, and it grows as the policy sharpens toward 0 (15.0 levels at
    a=0, 7.7 at a=15). Fixing the target to a number measured once at
    initialisation would silently make the arm explore less than the reference
    exactly when the reference explores most. Tracking it keeps the two matched
    at every point in training.

    Matched in the MEAN, not in shape -- and that is the point. Epsilon-greedy
    concentrates the same mean displacement into a tenth of the cells as large
    independent jumps; weight noise spreads it over every cell as one coherent
    shift. The difference between those two ways of spending the same
    displacement is the whole hypothesis.
    """
    a = q_ref.argmax(-1).to(th.float)
    n = float(n_actions)
    e = (a * (a + 1.0) / 2.0 + (n - 1.0 - a) * (n - a) / 2.0) / n
    return float(eps) * e.mean().item()


class ParameterNoise:
    """Adaptive additive weight noise on a private copy of the acting network.

    Lifecycle, one cycle per episode:
        ``refresh()``  -> copy policy weights in, draw one perturbation
        ``observe()``  -> per round, record the divergence from the policy
        ``finish()``   -> average the divergences, adapt the scale, report
    """

    def __init__(
        self,
        model,
        *,
        scale=0.0,
        target=EPS_MATCHED,
        eps=None,
        measure="mad",
        adapt_coef=1.01,
        relative=True,
        adapt=True,
        temperature=1.0,
        min_scale=1e-6,
        max_scale=100.0,
        dead_zone_steps=5,
    ):
        assert measure in MEASURES, f"measure must be one of {MEASURES}, got {measure}"
        assert scale >= 0.0, f"scale must be non-negative, got {scale}"
        assert adapt_coef > 1.0, f"adapt_coef must exceed 1, got {adapt_coef}"
        self.eps_matched = target == EPS_MATCHED
        if self.eps_matched:
            assert measure == "mad", (
                "the epsilon-matched target is a mean action displacement in "
                f"punishment levels, so it only means anything for 'mad', not {measure}"
            )
            assert eps is not None, "target 'eps_matched' needs the reference eps"
            self.eps = float(eps)
            self.target = float("nan")  # set per episode from the greedy policy
        else:
            assert target > 0.0, f"target must be positive, got {target}"
            self.eps = eps
            self.target = float(target)
        self.model = model
        self.scale = float(scale)
        self.measure = measure
        self.adapt_coef = float(adapt_coef)
        self.relative = bool(relative)
        self.adapt = bool(adapt)
        self.temperature = float(temperature)
        self.min_scale = float(min_scale)
        # Deliberately loose. A cap that binds is indistinguishable, in the
        # logs, from a mechanism that is working -- which is exactly the
        # failure the 200-step pilot found at a cap of 1.0. Measured there:
        # the scale reached 3.02 and was still climbing, holding only 0.365
        # punishment levels against a target of 1.10, because the policy had
        # collapsed to a constant action and its argmax barely moves. Letting
        # the search run means that if weight noise CANNOT match
        # epsilon-greedy's displacement while staying a local perturbation,
        # the logged scale says so instead of hiding it.
        self.max_scale = float(max_scale)
        # The ordinal measure has a dead zone. `mad` counts argmax flips, so
        # while the perturbation is too small to flip any argmax it reads
        # EXACTLY zero -- not small, zero -- and the paper's fixed 1% step is
        # climbing a signal that carries no information about how far it has
        # to go. Measured on the 200-step pilot: the policy had collapsed to a
        # constant action, `mad` was 0.000 while `l2` was 0.00003, and the
        # scale had walked from 0.05 to 0.331 in 200 episodes without the
        # mechanism ever engaging. Inside the dead zone the search is
        # therefore geometric at a coarser rate; outside it, it is exactly
        # Plappert's. `dead_zone_steps=0` restores the paper's behaviour.
        assert dead_zone_steps >= 0
        self.dead_zone_steps = int(dead_zone_steps)

        # The acting network. A deep copy so that nothing the optimiser owns
        # is aliased: `refresh` writes into these tensors, never into the
        # policy's. requires_grad off -- no gradient ever flows through the
        # perturbed weights, which is what keeps `update` unchanged.
        self.perturbed = copy.deepcopy(model)
        for p in self.perturbed.parameters():
            p.requires_grad_(False)
        # A deep-copied GRU loses cuDNN's flat weight buffer, and without this
        # every forward pass recompacts it and says so. Relayout only; the
        # parameter objects the perturbation writes into are unchanged.
        self._rnns = [
            m for m in self.perturbed.modules() if isinstance(m, th.nn.RNNBase)
        ]
        for rnn in self._rnns:
            rnn.flatten_parameters()

        self._acc = {m: [] for m in MEASURES}
        self._targets = []
        self.last = {m: float("nan") for m in MEASURES}
        self.refresh()

    def refresh(self):
        """Start an episode: reload the policy weights, draw one perturbation.

        The perturbation is drawn HERE and nowhere else, so it is fixed for
        every round of the episode that follows.
        """
        self.perturbed.train(self.model.training)
        with th.no_grad():
            for src, dst in zip(self.model.parameters(), self.perturbed.parameters()):
                dst.copy_(src)
                if self.scale <= 0.0:
                    continue
                s = self.scale
                if self.relative:
                    s = s * src.detach().float().pow(2).mean().sqrt().item()
                if s > 0.0:
                    dst.add_(th.randn_like(dst) * s)
            for src, dst in zip(self.model.buffers(), self.perturbed.buffers()):
                dst.copy_(src)
        for rnn in self._rnns:
            rnn.flatten_parameters()
        self._acc = {m: [] for m in MEASURES}
        self._targets = []

    def observe(self, q_ref, q_pert):
        """Record this round's divergence under all three measures."""
        for name, fn in DIVERGENCES.items():
            self._acc[name].append(fn(q_ref, q_pert, self.temperature))
        if self.eps_matched:
            self._targets.append(
                eps_greedy_displacement(q_ref, self.eps, q_ref.shape[-1])
            )

    def finish(self):
        """End the episode: average, adapt, and report what the episode saw.

        The reported scale is the one the episode was actually collected
        under, not the adapted one -- the adaptation applies to the next
        episode.
        """
        collected_scale = self.scale
        for name in MEASURES:
            vals = self._acc[name]
            self.last[name] = sum(vals) / len(vals) if vals else float("nan")
        if self.eps_matched and self._targets:
            self.target = sum(self._targets) / len(self._targets)
        d = self.last[self.measure]
        if self.adapt and self.scale > 0.0 and d == d and self.target == self.target:
            step = self.adapt_coef ** (1 + self.dead_zone_steps if d == 0.0 else 1)
            if d < self.target:
                self.scale = min(self.scale * step, self.max_scale)
            else:
                self.scale = max(self.scale / self.adapt_coef, self.min_scale)
        self._acc = {m: [] for m in MEASURES}
        self._targets = []
        return {
            "param_noise_scale": collected_scale,
            "param_noise_target": self.target,
            "param_noise_divergence": self.last[self.measure],
            "param_noise_divergence_mad": self.last["mad"],
            "param_noise_divergence_l2": self.last["l2"],
            "param_noise_divergence_w1": self.last["w1"],
        }
