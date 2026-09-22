"""Policy-shape and ensemble-diversity probes for the bootstrapped manager.

Two questions this arm has to answer, both cheap enough to run inside the
training loop on the forward pass the acting policy already made:

1. **Shape.** Mean punishment binned by the contribution it was aimed at.
   Human managers are monotone decreasing (4.76 at contribution 0 down to
   0.27 at 20); two of the three epsilon-greedy seeds came out monotone
   *increasing*. The bins are the evaluation suite's own RPA bins, imported
   from `aimanager.evaluation_suite.metrics` rather than restated, so this
   probe and `scripts/rl_two_worlds/measure.py` cut the axis identically and
   the arms stay comparable.

2. **Diversity.** Whether the K heads are actually different policies. The
   headline is `head_slope_sign_spread`: whether the heads disagree about the
   *sign* of the contribution-punishment relationship. Each head held one
   policy for a whole episode, so a sign disagreement is evidence that
   coherent contingent trajectories were generated and that the value
   function has not resolved their returns -- which is the trajectory-coverage
   claim this arm rests on, not the action-distribution one. See
   notes/autoresearch_log/rl-manager-bootstrapped-dqn.md.

Everything here is a pure function of tensors the rollout already has.
"""

import torch as th

from aimanager.evaluation_suite.metrics import RPA_EDGES, RPA_LABELS

N_RPA_BINS = len(RPA_LABELS)
# The finite interior edges of RPA_EDGES = [-1, 0, 5, 10, 15, 19, 20]. A cell
# lands in bin `sum_j (c > edge_j)`, which reproduces pandas' right-closed
# `pd.cut(c, RPA_EDGES, labels=RPA_LABELS)` on integer contributions:
# {0}, 1-5, 6-10, 11-15, 16-19, {20}.
_INNER_EDGES = tuple(RPA_EDGES[1:-1])


def rpa_bins(contribution):
    """Bin index 0..5 for each cell, matching RPA_LABELS."""
    out = th.zeros_like(contribution, dtype=th.long)
    for edge in _INNER_EDGES:
        out = out + (contribution > edge).long()
    return out


def binned_sum_count(values, bins, mask):
    """Per-bin sum and count of `values` over the cells `mask` keeps."""
    v = values.reshape(-1).to(th.float)
    b = bins.reshape(-1)
    m = mask.reshape(-1).to(th.float)
    zeros = th.zeros(N_RPA_BINS, device=v.device, dtype=th.float)
    total = zeros.clone().scatter_add_(0, b, v * m)
    count = zeros.clone().scatter_add_(0, b, m)
    return total, count


def binned_mean(total, count):
    """Per-bin mean, NaN where the bin is empty."""
    nan = th.full_like(total, float("nan"))
    return th.where(count > 0, total / count.clamp(min=1.0), nan)


def shape_metrics(punishment, contribution, mask, prefix="rpa"):
    """`{prefix}_mean_b<i>` and `{prefix}_n_b<i>` for the six RPA bins, plus
    `{prefix}_slope`: the top bin's mean minus the bottom bin's -- mean
    punishment at contribution 20 minus mean punishment at contribution 0.
    Negative is the human sign (humans: 0.27 - 4.76 = -4.49)."""
    bins = rpa_bins(contribution)
    total, count = binned_sum_count(punishment, bins, mask)
    mean = binned_mean(total, count)
    out = {}
    for i in range(N_RPA_BINS):
        out[f"{prefix}_mean_b{i}"] = mean[i].item()
        out[f"{prefix}_n_b{i}"] = count[i].item()
    out[f"{prefix}_slope"] = (mean[-1] - mean[0]).item()
    return out


def gather_to_own_group(per_group, agent_group):
    """(E, G, A, T, ...) -> (E, A, T, ...), keeping each agent's own group's
    slice. `agent_group` is (E, A, T)."""
    idx = agent_group
    for _ in range(per_group.dim() - 4):
        idx = idx.unsqueeze(-1)
    idx = idx.unsqueeze(1).expand(per_group.shape[0], 1, *per_group.shape[2:])
    return per_group.gather(1, idx).squeeze(1)


def consensus_action(q_values, agent_group):
    """The evaluation policy's action: argmax over the head-averaged Q."""
    return gather_to_own_group(q_values.mean(-2).argmax(-1), agent_group)


def vote_action(head_actions, n_actions):
    """Plurality vote over the per-head argmaxes, ties to the lowest level."""
    flat = head_actions.reshape(-1, head_actions.shape[-1])
    votes = th.zeros((flat.shape[0], n_actions), device=flat.device, dtype=th.float)
    votes.scatter_add_(1, flat, th.ones_like(flat, dtype=th.float))
    return votes.argmax(-1).reshape(head_actions.shape[:-1])


def head_metrics(q_values, head_actions, contribution, mask, agent_group):
    """Ensemble diversity on the cells `mask` keeps.

    `q_values`: (E, G, A, T, K, n_actions) straight from `get_action`.
    `head_actions`: (E, A, T, K), each head's greedy action gathered to the
    agent's own group.
    """
    n_heads = head_actions.shape[-1]
    if n_heads < 2:
        return {}
    m = mask.reshape(-1).to(th.float)
    denom = m.sum().clamp(min=1.0)
    a = head_actions.reshape(-1, n_heads).to(th.float)

    spread = a.max(-1)[0] - a.min(-1)[0]
    out = {
        "head_disagree_frac": ((spread > 0).to(th.float) * m).sum().item()
        / denom.item(),
        "head_action_spread": (spread * m).sum().item() / denom.item(),
    }
    per_head_mean = (a * m.unsqueeze(-1)).sum(0) / denom
    out["head_mean_spread"] = (per_head_mean.max() - per_head_mean.min()).item()

    # The headline: do the heads agree on the SIGN of the
    # contribution-punishment relationship? Per head, the top RPA bin's mean
    # action minus the bottom bin's.
    bins = rpa_bins(contribution).reshape(-1)
    slopes = []
    for k in range(n_heads):
        total, count = binned_sum_count(a[:, k], bins, m)
        mean = binned_mean(total, count)
        slope = (mean[-1] - mean[0]).item()
        slopes.append(slope)
        out[f"head_slope_h{k}"] = slope
    finite = [s for s in slopes if s == s]
    if finite:
        out["head_slope_spread"] = max(finite) - min(finite)
        n_neg = sum(1 for s in finite if s < 0)
        # 0 when every head agrees on the sign, 0.5 when they split evenly.
        out["head_slope_sign_spread"] = min(n_neg, len(finite) - n_neg) / len(finite)

    # Mean-of-Q consensus against plurality-vote consensus. They are not the
    # same rule on an ordinal action space; this measures how often they
    # differ on the states actually visited, so the choice of rule is
    # measured rather than asserted.
    mean_q = consensus_action(q_values, agent_group).reshape(-1).to(th.float)
    vote = vote_action(head_actions, q_values.shape[-1]).reshape(-1).to(th.float)
    out["consensus_vote_agree"] = (
        (mean_q == vote).to(th.float) * m
    ).sum().item() / denom.item()
    out["consensus_vote_gap"] = ((mean_q - vote).abs() * m).sum().item() / denom.item()
    return out
