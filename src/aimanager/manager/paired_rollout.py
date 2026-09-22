"""Batched paired rollouts: one manager per seat, many episodes at once.

`simulation/simulate.py` can already express a pairing (`pairings:`), but it
runs `batch_size=1` and loops episodes in Python, which costs about 77 seconds
per pairing at 100 episodes -- fine for a dozen named rules, hopeless for a
space-filling design over a five-parameter family. `ArtificialHumanEnv` runs
the whole batch dimension in parallel on one GPU, and the sigmoid rule is a
pure function of the served state, so a *different parameter vector per batch
element* costs nothing beyond the arithmetic. That is what makes a
thousand-point design affordable.

The rollout is the training-time game, not self-play: the focal manager holds
group 0, the rival holds group 1, and members move between them every
`switch_every` rounds, decided by the switch predictor. Both managers are
served `env.served_state()`, so a player who gave no input reads contribution
0 -- the value the game charged and everyone saw -- rather than the imputed
default `update_contribution` leaves in `env.state`.

What is recorded is `env.state`, i.e. exactly what `per_round.parquet`
carries: the raw contribution (imputed for a timed-out player) plus the
validity flag, so every summary below can zero those cells itself rather than
averaging an imputed 9 in.
"""

import torch as th

from aimanager.manager.environment import ArtificialHumanEnv

#: The evaluation suite's own contribution bins (evaluation_suite.metrics
#: RPA_EDGES / RPA_LABELS), so a policy shape measured here is directly
#: comparable with the human and clone rows it is plotted beside.
RPA_EDGES = (-1.0, 0.0, 5.0, 10.0, 15.0, 19.0, 20.0)
RPA_LABELS = ("{0}", "1-5", "6-10", "11-15", "16-19", "{20}")


class BatchedPairEnv(ArtificialHumanEnv):
    """`ArtificialHumanEnv` with the edge index built once per env.

    The base class rebuilds `batch_edge_index` inside `update_groups`, which
    the switch predictor calls every `switch_every` rounds -- a Python list
    comprehension over `batch_size * n_agents * (n_agents - 1)` pairs, which
    is 229k iterations at batch 4096 and would dominate the rollout. The
    index does not depend on the memberships at all (the graph is fully
    connected within an episode; membership enters the model as a
    `same_group` edge feature), so it is built once, vectorised, and reused.
    `test_paired_rollout.py::test_edge_index_matches_base` pins it against
    the base class's own construction.
    """

    def update_groups(self, agent_groups):
        self.agent_groups = agent_groups.unsqueeze(-1)
        if "_edge_index_cache" not in self.__dict__:
            a = th.arange(self.n_agents, device=self.device)
            pairs = th.cartesian_prod(a, a)
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            offset = (
                th.arange(self.batch_size, device=self.device) * self.n_agents
            ).view(-1, 1, 1)
            self.__dict__["_edge_index_cache"] = (
                (pairs.unsqueeze(0) + offset).reshape(-1, 2).T.contiguous()
            )
        self.batch_edge_index = self.__dict__["_edge_index_cache"]
        self.agent_group_mask = th.nn.functional.one_hot(
            agent_groups, num_classes=self.n_groups
        ).unsqueeze(-1)


def make_env(
    *,
    contribution_model,
    valid_model,
    switch_model,
    batch_size,
    device,
    n_agents=8,
    agent_groups=(0, 0, 0, 0, 1, 1, 1, 1),
    n_groups=2,
    switch_every=4,
    n_rounds=24,
    n_contributions=21,
    n_punishments=31,
):
    """The standard 2 x 8, 24-round, switch-every-4 protocol, batched."""
    return BatchedPairEnv(
        artifical_humans=contribution_model,
        artifical_humans_valid=valid_model,
        artifical_humans_switch=switch_model,
        switch_every=switch_every,
        batch_size=batch_size,
        n_agents=n_agents,
        agent_groups=list(agent_groups),
        n_groups=n_groups,
        n_contributions=n_contributions,
        n_punishments=n_punishments,
        n_rounds=n_rounds,
        device=device,
    )


def rollout(env, focal, rival, focal_group=0):
    """Run one full batch of episodes; return the recorded `(B, A, T)` frame.

    Keys mirror `per_round.parquet`: `contribution` is the raw recorded value
    (imputed where the player gave no input), `contribution_valid` says which
    those are, `punishment` is what the env realised (already zeroed on
    invalid cells by `punish`), `agent_group` is the membership that round.
    """
    env.reset()
    rec = {k: [] for k in ("contribution", "punishment", "valid", "group")}
    for t in range(env.n_rounds):
        state = env.served_state()
        kw = dict(reset_rnn=t == 0, edge_index=env.batch_edge_index)
        focal_p, _ = focal.predict(state, **kw)
        rival_p, _ = rival.predict(state, **kw)
        is_focal = env.agent_groups == focal_group
        env.punish(th.where(is_focal, focal_p, rival_p))
        rec["contribution"].append(env.state["contribution"].squeeze(-1))
        rec["punishment"].append(env.state["punishment"].squeeze(-1))
        rec["valid"].append(env.state["contribution_valid"].squeeze(-1))
        rec["group"].append(env.agent_groups.squeeze(-1))
        env.step()
    return {k: th.stack(v, dim=-1).cpu() for k, v in rec.items()}


def _seat_summary(rec, seat_group, prefix):
    """Per-episode, per-round totals for one seat.

    Every quantity is a seat TOTAL, not a per-member average, because that is
    what the competing setting prices: a manager that raises contributions
    but loses the members who make them has not gained anything. A timed-out
    player contributed nothing and was charged nothing, so their cell is
    zeroed rather than read at the imputed default.
    """
    c, p, v, g = (rec[k] for k in ("contribution", "punishment", "valid", "group"))
    seat = g == seat_group
    c_eff = th.where(v, c, th.zeros_like(c)).to(th.float)
    p_eff = th.where(v, p, th.zeros_like(p)).to(th.float)
    n_rounds = c.shape[-1]

    members = (seat.sum(dim=1)).to(th.float)  # (B, T)
    n_valid = (seat & v).sum(dim=1).to(th.float)
    sum_c = (c_eff * seat).sum(dim=1)
    sum_p = (p_eff * seat).sum(dim=1)
    out = {
        f"{prefix}_members": members.sum(-1) / n_rounds,
        f"{prefix}_n_valid": n_valid.sum(-1) / n_rounds,
        f"{prefix}_contribution": sum_c.sum(-1) / n_rounds,
        f"{prefix}_punishment": sum_p.sum(-1) / n_rounds,
        f"{prefix}_pool": (1.6 * sum_c - sum_p).sum(-1) / n_rounds,
        # numerators / denominators for per-member-round rates, kept apart so
        # they aggregate correctly across episodes
        f"{prefix}_p_num": sum_p.sum(-1),
        f"{prefix}_p_den": members.sum(-1),
        f"{prefix}_c_num": sum_c.sum(-1),
        f"{prefix}_c_den": n_valid.sum(-1),
    }
    return out


def _policy_shape(rec, seat_group):
    """Mean punishment per contribution bin on the focal seat, valid cells.

    Bins are the evaluation suite's `RPA_EDGES`. The cells a player timed out
    on are dropped rather than binned, because their recorded contribution is
    an imputed 9 the manager never saw -- binning them would put the rule's
    timeout behaviour in the `6-10` column.
    """
    c, p, v, g = (rec[k] for k in ("contribution", "punishment", "valid", "group"))
    keep = (g == seat_group) & v
    cf = c.to(th.float)
    idx = th.zeros_like(cf, dtype=th.int64)
    for edge in RPA_EDGES[1:-1]:
        idx = idx + (cf > edge).to(th.int64)
    out = {}
    for k, label in enumerate(RPA_LABELS):
        sel = keep & (idx == k)
        out[f"rpa_n_{label}"] = sel.sum(dim=(1, 2)).to(th.float)
        out[f"rpa_p_{label}"] = (p.to(th.float) * sel).sum(dim=(1, 2))
    return out


def _leaver_diagnostic(rec, seat_group, switch_every):
    """What leavers contributed minus what stayers did, on the focal seat.

    The switch is decided at the end of round s with `(s + 1) % switch_every
    == 0` and applied at s + 1, so the decision rounds are exactly those s,
    and a leaver is a member whose membership differs at s + 1. Restricted to
    cells where the player gave an input, which is the convention the earlier
    paired arms used (`rule_vs_clone_paired_report.who_leaves`) and which
    keeps the imputed contribution out of the average.

    Correctly-targeted managers run negative here (leavers contributed less
    than stayers, so the group that remains improves) and inverted ones go
    positive.
    """
    c, p, v, g = (rec[k] for k in ("contribution", "punishment", "valid", "group"))
    n_rounds = c.shape[-1]
    dec = [s for s in range(n_rounds - 1) if (s + 1) % switch_every == 0]
    here, nxt = g[:, :, dec], g[:, :, [s + 1 for s in dec]]
    keep = (here == seat_group) & v[:, :, dec]
    leaves = nxt != here
    cf, pf = c[:, :, dec].to(th.float), p[:, :, dec].to(th.float)
    out = {}
    for tag, sel in (("lv", keep & leaves), ("st", keep & ~leaves)):
        out[f"{tag}_n"] = sel.sum(dim=(1, 2)).to(th.float)
        out[f"{tag}_c"] = (cf * sel).sum(dim=(1, 2))
        out[f"{tag}_p"] = (pf * sel).sum(dim=(1, 2))
    return out


def summarise(rec, focal_group=0, rival_group=1, switch_every=4):
    """Every per-episode quantity this arm reports, as `(B,)` float tensors.

    Sums and counts are kept apart rather than pre-divided so a caller can
    pool episodes (or bootstrap over them) without re-weighting mistakes.
    """
    return {
        **_seat_summary(rec, focal_group, "focal"),
        **_seat_summary(rec, rival_group, "rival"),
        **_policy_shape(rec, focal_group),
        **_leaver_diagnostic(rec, focal_group, switch_every),
    }
