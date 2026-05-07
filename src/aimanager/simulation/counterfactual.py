"""Counter-factual probe helpers — focal selection + override resolution.

Used by ``aimanager.simulation.intervention_probe``. Two responsibilities:

- ``_select_focal_agents`` picks one focal agent per chosen episode based
  on a rule evaluated against the prefix history (``lowest_contributor``,
  ``highest_contributor``, ``most_punished``, ``random``, or a literal
  agent index).
- ``intervention_value`` resolves the override for the focal's slot at
  round t* — either an absolute ``new_value`` or a ``factor`` multiplied
  against a per-episode reference (the pilot's actual round-t* value).
  Result is rounded and clamped to a non-negative int so it stays a
  valid encoder action level.
"""

import torch as th


SELECTOR_RULES = (
    "lowest_contributor",
    "highest_contributor",
    "most_punished",
    "random",
)


def intervention_value(intervention, natural, max_value=None):
    """Resolve the focal-slot override.

    ``intervention`` is the YAML intervention dict; either ``new_value``
    (absolute) or ``factor`` (relative). ``natural`` is the per-episode
    reference value the factor multiplies against (e.g., the pilot's
    real round-t* value for the focal). ``max_value`` is the upper bound
    of the encoder's action level — clamps factor mode results so they
    stay valid.
    """
    if "factor" in intervention:
        v = max(0, round(natural * float(intervention["factor"])))
    else:
        v = int(intervention["new_value"])
    if max_value is not None:
        v = min(v, max_value)
    return v


def _masked_mean(value: th.Tensor, valid: th.Tensor) -> th.Tensor:
    """Per-(episode, agent) mean of ``value`` ignoring rows where
    ``valid`` is False, with a guard against all-invalid rows."""
    valid_f = valid.float()
    return (value.float() * valid_f).sum(dim=2) / valid_f.sum(dim=2).clamp(min=1)


def _select_focal_agents(
    prefix_data: dict,
    rule,
    intervention_round: int,
    n_chains: int,
    rng: th.Generator,
) -> th.Tensor:
    """Pick one focal agent per chosen episode based on prefix history.

    Selector rules are evaluated against rounds ``[0, intervention_round)`` —
    the actual prefix that influenced behaviour up to the intervention point.
    The resulting per-chosen-episode tensor is repeated by ``n_chains`` so
    each chain shares its slot's focal.

    Args:
        prefix_data: dict of (n_chosen, n_agents, intervention_round + 1)
            tensors before chain replication.
        rule: selector name or literal int agent index.
        intervention_round: round at which the intervention happens.
        n_chains: K — focal repeated across chains.
        rng: torch.Generator for reproducible "random" selection.

    Returns:
        focal_agents: (n_chosen * n_chains,) int64 tensor.
    """
    n_chosen, n_agents = prefix_data["contribution"].shape[:2]

    if isinstance(rule, int):
        if not 0 <= rule < n_agents:
            raise ValueError(f"agent_selector={rule} out of range [0, {n_agents})")
        focal_per_chosen = th.full((n_chosen,), rule, dtype=th.int64)
    elif rule in ("lowest_contributor", "highest_contributor"):
        m = _masked_mean(
            prefix_data["contribution"][:, :, :intervention_round],
            prefix_data["contribution_valid"][:, :, :intervention_round],
        )
        focal_per_chosen = (
            m.argmin(dim=1) if rule == "lowest_contributor" else m.argmax(dim=1)
        )
    elif rule == "most_punished":
        pun = prefix_data["punishment"][:, :, :intervention_round].float()
        valid = prefix_data["punishment_valid"][:, :, :intervention_round].float()
        focal_per_chosen = (pun * valid).sum(dim=2).argmax(dim=1)
    elif rule == "random":
        focal_per_chosen = th.randint(
            0, n_agents, (n_chosen,), generator=rng, dtype=th.int64
        )
    else:
        raise ValueError(
            f"Unknown agent_selector: {rule!r}; "
            f"expected one of {SELECTOR_RULES} or int"
        )

    return focal_per_chosen.repeat_interleave(n_chains).to(th.int64)


def _select_focal_groups(data: dict, rule, n_groups: int, rng=None) -> list:
    """Pick episodes by team-level mean contribution over the full game.

    Used by ``target=group`` interventions to select which episodes
    (8-agent teams) to probe. Mean is over (agents, rounds) of valid
    rows; deterministic given the data, so the same teams are reused
    across all scenarios in a manifest.

    Returns the K episode indices as a plain ``list[int]``.
    """
    contrib = data["contribution"].float()
    valid = data["contribution_valid"].float()
    team_mean = (contrib * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1)

    if rule == "lowest_contributor":
        return team_mean.argsort()[:n_groups].tolist()
    if rule == "highest_contributor":
        return team_mean.argsort(descending=True)[:n_groups].tolist()
    if rule == "random":
        if rng is None:
            rng = th.Generator()
        idx = th.randperm(team_mean.shape[0], generator=rng)[:n_groups]
        return idx.tolist()
    raise ValueError(
        f"Unknown group_selector: {rule!r}; "
        f"expected lowest_contributor, highest_contributor, or random"
    )
