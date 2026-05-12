"""Counter-factual probe helpers — focal selection + override resolution.

Used by ``aimanager.simulation.intervention_probe``. Three helpers:

- ``_select_focal_agents`` picks one focal agent per chosen episode for
  ``target=individual`` (selector evaluated at agent level over the
  prefix history).
- ``_select_focal_group`` picks the agent indices in one group per
  chosen episode for ``target=group`` (same selector rules but
  evaluated at group level using ``agent_group`` at t*).
- ``intervention_value`` resolves the override value for a single slot
  — either ``new_value`` (absolute) or ``factor × pilot[..., t*]``
  (relative, rounded, clamped to encoder y_levels).
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
            raise ValueError(f"selector={rule} out of range [0, {n_agents})")
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
            f"Unknown selector: {rule!r}; "
            f"expected one of {SELECTOR_RULES} or int"
        )

    return focal_per_chosen.repeat_interleave(n_chains).to(th.int64)


def _select_focal_group(
    prefix_data: dict,
    rule,
    intervention_round: int,
    rng: th.Generator,
) -> th.Tensor:
    """Pick the agent indices belonging to the selected group at t*.

    For ``target=group``: same selector rules as ``_select_focal_agents``,
    but evaluated at group level (the ``agent_group`` partition at round
    ``intervention_round``). Returns the int64 indices of every agent
    currently in the chosen group; the caller overrides all of them.

    Selector rules:
      - lowest_contributor / highest_contributor: per group mean
        contribution over rounds [0, intervention_round).
      - most_punished: per group cumulative punishment over the prefix.
      - random: uniform over groups.
    """
    ag_at_t = prefix_data["agent_group"][0, :, intervention_round - 1]
    contrib = prefix_data["contribution"][0, :, :intervention_round].float()
    contrib_valid = prefix_data["contribution_valid"][0, :, :intervention_round].float()
    pun = prefix_data["punishment"][0, :, :intervention_round].float()
    pun_valid = prefix_data["punishment_valid"][0, :, :intervention_round].float()

    group_ids = ag_at_t.unique().tolist()
    scores = {}
    for g in group_ids:
        in_g = ag_at_t == g
        if rule in ("lowest_contributor", "highest_contributor"):
            c = contrib[in_g] * contrib_valid[in_g]
            n = contrib_valid[in_g].sum().clamp(min=1)
            scores[g] = (c.sum() / n).item()
        elif rule == "most_punished":
            scores[g] = (pun[in_g] * pun_valid[in_g]).sum().item()
        elif rule == "random":
            scores[g] = 0.0  # unused
        else:
            raise ValueError(
                f"Unknown selector: {rule!r}; "
                f"expected one of {SELECTOR_RULES}"
            )

    if rule == "lowest_contributor":
        chosen = min(scores, key=scores.get)
    elif rule == "highest_contributor" or rule == "most_punished":
        chosen = max(scores, key=scores.get)
    elif rule == "random":
        idx = th.randint(0, len(group_ids), (1,), generator=rng).item()
        chosen = group_ids[idx]

    return (ag_at_t == chosen).nonzero(as_tuple=False).flatten().to(th.int64)
