"""Parameter-space noise for the RL manager (exploration arm: param noise).

Runs on Raven: `ArtificalManager` imports `GraphNetwork` -> `torch_scatter`.

What is proved here, in the order the brief asks for it:
  * the perturbation is resampled once per episode (`refresh`) and is fixed
    for every round inside it;
  * a zero noise scale reproduces the unperturbed agent exactly, and with no
    `param_noise` block the epsilon-greedy path is bit-identical to before;
  * the target network, the policy network and the optimiser's parameters are
    never written to by the noise;
  * the unweighted action-space divergence really is blind to the ordinal
    structure at 31 levels -- the measurement behind the choice of measure.
"""

import pytest
import torch as th

N_PUNISHMENTS = 31
N_CONTRIBUTIONS = 21
N_AGENTS = 8
N_BATCH = 4

MODEL_ARGS = {
    "hidden_size": 16,
    "add_rnn": True,
    "add_edge_model": True,
    "add_global_model": False,
    "x_encoding": [
        {"name": "contribution", "n_levels": N_CONTRIBUTIONS, "encoding": "numeric"},
        {"name": "prev_punishment", "n_levels": N_PUNISHMENTS, "encoding": "numeric"},
        {"etype": "bool", "name": "contribution_valid"},
        {"etype": "bool", "name": "in_group"},
    ],
    "b_encoding": [{"name": "round_number", "n_levels": 32, "encoding": "onehot"}],
}

DEFAULT_VALUES = {"contribution": 0, "punishment": 0, "prev_punishment": 0}

PARAM_NOISE = {
    "scale": 0.5,
    "target": 1.5,
    "measure": "mad",
    "adapt_coef": 1.01,
    "relative": True,
}


def _manager(param_noise=None, seed=0):
    from aimanager.manager.manager import ArtificalManager

    th.manual_seed(seed)
    return ArtificalManager(
        n_contributions=N_CONTRIBUTIONS,
        n_punishments=N_PUNISHMENTS,
        n_groups=2,
        default_values=DEFAULT_VALUES,
        model_args=MODEL_ARGS,
        opt_args={"lr": 2e-4},
        gamma=0.98,
        target_update_freq=1000,
        eps=0.1,
        param_noise=param_noise,
        device=th.device("cpu"),
    )


def _state(seed=0):
    g = th.Generator().manual_seed(seed)
    shape = (N_BATCH, N_AGENTS, 1)
    return {
        "contribution": th.randint(0, N_CONTRIBUTIONS, shape, generator=g),
        "prev_punishment": th.randint(0, N_PUNISHMENTS, shape, generator=g),
        "contribution_valid": th.ones(shape, dtype=th.bool),
        "punishment": th.randint(0, N_PUNISHMENTS, shape, generator=g),
        "agent_group": th.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        .view(1, N_AGENTS, 1)
        .expand(shape)
        .contiguous(),
        "round_number": th.zeros(shape, dtype=th.int64),
    }


def _edge_index():
    """`run_batch` calls `get_action` without one, so the model builds the
    fully connected graph itself -- the tests take the same path."""
    return None


def _params(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _same(a, b):
    return set(a) == set(b) and all(th.equal(a[k], b[k]) for k in a)


# ── per-episode resampling and fixity ────────────────────────────────


def test_refresh_draws_a_new_perturbation():
    m = _manager(PARAM_NOISE)
    th.manual_seed(1)
    m.param_noise.refresh()
    first = _params(m.param_noise.perturbed)
    m.param_noise.refresh()
    second = _params(m.param_noise.perturbed)
    assert not _same(first, second)


def test_perturbation_is_fixed_within_an_episode():
    """No `refresh` between rounds -> the acting weights do not move, and the
    same state re-presented to a reset RNN yields the same action."""
    m = _manager(PARAM_NOISE)
    state, ei = _state(), _edge_index()
    m.begin_behaviour_episode()
    before = _params(m.param_noise.perturbed)
    a0, _ = m.get_action(state, first=True, edge_index=ei, greedy=False)
    a1, _ = m.get_action(state, first=False, edge_index=ei, greedy=False)
    a2, _ = m.get_action(state, first=True, edge_index=ei, greedy=False)
    assert _same(before, _params(m.param_noise.perturbed))
    assert th.equal(a0, a2)  # same weights, same state, RNN reset both times
    assert a1.shape == a0.shape


def test_behaviour_is_deterministic_given_the_perturbation():
    """Two managers with identical weights and the same drawn perturbation act
    identically -- there is no residual action-level dithering."""
    m = _manager(PARAM_NOISE)
    state, ei = _state(), _edge_index()
    th.manual_seed(7)
    m.begin_behaviour_episode()
    a = m.get_action(state, first=True, edge_index=ei, greedy=False)[0]
    b = m.get_action(state, first=True, edge_index=ei, greedy=False)[0]
    assert th.equal(a, b)


# ── the zero-scale reproduction ──────────────────────────────────────


def test_zero_scale_is_exactly_the_unperturbed_policy():
    m = _manager({**PARAM_NOISE, "scale": 0.0})
    state, ei = _state(), _edge_index()
    m.begin_behaviour_episode()
    assert _same(_params(m.policy_model), _params(m.param_noise.perturbed))
    behaviour = m.get_action(state, first=True, edge_index=ei, greedy=False)[0]
    evaluated = m.get_action(state, first=True, edge_index=ei, greedy=True)[0]
    assert th.equal(behaviour, evaluated)


def test_zero_scale_draws_no_random_numbers():
    """Switching the mechanism off must not perturb the RNG stream either."""
    m = _manager({**PARAM_NOISE, "scale": 0.0})
    th.manual_seed(11)
    m.param_noise.refresh()
    after_refresh = th.rand(3)
    th.manual_seed(11)
    untouched = th.rand(3)
    assert th.equal(after_refresh, untouched)


def test_disabled_block_leaves_the_eps_greedy_path_bit_identical():
    """`enabled: false` (and an absent block) must reproduce the existing
    agent: same greedy action, same two RNG draws, same `where`."""
    off = _manager({**PARAM_NOISE, "enabled": False}, seed=3)
    assert off.param_noise is None
    state, ei = _state(), _edge_index()

    th.manual_seed(99)
    picked, q = off.get_action(state, first=True, edge_index=ei, greedy=False)

    th.manual_seed(99)
    greedy = off.get_action(state, first=True, edge_index=ei, greedy=True)[0]
    # Exactly the draws manager.get_action makes, in the same order.
    rnd = th.randint(0, q.shape[-1], size=greedy.shape)
    sel = th.rand(size=greedy.shape) < off.eps
    assert th.equal(picked, th.where(sel, rnd, greedy))


def test_behaviour_label_names_the_mechanism():
    assert _manager(PARAM_NOISE).behaviour_label == "param-noise"
    assert _manager(None).behaviour_label == "eps-greedy"


# ── nothing but the acting network is perturbed ──────────────────────


def test_target_and_policy_networks_are_never_perturbed():
    m = _manager(PARAM_NOISE)
    state, ei = _state(), _edge_index()
    policy_before = _params(m.policy_model)
    target_before = _params(m.target_model)
    for _ in range(3):
        m.begin_behaviour_episode()
        for r in range(4):
            m.get_action(state, first=r == 0, edge_index=ei, greedy=False)
        m.end_behaviour_episode()
    assert _same(policy_before, _params(m.policy_model))
    assert _same(target_before, _params(m.target_model))


def test_the_acting_copy_shares_no_storage_with_anything_else():
    """`refresh` writes only into the perturbed copy's tensors, so what it can
    reach is decided by aliasing: no tensor of the acting copy may share
    storage with the policy net, the target net or any other module in the
    process (the opponent's real parameters are checked on the live stack by
    scripts/rl_param_noise/probe.py)."""
    m = _manager(PARAM_NOISE)
    stranger = _manager(None, seed=17).policy_model  # stands in for the opponent
    others = {
        p.data_ptr()
        for mod in (m.policy_model, m.target_model, stranger)
        for p in list(mod.parameters()) + list(mod.buffers())
    }
    acting = list(m.param_noise.perturbed.parameters())
    assert acting
    for p in acting + list(m.param_noise.perturbed.buffers()):
        assert p.data_ptr() not in others
    for p in acting:
        assert not p.requires_grad


def test_refresh_reloads_the_trained_weights():
    """The acting copy tracks training: after the policy moves, the next
    episode's perturbation is centred on the moved weights."""
    m = _manager({**PARAM_NOISE, "scale": 0.0})
    with th.no_grad():
        for p in m.policy_model.parameters():
            p.add_(1.0)
    m.begin_behaviour_episode()
    assert _same(_params(m.policy_model), _params(m.param_noise.perturbed))


# ── the divergence measure, and why it is ordinal ────────────────────


def _q_shifted(shift, n_states=64):
    """A near-deterministic Q whose argmax sits at 0, and the same Q with the
    argmax moved `shift` levels up."""
    base = th.full((n_states, N_PUNISHMENTS), -10.0)
    ref = base.clone()
    ref[:, 0] = 10.0
    pert = base.clone()
    pert[:, shift] = 10.0
    return ref, pert


def test_unweighted_divergence_is_blind_to_the_size_of_the_move():
    """Plappert's DQN distance is defined on an unweighted action set. At 31
    ordinal punishment levels a one-level slip and a thirty-level jump are
    the same number to it -- the measurement behind choosing an ordinal
    measure for the adaptation target."""
    from aimanager.manager.param_noise import (
        divergence_l2,
        divergence_mad,
        divergence_w1,
    )

    ref, near = _q_shifted(1)
    _, far = _q_shifted(30)
    assert divergence_l2(ref, far) == pytest.approx(divergence_l2(ref, near), rel=1e-3)
    assert divergence_mad(ref, near) == pytest.approx(1.0)
    assert divergence_mad(ref, far) == pytest.approx(30.0)
    assert divergence_w1(ref, far) > 20 * divergence_w1(ref, near)


def test_mad_is_in_punishment_levels():
    from aimanager.manager.param_noise import divergence_mad

    ref, pert = _q_shifted(7)
    assert divergence_mad(ref, pert) == pytest.approx(7.0)


# ── the adaptive scale ───────────────────────────────────────────────


def test_scale_grows_below_target_and_shrinks_above():
    from aimanager.manager.param_noise import ParameterNoise

    m = _manager(None)
    pn = ParameterNoise(m.policy_model, scale=0.1, target=1.5, adapt_coef=1.01)
    ref, near = _q_shifted(1)  # mad 1.0 < target
    pn.observe(ref, near)
    stats = pn.finish()
    assert stats["param_noise_scale"] == pytest.approx(0.1)
    assert stats["param_noise_divergence"] == pytest.approx(1.0)
    assert pn.scale > 0.1

    _, far = _q_shifted(30)  # mad 30.0 > target
    pn.observe(ref, far)
    pn.finish()
    assert pn.scale < 0.1 * 1.01


def test_zero_scale_never_adapts_off_zero():
    from aimanager.manager.param_noise import ParameterNoise

    m = _manager(None)
    pn = ParameterNoise(m.policy_model, scale=0.0, target=1.5)
    ref, near = _q_shifted(1)
    pn.observe(ref, near)
    pn.finish()
    assert pn.scale == 0.0


def test_finish_reports_all_three_measures():
    from aimanager.manager.param_noise import ParameterNoise

    m = _manager(None)
    pn = ParameterNoise(m.policy_model, scale=0.1, target=1.5)
    ref, near = _q_shifted(3)
    pn.observe(ref, near)
    stats = pn.finish()
    assert set(stats) == {
        "param_noise_scale",
        "param_noise_target",
        "param_noise_divergence",
        "param_noise_divergence_mad",
        "param_noise_divergence_l2",
        "param_noise_divergence_w1",
    }
    assert stats["param_noise_divergence_mad"] == pytest.approx(3.0)


# ── the epsilon-matched target ───────────────────────────────────────


def test_eps_greedy_displacement_is_exact():
    """Closed form against the definition: eps * E_u|u - a| for u uniform."""
    from aimanager.manager.param_noise import eps_greedy_displacement

    for a in (0, 5, 15, 30):
        q = th.full((1, N_PUNISHMENTS), -10.0)
        q[0, a] = 10.0
        want = 0.1 * sum(abs(u - a) for u in range(N_PUNISHMENTS)) / N_PUNISHMENTS
        assert eps_greedy_displacement(q, 0.1, N_PUNISHMENTS) == pytest.approx(want)


def test_eps_matched_target_follows_the_greedy_policy():
    """The displacement epsilon-greedy injects is not a constant: it grows as
    the policy sharpens toward 0. The target has to follow it, or the arm
    explores less than the reference exactly when the reference explores
    most."""
    from aimanager.manager.param_noise import ParameterNoise

    m = _manager(None)
    pn = ParameterNoise(m.policy_model, scale=0.1, target="eps_matched", eps=0.1)
    centre, _ = _q_shifted(0)  # argmax at 0 -> eps-greedy displaces by 1.5
    pn.observe(centre, centre)
    assert pn.finish()["param_noise_target"] == pytest.approx(1.5)

    mid = th.full((8, N_PUNISHMENTS), -10.0)
    mid[:, 15] = 10.0
    pn.observe(mid, mid)
    assert pn.finish()["param_noise_target"] == pytest.approx(240 / 31 * 0.1)


def test_eps_matched_target_rejects_a_non_ordinal_measure():
    from aimanager.manager.param_noise import ParameterNoise

    m = _manager(None)
    with pytest.raises(AssertionError):
        ParameterNoise(
            m.policy_model, scale=0.1, target="eps_matched", eps=0.1, measure="l2"
        )


def test_manager_hands_its_own_eps_to_the_matched_target():
    """One source of truth: the matched target must use the same epsilon the
    reference arm explores with, not a number restated in the block."""
    m = _manager({"scale": 0.1, "target": "eps_matched", "measure": "mad"})
    assert m.param_noise.eps == m.eps == 0.1


def test_bad_measure_is_rejected():
    from aimanager.manager.param_noise import ParameterNoise

    m = _manager(None)
    with pytest.raises(AssertionError):
        ParameterNoise(m.policy_model, scale=0.1, target=1.5, measure="kl")


# ── the policy-shape rows ────────────────────────────────────────────


def test_rpa_shape_matches_the_evaluation_suite_binning():
    """The shape rows must sit on the suite's own RPA bins, not on a
    re-invented copy, or the arms are not comparable."""
    import pandas as pd

    from aimanager.evaluation_suite.metrics import RPA_EDGES, RPA_LABELS
    from aimanager.rl_manager import rpa_shape

    g = th.Generator().manual_seed(5)
    c = th.randint(0, 21, (32, 8, 1), generator=g)
    p = th.randint(0, 31, (32, 8, 1), generator=g)
    valid = th.rand((32, 8, 1), generator=g) > 0.1
    groups = th.randint(0, 2, (32, 8), generator=g)
    recorded = {"contribution": c, "punishment": p, "contribution_valid": valid}

    out = rpa_shape(recorded, groups, 0)

    keep = (valid.squeeze(-1) & (groups == 0)).flatten().numpy()
    df = pd.DataFrame(
        {
            "contribution": c.squeeze(-1).flatten().numpy(),
            "punishment": p.squeeze(-1).flatten().numpy(),
        }
    )[keep]
    bins = pd.cut(df["contribution"], RPA_EDGES, labels=RPA_LABELS)
    ref = df.groupby(bins.astype(str))["punishment"].agg(["mean", "size"])
    for label in RPA_LABELS:
        assert out[f"rpa_n_{label}"] == pytest.approx(float(ref.loc[label, "size"]))
        assert out[f"rpa_mean_{label}"] == pytest.approx(float(ref.loc[label, "mean"]))


def test_rpa_shape_drops_timed_out_contributors():
    """An imputed contribution would land in the {0} bin and fake the very
    number the inverted-policy result turns on."""
    from aimanager.rl_manager import rpa_shape

    c = th.zeros((2, 4, 1), dtype=th.int64)
    p = th.full((2, 4, 1), 9, dtype=th.int64)
    valid = th.zeros((2, 4, 1), dtype=th.bool)
    out = rpa_shape(
        {"contribution": c, "punishment": p, "contribution_valid": valid}, None, None
    )
    assert out["rpa_n_{0}"] == 0
    assert out["rpa_mean_{0}"] != out["rpa_mean_{0}"]  # NaN, not a fabricated 9
