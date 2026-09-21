"""Tests for the GATED immediate-stimulus skip in `GraphNetwork`
(contributor-gated-skip): instead of handing `op2` both the post-`op1`
embedding and the post-RNN one side by side -- which weights stimulus against
memory at one fixed ratio for every round -- a per-agent, per-round scalar
gate read off the stimulus mixes them,

    x = g * x_skip + (1 - g) * x_rnn,   g = sigmoid(w . x_skip + b),

so the immediate stimulus can dominate on the rounds where something happened
to the player and the carried memory on the quiet ones.

Runs locally on macOS with plain pytest:
    PYTHONPATH=$PWD/src <venv python> -m pytest -q tests/skip/test_stimulus_gate.py

The PyG stand-ins, the model/data builders and the forward helpers are
imported from the sibling `test_stimulus_skip.py` rather than copied, so both
suites exercise the same construction under the same seed.

Context: notes/autoresearch_log/contributor-gated-skip.md.
"""

import os
import sys
import tempfile

import pytest
import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_stimulus_skip import (  # noqa: E402
    COPULA_PHI,
    COPULA_RHO,
    COPULA_SWITCH_EVERY,
    HIDDEN,
    N_AGENTS,
    N_LEVELS,
    SEED,
    VNODE_HIDDEN,
    _forward,
    _make_carry_data,
    _op2_in_features,
    _op2_weight,
    _slice_round,
    make_data,
    make_model,
)

from aimanager.generic.graph import GraphNetwork  # noqa: E402

GATED = dict(
    stimulus_skip=True,
    stimulus_gate=True,
    group_vnode=True,
    group_vnode_hidden=VNODE_HIDDEN,
)


# --------------------------------------------------------------------------- #
# (a) off changes nothing
# --------------------------------------------------------------------------- #
def test_gate_off_matches_the_flag_absent_bit_for_bit():
    """`stimulus_gate=False` passed explicitly and the kwarg absent are the
    same model, with and without the ungated skip: construction draws no RNG
    with the gate off, so every artifact trained before this change is
    reproduced weight for weight."""
    for extra in ({}, {"stimulus_skip": True}):
        absent = make_model(**extra)
        off = make_model(stimulus_gate=False, **extra)
        assert absent.stimulus_gate is False
        assert off.stimulus_gate is False
        assert absent.stimulus_gate_module is None
        absent_state = absent.state_dict()
        off_state = off.state_dict()
        assert set(absent_state) == set(off_state)
        assert not [k for k in absent_state if "stimulus_gate" in k]
        for key, value in absent_state.items():
            assert th.equal(value, off_state[key]), key
        data = make_data(n_batch=2, round_number=3)
        assert th.equal(_forward(absent, data), _forward(off, data))


# --------------------------------------------------------------------------- #
# (b) on: the width, the module, the gradient
# --------------------------------------------------------------------------- #
def test_the_gate_replaces_the_widening_rather_than_adding_to_it():
    """The gate MIXES, so op2 goes back to the un-skipped width: the ungated
    skip's extra `hidden_size` columns are gone and the group state keeps
    exactly the slice it has today. The gate itself is one scalar readout of
    the post-op1 embedding."""
    plain = make_model(group_vnode=True, group_vnode_hidden=VNODE_HIDDEN)
    ungated = make_model(
        stimulus_skip=True, group_vnode=True, group_vnode_hidden=VNODE_HIDDEN
    )
    gated = make_model(**GATED)

    assert _op2_in_features(plain) == HIDDEN + VNODE_HIDDEN
    assert _op2_in_features(ungated) == HIDDEN + VNODE_HIDDEN + HIDDEN
    assert _op2_in_features(gated) == HIDDEN + VNODE_HIDDEN
    assert _op2_weight(gated).shape[1] == _op2_in_features(gated)

    assert gated.stimulus_gate_module.weight.shape == (1, HIDDEN)
    assert gated.stimulus_gate_module.bias.shape == (1,)

    # everything built before the gate is untouched under the same seed --
    # the gate is constructed LAST, after the virtual node
    plain_state = plain.state_dict()
    gated_state = gated.state_dict()
    for prefix in ("op1.", "rnn_n.", "op2.", "group_vnode_module."):
        matched = [k for k in plain_state if k.startswith(prefix)]
        assert matched, f"no parameters matched prefix {prefix!r}"
        for key in matched:
            assert th.equal(plain_state[key], gated_state[key]), key
    assert set(gated_state) - set(plain_state) == {
        "stimulus_gate_module.weight",
        "stimulus_gate_module.bias",
    }


def test_op2_receives_exactly_the_gated_mix():
    """What `op2` actually sees is `g * x_skip + (1 - g) * x_rnn` on the
    per-agent slice, with `g = sigmoid(gate(x_skip))` broadcast over the
    channels -- recomputed here from op1 and the GRU and compared against a
    forward pre-hook capture, so the mix is checked and not assumed."""
    model = make_model(**GATED)
    assert model.u_encoder.size == 0
    assert model.op2.edge_model is None

    data = make_data(n_batch=2, round_number=3)
    model.eval()
    encoded = model.encode(data, y_encode=False, device="cpu")

    captured = {}

    def hook(_module, args):
        captured["input"] = args[0]

    handle = model.op2.node_model.node_mlp.register_forward_pre_hook(hook)
    try:
        model(encoded)
    finally:
        handle.remove()

    post_op1, _, _ = model.op1(
        encoded["x"],
        encoded["edge_index"],
        encoded["edge_attr"],
        encoded["u"],
        encoded["batch"],
    )
    post_rnn, _ = model.rnn_n(post_op1, None)
    gate = th.sigmoid(model.stimulus_gate_module(post_op1))
    assert gate.shape == post_op1.shape[:-1] + (1,)
    mixed = gate * post_op1 + (1.0 - gate) * post_rnn

    op2_input = captured["input"]
    assert op2_input.shape[-1] == HIDDEN + VNODE_HIDDEN
    assert th.allclose(op2_input[..., :HIDDEN], mixed, atol=1e-6)
    # the mix is neither endpoint: a gate stuck at 0 or 1 would be untestable
    assert not th.allclose(op2_input[..., :HIDDEN], post_rnn, atol=1e-6)
    assert not th.allclose(op2_input[..., :HIDDEN], post_op1, atol=1e-6)
    assert 0.0 < float(gate.min()) and float(gate.max()) < 1.0


def test_gradient_reaches_the_gate():
    """Gradient flows into the gate's own weight and bias, and still into op1
    and op2 -- a gate that never receives gradient would train as a constant
    mix, which is the thing this change exists to replace."""
    model = make_model(**GATED)
    model.train()
    data = make_data(n_batch=2, n_rounds=3, round_number=0)
    logit = model(model.encode(data, y_encode=False, device="cpu"))

    th.manual_seed(SEED + 2)
    target = th.randint(0, N_LEVELS, logit.shape[:-1]).reshape(-1)
    th.nn.functional.cross_entropy(logit.reshape(-1, N_LEVELS), target).backward()

    for name, param in model.stimulus_gate_module.named_parameters():
        assert param.grad is not None, name
        assert th.isfinite(param.grad).all(), name
        assert param.grad.abs().max() > 0, name
    assert _op2_weight(model).grad.abs().max() > 0
    assert model.op1.node_model.node_mlp[0].weight.grad.abs().max() > 0


def test_forward_differs_from_both_the_ungated_skip_and_no_skip():
    """Under the same seed the gated model must differ observably from the
    plain trunk and from the ungated concatenation."""
    plain = make_model(group_vnode=True, group_vnode_hidden=VNODE_HIDDEN)
    ungated = make_model(
        stimulus_skip=True, group_vnode=True, group_vnode_hidden=VNODE_HIDDEN
    )
    gated = make_model(**GATED)
    data = make_data(n_batch=2, round_number=5)
    logit_plain = _forward(plain, data)
    logit_ungated = _forward(ungated, data)
    logit_gated = _forward(gated, data)
    assert logit_gated.shape == logit_plain.shape == logit_ungated.shape
    assert not th.equal(logit_gated, logit_plain)
    assert not th.equal(logit_gated, logit_ungated)


# --------------------------------------------------------------------------- #
# (c) the simulation's calling convention
# --------------------------------------------------------------------------- #
def test_per_round_calls_reproduce_the_single_call():
    """24 single-round `predict_independent` calls with `reset_rnn` only at
    round 0 reproduce one 24-round call -- the property `simulate.py` depends
    on. The gate is a pointwise function of the round's own embedding, so it
    must not introduce any cross-round coupling of its own."""
    n_batch, n_rounds = 2, 24
    kwargs = dict(
        **GATED,
        copula_rho=COPULA_RHO,
        copula_phi=COPULA_PHI,
        copula_switch_every=COPULA_SWITCH_EVERY,
    )
    model_full = make_model(**kwargs)
    model_chunks = make_model(**kwargs)
    for key, value in model_full.state_dict().items():
        assert th.equal(value, model_chunks.state_dict()[key]), key

    data = _make_carry_data(n_batch, n_rounds)
    edge_index = model_full.create_fully_connected(N_AGENTS, n_batch=n_batch)

    th.manual_seed(SEED + 1)
    pred_full, proba_full = model_full.predict_independent(
        data, sample=True, reset_rnn=True, edge_index=edge_index
    )

    th.manual_seed(SEED + 1)
    pred_chunks = th.empty_like(pred_full)
    proba_chunks = th.empty_like(proba_full)
    for r in range(n_rounds):
        p, pr = model_chunks.predict_independent(
            _slice_round(data, r),
            sample=True,
            reset_rnn=(r == 0),
            edge_index=edge_index,
        )
        pred_chunks[:, :, r] = p[:, :, 0]
        proba_chunks[:, :, r] = pr[:, :, 0]

    proba_diff = (proba_chunks - proba_full).abs().max().item()
    assert proba_diff < 1e-6, proba_diff
    assert th.equal(pred_chunks, pred_full)


# --------------------------------------------------------------------------- #
# (d) save / load round-trip
# --------------------------------------------------------------------------- #
def test_save_load_round_trips_the_gate():
    model = make_model(**GATED)
    data = make_data(n_batch=2, round_number=3)
    ref_logit = _forward(model, data)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        saved = th.load(path, map_location="cpu")
        assert saved["stimulus_gate"] is True
        assert saved["stimulus_gate_module"] is not None
        loaded = GraphNetwork.load(path, device="cpu")

    assert loaded.stimulus_gate is True
    assert _op2_in_features(loaded) == _op2_in_features(model)
    ref_state = model.state_dict()
    loaded_state = loaded.state_dict()
    assert set(ref_state) == set(loaded_state)
    for key, value in ref_state.items():
        assert th.equal(value, loaded_state[key]), key
    assert th.equal(_forward(loaded, data), ref_logit)


def test_artifact_without_the_gate_keys_loads_with_the_gate_absent():
    """The frontier's own artifact carries `stimulus_skip` but no gate keys.
    It must keep loading as the ungated concatenation it was trained as."""
    model = make_model(
        stimulus_skip=True, group_vnode=True, group_vnode_hidden=VNODE_HIDDEN
    )
    data = make_data(n_batch=2, round_number=3)
    ref_logit = _forward(model, data)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        model.save(path)
        saved = th.load(path, map_location="cpu")
        del saved["stimulus_gate"]
        del saved["stimulus_gate_module"]
        legacy_path = os.path.join(d, "legacy.pt")
        th.save(saved, legacy_path)
        legacy = GraphNetwork.load(legacy_path, device="cpu")

    assert legacy.stimulus_gate is False
    assert legacy.stimulus_skip is True
    assert _op2_in_features(legacy) == HIDDEN + VNODE_HIDDEN + HIDDEN
    assert th.equal(_forward(legacy, data), ref_logit)


# --------------------------------------------------------------------------- #
# (e) the guards
# --------------------------------------------------------------------------- #
def test_gate_without_the_skip_raises():
    with pytest.raises(AssertionError, match="stimulus_gate requires stimulus_skip"):
        make_model(stimulus_gate=True)


def test_non_bool_stimulus_gate_raises():
    with pytest.raises(AssertionError, match="stimulus_gate must be a bool"):
        make_model(stimulus_skip=True, stimulus_gate=1)
