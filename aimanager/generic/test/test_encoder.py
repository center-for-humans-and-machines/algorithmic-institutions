import torch as th
import pytest
from aimanager.generic.encoder import IntEncoder, FloatEncoder, BoolEncoder, Encoder


@pytest.fixture
def example_state_results():
    state = {
        "int_tensor": th.tensor([1, 2, 3], dtype=th.int64),
        "float_tensor": th.tensor([1.0, 2.0, 3.0]),
        "bool_tensor": th.tensor([True, False, True]),
        "batch": th.tensor([0, 1, 0]),
    }
    results = {
        "int_tensor": {
            "ordinal": th.tensor(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]],
                dtype=th.float,
            ),
            "onehot": th.tensor(
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                ],
                dtype=th.float,
            ),
            "numeric": th.tensor([[1.0], [2.0], [3.0]], dtype=th.float) / 4,
            "projection": th.tensor(
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                dtype=th.float,
            ),
        },
        "float_tensor": th.tensor([[1.0], [2.0], [3.0]], dtype=th.float) / 3,
        "bool_tensor": th.tensor([[1.0], [0.0], [1.0]], dtype=th.float),
    }
    return state, results


def test_int_encoder(example_state_results):
    example_state, results = example_state_results
    encoder = IntEncoder(encoding="ordinal", name="int_tensor", n_levels=5)
    output = encoder(**example_state)
    expected_output = results["int_tensor"]["ordinal"]
    assert th.allclose(output, expected_output, rtol=1e-03, atol=1e-03)

    encoder = IntEncoder(encoding="onehot", name="int_tensor", n_levels=5)
    output = encoder(**example_state)
    expected_output = results["int_tensor"]["onehot"]
    assert th.allclose(output, expected_output, rtol=1e-03, atol=1e-03)

    encoder = IntEncoder(encoding="numeric", name="int_tensor", n_levels=5)
    output = encoder(**example_state)
    expected_output = results["int_tensor"]["numeric"]
    assert th.allclose(output, expected_output, rtol=1e-03, atol=1e-03)


def test_float_encoder(example_state_results):
    example_state, results = example_state_results
    encoder = FloatEncoder(norm=3.0, name="float_tensor")
    output = encoder(**example_state)
    expected_output = results["float_tensor"]
    assert th.allclose(output, expected_output, rtol=1e-03, atol=1e-03)


def test_bool_encoder(example_state_results):
    example_state, results = example_state_results
    encoder = BoolEncoder(name="bool_tensor")
    output = encoder(**example_state)
    expected_output = results["bool_tensor"]
    assert th.allclose(output, expected_output, rtol=1e-03, atol=1e-03)


def test_encoder(example_state_results):
    example_state, results = example_state_results
    encodings = [
        {"encoding": "ordinal", "name": "int_tensor", "n_levels": 5, "etype": "int"},
        {"encoding": "numeric", "name": "int_tensor", "n_levels": 5, "etype": "int"},
        {"encoding": "onehot", "name": "int_tensor", "n_levels": 5, "etype": "int"},
        {"name": "bool_tensor", "etype": "bool"},
        {"name": "float_tensor", "etype": "float", "norm": 3.0},
    ]
    encoder = Encoder(encodings, aggregation="mean", keepdim=False)
    # add two more dimensions to state
    _example_state = {k: v.unsqueeze(0).unsqueeze(0) for k, v in example_state.items()}
    output = encoder(**_example_state)
    _output = output.squeeze(0).squeeze(0)
    expected_output = th.cat(
        [
            results["int_tensor"]["ordinal"],
            results["int_tensor"]["numeric"],
            results["int_tensor"]["onehot"],
            results["bool_tensor"],
            results["float_tensor"],
        ],
        dim=1,
    )
    assert th.allclose(_output, expected_output, rtol=1e-03, atol=1e-03)
