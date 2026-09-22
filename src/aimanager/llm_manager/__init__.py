"""Evaluating a manager that is expensive to call.

`battery` is the measurement: pure torch/numpy/pandas over a recorded
rollout, no torch_geometric, so it runs and is tested locally. `harness`
drives the paired competing setting and needs the artificial humans, so it
only runs where PyG does. `stub` is the manager this was all built against.
"""
