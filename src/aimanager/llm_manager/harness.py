"""Run a manager in the paired competing setting and score it.

The setting is the one #209 established and #217 and #219 used: the manager
under test holds group 0, the behavioural clone holds group 1, and the eight
players move between the two every fourth round. It is not self-play,
because self-play rankings do not survive competition in this project -- a
manager can look strong against a copy of itself and lose its members the
moment there is somewhere else to go.

The baselines run **in this same invocation**, against the same rival, on
the same stack, from the same seed schedule: the clone, `thr9_p10`,
never-punish, and the capped sigmoid rule of PR #219. Quoting them from
another arm's table would compare across two different RNG streams and two
different artifact sets, which is the mistake #217 avoided by re-running
`thr9_p10` rather than citing #209's number for it.

`CAPPED_SIGMOID` is the rule whose `P_max` is capped, not the one whose
realised severity is. The two filters exist in #219 and select different
rules; the ceiling is the one that binds, because it caps the largest
punishment the rule can ever issue and so keeps every decision inside the
contribution model's evidence, whereas a severity constraint leaves the
heavy early rounds in.

Everything here is CPU-bound -- a 192-episode rollout is about 3 s on four
CPU threads (#219) -- which matters because the language model will want
the GPU.
"""

import time

import numpy as np
import pandas as pd
import torch as th

from aimanager.llm_manager import battery as bat
from aimanager.llm_manager.lfs import assert_real_file
from aimanager.llm_manager.stub import StubManager, collect_telemetry, reset_telemetry
from aimanager.manager.paired_rollout import contingency, make_env, rollout
from aimanager.manager.sigmoid_rule import ConstantManager, SigmoidRuleBatch

#: The frontier stack, identical to #217's and #219's
#: (configs/simulation/manager_testing/26_rule_inverted_targeting_s42.yml), so
#: a number measured here is comparable with their published tables.
DEFAULT_STACK = {
    "contribution_model": (
        "artifacts/artificial_humans/"
        "group_switching_contribution_50ep_vnode_stimulus_skip_herding_copula/"
        "model/architecture_node+edge+rnn__dataset_50ep__epochs_575.pt"
    ),
    "valid_model": (
        "artifacts/artificial_humans/raven_script_22/model/"
        "rnn_False__dataset_full.pt"
    ),
    "switch_model": (
        "artifacts/artificial_humans/switch_joint_exodus/model/"
        "architecture_mlp+rnn+edge__dataset_50ep_doubled.pt"
    ),
    "clone": (
        "artifacts/baselines/punishment_multinomial_timeout_severity_copula.joblib"
    ),
}

#: `thr9_p10` as a member of the sigmoid family: `tau -> 0` is a hard step,
#: and 1e-6 saturates the logistic in float32 at every integer contribution,
#: so this row IS `RuleBasedManager(rule="threshold", threshold=9,
#: amount=10)` (pinned cell by cell in `test_sigmoid_rule.py`).
THR9_P10 = dict(p_max=10.0, c0=9.5, tau=1e-6, gamma_ep=0.0, gamma_sw=0.0)

#: #219's `best_cap10_pool`: the best rule in the design that can never
#: issue a punishment above 10. It beat `thr9_p10` by +5.00 on the pool at a
#: third of its spend.
CAPPED_SIGMOID = dict(
    p_max=9.338554367423058,
    c0=9.030428379774094,
    tau=0.4756534294965072,
    gamma_ep=1.0905728470534086,
    gamma_sw=2.342363149859011,
)


def load_models(stack=None, device="cpu"):
    """Load the four artifacts. Needs torch_geometric, so: Raven."""
    import joblib

    from aimanager.artificial_humans import GraphNetwork

    stack = {**DEFAULT_STACK, **(stack or {})}
    for k, p in stack.items():
        assert_real_file(p, k)
    device = th.device(device)
    models = {
        k: GraphNetwork.load(stack[k], device=device).to(device)
        for k in ("contribution_model", "valid_model", "switch_model")
    }
    models["clone_bundle"] = joblib.load(stack["clone"])
    models["_stack"] = stack
    return models


def _sigmoid(params, batch_size, device):
    theta = th.tensor(
        [
            [
                params["p_max"],
                params["c0"],
                params["tau"],
                params["gamma_ep"],
                params["gamma_sw"],
            ]
        ],
        dtype=th.float,
    ).repeat(batch_size, 1)
    return SigmoidRuleBatch(theta).to(device)


class ApiManagerSeat:
    """Any `get_punishments(data)` manager behind the `predict(state)` seat.

    `api_manager.RuleBasedManager` is the implementation #217 ran, so the
    inverted mirrors it published are re-run here through that code rather
    than through a second copy of the same arithmetic. The served state
    carries every key it reads (`contribution`, `round_number`,
    `contribution_valid`, and `punishment` for the dtype).
    """

    autoregressive = False

    def __init__(self, manager):
        self.manager = manager
        self.default_values = manager.default_values

    def to(self, device):
        return self

    def predict(self, state, **_):
        return self.manager.get_punishments(state).to(th.int64), None


def build_manager(spec, batch_size, models, device):
    """A seat from a name or a `{"kind": ...}` spec.

    An object that already exposes `predict(state) -> (p, None)` is passed
    straight through, which is how the language-model manager arrives.
    """
    if hasattr(spec, "predict"):
        return spec
    if isinstance(spec, str):
        spec = {"kind": spec}
    spec = dict(spec)
    kind = spec.pop("kind")
    if kind == "clone":
        from aimanager.manager.linear_opponent import LinearPunisherOpponent

        return LinearPunisherOpponent(models["clone_bundle"], device=device)
    if kind == "never":
        return ConstantManager(0)
    if kind == "constant":
        return ConstantManager(int(spec.get("amount", 0)))
    if kind == "thr9_p10":
        return _sigmoid(THR9_P10, batch_size, device)
    if kind == "capped_sigmoid":
        return _sigmoid(CAPPED_SIGMOID, batch_size, device)
    if kind == "sigmoid":
        return _sigmoid(spec, batch_size, device)
    if kind == "stub":
        return StubManager(**spec)
    if kind == "rule":
        # imported lazily: api_manager pulls in torch_geometric
        from aimanager.manager.api_manager import RuleBasedManager

        return ApiManagerSeat(RuleBasedManager(**spec))
    raise ValueError(f"unknown manager spec {kind!r}")


def _as_seeds(seeds):
    return (int(seeds),) if isinstance(seeds, (int, np.integer)) else tuple(seeds)


def run_arm(
    name,
    focal_spec,
    rival_spec,
    models,
    *,
    episodes=200,
    chunk=200,
    seeds=42,
    device="cpu",
    n_rounds=24,
    switch_every=4,
):
    """One arm: `episodes` episodes per seed of focal-against-rival, scored.

    Episodes are split into `ceil(episodes / chunk)` batched rollouts, each
    reseeded from `(seed, rep)` only -- not from the arm -- so every arm in
    a battery faces the same initial draw and the streams only diverge once
    the managers actually act differently.

    Seeds are the unit of replication, as they are throughout this project:
    `seeds=(42, 43, 44)` at `episodes=100` is #217's 300-episode budget,
    pooled here into one frame whose rows are still independent episodes.

    Returns the per-episode frame, the (21, 31) contingency table of the
    focal seat's valid decisions, the arm's telemetry and the wall clock.
    """
    device = th.device(device)
    n_reps = int(np.ceil(episodes / chunk))
    frames, counts, wall = [], None, 0.0
    reset_telemetry(focal_spec)
    for seed in _as_seeds(seeds):
        for rep in range(n_reps):
            b = min(chunk, episodes - rep * chunk)
            env = make_env(
                contribution_model=models["contribution_model"],
                valid_model=models["valid_model"],
                switch_model=models["switch_model"],
                batch_size=b,
                device=device,
                n_rounds=n_rounds,
                switch_every=switch_every,
            )
            focal = build_manager(focal_spec, b, models, device)
            rival = build_manager(rival_spec, b, models, device)
            th.manual_seed(seed * 1_000_003 + rep)
            np.random.seed((seed * 7919 + rep) % (2**31))
            t0 = time.perf_counter()
            rec = rollout(env, focal, rival)
            wall += time.perf_counter() - t0
            f = bat.episode_frame(rec, switch_every=switch_every)
            f.insert(0, "seed", seed)
            f.insert(1, "rep", rep)
            frames.append(f)
            c = contingency(rec, th.zeros(b, dtype=th.int64), 1)[0].numpy()
            counts = c if counts is None else counts + c
            del env
            if device.type == "cuda":
                th.cuda.empty_cache()
    ep = pd.concat(frames, ignore_index=True)
    ep["episode"] = np.arange(len(ep), dtype=np.int64)
    ep.insert(0, "arm", name)
    # who the other seat held, carried on every row: a contrast between two
    # arms that did not face the same rival is not a contrast between their
    # managers, and `battery.contrasts` uses this to refuse one.
    ep.insert(1, "rival", spec_name(rival_spec))
    tel = collect_telemetry(focal_spec)
    return ep, counts, tel, wall


def spec_name(spec):
    """A short, stable name for whatever was put in a seat."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return str(spec.get("kind", "spec"))
    return type(spec).__name__


def run_battery(
    arms,
    models,
    *,
    rival="clone",
    episodes=200,
    chunk=200,
    seeds=42,
    device="cpu",
    verbose=True,
):
    """Every arm through `run_arm`, then one battery row each.

    `arms` is `{name: spec}`. `rival` is shared unless an arm's spec is a
    `(focal, rival)` pair, which is how the symmetric controls are asked
    for: `{"clone_vs_clone": ("clone", "clone")}`.
    """
    rows, episode_frames, shape = [], [], {}
    for name, spec in arms.items():
        focal_spec, rival_spec = spec if isinstance(spec, tuple) else (spec, rival)
        ep, counts, tel, wall = run_arm(
            name,
            focal_spec,
            rival_spec,
            models,
            episodes=episodes,
            chunk=chunk,
            seeds=seeds,
            device=device,
        )
        rows.append(bat.battery_row(name, ep, counts, tel, wall))
        rows[-1]["rival"] = spec_name(rival_spec)
        episode_frames.append(ep)
        shape[name] = counts
        if verbose:
            r = rows[-1]
            print(
                f"{name:>18}  pool {r['focal_pool']:7.2f}  "
                f"contr {r['focal_contribution']:7.2f}  "
                f"members {r['focal_members']:5.2f}  "
                f"p {r['focal_mean_punishment']:5.2f}  "
                f"rho {r['rho']:6.3f}  {wall:6.1f}s",
                flush=True,
            )
    return {
        "battery": pd.DataFrame(rows),
        "episodes": pd.concat(episode_frames, ignore_index=True),
        "contingency": shape,
    }


def contingency_frame(shape):
    """`{arm: (21, 31) counts}` -> a long frame of the non-zero cells."""
    out = []
    for name, cnt in shape.items():
        ci, pi = np.nonzero(cnt)
        out.append(
            pd.DataFrame(
                {
                    "arm": name,
                    "contribution": ci.astype(np.int16),
                    "punishment": pi.astype(np.int16),
                    "count": cnt[ci, pi].astype(np.int64),
                }
            )
        )
    return pd.concat(out, ignore_index=True)
