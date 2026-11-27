"""The AH and punisher models' forward pass on fixed inputs, to compare two
images (torch 1.11 and torch 2.5) on the CPU, where they should agree to
numerical precision (docs/reports/dais_gpu_sim_equivalence.md).

    python infra/dais/gpusim_forward.py capture CONFIG OUT.pkl [--episodes 64]
        plays a small batched sim of CONFIG (its first pairing) on the CPU
        and records every input the models see: the env state given to each
        AH model's predict, and the feature matrix given to each linear
        bundle's scaler. Plain numpy arrays, so either image can read it.
    python infra/dais/gpusim_forward.py replay CONFIG CAPTURE.pkl OUT.npz
        [--device cpu]
        loads CONFIG's models and recomputes, call by call in the captured
        order (the RNN state carries over as in the sim), each AH model's
        class probabilities (predict with sample=False) and each linear
        bundle's predict_proba. Deterministic: no sampling.
    python infra/dais/gpusim_forward.py compare A.npz B.npz
        largest absolute difference per output, and how often the most
        likely class differs. Exits 1 if any output is not finite.
"""

import argparse
import pickle
import sys

import numpy as np
import torch as th
import yaml


def _models(config, device):
    """CONFIG's AH models by role, on `device`."""
    from aimanager.simulation.simulate import load_humans

    humans = next(iter(config["artificial_humans"]))
    ah, ah_val, ah_switch = load_humans(config, humans, device)
    models = {"contribution": ah, "valid": ah_val, "switch": ah_switch}
    return {k: v for k, v in models.items() if v is not None}


def capture(a):
    from aimanager.simulation import simulate

    with open(a.config) as f:
        config = yaml.safe_load(f)
    config["pairings"] = config["pairings"][:1]
    config.update(n_episodes=a.episodes, episode_batch_size=a.episodes, seed=a.seed)
    th.manual_seed(a.seed)
    np.random.seed(a.seed)
    record = {"gnn": [], "linear": []}

    def tap_model(role, model):
        original = model.predict

        def predict(state, **kwargs):
            record["gnn"].append(
                {
                    "role": role,
                    # copies, not views: the env updates its state in place
                    "state": {
                        k: v.detach().cpu().numpy().copy()
                        for k, v in state.items()
                        if isinstance(v, th.Tensor)
                    },
                    "reset_rnn": bool(kwargs.get("reset_rnn", True)),
                    "edge_index": kwargs["edge_index"].cpu().numpy().copy(),
                }
            )
            return original(state, **kwargs)

        model.predict = predict
        return model

    load_humans = simulate.load_humans

    def tapped_humans(config, humans, device):
        ah, val, switch = load_humans(config, humans, device)
        roles = (("contribution", ah), ("valid", val), ("switch", switch))
        return tuple(m if m is None else tap_model(r, m) for r, m in roles)

    simulate.load_humans = tapped_humans

    import joblib

    load = joblib.load

    def tapped_load(path, *args, **kwargs):
        bundle = load(path, *args, **kwargs)
        if isinstance(bundle, dict) and "scaler" in bundle:
            scaler, name = bundle["scaler"], str(path)

            def transform(X, _orig=scaler.transform):
                record["linear"].append({"path": name, "X": np.array(X, copy=True)})
                return _orig(X)

            scaler.transform = transform
        return bundle

    # every loader imports joblib inside the function, so the tap reaches it
    joblib.load = tapped_load
    simulate.run_simulation(config, output_dir=None)
    with open(a.out, "wb") as f:
        pickle.dump(record, f, protocol=4)
    print(
        f"captured {len(record['gnn'])} AH calls,"
        f" {len(record['linear'])} linear calls -> {a.out}"
    )


def replay(a):
    with open(a.config) as f:
        config = yaml.safe_load(f)
    with open(a.capture, "rb") as f:
        record = pickle.load(f)
    import joblib

    device = th.device(a.device)
    models = _models(config, device)
    # the bundles under the paths the sim loaded them from (captured)
    bundles = {p: joblib.load(p) for p in {c["path"] for c in record["linear"]}}
    out = {}
    counts = {}
    with th.no_grad():
        for call in record["gnn"]:
            model = models[call["role"]]
            state = {k: th.from_numpy(v).to(device) for k, v in call["state"].items()}
            _, proba = model.predict(
                state,
                sample=False,
                reset_rnn=call["reset_rnn"],
                edge_index=th.from_numpy(call["edge_index"]).to(device),
            )
            i = counts[call["role"]] = counts.get(call["role"], -1) + 1
            out[f"{call['role']}/{i:03d}"] = proba.cpu().numpy().astype(np.float64)
    for i, call in enumerate(record["linear"]):
        b = bundles[call["path"]]
        out[f"linear/{i:03d}"] = b["estimator"].predict_proba(
            b["scaler"].transform(call["X"])
        )
    np.savez_compressed(a.out, **out)
    import sklearn

    print(
        "torch",
        th.__version__,
        "sklearn",
        sklearn.__version__,
        "replayed",
        {k: v + 1 for k, v in counts.items()},
        "AH calls,",
        len(record["linear"]),
        "linear calls ->",
        a.out,
    )


def compare(a):
    x, y = np.load(a.a), np.load(a.b)
    if sorted(x.files) != sorted(y.files):
        sys.exit("the two replays hold different outputs")
    ok = True
    for role in sorted({k.split("/")[0] for k in x.files}):
        keys = [k for k in x.files if k.startswith(role + "/")]
        p = np.concatenate([x[k].reshape(-1, x[k].shape[-1]) for k in keys])
        q = np.concatenate([y[k].reshape(-1, y[k].shape[-1]) for k in keys])
        finite = bool(np.isfinite(p).all() and np.isfinite(q).all())
        ok &= finite
        diff = np.abs(p - q)
        flips = float((p.argmax(-1) != q.argmax(-1)).mean())
        print(
            f"{role}: calls {len(keys)} rows {len(p)} classes {p.shape[-1]}"
            f" finite {finite} max_abs_diff {diff.max():.3e}"
            f" mean_abs_diff {diff.mean():.3e} argmax_differs {flips:.5f}"
        )
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("config")
    c.add_argument("out")
    c.add_argument("--episodes", type=int, default=64)
    c.add_argument("--seed", type=int, default=7)
    r = sub.add_parser("replay")
    r.add_argument("config")
    r.add_argument("capture")
    r.add_argument("out")
    r.add_argument("--device", default="cpu")
    m = sub.add_parser("compare")
    m.add_argument("a")
    m.add_argument("b")
    a = p.parse_args()
    {"capture": capture, "replay": replay, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
