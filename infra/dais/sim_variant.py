"""Helpers for the DAIS sim checks; the sims themselves run through the repo's
own entry point, `python -m aimanager simulate <config>`.

    python infra/dais/sim_variant.py check-gpu
        torch, CUDA and PyG on the GPU (runs inside the sim image)
    python infra/dais/sim_variant.py config SWEEP_CONFIG OUT_CONFIG
        [--points N] [--episodes E] [--batch B] [--output-dir DIR]
        a copy of a policy-finder sweep config (as generate_sim_config.py
        writes it) with its rule-based point replicated N times and a few
        keys changed: the shape of a real sweep (N points x E episodes)
"""

import argparse

import yaml


def check_gpu():
    import torch as th
    import torch_geometric
    import torch_scatter

    print("torch", th.__version__, "cuda", th.version.cuda)
    print("arch list", th.cuda.get_arch_list())
    print("available", th.cuda.is_available())
    print("device", th.cuda.get_device_name(0), th.cuda.get_device_capability(0))
    x = th.randn(2048, 2048, device="cuda")
    print("matmul finite", bool(th.isfinite(x @ x).all()))
    src = th.arange(8.0, device="cuda")
    idx = th.tensor([0, 0, 1, 1, 2, 2, 3, 3], device="cuda")
    print("torch_scatter", torch_scatter.scatter_add(src, idx).tolist())
    print("torch_geometric", torch_geometric.__version__)


def write_config(a):
    with open(a.config) as f:
        config = yaml.safe_load(f)
    if a.points:
        rule = next(
            k for k, v in config["managers"].items() if v["type"] == "rule_based"
        )
        point = config["managers"][rule]
        config["managers"] = {"ah": config["managers"]["ah"]}
        config["pairings"] = []
        for i in range(a.points):
            name = f"p{i:03d}"
            config["managers"][name] = dict(point)
            config["pairings"].append(
                {"name": f"{name}_vs_ah", "group_0": name, "group_1": "ah"}
            )
    for key, value in (
        ("n_episodes", a.episodes),
        ("episode_batch_size", a.batch),
        ("output_dir", a.output_dir),
    ):
        if value is not None:
            config[key] = value
    with open(a.out, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-gpu")
    c = sub.add_parser("config")
    c.add_argument("config")
    c.add_argument("out")
    c.add_argument("--points", type=int)
    c.add_argument("--episodes", type=int)
    c.add_argument("--batch", type=int)
    c.add_argument("--output-dir")
    a = p.parse_args()
    check_gpu() if a.cmd == "check-gpu" else write_config(a)


if __name__ == "__main__":
    main()
