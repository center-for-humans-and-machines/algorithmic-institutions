"""Intervention probe — multi-scenario.

Per scenario × episode × n_seeds, runs the AH stack twice (baseline,
treatment) and records mean metrics over the targeted agents at rounds
t* and t*+1: pun_t, pun_t1, contrib_t, contrib_t1, switch_t1. Aggregated
mean ± std is written to ``<output_dir>/scenarios.csv``.

Two targets:
- ``individual`` — overrides one focal agent (selected per-episode by
  ``agent_selector``); metrics are the focal's values.
- ``group`` — overrides all agents in the team; metrics are averaged
  across agents. Episodes can be auto-picked by ``group_selector`` +
  ``n_groups`` (lowest/highest/random by full-game team contribution).

Manifests support an explicit ``scenarios:`` list and/or a compact
``grids:`` form whose parameter cross-product is expanded at load time.
Treatment uses ``new_value`` (absolute) or ``factor`` (scales pilot's
round-t* value, clamped to the encoder level count).
"""

import argparse
import itertools
import math
import os

import pandas as pd
import torch as th
import yaml

from aimanager.artificial_humans import GraphNetwork
from aimanager.generic.data import create_torch_data
from aimanager.simulation.counterfactual import (
    _select_focal_agents,
    _select_focal_groups,
    intervention_value,
)


def _mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / max(n - 1, 1)
    return m, math.sqrt(var)


TRACE = False


def _trace(msg):
    if TRACE:
        print(msg, flush=True)


def _run_seed(
    ep, agents_idx, t_star, data, models, device, intervention=None, label=""
):
    """One stochastic draw → mean metrics over ``agents_idx`` at rounds t* and t*+1.

    ``agents_idx`` is a 1-D int64 tensor of agent indices to intervene on
    and average over. Pass a single index for individual mode, all
    agents for group mode.

    Real-data backfill design (unchanged across modes):
    - Round 0..t* values come from pilot data verbatim. Treatment
      overrides each agent in ``agents_idx`` at t* to either
      ``new_value`` or ``factor × pilot[agent, t*]``.
    - Override lands as ``prev_<feature>[t*+1, agent]`` so the AH at
      t*+1 reads the perturbation. Non-targeted prev_* stays at the
      data's natural shift values.
    - Round-t*+1 outputs come from a single AH-stack forward.

    When module-level ``TRACE`` is enabled (via ``--trace``) every step
    is logged so a small run can be eyeballed against the design.
    """
    contrib_ah, valid_ah, switch_ah, pun_ah = models
    agents_list = agents_idx.tolist()
    _trace(
        f"  [trace {label}] _run_seed begin: ep={ep} agents={agents_list} "
        f"t*={t_star}"
    )

    full = {
        k: t[ep : ep + 1, :, : t_star + 2].clone().to(device) for k, t in data.items()
    }
    _trace(
        f"  [trace {label}] STEP 1 prefix slice [0..t*+1] from pilot tensor; "
        f"contribution shape={tuple(full['contribution'].shape)}"
    )

    if intervention is not None:
        f = intervention["feature"]
        max_val = (
            pun_ah.y_levels - 1 if f == "punishment" else contrib_ah.y_levels - 1
        )
        for a in agents_list:
            pilot_ref = int(data[f][ep, a, t_star].item())
            natural_prev = int(full[f"prev_{f}"][0, a, t_star + 1].item())
            v = intervention_value(intervention, pilot_ref, max_value=max_val)
            full[f"prev_{f}"][0, a, t_star + 1] = v
            full[f"prev_{f}_valid"][0, a, t_star + 1] = True
            _trace(
                f"  [trace {label}] STEP 3 override agent={a}: "
                f"pilot_ref={pilot_ref}  resolved={v}  "
                f"prev_{f}[t*+1] {natural_prev} -> {v}"
            )
        prev_all = [int(v) for v in full[f"prev_{f}"][0, :, t_star + 1].tolist()]
        _trace(
            f"  [trace {label}] STEP 3 prev_{f}[t*+1] all-agents post-override: "
            f"{prev_all}"
        )
    else:
        _trace(f"  [trace {label}] STEP 3 baseline: no override applied")

    contrib_pred, _ = contrib_ah.predict_independent(full, sample=True, reset_rnn=True)
    valid_pred, _ = valid_ah.predict_independent(full, sample=True, reset_rnn=True)
    switch_pred, _ = switch_ah.predict_independent(full, sample=True, reset_rnn=True)
    _trace(
        f"  [trace {label}] STEP 4 AH stack forward (contrib, valid, switch) "
        f"with reset_rnn=True over rounds [0..{t_star + 1}]"
    )

    default_c = int(contrib_ah.default_values["contribution"])
    ah_contrib_t1 = contrib_pred[0, :, t_star + 1]
    ah_valid_t1 = valid_pred[0, :, t_star + 1].to(th.bool)
    ah_contrib_t1 = th.where(
        ah_valid_t1, ah_contrib_t1, th.full_like(ah_contrib_t1, default_c)
    )

    pun_input_t1 = {k: t[:, :, : t_star + 2].clone() for k, t in full.items()}
    pun_input_t1["contribution"][0, :, t_star + 1] = ah_contrib_t1.to(th.int64)
    pun_input_t1["contribution_valid"][0, :, t_star + 1] = ah_valid_t1
    pun_pred_t1, _ = pun_ah.predict(pun_input_t1, sample=True)

    # pun_t / contrib_t read from full[prev_*][t*+1]: equals override for
    # the intervened feature, equals pilot[t*] for the non-intervened one
    # and for baseline runs.
    pun_t = float(full["prev_punishment"][0, agents_idx, t_star + 1].float().mean())
    contrib_t = float(
        full["prev_contribution"][0, agents_idx, t_star + 1].float().mean()
    )
    result = {
        "pun_t": pun_t,
        "pun_t1": float(pun_pred_t1[0, agents_idx, t_star + 1].float().mean()),
        "contrib_t": contrib_t,
        "contrib_t1": float(ah_contrib_t1[agents_idx].float().mean()),
        "switch_t1": float(switch_pred[0, agents_idx, t_star + 1].float().mean()),
    }
    _trace(f"  [trace {label}] result={result}")
    return result


METRICS = ("pun_t", "pun_t1", "contrib_t", "contrib_t1", "switch_t1")


_FEATURE_SHORT = {"punishment": "pun", "contribution": "contrib"}
_SELECTOR_SHORT = {
    "lowest_contributor": "low",
    "highest_contributor": "high",
    "most_punished": "punished",
    "random": "rand",
}


def _auto_name(intervention_round, intervention):
    """Generate a scenario name from the parameter combo."""
    feat = _FEATURE_SHORT.get(intervention["feature"], intervention["feature"])
    if intervention.get("target") == "group":
        gs = intervention.get("group_selector", "group")
        sel_s = "group" + _SELECTOR_SHORT.get(gs, gs)
    else:
        sel = intervention.get("agent_selector")
        if isinstance(sel, int):
            sel_s = f"a{sel}"
        elif sel is None:
            sel_s = intervention.get("target", "")
        else:
            sel_s = _SELECTOR_SHORT.get(sel, sel)
    if "factor" in intervention:
        mod = f"x{intervention['factor']}"
    else:
        mod = f"v{intervention['new_value']}"
    return f"{feat}_{sel_s}_t{intervention_round}_{mod}"


_INTERVENTION_KEYS = (
    "feature",
    "target",
    "agent_selector",
    "group_selector",
    "n_groups",
    "factor",
    "new_value",
)


def _expand_grid(grid):
    """Cross-product of grid params → list of scenario dicts.

    Recognised keys (each maps to a list of values to sweep):
      intervention_round, feature, target, agent_selector, group_selector,
      n_groups, factor, new_value
    Either ``factor`` or ``new_value`` should be set, not both.
    """
    keys = ["intervention_round", *_INTERVENTION_KEYS]
    sweep_keys = [k for k in keys if k in grid]
    sweep_values = [grid[k] for k in sweep_keys]
    out = []
    for combo in itertools.product(*sweep_values):
        params = dict(zip(sweep_keys, combo))
        intervention = {k: params[k] for k in _INTERVENTION_KEYS if k in params}
        scenario = {
            "intervention_round": params["intervention_round"],
            "intervention": intervention,
        }
        scenario["name"] = grid.get("name") or _auto_name(
            scenario["intervention_round"], intervention
        )
        out.append(scenario)
    return out


def _run_scenario(scen, data, models, chosen, n_seeds, device, rng):
    """Compute baseline + treatment metrics for one scenario.

    Returns a list of per-(scenario, episode) row dicts ready for CSV.
    """
    t_star = scen["intervention_round"]
    iv = scen["intervention"]
    target = iv["target"]
    feature = iv["feature"]
    selector = iv.get("agent_selector") if target == "individual" else None
    is_decision = bool(data["switch_mask"][0, 0, t_star + 1].item())
    n_agents = data["contribution"].shape[1]

    rows = []
    for ep in chosen:
        if target == "group":
            focal = -1
            agents_idx = th.arange(n_agents, dtype=th.int64)
        else:
            ep_prefix = {k: t[ep : ep + 1, :, : t_star + 1] for k, t in data.items()}
            if selector is not None:
                focal = int(
                    _select_focal_agents(ep_prefix, selector, t_star, 1, rng).item()
                )
            else:
                focal = 0
            agents_idx = th.tensor([focal], dtype=th.int64)

        baseline = {k: [] for k in METRICS}
        treatment = {k: [] for k in METRICS}
        for s in range(n_seeds):
            label_b = f"{scen['name']}|ep={ep}|seed={s}|baseline"
            label_t = f"{scen['name']}|ep={ep}|seed={s}|treatment"
            r = _run_seed(ep, agents_idx, t_star, data, models, device, label=label_b)
            for k in METRICS:
                baseline[k].append(float(r[k]))
            r = _run_seed(
                ep, agents_idx, t_star, data, models, device, iv, label=label_t
            )
            for k in METRICS:
                treatment[k].append(float(r[k]))

        row = {
            "scenario": scen["name"],
            "t_star": t_star,
            "is_decision": is_decision,
            "feature": feature,
            "target": target,
            "selector": selector,
            "group_selector": iv.get("group_selector"),
            "new_value": iv.get("new_value"),
            "factor": iv.get("factor"),
            "ep": ep,
            "focal": focal,
            "real_pun_t": float(
                data["punishment"][ep, agents_idx, t_star].float().mean()
            ),
            "real_pun_t1": float(
                data["punishment"][ep, agents_idx, t_star + 1].float().mean()
            ),
            "real_contrib_t": float(
                data["contribution"][ep, agents_idx, t_star].float().mean()
            ),
            "real_contrib_t1": float(
                data["contribution"][ep, agents_idx, t_star + 1].float().mean()
            ),
            "real_switch_t1": float(
                data["does_switch"][ep, agents_idx, t_star + 1].float().mean()
            ),
        }
        for k in METRICS:
            mb, sb = _mean_std(baseline[k])
            mt, st = _mean_std(treatment[k])
            row[f"{k}_baseline_mean"] = mb
            row[f"{k}_baseline_std"] = sb
            row[f"{k}_treatment_mean"] = mt
            row[f"{k}_treatment_std"] = st
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the manifest's output_dir (CSV destination).",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Emit per-seed step-by-step logs verifying the mechanism. "
        "Only practical for small runs (a few scenarios × episodes × seeds); "
        "for production sweeps leave it off.",
    )
    args = parser.parse_args()
    if args.trace:
        global TRACE
        TRACE = True

    cfg = yaml.safe_load(open(args.config))
    if "scenarios" not in cfg and "grids" not in cfg:
        raise ValueError(
            "manifest must have a `scenarios` list or a `grids` list (or both)"
        )
    base = yaml.safe_load(open(cfg["base_config"]))
    chosen = cfg.get("chosen_episodes")
    n_seeds = args.n_seeds or cfg.get("n_seeds", base.get("n_episodes"))
    scenarios = list(cfg.get("scenarios", []))
    for grid in cfg.get("grids", []):
        scenarios.extend(_expand_grid(grid))
    cf_seed = cfg.get("seed")
    output_dir = args.output_dir or cfg.get("output_dir")
    if output_dir is None:
        raise ValueError("output_dir must be set in the manifest (or via --output-dir)")

    basedir = base.get("basedir", ".")
    switch_every = base.get("switch_every")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    print(f"[setup] device={device}  output_dir={output_dir}")
    print(f"[setup] {len(scenarios)} scenarios × {n_seeds} seeds")

    print("[load] tensorizing pilot data...")
    df = pd.read_csv(os.path.join(basedir, base["pilot_data_file"]))
    data, _ = create_torch_data(df, switch_every=switch_every)

    print("[load] loading AH artifacts...")
    ahs = base["artificial_humans"]["group_switching"]
    models = (
        GraphNetwork.load(
            os.path.join(basedir, ahs["contribution_model"]), device=device
        ),
        GraphNetwork.load(os.path.join(basedir, ahs["valid_model"]), device=device),
        GraphNetwork.load(os.path.join(basedir, ahs["switch_model"]), device=device),
        GraphNetwork.load(
            os.path.join(basedir, base["managers"]["punishment_human_manager"]["path"]),
            device=device,
        ),
    )

    rng = th.Generator()
    if cf_seed is not None:
        rng.manual_seed(cf_seed)

    group_cache = {}

    def _episodes_for(iv):
        # group_selector takes precedence; otherwise fall back to manifest's
        # explicit chosen_episodes.
        gs = iv.get("group_selector") if iv["target"] == "group" else None
        if gs is not None:
            n_groups = int(iv.get("n_groups", 5))
            key = (gs, n_groups)
            if key not in group_cache:
                group_cache[key] = _select_focal_groups(data, gs, n_groups, rng)
            return group_cache[key]
        if chosen is None:
            raise ValueError(
                "manifest must set `chosen_episodes` or supply a "
                "`group_selector` per group-target scenario"
            )
        return chosen

    all_rows = []
    for i, scen in enumerate(scenarios, start=1):
        iv = scen["intervention"]
        if iv["target"] == "group":
            sel = iv.get("group_selector")
        else:
            sel = iv.get("agent_selector")
        mod = (
            f"factor={iv['factor']}"
            if "factor" in iv
            else f"new_value={iv['new_value']}"
        )
        eps = _episodes_for(iv)
        print(
            f"[scenario {i}/{len(scenarios)}] {scen['name']}  "
            f"t*={scen['intervention_round']}  "
            f"feature={iv['feature']}  target={iv['target']}  "
            f"selector={sel}  {mod}  episodes={eps}"
        )
        all_rows.extend(_run_scenario(scen, data, models, eps, n_seeds, device, rng))

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "scenarios.csv")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"[done] wrote {len(all_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
