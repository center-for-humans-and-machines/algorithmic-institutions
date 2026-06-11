"""Intervention probe — multi-scenario.

Per scenario × episode × n_seeds, runs the AH stack twice (baseline,
treatment) and records mean metrics over the targeted agents at rounds
t* and t*+1: pun_t, pun_t1, contrib_t, contrib_t1, switch_t1. Aggregated
mean ± std is written to ``<output_dir>/scenarios.csv``.

Two targets, common ``selector`` field:
- ``individual`` — overrides one focal agent picked by ``selector`` from
  every chosen episode; metrics are the focal's values.
- ``group`` — overrides every agent in the group picked by ``selector``
  at t* (the agent_group partition; in 2g8a runs, 4 of 8 agents);
  metrics are averaged across the targeted group.

Manifests give episodes via top-level ``chosen_episodes`` for both targets.
They support an explicit ``scenarios:`` list and/or a compact ``grids:``
form whose parameter cross-product is expanded at load time. Treatment
uses ``new_value`` (absolute) or ``factor`` (scales pilot's round-t*
value, clamped to the encoder level count).
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
    _select_focal_group,
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
        max_val = pun_ah.y_levels - 1 if f == "punishment" else contrib_ah.y_levels - 1
        for a in agents_list:
            pilot_ref = int(data[f][ep, a, t_star].item())
            natural_prev = int(full[f"prev_{f}"][0, a, t_star + 1].item())
            v = intervention_value(intervention, pilot_ref, max_value=max_val)
            # Mirror the override into both the prev_*[t*+1] slot the
            # AH reads and the round-t* slot the common_good recompute
            # below sums over, so prev_X = X-shifted-by-1 stays consistent.
            full[f"prev_{f}"][0, a, t_star + 1] = v
            full[f"prev_{f}_valid"][0, a, t_star + 1] = True
            full[f][0, a, t_star] = v
            full[f"{f}_valid"][0, a, t_star] = True
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

        # STEP 3.5: recompute common_good[t*] per sub-group with the
        # overridden values, then propagate to prev_common_good[t*+1].
        # Sub-groups are determined by agent_group at t* (switching is
        # frozen within a block, so this is well-defined).
        ag_at_t = full["agent_group"][0, :, t_star]
        c_t = full["contribution"][0, :, t_star].float()
        p_t = full["punishment"][0, :, t_star].float()
        c_valid = full["contribution_valid"][0, :, t_star].float()
        old_cg = full["common_good"][0, :, t_star].clone()
        new_cg = th.zeros_like(old_cg)
        per_group = {}
        for g in ag_at_t.unique().tolist():
            in_g = ag_at_t == g
            n_valid = c_valid[in_g].sum().clamp(min=1)
            pool = (1.6 * c_t[in_g] - p_t[in_g]).sum()
            per_capita = pool / n_valid
            new_cg = th.where(in_g, per_capita, new_cg)
            per_group[g] = (
                int(in_g.sum().item()),
                int(c_valid[in_g].sum().item()),
                float(c_t[in_g].sum().item()),
                float(p_t[in_g].sum().item()),
                float(per_capita.item()),
            )
        full["common_good"][0, :, t_star] = new_cg
        full["prev_common_good"][0, :, t_star + 1] = new_cg
        for g, (n, nv, sc, sp, pc) in per_group.items():
            _trace(
                f"  [trace {label}] STEP 3.5 cg group={g}: members={n} "
                f"valid={nv}  sum_c={sc:.0f}  sum_p={sp:.0f}  "
                f"new_cg={pc:.3f}"
            )
        cg_pairs = [
            (int(ag_at_t[i].item()), float(new_cg[i].item()))
            for i in range(new_cg.shape[0])
        ]
        old_pairs = [float(v) for v in old_cg.tolist()]
        _trace(
            f"  [trace {label}] STEP 3.5 prev_common_good[t*+1] per agent "
            f"(group, new): {cg_pairs}  (was {old_pairs})"
        )
    else:
        _trace(f"  [trace {label}] STEP 3 baseline: no override applied")

    contrib_pred, _ = contrib_ah.predict_independent(full, sample=True, reset_rnn=True)
    valid_pred, _ = valid_ah.predict_independent(full, sample=True, reset_rnn=True)
    switch_pred, switch_proba = switch_ah.predict_independent(
        full, sample=True, reset_rnn=True
    )
    _trace(
        f"  [trace {label}] STEP 4 AH stack forward (contrib, valid, switch) "
        f"with reset_rnn=True over rounds [0..{t_star + 1}]"
    )
    # Per-agent diagnostic for the switch panel: feature inputs the
    # switch AH actually sees at t*+1 (prev_common_good, prev_punishment,
    # agent_group) and the predicted P(switch=1). Probabilities are
    # deterministic given inputs — sampling only adds Bernoulli noise.
    if TRACE:
        ag = full["agent_group"][0, :, t_star + 1].tolist()
        p_p = full["prev_punishment"][0, :, t_star + 1].tolist()
        p_cg = full["prev_common_good"][0, :, t_star + 1].tolist()
        p_sw = switch_proba[0, :, t_star + 1, 1].tolist()
        rows = [
            f"a{i}: g={int(ag[i])}  prev_p={int(p_p[i])}  "
            f"prev_cg={p_cg[i]:.3f}  P(sw)={p_sw[i]:.3f}"
            for i in range(len(ag))
        ]
        _trace(f"  [trace {label}] STEP 4 switch inputs+P at t*+1:")
        for r in rows:
            _trace(f"    {r}")

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
    sel = intervention.get("selector")
    target = intervention.get("target", "")
    if isinstance(sel, int):
        sel_s = f"a{sel}"
    elif sel is None:
        sel_s = target
    else:
        sel_s = _SELECTOR_SHORT.get(sel, sel)
    if target == "group":
        sel_s = "group" + sel_s
    if "factor" in intervention:
        mod = f"x{intervention['factor']}"
    else:
        mod = f"v{intervention['new_value']}"
    return f"{feat}_{sel_s}_t{intervention_round}_{mod}"


_INTERVENTION_KEYS = (
    "feature",
    "target",
    "selector",
    "factor",
    "new_value",
)


def _expand_grid(grid):
    """Cross-product of grid params → list of scenario dicts.

    Recognised keys (each maps to a list of values to sweep):
      intervention_round, feature, target, selector, factor, new_value
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
    selector = iv.get("selector")
    is_decision = bool(data["switch_mask"][0, 0, t_star + 1].item())

    rows = []
    for ep in chosen:
        ep_prefix = {k: t[ep : ep + 1, :, : t_star + 1] for k, t in data.items()}
        if target == "group":
            agents_idx = _select_focal_group(ep_prefix, selector, t_star, rng)
            focal = -1
        elif target == "individual":
            if selector is not None:
                focal = int(
                    _select_focal_agents(ep_prefix, selector, t_star, 1, rng).item()
                )
            else:
                focal = 0
            agents_idx = th.tensor([focal], dtype=th.int64)
        else:
            raise ValueError(
                f"Unknown target: {target!r}; expected 'individual' or 'group'"
            )

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
            "new_value": iv.get("new_value"),
            "factor": iv.get("factor"),
            "ep": ep,
            "focal": focal,
            "agents": agents_idx.tolist(),
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
    if chosen is None:
        raise ValueError("manifest must set `chosen_episodes`")
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
    print(
        f"[setup] {len(scenarios)} scenarios × {len(chosen)} episodes "
        f"× {n_seeds} seeds (chosen={chosen})"
    )

    print("[load] tensorizing pilot data...")
    df = pd.read_csv(os.path.join(basedir, base["pilot_data_file"]))
    data, _, _ = create_torch_data(df, switch_every=switch_every)

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

    all_rows = []
    for i, scen in enumerate(scenarios, start=1):
        iv = scen["intervention"]
        sel = iv.get("selector")
        mod = (
            f"factor={iv['factor']}"
            if "factor" in iv
            else f"new_value={iv['new_value']}"
        )
        print(
            f"[scenario {i}/{len(scenarios)}] {scen['name']}  "
            f"t*={scen['intervention_round']}  "
            f"feature={iv['feature']}  target={iv['target']}  "
            f"selector={sel}  {mod}"
        )
        all_rows.extend(_run_scenario(scen, data, models, chosen, n_seeds, device, rng))

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "scenarios.csv")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"[done] wrote {len(all_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
