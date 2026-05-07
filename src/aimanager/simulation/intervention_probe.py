"""Focal-only intervention probe — multi-scenario.

Per scenario × chosen_episode × n_seeds, runs the AH stack twice
(baseline, treatment) and records the focal's metrics at rounds t* and
t*+1: pun_t, pun_t1, contrib_t, contrib_t1, switch_t1. Aggregated
mean ± std is written to ``<output_dir>/scenarios.csv``.

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
    ep, focal, t_star, data, models, device, intervention=None, label=""
):
    """One stochastic draw → focal's metrics at rounds t* and t*+1.

    Real-data backfill design:
    - Round 0..t* values come from pilot data verbatim. No AH manager
      prediction at t* — the human's actual action is the reference.
      Treatment intervention overrides only the focal's slot at t* to
      either ``new_value`` (absolute) or ``factor × real`` (factor mode).
    - The intervened value lands as ``prev_<feature>[t*+1, focal]`` so the
      AH at t*+1 reads the perturbation. Non-focals' prev_*[t*+1] stays
      at the data's natural shift values.
    - Round-t*+1 outputs come from a single AH-stack forward — that's
      the response we're measuring. The punishment AH at t*+1 sees
      AH-predicted round-t*+1 contributions plus the (intervened or
      natural) prev_* values.

    Under this design, baseline at round t* equals real pilot exactly
    (no AH involvement at t*). The AH-vs-pilot fidelity gap only shows
    up at round-t*+1 metrics.

    When module-level ``TRACE`` is enabled (via ``--trace`` on the CLI)
    every step is logged to stdout so a small run can be eyeballed against
    the design described above.
    """
    contrib_ah, valid_ah, switch_ah, pun_ah = models
    _trace(f"  [trace {label}] _run_seed begin: ep={ep} focal={focal} t*={t_star}")

    full = {
        k: t[ep : ep + 1, :, : t_star + 2].clone().to(device) for k, t in data.items()
    }
    _trace(
        f"  [trace {label}] STEP 1 prefix slice [0..t*+1] from pilot tensor; "
        f"contribution shape={tuple(full['contribution'].shape)}"
    )

    pun_t = int(data["punishment"][ep, focal, t_star].item())
    contrib_t = int(data["contribution"][ep, focal, t_star].item())
    _trace(
        f"  [trace {label}] STEP 2 round-t* values from pilot data: "
        f"pun_t={pun_t}  contrib_t={contrib_t}"
    )

    if intervention is not None:
        f = intervention["feature"]
        pilot_ref = int(data[f][ep, focal, t_star].item())
        natural_prev = int(full[f"prev_{f}"][0, focal, t_star + 1].item())
        if f == "punishment":
            pun_t = intervention_value(
                intervention, pilot_ref, max_value=pun_ah.y_levels - 1
            )
            full["prev_punishment"][0, focal, t_star + 1] = pun_t
            full["prev_punishment_valid"][0, focal, t_star + 1] = True
            _trace(
                f"  [trace {label}] STEP 3 treatment override: "
                f"feature={f}  pilot_ref={pilot_ref}  resolved={pun_t}; "
                f"set full[prev_punishment][0,{focal},{t_star + 1}] "
                f"{natural_prev} -> {pun_t}"
            )
        elif f == "contribution":
            contrib_t = intervention_value(
                intervention, pilot_ref, max_value=contrib_ah.y_levels - 1
            )
            full["prev_contribution"][0, focal, t_star + 1] = contrib_t
            full["prev_contribution_valid"][0, focal, t_star + 1] = True
            _trace(
                f"  [trace {label}] STEP 3 treatment override: "
                f"feature={f}  pilot_ref={pilot_ref}  resolved={contrib_t}; "
                f"set full[prev_contribution][0,{focal},{t_star + 1}] "
                f"{natural_prev} -> {contrib_t}"
            )
        prev_all = [int(v) for v in full[f"prev_{f}"][0, :, t_star + 1].tolist()]
        _trace(
            f"  [trace {label}] STEP 3 prev_{f}[t*+1] all-agents post-override: "
            f"{prev_all}  (only focal={focal} should differ from data shift)"
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
    _trace(
        f"  [trace {label}] STEP 5 pun AH inputs at t*+1={t_star + 1}: "
        f"contribution[t*+1]=AH-predicted "
        f"({[int(v) for v in ah_contrib_t1.tolist()]}); "
        f"prev_punishment[t*+1, focal={focal}]="
        f"{int(pun_input_t1['prev_punishment'][0, focal, t_star + 1].item())}; "
        f"prev_contribution[t*+1, focal={focal}]="
        f"{int(pun_input_t1['prev_contribution'][0, focal, t_star + 1].item())}"
    )
    pun_pred_t1, _ = pun_ah.predict(pun_input_t1, sample=True)
    pun_t1 = int(pun_pred_t1[0, focal, t_star + 1].item())

    result = {
        "pun_t": pun_t,
        "pun_t1": pun_t1,
        "contrib_t": contrib_t,
        "contrib_t1": int(ah_contrib_t1[focal].item()),
        "switch_t1": bool(switch_pred[0, focal, t_star + 1].item()),
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


def _expand_grid(grid):
    """Cross-product of grid params → list of scenario dicts.

    Recognised keys (each maps to a list of values to sweep):
      intervention_round, feature, target, agent_selector, factor, new_value
    Either ``factor`` or ``new_value`` should be set, not both.
    """
    keys = [
        "intervention_round",
        "feature",
        "target",
        "agent_selector",
        "factor",
        "new_value",
    ]
    sweep_keys = [k for k in keys if k in grid]
    sweep_values = [grid[k] for k in sweep_keys]
    out = []
    for combo in itertools.product(*sweep_values):
        params = dict(zip(sweep_keys, combo))
        intervention = {
            k: params[k]
            for k in ("feature", "target", "agent_selector", "factor", "new_value")
            if k in params
        }
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

    rows = []
    for ep in chosen:
        ep_prefix = {k: t[ep : ep + 1, :, : t_star + 1] for k, t in data.items()}
        if selector is not None:
            focal = int(
                _select_focal_agents(ep_prefix, selector, t_star, 1, rng).item()
            )
        else:
            focal = 0

        baseline = {k: [] for k in METRICS}
        treatment = {k: [] for k in METRICS}
        for s in range(n_seeds):
            label_b = f"{scen['name']}|ep={ep}|seed={s}|baseline"
            label_t = f"{scen['name']}|ep={ep}|seed={s}|treatment"
            r = _run_seed(ep, focal, t_star, data, models, device, label=label_b)
            for k in METRICS:
                baseline[k].append(int(r[k]))
            r = _run_seed(
                ep, focal, t_star, data, models, device, iv, label=label_t
            )
            for k in METRICS:
                treatment[k].append(int(r[k]))

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
            "real_pun_t": int(data["punishment"][ep, focal, t_star].item()),
            "real_pun_t1": int(data["punishment"][ep, focal, t_star + 1].item()),
            "real_contrib_t": int(data["contribution"][ep, focal, t_star].item()),
            "real_contrib_t1": int(data["contribution"][ep, focal, t_star + 1].item()),
            "real_switch_t1": bool(data["does_switch"][ep, focal, t_star + 1].item()),
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
    chosen = cfg["chosen_episodes"]
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

    all_rows = []
    for i, scen in enumerate(scenarios, start=1):
        iv = scen["intervention"]
        sel = iv.get("agent_selector") if iv["target"] == "individual" else None
        mod = (
            f"factor={iv['factor']}"
            if "factor" in iv
            else f"new_value={iv['new_value']}"
        )
        print(
            f"[scenario {i}/{len(scenarios)}] {scen['name']}  "
            f"t*={scen['intervention_round']}  "
            f"feature={iv['feature']}  selector={sel}  {mod}"
        )
        all_rows.extend(_run_scenario(scen, data, models, chosen, n_seeds, device, rng))

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "scenarios.csv")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"[done] wrote {len(all_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
