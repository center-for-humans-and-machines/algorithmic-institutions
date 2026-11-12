"""overview_charts.json for the exp2_diff policy-finder page: the campaign's top rules,
zero and the earlier campaigns' bests, each against ah on fresh seeds (34_DIFF_h2h).

Reads plots/simulation/34_DIFF_h2h/h2h_summary.json (scripts/data_analysis/exp2_diff_h2h.py)
and writes Vega-Lite specs that the observation page places at its overview's
<!--chart:name--> markers.

    python scripts/data_analysis/exp2_diff_charts.py <worktree root>/overview_charts.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "overview_charts.json")
SUMMARY = REPO / "plots/simulation/34_DIFF_h2h/h2h_summary.json"
INK, MUTED, AH = "#141413", "#87867f", "#eb6834"
NAMES = {
    "pf_2_v7": "pf-20261112-2-v7", "pf_6_v6": "pf-20261112-6-v6",
    "pf_5_v7": "pf-20261112-5-v7", "pf_9_w0": "multi -9-v5, w = 0",
    "prev_best": "prev_best (exp2)", "cem_best": "Cem's best", "zero": "zero",
}
#: the lines charts: the new rules, the earlier best lead, and zero
LINES = ["pf_2_v7", "pf_6_v6", "pf_5_v7", "pf_9_w0", "zero"]
COLORS = ["#2a78d6", "#1baf7a", "#8a5cd6", INK, MUTED]
FONT = {"labelFont": "Geist", "titleFont": "Geist", "labelColor": MUTED, "titleColor": MUTED}


def records(frame):
    return json.loads(frame.round(3).to_json(orient="records"))


def color(keys):
    names = [NAMES[k] for k in keys]
    return {"field": "manager", "type": "nominal", "title": None,
            "scale": {"domain": names, "range": COLORS[: len(names)]},
            "legend": {"orient": "top", "direction": "horizontal", "labelFont": "Geist",
                       "labelColor": INK, "symbolStrokeWidth": 3, "columns": 3}}


def lines(data, y, y_title, fmt=".2f", height=240):
    hover = {"name": "hover", "select": {"type": "point", "fields": ["round"], "nearest": True,
                                        "on": "pointerover", "clear": "pointerout"}}
    c = color(LINES)
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "height": height, "data": {"values": data},
        "config": {"axis": FONT, "view": {"stroke": None}},
        "encoding": {"x": {"field": "round", "type": "quantitative", "title": "round",
                           "axis": {"tickMinStep": 1, "grid": False}},
                     "y": {"field": y, "type": "quantitative", "title": y_title}},
        "layer": [
            {"mark": {"type": "line", "strokeWidth": 2}, "encoding": {"color": c}},
            {"params": [hover], "mark": {"type": "point", "filled": True, "size": 80},
             "encoding": {"color": c,
                          "opacity": {"condition": {"param": "hover", "empty": False,
                                                    "value": 1}, "value": 0},
                          "tooltip": [{"field": "manager"}, {"field": "round"},
                                      {"field": y, "title": y_title, "format": fmt}]}},
            {"transform": [{"filter": {"param": "hover", "empty": False}}],
             "mark": {"type": "rule", "color": MUTED, "strokeDash": [3, 3]},
             "encoding": {"x": {"field": "round", "type": "quantitative"}}},
        ],
    }


def main():
    summary = json.loads(SUMMARY.read_text())
    rows = summary["rows"]
    order = sorted(rows, key=lambda k: -rows[k]["lead"]["mean"])
    specs = {}

    # 1 · the lead per seed, and pooled with its 95% interval
    seeds = pd.DataFrame([{"manager": NAMES[k], "seed": s, "lead": v}
                          for k in order for s, v in rows[k]["by_seed"].items()])
    pooled = pd.DataFrame([{
        "manager": NAMES[k], "lead": rows[k]["lead"]["mean"],
        "lo": rows[k]["lead"]["mean"] - 1.96 * rows[k]["lead"]["se"],
        "hi": rows[k]["lead"]["mean"] + 1.96 * rows[k]["lead"]["se"],
        "own": rows[k]["own"]["mean"], "ah": rows[k]["other"]["mean"]} for k in order])
    y = {"field": "manager", "type": "nominal", "title": None,
         "sort": [NAMES[k] for k in order], "axis": {"labelFont": "Geist", "labelColor": INK}}
    specs["lead-seeds"] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "height": {"step": 34}, "config": {"axis": FONT, "view": {"stroke": None}},
        "layer": [
            {"data": {"values": records(seeds)},
             "mark": {"type": "circle", "size": 55, "opacity": 0.45, "color": MUTED},
             "encoding": {"y": y, "x": {"field": "lead", "type": "quantitative",
                                        "title": "lead over ah's group (pool points)",
                                        "scale": {"zero": False}},
                          "tooltip": [{"field": "manager"}, {"field": "seed"},
                                      {"field": "lead", "format": ".2f"}]}},
            {"data": {"values": records(pooled)},
             "mark": {"type": "rule", "strokeWidth": 3, "color": INK},
             "encoding": {"y": y, "x": {"field": "lo", "type": "quantitative"},
                          "x2": {"field": "hi"}}},
            {"data": {"values": records(pooled)},
             "mark": {"type": "tick", "thickness": 3, "size": 20, "color": INK},
             "encoding": {"y": y, "x": {"field": "lead", "type": "quantitative"},
                          "tooltip": [{"field": "manager"},
                                      {"field": "lead", "title": "lead", "format": ".2f"},
                                      {"field": "lo", "title": "95% from", "format": ".2f"},
                                      {"field": "hi", "title": "95% to", "format": ".2f"}]}},
        ],
    }

    # 2 · both groups' pools: the lead is the distance between the two dots
    both = pd.DataFrame([{"manager": NAMES[k], "group": g, "pool": rows[k][f]["mean"],
                          "lead": rows[k]["lead"]["mean"]}
                         for k in order for g, f in (("the rule's group", "own"),
                                                     ("ah's group", "other"))])
    specs["both-pools"] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "height": {"step": 34}, "data": {"values": records(both)},
        "config": {"axis": FONT, "view": {"stroke": None}},
        "encoding": {"y": y},
        "layer": [
            {"mark": {"type": "rule", "color": "#d6d2c8", "strokeWidth": 2},
             "encoding": {"x": {"aggregate": "min", "field": "pool", "type": "quantitative"},
                          "x2": {"aggregate": "max", "field": "pool"}}},
            {"mark": {"type": "circle", "size": 110, "opacity": 1},
             "encoding": {"x": {"field": "pool", "type": "quantitative",
                                "title": "common pool per round", "scale": {"zero": False}},
                          "color": {"field": "group", "type": "nominal", "title": None,
                                    "scale": {"domain": ["the rule's group", "ah's group"],
                                              "range": [INK, AH]},
                                    "legend": {"orient": "top", "labelFont": "Geist",
                                               "labelColor": INK}},
                          "tooltip": [{"field": "manager"}, {"field": "group"},
                                      {"field": "pool", "format": ".2f"},
                                      {"field": "lead", "title": "lead", "format": ".2f"}]}},
        ],
    }

    # 3-5 · per round
    pr = pd.DataFrame(summary["per_round"])
    pr = pr[pr["row"].isin(LINES)].copy()
    pr["manager"] = pr["row"].map(NAMES)
    pr["round"] = pr["round_number"]
    pr["fine"] = pr["own_fines"] / pr["own_n"].where(pr["own_n"] > 0)
    pr["lead"] = pr["own_pool"] - pr["other_pool"]
    pr = pr.sort_values(["row", "round"])
    pr["lead_cum"] = pr.groupby("row")["lead"].cumsum() / 24
    zero = pr[pr["row"] == "zero"].set_index("round")["lead_cum"]
    pr["vs_zero"] = pr["lead_cum"] - pr["round"].map(zero)
    cols = ["manager", "round", "fine", "own_n", "lead", "vs_zero"]
    data = records(pr[cols])
    specs["fine-by-round"] = lines(data, "fine", "fine per member of the rule's group")
    specs["members-by-round"] = lines(data, "own_n", "members of the rule's group")
    specs["lead-vs-zero"] = lines(data, "vs_zero", "lead gained over zero, cumulative / 24")
    OUT.write_text(json.dumps(specs) + "\n")
    print("wrote", OUT, list(specs))


if __name__ == "__main__":
    main()
