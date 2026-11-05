"""overview_charts.json for the exp2_multi policy-finder page: the campaign's top 3 rules
against prev_best (pf-20261104-12-v7), the earlier exp2 champion, opponent by opponent.

Reads the fresh-seed head-to-head (33_MULTI_h2h) and its summary from
scripts/data_analysis/exp2_multi_h2h.py, and writes Vega-Lite specs that the observation
page places at its overview's <!--chart:name--> markers.

    python scripts/data_analysis/exp2_multi_charts.py <worktree root>/overview_charts.json

Report: docs/reports/policy_finder_multi_opponent.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "overview_charts.json")
ROOT = REPO / "plots/simulation/33_MULTI_h2h"
INK, MUTED = "#141413", "#87867f"
NAMES = {"prev_best": "prev_best (exp2)", "pf_7": "pf-20261106-7-v5",
         "pf_11": "pf-20261106-11-v7", "pf_9": "pf-20261106-9-v5"}
ORDER = list(NAMES.values())
COLORS = [INK, "#2a78d6", "#eb6834", "#1baf7a"]
OPPONENTS = {"levin": "Levin's rule", "ah": "ah (human clone)", "prev_best": "prev_best",
             "zero": "zero (never punishes)", "cem_best": "Cem's best (held out)",
             "thr9_p10": "thr9_p10 (held out)", "self": "itself"}
TRAINING = ["levin", "ah", "prev_best", "zero"]

color = {"field": "manager", "type": "nominal", "title": None,
         "scale": {"domain": ORDER, "range": COLORS},
         "legend": {"orient": "top", "direction": "horizontal", "labelFont": "Geist",
                    "labelColor": INK, "symbolStrokeWidth": 3}}
width = {"condition": {"test": f"datum.manager === '{ORDER[0]}'", "value": 3}, "value": 2}


def rows(frame):
    return json.loads(frame.round(3).to_json(orient="records"))


def lines(data, x, y, x_title, y_title, tooltip_y, height=260):
    hover = {"name": "hover", "select": {"type": "point", "fields": [x], "nearest": True,
                                        "on": "pointerover", "clear": "pointerout"}}
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "height": height,
        "data": {"values": data},
        "encoding": {"x": {"field": x, "type": "quantitative", "title": x_title,
                           "axis": {"tickMinStep": 1, "grid": False}},
                     "y": {"field": y, "type": "quantitative", "title": y_title}},
        "layer": [
            {"mark": {"type": "line", "interpolate": "linear"},
             "encoding": {"color": color, "strokeWidth": width}},
            {"params": [hover],
             "mark": {"type": "point", "filled": True, "size": 90},
             "encoding": {"color": color,
                          "opacity": {"condition": {"param": "hover", "empty": False,
                                                    "value": 1}, "value": 0},
                          "tooltip": [{"field": "manager", "title": "manager"},
                                      {"field": x, "title": x_title},
                                      {"field": y, "title": tooltip_y, "format": ".2f"}]}},
            {"transform": [{"filter": {"param": "hover", "empty": False}}],
             "mark": {"type": "rule", "color": MUTED, "strokeDash": [3, 3]},
             "encoding": {"x": {"field": x, "type": "quantitative"}}},
        ],
    }


def by_opponent(spec, name="opponent"):
    """Show one opponent at a time, picked with radio buttons."""
    labels = [OPPONENTS[o] for o in TRAINING]
    spec["params"] = [{"name": name, "value": labels[0],
                       "bind": {"input": "radio", "options": labels, "name": "Opponent  "}}]
    spec["transform"] = [{"filter": f"datum.opponent === {name}"}]
    return spec


def per_round():
    """(manager, opponent, round) -> mean fine per member, members, pool of its group."""
    import pyarrow.parquet as pq

    acc = []
    for f in sorted(ROOT.glob("seed_*/per_round.parquet")):
        tab = pq.read_table(f, columns=["episode", "round_number", "agent_group",
                                        "contribution", "punishment",
                                        "contribution_valid", "run"])
        i = tab.schema.get_field_index("run")
        tab = tab.set_column(i, "run", tab.column("run").dictionary_encode())
        d = tab.to_pandas()  # run comes out categorical: one string per pairing
        del tab
        d["pool"] = 1.6 * d.contribution - d.punishment
        keys = ["run", "agent_group", "round_number"]
        s = d.groupby(keys + ["episode"], observed=True).pool.agg(["sum", "size"])
        s = s.groupby(keys, observed=True).sum().rename(
            columns={"sum": "pool", "size": "members"})
        valid = d.contribution_valid.astype(bool)
        fines = d[valid].groupby(keys, observed=True).punishment.agg(["sum", "size"])
        games = d.groupby("run", observed=True).episode.nunique().rename("games")
        del d
        t = s.join(fines).reset_index().merge(games.reset_index(), on="run")
        pair = t["run"].astype(str).str.split("managed by ").str[-1].str.split("_vs_")
        g0, g1 = pair.str[0], pair.str[1]
        t["manager"] = g0.where(t.agent_group == 0, g1)
        other = g1.where(t.agent_group == 0, g0)
        t["opponent"] = other.where(t.manager != other, "self")
        acc.append(t[t.manager.isin(NAMES)].drop(columns=["run", "agent_group"]))
    t = pd.concat(acc).groupby(["manager", "opponent", "round_number"]).sum().reset_index()
    t["fine"] = t["sum"] / t["size"]
    # empty group-rounds count 0 for pool and members: divide by all games
    t["pool"] = t["pool"] / t["games"]
    t["members"] = t["members"] / t["games"]
    return t


specs = {}
summary = json.loads((ROOT / "h2h_summary.json").read_text())
table = summary["rows"]

# 1. gap to prev_best, opponent by opponent, with 95% intervals
gap = []
for m in ["pf_7", "pf_11", "pf_9"]:
    for o in TRAINING + ["cem_best", "thr9_p10", "self"]:
        a, b = table[m][o], table["prev_best"][o]
        g, se = a["pool"] - b["pool"], float(np.hypot(a["se"], b["se"]))
        gap.append({"manager": NAMES[m], "opponent": OPPONENTS[o], "gap": g,
                    "lo": g - 1.96 * se, "hi": g + 1.96 * se, "pool": a["pool"],
                    "prev_best": b["pool"]})
for m in ["pf_7", "pf_11", "pf_9"]:
    d = table[m]["training_vs_prev_best"]
    gap.append({"manager": NAMES[m], "opponent": "mean of the four", "gap": d["pool"],
                "lo": d["pool"] - 1.96 * d["se"], "hi": d["pool"] + 1.96 * d["se"],
                "pool": table[m]["training"]["pool"],
                "prev_best": table["prev_best"]["training"]["pool"]})
x_order = ["mean of the four"] + [OPPONENTS[o] for o in TRAINING + ["cem_best", "thr9_p10",
                                                                    "self"]]
gap_color = dict(color, scale={"domain": ORDER[1:], "range": COLORS[1:]})
x = {"field": "opponent", "type": "nominal", "title": None, "sort": x_order,
     "axis": {"labelAngle": -35, "labelFont": "Geist", "labelColor": INK, "labelLimit": 160}}
specs["gap-to-prev-best"] = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "height": 280,
    "data": {"values": gap},
    "encoding": {"x": x, "xOffset": {"field": "manager", "sort": ORDER[1:]}},
    "layer": [
        {"mark": {"type": "rule", "color": MUTED},
         "encoding": {"y": {"datum": 0, "type": "quantitative"}, "xOffset": None}},
        {"mark": {"type": "rule", "strokeWidth": 2, "strokeCap": "round"},
         "encoding": {"y": {"field": "lo", "type": "quantitative",
                            "title": "pool minus prev_best's"},
                      "y2": {"field": "hi"}, "color": gap_color}},
        {"mark": {"type": "point", "filled": True, "size": 70},
         "encoding": {"y": {"field": "gap", "type": "quantitative"}, "color": gap_color,
                      "tooltip": [{"field": "manager"}, {"field": "opponent"},
                                  {"field": "pool", "format": ".2f"},
                                  {"field": "prev_best", "format": ".2f"},
                                  {"field": "gap", "format": "+.2f"},
                                  {"field": "lo", "title": "95% low", "format": "+.2f"},
                                  {"field": "hi", "title": "95% high", "format": "+.2f"}]}},
    ],
}

# 2. score per seed (mean of the four), and pooled
seeds = summary["seeds"]
pts, pooled = [], []
for m in NAMES:
    for s in seeds:
        vals = [table[m][o + "_by_seed"][s] for o in TRAINING]
        pts.append({"manager": NAMES[m], "seed": int(s), "score": sum(vals) / 4})
    t = table[m]["training"]
    pooled.append({"manager": NAMES[m], "score": t["pool"], "lo": t["pool"] - 1.96 * t["se"],
                   "hi": t["pool"] + 1.96 * t["se"], "games": 4 * table[m]["levin"]["n"]})
y = {"field": "manager", "type": "nominal", "title": None, "sort": ORDER,
     "axis": {"labelFont": "Geist Mono", "labelColor": INK}}
lo = min(p["score"] for p in pts) - 0.5
hi = max(p["score"] for p in pts) + 0.5
specs["score-seeds"] = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "height": {"step": 46},
    "encoding": {"y": y},
    "layer": [
        {"data": {"values": pooled},
         "mark": {"type": "rule", "strokeWidth": 3, "strokeCap": "round"},
         "encoding": {"x": {"field": "lo", "type": "quantitative",
                            "title": "mean pool over the four opponents",
                            "scale": {"zero": False, "domain": [np.floor(lo), np.ceil(hi)]}},
                      "x2": {"field": "hi"}, "color": dict(color, legend=None)}},
        {"data": {"values": pts},
         "mark": {"type": "point", "size": 60, "strokeWidth": 1.5, "yOffset": 12},
         "encoding": {"x": {"field": "score", "type": "quantitative"},
                      "color": dict(color, legend=None),
                      "tooltip": [{"field": "manager"}, {"field": "seed"},
                                  {"field": "score", "format": ".2f"}]}},
        {"data": {"values": pooled},
         "mark": {"type": "tick", "thickness": 3, "size": 18},
         "encoding": {"x": {"field": "score", "type": "quantitative"},
                      "color": dict(color, legend=None),
                      "tooltip": [{"field": "manager"},
                                  {"field": "score", "title": "all seeds", "format": ".2f"},
                                  {"field": "lo", "title": "95% low", "format": ".2f"},
                                  {"field": "hi", "title": "95% high", "format": ".2f"},
                                  {"field": "games"}]}},
    ],
}

# 3-5. by round, one training opponent at a time
t = per_round()
t = t[t.opponent.isin(TRAINING + ["self"])].copy()
t.loc[(t.manager == "prev_best") & (t.opponent == "self"), "opponent"] = "prev_best"
t = t[t.opponent.isin(TRAINING)]
t["manager"] = t.manager.map(NAMES)
t["opponent"] = t.opponent.map(OPPONENTS)
t = t.rename(columns={"round_number": "round"})
for key, col, title in [("fine-by-round", "fine", "mean fine per member"),
                        ("members-by-round", "members", "members")]:
    spec = lines(rows(t[["manager", "opponent", "round", col]]), "round", col, "round",
                 title, title)
    if col != "fine":
        spec["encoding"]["y"]["scale"] = {"zero": False}
    specs[key] = by_opponent(spec)

# 5. where the gap builds: the pool difference to prev_best, summed over rounds and
# divided by 24, so the line ends at the score gap
wide = t.pivot_table(index=["opponent", "round"], columns="manager", values="pool")
gap_r = wide.sub(wide[ORDER[0]], axis=0).drop(columns=ORDER[0]) / 24
gap_r = gap_r.groupby(level="opponent").cumsum().stack().rename("gap").reset_index()
spec = lines(rows(gap_r), "round", "gap", "round", "score gap to prev_best so far",
             "gap so far")
for layer in spec["layer"][:2]:
    layer["encoding"]["color"] = gap_color
# one row per opponent, so the radio filter keeps exactly one
spec["layer"].insert(0, {"data": {"values": [{"opponent": OPPONENTS[o]} for o in TRAINING]},
                         "mark": {"type": "rule", "color": MUTED},
                         "encoding": {"y": {"datum": 0, "type": "quantitative"},
                                      "x": {"value": 0}, "x2": {"value": "width"}}})
specs["pool-gap"] = by_opponent(spec)

OUT.write_text(json.dumps(specs))
print("wrote", OUT, OUT.stat().st_size, "bytes;", list(specs))
print(t.groupby(["opponent", "manager"])[["fine", "members", "pool"]].mean().round(2)
      .to_string())
