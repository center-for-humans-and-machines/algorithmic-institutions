"""overview_charts.json for the exp1 policy-finder page: the top rules against the RL manager.

Reads every fresh-seed head-to-head run (30_EXP1_pf_vs_rl, 31_EXP1_confirm local and Raven)
and writes Vega-Lite specs the observation page places at its overview's <!--chart:name-->
markers. Usage: python scripts/data_analysis/exp1_pf_vs_rl_charts.py <worktree root>/overview_charts.json
Report: docs/reports/policy_finder_exp1_vs_rl.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "overview_charts.json")
SIM = REPO / "plots/simulation"
INK, MUTED, LINE = "#141413", "#87867f", "#e3dacc"
NAMES = {"rl_deployed": "RL (deployed)", "pf_3": "pf-20261105-3-v4", "pf_7": "pf-20261105-7-v3", "pf_8": "pf-20261105-8-v7"}
ORDER = list(NAMES.values())
COLORS = [INK, "#2a78d6", "#eb6834", "#1baf7a"]

runs = [(SIM / "30_EXP1_pf_vs_rl", "seed 6000, Raven", 6000)]
runs += [(SIM / f"31_EXP1_confirm/seed_{s}", "seeds 7000–7004, local", s) for s in range(7000, 7005)]
runs += [(SIM / f"31_EXP1_confirm_raven/seed_{s}", "seeds 7000–7004, Raven", s) for s in range(7000, 7005)]
frames = []
for path, platform, seed in runs:
    f = path / "per_round.parquet"
    if not f.exists():
        print("missing", f, file=sys.stderr)
        continue
    d = pd.read_parquet(f)
    d["manager"] = d["run"].str.split(" managed by ").str[-1]
    d = d[d.manager.isin(NAMES)]
    d["block"] = f"{platform}:{seed}"
    d["platform"], d["seed"] = platform, seed
    frames.append(d)
d = pd.concat(frames, ignore_index=True)
d["name"] = d.manager.map(NAMES)
v = d[d.contribution_valid.astype(bool)]
n_games = d.groupby("name")[["block", "episode"]].apply(lambda g: len(g.drop_duplicates())).to_dict()
print("games per manager", n_games)

color = {"field": "manager", "type": "nominal", "title": None,
         "scale": {"domain": ORDER, "range": COLORS},
         "legend": {"orient": "top", "direction": "horizontal", "labelFont": "Geist",
                    "labelColor": INK, "symbolStrokeWidth": 3}}
width = {"condition": {"test": "datum.manager === 'RL (deployed)'", "value": 3}, "value": 2}


def rows(frame):
    return json.loads(frame.round(3).to_json(orient="records"))


def lines(data, x, y, x_title, y_title, tooltip_y, height=260, extra=()):
    hover = {"name": "hover", "select": {"type": "point", "fields": [x], "nearest": True,
                                        "on": "pointerover", "clear": "pointerout"}}
    spec = {
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
                          "opacity": {"condition": {"param": "hover", "empty": False, "value": 1},
                                      "value": 0},
                          "tooltip": [{"field": "manager", "title": "manager"},
                                      {"field": x, "title": x_title},
                                      {"field": y, "title": tooltip_y, "format": ".2f"}]}},
            {"transform": [{"filter": {"param": "hover", "empty": False}}],
             "mark": {"type": "rule", "color": MUTED, "strokeDash": [3, 3]},
             "encoding": {"x": {"field": x, "type": "quantitative"}}},
        ],
    }
    if extra:  # layers without the data's x and y: keep the shared encoding off them
        shared = spec.pop("encoding")
        for layer in spec["layer"]:
            layer["encoding"] = {**shared, **layer.get("encoding", {})}
        spec["layer"] = [*extra, *spec["layer"]]
    return spec


specs = {}

# 1. punishment by round
p = v.groupby(["name", "round_number"]).punishment.mean().reset_index()
p.columns = ["manager", "round", "punishment"]
specs["punishment-by-round"] = lines(rows(p), "round", "punishment", "round",
                                     "mean fine per player", "mean fine")

# 2. fine against contribution, by phase (radio-bound parameter)
phases = [("rounds 0–3", 0, 3), ("rounds 4–6", 4, 6), ("rounds 7–15", 7, 15), ("rounds 16–23", 16, 23)]
parts = []
for label, lo, hi in phases:
    g = v[(v.round_number >= lo) & (v.round_number <= hi)]
    a = g.groupby(["name", "contribution"]).punishment.agg(["mean", "size"]).reset_index()
    a = a[a["size"] >= 300]
    a["phase"] = label
    parts.append(a)
dose = pd.concat(parts).rename(columns={"name": "manager", "mean": "punishment", "size": "n"})
dose_spec = lines(rows(dose), "contribution", "punishment", "contribution this round",
                  "mean fine", "mean fine")
dose_spec["params"] = [{"name": "phase", "value": "rounds 0–3",
                        "bind": {"input": "radio", "options": [p[0] for p in phases], "name": "Phase  "}}]
dose_spec["transform"] = [{"filter": "datum.phase === phase"}]
dose_spec["encoding"]["y"]["scale"] = {"domain": [0, 30]}
dose_spec["encoding"]["x"]["scale"] = {"domain": [0, 20]}
dose_spec["layer"][1]["encoding"]["tooltip"].append({"field": "n", "title": "player-rounds"})
specs["dose-by-phase"] = dose_spec

# 3. contribution by round
c = v.groupby(["name", "round_number"]).contribution.mean().reset_index()
c.columns = ["manager", "round", "contribution"]
cs = lines(rows(c), "round", "contribution", "round", "mean contribution", "mean contribution")
cs["encoding"]["y"]["scale"] = {"zero": False}
specs["contribution-by-round"] = cs

# 4. cumulative payoff gap to RL by round
per = v.groupby(["name", "block", "episode", "round_number"]).payoff.mean()
games = d[["name", "block", "episode"]].drop_duplicates()
rounds = pd.DataFrame({"round_number": range(24)})
full = games.merge(rounds, how="cross").set_index(["name", "block", "episode", "round_number"])
full = full.join(per).fillna(0.0)
by_round = full.groupby(["name", "round_number"]).payoff.mean().unstack(0)
gap = by_round.sub(by_round["RL (deployed)"], axis=0).cumsum()
gap = gap.drop(columns="RL (deployed)").stack().reset_index()
gap.columns = ["round", "manager", "gap"]
ends = gap[gap["round"] == 23].copy()
ends["label"] = ends.gap.map(lambda x: f"{x:+.1f}".replace("-", "−"))
zero_rule = {"mark": {"type": "rule", "color": MUTED},
             "encoding": {"y": {"datum": 0, "type": "quantitative"}, "x": {"value": 0},
                          "x2": {"value": "width"}}, "data": {"values": [{}]}}
zero_text = {"mark": {"type": "text", "color": MUTED, "align": "right", "dx": -2, "dy": -7,
                      "font": "Geist", "fontSize": 11},
             "encoding": {"x": {"value": "width"}, "y": {"datum": 0, "type": "quantitative"},
                          "text": {"value": "RL (deployed)"}},
             "data": {"values": [{}]}}
gs = lines(rows(gap), "round", "gap", "round", "cumulative payoff vs RL", "gap so far",
           extra=[zero_rule, zero_text])
gap_color = dict(color, scale={"domain": ORDER[1:], "range": COLORS[1:]})
for layer in gs["layer"][2:]:
    if "color" in layer["encoding"]:
        layer["encoding"]["color"] = gap_color
    layer["encoding"]["x"] = dict(layer["encoding"]["x"], scale={"domain": [0, 25.5]})
gs["layer"].append({"data": {"values": rows(ends)}, "mark": {"type": "text", "align": "left", "dx": 6,
                    "font": "Geist Mono", "fontSize": 11, "color": INK},
                    "encoding": {"x": {"field": "round", "type": "quantitative"},
                                 "y": {"field": "gap", "type": "quantitative"},
                                 "text": {"field": "label"}}})
specs["payoff-gap"] = gs

# 5. score per 2000-game block, and pooled
ep = full.groupby(["name", "block", "episode"]).payoff.sum().reset_index()
blocks = ep.groupby(["name", "block"]).payoff.agg(["mean", "size"]).reset_index()
blocks["platform"] = blocks.block.str.split(":").str[0]
blocks["seed"] = blocks.block.str.split(":").str[1].astype(int)
blocks = blocks.rename(columns={"name": "manager", "mean": "score", "size": "games"})
pooled = ep.groupby("name").payoff.agg(["mean", "std", "size"]).reset_index()
pooled["lo"] = pooled["mean"] - 1.96 * pooled["std"] / np.sqrt(pooled["size"])
pooled["hi"] = pooled["mean"] + 1.96 * pooled["std"] / np.sqrt(pooled["size"])
pooled = pooled.rename(columns={"name": "manager", "mean": "score", "size": "games"})
print(pooled.round(2).to_string())
y = {"field": "manager", "type": "nominal", "title": None, "sort": ORDER,
     "axis": {"labelFont": "Geist Mono", "labelColor": INK}}
specs["score-blocks"] = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "height": {"step": 46},
    "encoding": {"y": y},
    "layer": [
        {"data": {"values": rows(pooled)},
         "mark": {"type": "rule", "strokeWidth": 3, "strokeCap": "round"},
         "encoding": {"x": {"field": "lo", "type": "quantitative", "title": "average payoff", "axis": {"tickCount": 6},
                            "scale": {"zero": False, "domain": [651, 661]}},
                      "x2": {"field": "hi"}, "color": dict(color, legend=None)}},
        {"data": {"values": rows(blocks)},
         "mark": {"type": "point", "size": 70, "strokeWidth": 1.5, "yOffset": 12},
         "encoding": {"x": {"field": "score", "type": "quantitative"},
                      "color": dict(color, legend=None),
                      "shape": {"field": "platform", "title": None,
                                "scale": {"domain": ["seed 6000, Raven", "seeds 7000–7004, local",
                                                     "seeds 7000–7004, Raven"],
                                          "range": ["diamond", "circle", "square"]},
                                "legend": {"orient": "top", "labelFont": "Geist", "labelColor": INK}},
                      "tooltip": [{"field": "manager"}, {"field": "seed"}, {"field": "platform"},
                                  {"field": "score", "format": ".1f"}, {"field": "games"}]}},
        {"data": {"values": rows(pooled)},
         "mark": {"type": "tick", "thickness": 3, "size": 18},
         "encoding": {"x": {"field": "score", "type": "quantitative"},
                      "color": dict(color, legend=None),
                      "tooltip": [{"field": "manager"},
                                  {"field": "score", "title": "all fresh seeds", "format": ".2f"},
                                  {"field": "lo", "title": "95% low", "format": ".2f"},
                                  {"field": "hi", "title": "95% high", "format": ".2f"},
                                  {"field": "games"}]}},
    ],
}
OUT.write_text(json.dumps(specs))
print("wrote", OUT, OUT.stat().st_size, "bytes;", list(specs))
