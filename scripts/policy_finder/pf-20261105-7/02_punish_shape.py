"""How did each manager kind punish, as a function of contribution; per round punishment."""
import pandas as pd, numpy as np
df = pd.read_csv("experiments/pilot_random1_player_round_slim.csv")
v = df[df.player_no_input==0].copy()
v["cbin"] = pd.cut(v.contribution,[-1,0,4,9,14,19,20])
print(v.groupby(["experiment_name","cbin"]).punishment.agg(["mean","count",lambda s:(s>0).mean()]).round(2))
print(v.groupby(["experiment_name","round_number"]).punishment.mean().round(2).unstack(0))
# random_1: per group, fit p = a + b*(20-c) to see the rule family
r = v[v.experiment_name=="random_1"]
for g, s in list(r.groupby("global_group_id"))[:12]:
    tab = s.groupby("contribution").punishment.agg(lambda x: sorted(set(x)))
    print(g, dict(tab))
