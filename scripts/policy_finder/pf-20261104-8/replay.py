"""Replay the rule on reference-sim histories (zero_vs_zero, group 0 as mine) to see
how much it would punish, per member-round, compared with ah's ~1.8; for a few
param sets and over the sweep design."""
import sys
import numpy as np
import pandas as pd
import torch as th
from aimanager.manager.rule import read_rule, run_rule, rule_inputs, sobol_design

path = "configs/managers/rule_based/pf-20261104-8.yml"
rule, code = read_rule(path)
d = pd.read_parquet("plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet")
d = d[d.run.str.contains("zero_vs_zero") & (d.episode < 300)]
d["pid"] = d.participant_code.str.split("_").str[0].astype(int)
def arr(col):
    return th.tensor(d.pivot_table(index=["episode", "pid"], columns="round_number", values=col).values.reshape(-1, 8, 24), dtype=th.float)
C, V, G = arr("contribution"), arr("contribution_valid"), arr("agent_group")
P = th.zeros_like(C)
# warm up torch ops outside the builtins-free scope (first calls import lazily)
_x = th.ones(2, 3); _x.sum(-1, keepdim=True); th.clamp(_x, min=1.0); th.remainder(th.tensor(3.0), 4.0)

def mean_pun(params):
    out = []
    for t in range(24):
        inp = rule_inputs(C[..., : t + 1], V[..., : t + 1] > 0, P[..., :t], G[..., : t + 1].long(), 0)
        p = run_rule(code, params, inp).clamp(0, 30).floor()
        m = inp["in_group"][..., -1] * inp["valid"][..., -1]
        out.append(((p * m).sum() / m.sum()).item())
    return np.array(out)

for params in [dict(k=0.3, tol=3, cap=5, quiet=1, t_end=20), dict(k=1, tol=1, cap=10, quiet=0, t_end=23), dict(k=0.15, tol=4, cap=3, quiet=2, t_end=16)]:
    m = mean_pun(params)
    print(params, "mean %.2f" % m.mean(), np.round(m, 1))
design = sobol_design(rule, 256)
means = np.array([mean_pun(p).mean() for p in design[::2]])
print("design: mean punishment per member-round quantiles", np.quantile(means, [0, .1, .25, .5, .75, .9, 1]).round(2))
