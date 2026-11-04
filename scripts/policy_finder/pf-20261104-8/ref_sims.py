"""Per-side pool, members and contribution in the reference sims (ah vs zero)."""
import sys
import pandas as pd
sys.path.insert(0, "src")
from aimanager.simulation.pool_scores import add_pool, against_anchor, group_round_scores

d = pd.read_parquet("plots/simulation/25_LEVIN_run1_ah_zero_pairings_batched/per_round.parquet")
add_pool(d)
res = against_anchor(d, "ah")
for m, f in res.items():
    print(m, "pool %.2f +- %.2f members %.2f" % (f.pool.mean(), f.pool.std() / len(f) ** .5, f.members.mean()))

# per-round, per-side for zero_vs_ah (group 0 = zero, group 1 = ah)
for run in d.run.unique():
    sub = d[d.run == run]
    g = sub.groupby(["round_number", "agent_group"]).agg(c=("contribution", "mean"), p=("punishment", "mean"))
    n = sub.groupby(["episode", "round_number", "agent_group"]).size().groupby(["round_number", "agent_group"]).mean()
    pool = group_round_scores(sub, "pool", "sum").groupby("round_number").mean()
    print(run)
    t = g.unstack()
    t["n0"] = n.unstack()[0]; t["n1"] = n.unstack()[1]
    t["pool0"] = pool[0]; t["pool1"] = pool[1]
    print(t.round(2).to_string())
