"""v1 final sims: per round of the rule's group: members, contribution, punishment, share punished, pool."""
import sys, pandas as pd, numpy as np
from aimanager.simulation.pool_scores import add_pool, against_anchor, group_round_scores
f = sys.argv[1] if len(sys.argv) > 1 else 'plots/simulation/policy_finder/pf-20261104-12_sweep/final/per_round.parquet'
d = pd.read_parquet(f)
add_pool(d)
print({m: (round(x.pool.mean(),2), round(x.members.mean(),2)) for m, x in against_anchor(d, 'ah').items()})
for run in d.run.unique()[:int(sys.argv[2]) if len(sys.argv) > 2 else 2]:
    s = d[d.run == run]
    print(run, s.groupby('agent_group').size().to_dict())
    rows = []
    for g in (0, 1):
        x = s[s.agent_group == g]
        r = x.groupby('round_number').agg(c=('contribution','mean'), p=('punishment','mean'), pun=('punishment', lambda v: (v>0).mean()))
        r['n'] = x.groupby(['round_number','episode']).size().groupby('round_number').sum() / s.episode.nunique()
        r['pool'] = r.n * (1.6*r.c - r.p)
        rows.append(r.add_suffix(str(g)))
    print(pd.concat(rows, axis=1).round(2).to_string())
