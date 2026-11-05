# v3: both groups' pools per round block in v2's final run, per opponent
import pandas as pd, numpy as np, sys
f = sys.argv[1] if len(sys.argv)>1 else '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-7-v2_sweep/final/per_round.parquet'
point = sys.argv[2] if len(sys.argv)>2 else 'pf-20261106-7-v2_gp'
d = pd.read_parquet(f, columns=['run','episode','round_number','group_id','contribution','punishment'])
d = d[d.run.str.contains(point)]
d['seat'] = np.where(d.run.str.contains(point+'_vs_'), 0, 1)
d['opp'] = d.run.str.extract(r'(opponent_\d+)')[0]
d['mine'] = d.group_id == d.seat
d['pool'] = 1.6*d.contribution - d.punishment
g = d.groupby(['opp','run','episode','round_number','mine']).pool.sum().unstack('mine').fillna(0)
g = g.reset_index(); g['block'] = g.round_number//4
print(g.groupby(['opp','block'])[[False, True]].mean().unstack('block').round(1).to_string())
print(g.groupby(['opp'])[[False, True]].mean().round(2))
