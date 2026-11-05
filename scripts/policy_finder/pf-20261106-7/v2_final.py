# v2: per-round view of v1's final runs: my group vs opponent group, per opponent
import pandas as pd, numpy as np, sys
f = sys.argv[1] if len(sys.argv)>1 else '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-7_sweep/final/per_round.parquet'
point = sys.argv[2] if len(sys.argv)>2 else 'pf-20261106-7_gp'
d = pd.read_parquet(f)
d = d[d.run.str.contains(point+'_vs_') | d.run.str.endswith('_vs_'+point)]
d['mine_seat'] = np.where(d.run.str.contains(point+'_vs_'), 0, 1)
d['opp'] = d.run.str.extract(r'(opponent_\d+)')[0]
d['mine'] = (d.group_id.astype(str).str.extract(r'(\d)')[0].astype(int) == d.mine_seat) if d.group_id.dtype==object else (d.group_id==d.mine_seat)
print(d.group_id.unique()[:5])
for opp, g in d.groupby('opp'):
    r = g.groupby(['round_number','mine']).agg(n=('contribution','size'), c=('contribution','mean'), p=('punishment','mean')).reset_index()
    neps = g.groupby(['run','episode']).ngroups
    r['n'] = r.n/neps
    t = r.pivot(index='round_number', columns='mine', values=['n','c','p']).round(2)
    print('==', opp); print(t.to_string())
