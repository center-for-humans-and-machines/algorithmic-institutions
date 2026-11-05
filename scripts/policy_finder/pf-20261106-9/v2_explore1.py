# v1 final: per-opponent behaviour of our group vs opponent group, per round.
import pandas as pd, numpy as np, sys
f = sys.argv[1] if len(sys.argv) > 1 else 'plots/simulation/policy_finder/pf-20261106-9_sweep/final/per_round.parquet'
d = pd.read_parquet(f)
d = d[d.run.str.contains('_gp')]
d['pair'] = d.run.str.extract(r'managed by (.*)$')[0]
d['ours_g0'] = d.pair.str.startswith('pf-')
d['opp'] = d.pair.str.extract(r'(opponent_\d+)')[0]
d['ours'] = np.where(d.ours_g0, d.group_id == 0, d.group_id == 1)
for opp, g in d.groupby('opp'):
    print('=====', opp)
    s = g.groupby(['ours', 'round_number']).agg(n=('contribution', 'size'), c=('contribution', 'mean'),
                                               p=('punishment', 'mean')).reset_index()
    ne = g.groupby(['ours']).episode.nunique()
    # per episode count; group by run too since episode per pairing
    nep = g.groupby('ours').apply(lambda x: x[['run', 'episode']].drop_duplicates().shape[0])
    s['members'] = s.n / s.ours.map(nep)
    s['pool'] = s.members * (1.6 * s.c - s.p)
    piv = s.pivot(index='round_number', columns='ours', values=['members', 'c', 'p', 'pool']).round(2)
    print(piv.to_string())
