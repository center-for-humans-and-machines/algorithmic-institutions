# v3: per-round trajectory of a final candidate vs each opponent: my group / their group
# members, contribution per member, punishment per member, pool (1.6*sum c - sum p)
import pandas as pd, numpy as np, re, sys
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
ver, cand = sys.argv[1], sys.argv[2]
d = pd.read_parquet(base + ver + '_sweep/final/per_round.parquet',
                    columns=['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'group_id', 'run'])
d = d[d.run.str.contains(cand + '_vs_') | d.run.str.endswith('_vs_' + cand)].copy()
def parse(r):
    a, b = re.search(r'managed by (.+)_vs_(.+)$', r).groups()
    return (b, 0) if a.startswith('pf-') else (a, 1)
m = {r: parse(r) for r in d.run.unique()}
d['opp'] = d.run.map(lambda r: m[r][0]); d['seat'] = d.run.map(lambda r: m[r][1])
d['mine'] = (d.group_id == d.seat).astype(int)
g = d.groupby(['opp', 'run', 'episode', 'round_number', 'mine']).agg(n=('contribution', 'size'), c=('contribution', 'sum'), p=('punishment', 'sum')).reset_index()
g['pool'] = 1.6 * g.c - g.p
# fill empty groups
full = g.set_index(['opp', 'run', 'episode', 'round_number', 'mine']).unstack('mine', fill_value=0).stack('mine', future_stack=True).fillna(0).reset_index()
r = full.groupby(['opp', 'mine', 'round_number'])[['n', 'c', 'p', 'pool']].mean()
r['cpm'] = r.c / r.n; r['ppm'] = r.p / r.n
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
for o in sorted(d.opp.unique()):
    x = r.loc[o]
    t = pd.concat([x.loc[1][['n', 'cpm', 'ppm', 'pool']].add_prefix('me_'), x.loc[0][['n', 'cpm', 'ppm', 'pool']].add_prefix('op_')], axis=1)
    print(o, 'mean pool me', round(x.loc[1].pool.mean(), 2), 'them', round(x.loc[0].pool.mean(), 2))
    print(t.round(2).T.to_string())
