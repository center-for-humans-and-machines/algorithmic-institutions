# v6: in the v5 final, compare my group and the opponent's group per opponent:
# pool, members, contribution per head, punishment per head, per round.
import pandas as pd, numpy as np, sys
f = sys.argv[1] if len(sys.argv) > 1 else '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-v5_sweep/final/per_round.parquet'
df = pd.read_parquet(f)
df = df[df.run.str.contains('_gp')]
def parse(r):
    s = r.split('managed by ')[1]
    a, b = s.split('_vs_')
    return (b, 0) if a.startswith('pf-') else (a, 1)
m = {r: parse(r) for r in df.run.unique()}
df['opp'] = df.run.map(lambda r: m[r][0]); df['myg'] = df.run.map(lambda r: m[r][1])
print(df.group_id.unique()[:5], df.agent_group.unique()[:5] if 'agent_group' in df else '')
df['g'] = df.group_id.astype(str)
# figure group label
df['mine'] = (df.g.str[-1].astype(int) == df.myg) if df.g.str[-1].str.isdigit().all() else None
g = df.groupby(['opp','mine','run','episode','round_number']).agg(n=('contribution','size'), c=('contribution','sum'), p=('punishment','sum')).reset_index()
g['pool'] = 1.6*g.c - g.p
s = g.groupby(['opp','mine']).agg(pool=('pool','mean'), n=('n','mean'), c=('c','mean'), p=('p','mean'))
s['c_head'] = s.c/s.n; s['p_head'] = s.p/s.n
print(s.round(2))
r = g.groupby(['opp','mine','round_number']).agg(pool=('pool','mean'), n=('n','mean'), c=('c','sum'), p=('p','sum'), k=('n','sum')).reset_index()
r['ch'] = r.c/r.k; r['ph'] = r.p/r.k
for o in sorted(r.opp.unique()):
    x = r[r.opp==o].pivot(index='round_number', columns='mine', values=['pool','n','ch','ph']).round(1)
    print(o); print(x.to_string())
