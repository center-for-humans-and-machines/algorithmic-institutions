"""Per-round summary of v3's final run: members, contribution/member, punishment/member, pool, both groups."""
import pandas as pd, sys
fn = sys.argv[1] if len(sys.argv) > 1 else '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261104-9-v3_sweep/final/per_round.parquet'
d = pd.read_parquet(fn, columns=['run', 'episode', 'round_number', 'group_id', 'contribution', 'punishment', 'contribution_valid'])
print(d.run.unique())
d = d[d.run == d.run.unique()[0]]
d['ci'] = d.contribution * d.contribution_valid
g = d.groupby(['episode', 'round_number', 'group_id']).agg(n=('ci', 'size'), c=('ci', 'sum'), p=('punishment', 'sum')).reset_index()
g['pool'] = 1.6 * g.c - g.p
r = g.groupby(['group_id', 'round_number'])[['n', 'c', 'p', 'pool']].mean()
r['c_m'] = r.c / r.n; r['p_m'] = r.p / r.n
pd.set_option('display.width', 200)
print(r.unstack(0).round(2)[['n', 'c_m', 'p_m', 'pool']].to_string())
