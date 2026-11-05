# v6: distribution of my group's size per block (incl. empty), per opponent, v5 final
import pandas as pd, numpy as np, sys
f = sys.argv[1] if len(sys.argv) > 1 else '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-11-v5_sweep/final/per_round.parquet'
df = pd.read_parquet(f)
df = df[df.run.str.contains('_gp')]
def parse(r):
    a, b = r.split('managed by ')[1].split('_vs_')
    return (b, 0) if a.startswith('pf-') else (a, 1)
m = {r: parse(r) for r in df.run.unique()}
df['opp'] = df.run.map(lambda r: m[r][0]); df['myg'] = df.run.map(lambda r: m[r][1])
df['mine'] = (df.group_id == df.myg).astype(int)
g = df.groupby(['opp','run','episode','round_number']).mine.sum().reset_index()
g['blk'] = g.round_number//4
t = g[g.round_number%4==0].groupby(['opp','blk']).mine.value_counts(normalize=True).unstack().fillna(0).round(3)
print(t.to_string())
