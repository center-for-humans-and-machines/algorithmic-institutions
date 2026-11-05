"""v1 final: per-round contribution/punishment/payoff for the final points vs zero/humanlike."""
import pandas as pd, glob
F='plots/simulation/policy_finder/pf-20261105-7_sweep/final/per_round.parquet'
df=pd.read_parquet(F)
ref=pd.read_parquet('plots/simulation/29_EXP1_reference/per_round.parquet')
print(ref.run.unique())
df=pd.concat([df,ref])
v=df[df.contribution_valid.astype(bool)]
g=v.groupby(['run','round_number']).agg(c=('contribution','mean'),p=('punishment','mean'),pay=('payoff','mean')).reset_index()
for r in g.run.unique():
    s=g[g.run==r]
    print(r[-28:], 'total pay', round(s.pay.sum(),1))
    print('  c ', ' '.join(f'{x:4.1f}' for x in s.c))
    print('  p ', ' '.join(f'{x:4.1f}' for x in s.p))
