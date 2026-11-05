"""v2: who pays the punishment bill in v1's final best point, by round phase and contribution bin;
and next-round response of each bin (punished vs not), during and after punishment."""
import pandas as pd, numpy as np
d = pd.read_parquet('plots/simulation/policy_finder/pf-20261105-11_sweep/final/per_round.parquet')
d = d[d.run.str.endswith('_gp')].sort_values(['episode', 'participant_code', 'round_number'])
d['c_next'] = d.groupby(['episode', 'participant_code']).contribution.shift(-1)
d['v_next'] = d.groupby(['episode', 'participant_code']).contribution_valid.shift(-1)
v = d[d.contribution_valid]
v = v.assign(bin=pd.cut(v.contribution, [-1, 0, 4, 9, 12, 14, 15, 17, 19, 20]),
             phase=pd.cut(v.round_number, [-1, 0, 2, 5, 9, 13, 18, 23]))
tot = v.groupby('phase').punishment.sum()
b = v.groupby(['phase', 'bin']).agg(share=('contribution', 'size'), p=('punishment', 'mean'), bill=('punishment', 'sum'))
b['share'] = b.share / b.groupby(level=0).share.transform('sum')
b['bill_share'] = b.bill / b.groupby(level=0).bill.transform('sum')
print(b[['share', 'p', 'bill_share']].round(3).to_string())
w = v[v.v_next == True]
w = w.assign(dc=w.c_next - w.contribution, pun=w.punishment > 0)
print(w.groupby(['phase', 'bin', 'pun']).dc.agg(['mean', 'size']).round(2).unstack('pun').to_string())
