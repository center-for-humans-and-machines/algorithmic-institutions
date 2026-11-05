# v2: newcomers after a switch in the simulated games: their contributions, and their
# response to punishment in their first rounds, vs incumbents.
import pandas as pd, numpy as np, pyarrow.parquet as pq
fn = '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-5_sweep/final/per_round.parquet'
cols = ['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'contribution_valid', 'group_id', 'run']
pf = pq.ParquetFile(fn); out = []
for i in range(min(pf.num_row_groups, 4)):
    d = pf.read_row_group(i, columns=cols).to_pandas()
    d['run'] = d.run.astype('category').cat.codes
    d['participant_code'] = d.participant_code.astype('category').cat.codes
    d = d.sort_values(['run', 'episode', 'participant_code', 'round_number']).reset_index(drop=True)
    g = d.groupby(['run', 'episode', 'participant_code'], sort=False)
    d['pg'] = g.group_id.shift(1); d['ng'] = g.group_id.shift(-1)
    d['nc'] = g.contribution.shift(-1); d['c3'] = g.contribution.shift(-3); d['pc'] = g.contribution.shift(1)
    d = d[d.round_number.isin([4, 8, 12, 16, 20])]
    d['new'] = (d.pg != d.group_id).astype(int)
    out.append(d)
d = pd.concat(out)
pd.set_option('display.width', 250)
print('contribution at the first round of a block, newcomers vs incumbents; and their previous-round c')
print(d.groupby(['round_number', 'new'])[['contribution', 'pc']].mean().round(2).unstack())
d['pb'] = pd.cut(d.punishment, [-1, 0, 5, 10, 30]); d['cb'] = pd.cut(d.contribution, [-1, 5, 10, 15, 20])
st = d[(d.ng == d.group_id) & d.contribution_valid]
print('3-round-later change in c, by new, own c and punishment at block start')
print(st.assign(dc=st.c3 - st.contribution).pivot_table(index=['new', 'cb'], columns='pb', values='dc', aggfunc='mean', observed=True).round(2))
