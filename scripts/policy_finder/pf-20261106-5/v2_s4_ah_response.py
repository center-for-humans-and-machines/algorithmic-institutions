# v2: how do the artificial humans respond to punishment, in the simulated games?
# Uses one sweep part file (row groups chunked) - all managers, both groups.
import pandas as pd, numpy as np, pyarrow.parquet as pq, sys
fn = sys.argv[1] if len(sys.argv) > 1 else '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-5_sweep/final/per_round.parquet'
cols = ['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'contribution_valid', 'group_id', 'run']
pf = pq.ParquetFile(fn)
resp, sw, pers = [], [], []
for i in range(min(pf.num_row_groups, 4)):
    d = pf.read_row_group(i, columns=cols).to_pandas()
    d['run'] = d.run.astype('category').cat.codes
    d['participant_code'] = d.participant_code.astype('category').cat.codes
    d = d.sort_values(['run', 'episode', 'participant_code', 'round_number']).reset_index(drop=True)
    g = d.groupby(['run', 'episode', 'participant_code'], sort=False)
    d['nc'] = g.contribution.shift(-1)
    d['c4'] = g.contribution.shift(-4)
    d['ng'] = g.group_id.shift(-1)
    gr = d.groupby(['run', 'episode', 'round_number', 'group_id'])
    d['gp'] = gr.punishment.transform('mean'); d['gc'] = gr.contribution.transform('mean'); d['gn'] = gr.punishment.transform('size')
    # other group's mean punishment this round
    tot = d.groupby(['run', 'episode', 'round_number']).punishment.transform('sum')
    d['op'] = (tot - d.gp * d.gn) / (8 - d.gn).clip(lower=1)
    # block-summed own punishment (4 rounds ending at decision)
    d['blk'] = d.round_number // 4
    d['bp'] = d.groupby(['run', 'episode', 'participant_code', 'blk']).punishment.transform('sum')
    d['bgp'] = d.groupby(['run', 'episode', 'participant_code', 'blk']).gp.transform('mean')
    d['bop'] = d.groupby(['run', 'episode', 'participant_code', 'blk']).op.transform('mean')
    d['bc'] = d.groupby(['run', 'episode', 'participant_code', 'blk']).contribution.transform('mean')
    st = d[(d.ng == d.group_id) & d.contribution_valid & d.nc.notna()]
    resp.append(st[['contribution', 'punishment', 'nc', 'c4', 'gc', 'round_number']])
    dec = d[d.round_number.isin([3, 7, 11, 15, 19])]
    sw.append(dec.assign(sw=(dec.ng != dec.group_id).astype(int))[['sw', 'bp', 'bgp', 'bop', 'bc', 'round_number', 'gn']])
r = pd.concat(resp); s = pd.concat(sw)
r['cb'] = pd.cut(r.contribution, [-1, 0, 5, 10, 15, 19, 20])
r['pb'] = pd.cut(r.punishment, [-1, 0, 2, 5, 10, 30])
r['dc'] = r.nc - r.contribution; r['dc4'] = r.c4 - r.contribution
pd.set_option('display.width', 250)
print('next-round change in contribution (stayers), by own c and own punishment')
print(r.pivot_table(index='cb', columns='pb', values='dc', aggfunc='mean', observed=True).round(2))
print(r.pivot_table(index='cb', columns='pb', values='dc', aggfunc='size', observed=True))
print('change after 4 rounds')
print(r.pivot_table(index='cb', columns='pb', values='dc4', aggfunc='mean', observed=True).round(2))
r['dev'] = pd.cut(r.contribution - r.gc, [-21, -8, -4, -1, 1, 4, 21])
print('next change by deviation from group mean and punishment')
print(r.pivot_table(index='dev', columns='pb', values='dc', aggfunc='mean', observed=True).round(2))
print('switch rate by block own punishment sum')
s['bpb'] = pd.cut(s.bp, [-1, 0, 4, 8, 20, 40, 200])
print(s.groupby('bpb', observed=True).sw.agg(['mean', 'size']).round(3))
s['rel'] = pd.cut(s.bgp - s.bop, [-30, -3, -1, -0.25, 0.25, 1, 3, 30])
print('switch rate by own group mean pun - other group mean pun (block mean)')
print(s.groupby('rel', observed=True).sw.agg(['mean', 'size']).round(3))
s['bcb'] = pd.cut(s.bc, [-1, 3, 7, 11, 15, 20])
print(s.pivot_table(index='bcb', columns='bpb', values='sw', aggfunc='mean', observed=True).round(3))
import statsmodels.formula.api as smf
m = smf.logit('sw ~ bp + bgp + bop + bc + C(round_number) + gn', s.sample(min(len(s), 300000), random_state=0)).fit(disp=0)
print(m.params.round(3))
