"""v1 sweep, players starting in the rule's group: later contribution (rounds 8-23) and leaving at
round 4 regressed on own punishment in each of rounds 0-3, controlling own contributions in rounds
0-3 and the group's mean contribution in round 0. Which round's point buys most lift per departure?"""
import pandas as pd, numpy as np, pyarrow.parquet as pq, statsmodels.formula.api as smf
base = 'plots/simulation/policy_finder/pf-20261104-11_sweep/'
out = []
for f in ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']:
    d = pq.read_table(base + f, columns=['episode', 'participant_code', 'round_number', 'group_id', 'contribution', 'punishment', 'run'], read_dictionary=['run', 'participant_code']).to_pandas()
    d['pt'] = d.run.cat.codes.astype(np.int32) + (0 if 'p1' in f else 1000)
    d['pc'] = d.participant_code.cat.codes
    d = d.drop(columns=['run', 'participant_code'])
    k = ['pt', 'episode', 'pc']
    e = d[d.round_number <= 3].pivot_table(index=k, columns='round_number', values=['contribution', 'punishment', 'group_id'])
    e.columns = [f'{a[0][0]}{a[1]}' for a in e.columns]
    e = e[(e[['g0', 'g1', 'g2', 'g3']] == 0).all(1)]
    e['g4'] = d[d.round_number == 4].set_index(k).group_id
    e['c_late'] = d[d.round_number >= 8].groupby(k).contribution.mean()
    e['c_47'] = d[(d.round_number >= 4) & (d.round_number < 8)].groupby(k).contribution.mean()
    e['gm0'] = e.groupby(['pt', 'episode']).c0.transform('mean')
    out.append(e.reset_index())
X = pd.concat(out)
X['left'] = (X.g4 == 1).astype(float)
for y in ['c_47', 'c_late', 'left']:
    m = smf.ols(f'{y} ~ p0 + p1 + p2 + p3 + c0 + c1 + c2 + c3 + gm0', X).fit()
    print(y, m.params.round(4).to_dict())
# stayers only for c_late
m = smf.ols('c_late ~ p0 + p1 + p2 + p3 + c0 + c1 + c2 + c3 + gm0', X[X.left == 0]).fit()
print('c_late stayers', m.params.round(4).to_dict())
print('sequential: effect of p_k controlling only for what precedes it (c_0..c_k, p_0..p_k-1)')
for k in range(4):
    ctr = ' + '.join([f'c{j}' for j in range(k + 1)] + [f'p{j}' for j in range(k)] + ['gm0'])
    for y in ['c_late', 'left']:
        m = smf.ols(f'{y} ~ p{k} + {ctr}', X).fit()
        print(k, y, round(m.params[f'p{k}'], 4), '+-', round(m.bse[f'p{k}'], 4))
