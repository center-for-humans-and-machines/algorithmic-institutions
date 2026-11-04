"""v1 sweep, player level: players who start in the rule's group; their early punishment (rounds 0-2)
by their round-0 contribution rank, vs later contribution (rounds 8-23, wherever they are), leaving at
round 4; also the spillover: unpunished players' later contribution vs the dose dealt to the others."""
import pandas as pd, numpy as np, pyarrow.parquet as pq, statsmodels.formula.api as smf
base = 'plots/simulation/policy_finder/pf-20261104-11_sweep/'
out = []
for f in ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']:
    d = pq.read_table(base + f, columns=['episode', 'participant_code', 'round_number', 'group_id', 'contribution', 'punishment', 'run'], read_dictionary=['run', 'participant_code']).to_pandas()
    d['pt'] = d.run.cat.codes.astype(np.int32) + (0 if 'p1' in f else 1000)
    d['pc'] = d.participant_code.cat.codes
    d = d.drop(columns=['run', 'participant_code'])
    d = d[d.round_number.isin([0, 1, 2, 3, 4]) | (d.round_number >= 8)]
    k = ['pt', 'episode', 'pc']
    r0 = d[d.round_number == 0].set_index(k)
    r0 = r0[r0.group_id == 0][['contribution']].rename(columns={'contribution': 'c0'})
    e = d[d.round_number <= 2].groupby(k).agg(p02=('punishment', 'sum'), c02=('contribution', 'mean'))
    r4 = d[d.round_number == 4].set_index(k).group_id.rename('g4')
    late = d[d.round_number >= 8].groupby(k).contribution.mean().rename('c_late')
    x = r0.join(e).join(r4).join(late)
    # group dose to others
    x['gp'] = x.groupby(['pt', 'episode']).p02.transform('sum') - x.p02
    x['rank'] = x.groupby(['pt', 'episode']).c0.rank(method='average', ascending=False)
    out.append(x.reset_index())
X = pd.concat(out)
X['left'] = (X.g4 == 1).astype(float)
X['cb'] = pd.cut(X.c0, [-1, 0, 5, 10, 15, 20])
X['pb'] = pd.cut(X.p02 / 3, [-1, 0, 1, 3, 6, 10, 30])
X['gb'] = pd.cut(X.gp / 3, [-1, 0.5, 3, 8, 15, 90])
pd.set_option('display.width', 250)
print(X.groupby(['cb', 'pb'], observed=True)[['c_late', 'left']].agg(['mean', 'count']).round(2).to_string())
print('unpunished players, by dose to the others (per round)')
print(X[X.p02 == 0].groupby(['cb', 'gb'], observed=True)[['c_late', 'left']].agg(['mean', 'count']).round(2).to_string())
m = smf.ols('c_late ~ C(cb) + C(cb):p02 + gp + c02', X).fit()
print(m.params.round(4))
