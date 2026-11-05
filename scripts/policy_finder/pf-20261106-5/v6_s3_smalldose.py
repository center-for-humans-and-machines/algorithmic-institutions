# v6: dose-response of the round-0 punishment, esp. small doses (1-5): next-round and rounds 9-11 contribution, staying
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/'
pd.set_option('display.width', 250)
files = sys.argv[1:]
out = []
for f in files:
    d = pq.read_table(base + f, columns=['episode', 'participant_code', 'round_number', 'punishment', 'contribution', 'group_id', 'run'],
                      filters=[('episode', '<', 16), ('round_number', 'in', [0, 1, 4, 9, 10, 11])]).to_pandas()
    d['seat'] = np.where(d.run.str.contains('managed by pf-'), 0, 1)
    w = d.pivot_table(index=['run', 'episode', 'participant_code', 'seat'], columns='round_number', values=['contribution', 'punishment', 'group_id']).reset_index()
    w.columns = [a if b == '' else f'{a[0]}{b}' for a, b in w.columns]
    w = w[w.g0 == w.seat]
    out.append(w)
w = pd.concat(out)
w['stay'] = (w.g4 == w.g0).astype(float)
w['c911'] = w[['c9', 'c10', 'c11']].mean(axis=1)
w['cb'] = pd.cut(w.c0, [-1, 2, 5, 8, 11, 14, 20])
w['pb'] = pd.cut(w.p0, [-1, 0, 2, 5, 9, 14, 20, 30])
print(w.groupby(['cb', 'pb'], observed=True).agg(n=('c1', 'size'), c1=('c1', 'mean'), stay=('stay', 'mean')).unstack('pb').round(2).to_string())
print(w[w.stay == 1].groupby(['cb', 'pb'], observed=True).c911.mean().unstack('pb').round(2).to_string())
