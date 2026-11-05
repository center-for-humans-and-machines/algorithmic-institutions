# v2: from v1 sweep per-round data: per point x opponent, my group's realised punishment,
# members, contributions (early rounds 0-7 / late 8-23), and the opponent group's.
import pandas as pd, numpy as np, re, pyarrow.parquet as pq, sys
base = sys.argv[1] if len(sys.argv) > 1 else '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-5_sweep/'
files = [x for x in sys.argv[2:] if not x.endswith('.csv')] or ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']
cols = ['episode', 'round_number', 'punishment', 'contribution', 'group_id', 'run']
def parse(r):
    m = re.search(r'managed by (.+)_vs_(.+)$', r)
    a, b = m.group(1), m.group(2)
    return (a, b, 0) if a.startswith('pf-') else (b, a, 1)
out = []
for fn in files:
    pf = pq.ParquetFile(base + fn)
    for i in range(pf.num_row_groups):
        d = pf.read_row_group(i, columns=cols).to_pandas()
        d['run'] = d.run.astype('category')
        runs = pd.DataFrame([(r,) + parse(r) for r in d.run.cat.categories], columns=['run', 'pt', 'opp', 'seat'])
        d = d.merge(runs, on='run')
        d['mine'] = (d.group_id == d.seat).astype(int)
        d['phase'] = np.select([d.round_number < 4, d.round_number < 8, d.round_number < 16], [0, 1, 2], 3)
        a = d.groupby(['pt', 'opp', 'mine', 'phase']).agg(p=('punishment', 'sum'), c=('contribution', 'sum'), n=('punishment', 'size')).reset_index()
        e = d.groupby(['pt', 'opp'])[['run', 'episode']].apply(lambda x: x.drop_duplicates().shape[0]).rename('neps').reset_index()
        out.append(a.merge(e, on=['pt', 'opp']))
        print(fn, i, flush=True)
a = pd.concat(out)
a = a.groupby(['pt', 'opp', 'mine', 'phase']).sum().reset_index()
a.to_csv(sys.argv[-1] if sys.argv[-1].endswith('.csv') else '/tmp/v1agg.csv', index=False)
