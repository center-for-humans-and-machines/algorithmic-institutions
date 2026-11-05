# v4: per point and opponent (v3 sweep): punishment dealt per member-round by my rule in rounds 0-11,
# members after the first decision, pool; dose-response curve per opponent
import pandas as pd, numpy as np, pyarrow.parquet as pq, re
base = '/opt/algorithmic-institutions/plots/simulation/policy_finder/pf-20261106-5-v3_sweep/'
cols = ['episode', 'round_number', 'punishment', 'contribution', 'group_id', 'run']
rows = []
for f in ['per_round_p1of2.parquet', 'per_round_p2of2.parquet']:
    tb = pq.read_table(base + f, columns=cols, read_dictionary=['run'])
    d = tb.to_pandas(); del tb
    cats = d.run.cat.categories
    info = []
    for r in cats:
        a, b = re.search(r'managed by (.+)_vs_(.+)$', r).groups()
        info.append((a, b, 0) if a.startswith('pf-') else (b, a, 1))
    d['run'] = d.run.cat.codes
    d['seat'] = np.array([i[2] for i in info])[d.run.values]
    d['mine'] = (d.group_id == d.seat)
    d['early'] = d.round_number < 12
    d['pe'] = d.punishment * d.early * d.mine
    d['ne'] = d.early * d.mine
    d['pool'] = (1.6 * d.contribution - d.punishment) * d.mine
    d['opp_pe'] = d.punishment * d.early * (~d.mine)
    d['m4'] = d.mine * (d.round_number == 4)
    g = d.groupby('run')[['pe', 'ne', 'pool', 'm4', 'opp_pe']].sum()
    ne_tot = d.groupby('run').size()
    g['episodes'] = d.groupby('run').episode.nunique()
    del d
    for r, x in g.iterrows():
        rows.append(dict(point=info[r][0], opp=info[r][1], seat=info[r][2], pe=x['pe'], ne=x['ne'], pool=x['pool'] / 24 / x['episodes'],
                         m4=x['m4'] / x['episodes'], opp_pe=x['opp_pe'], eps=x['episodes']))
D = pd.DataFrame(rows); D.to_csv('/tmp/w/dose_raw.csv', index=False)
A = D.groupby(['point', 'opp']).agg(pe=('pe', 'sum'), ne=('ne', 'sum'), pool=('pool', 'mean'), m4=('m4', 'mean')).reset_index()
A['ppm'] = A['pe'] / A['ne']
A.to_csv('/tmp/w/dose.csv', index=False)
A['bin'] = pd.cut(A.ppm, [-0.01, 0.5, 1, 2, 3, 4, 5, 6, 8, 20])
pd.set_option('display.width', 250)
print(A.pivot_table(index='bin', columns='opp', values='pool', aggfunc='mean', observed=True).round(1))
print(A.pivot_table(index='bin', columns='opp', values='m4', aggfunc='mean', observed=True).round(2))
print(A.pivot_table(index='bin', columns='opp', values='pool', aggfunc='size', observed=True))
